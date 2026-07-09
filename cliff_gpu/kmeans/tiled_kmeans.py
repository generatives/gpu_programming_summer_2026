import time

import numpy as np
import warp as wp

wp.config.enable_backward = False

BLOCK_SIZE = 512
MAX_DATA_SIZE = 4096
NUM_CLUSTERS = 3
MAX_ITERATIONS = 100
PROGRESS_TOLERANCE = 1e-4

@wp.func
def sq_length(v: wp.vec2f):
    return wp.dot(v, v)

@wp.func
def index_if_above(val: float, idx: int, threshold: float):
    if val > threshold:
        return idx
    return 0x7FFFFFFF   # sentinel: larger than any real index

@wp.func
def first_idx_above(data_tile: wp.tile[wp.float32, MAX_DATA_SIZE], threshold: float):
    idx_tile = wp.tile_arange(MAX_DATA_SIZE, dtype=int)
    candidate_indices = wp.tile_map(index_if_above, data_tile, idx_tile, threshold)
    first_idx = wp.tile_min(candidate_indices)   # smallest index where condition held, or sentinel if none did
    return first_idx

@wp.kernel
def kmean_iteration(X: wp.array[wp.vec2f],
                centroids: wp.array[wp.vec2f],
                offsets: wp.array[wp.int32],
                n_clusters: wp.int32):
    
    block_id, i = wp.tid()
    model_idx = block_id

    data_start = offsets[model_idx]
    data_end = offsets[model_idx+1]
    actual_data_size = data_end - data_start

    model_offset = model_idx * n_clusters

    model_centroids = wp.tile_load(centroids, shape=(NUM_CLUSTERS,), offset=(model_offset,))

    data_tile = wp.tile_empty((MAX_DATA_SIZE,), dtype=wp.vec2f)
    label_tile = wp.tile_empty((MAX_DATA_SIZE,), dtype=wp.int8)

    # load the needed data into shared memory
    tile_data_idx = i
    while tile_data_idx < MAX_DATA_SIZE:
        buffer_data_idx = tile_data_idx + data_start
        data_tile[tile_data_idx] = X[buffer_data_idx] if buffer_data_idx < data_end else wp.vec2f(0.0, 0.0)
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    iteration = int(0)
    while iteration < MAX_ITERATIONS:
        # label each data point
        tile_data_idx = i
        while tile_data_idx < MAX_DATA_SIZE:
            min_sq_dist = float(-1.0)
            label = int(-1)
            for j in range(n_clusters):
                diff = data_tile[tile_data_idx] - model_centroids[j]
                sq_dist = wp.dot(diff, diff)
                if sq_dist < min_sq_dist or label == -1:
                    min_sq_dist = sq_dist
                    label = j
            label_tile[tile_data_idx] = label
            
            tile_data_idx = tile_data_idx + BLOCK_SIZE

        # accumulate updated centroids and counts
        new_model_centroids = wp.tile_zeros((NUM_CLUSTERS,), dtype=wp.vec2f)
        new_model_counts = wp.tile_zeros((NUM_CLUSTERS,), dtype=wp.float32)

        tile_data_idx = i
        while tile_data_idx < MAX_DATA_SIZE:
            label = wp.int32(label_tile[tile_data_idx])
            data = data_tile[tile_data_idx]
            wp.tile_scatter_add(new_model_centroids, label, data, has_value=tile_data_idx<actual_data_size, atomic=True)
            wp.tile_scatter_add(new_model_counts, label, 1.0, has_value=tile_data_idx<actual_data_size, atomic=True)
            
            tile_data_idx = tile_data_idx + BLOCK_SIZE

        # normalize centroids
        new_model_centroids = new_model_centroids / new_model_counts

        # frobenius norm of the difference
        centroid_diff = model_centroids - new_model_centroids
        centroid_vec_sq_length = wp.tile_map(sq_length, centroid_diff)
        sum_sq_centroid_diff = wp.tile_sum(centroid_vec_sq_length)
        
        progress = wp.sqrt(sum_sq_centroid_diff[0])
        if progress < PROGRESS_TOLERANCE:
            wp.tile_store(centroids, model_centroids, offset=(model_offset,))
            break
        else:
            model_centroids = new_model_centroids
            iteration = iteration + 1


class TiledKMeans:
    def __init__(self, n_models, n_clusters, max_iter=300, tol=1e-4, verbose=False):
        self.n_models = n_models
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None
        self.labels = None

    def _single_kinit(self, X, model_idx):
        # Randomly initialize with well spaced selections
        n_samples = X.shape[0]
        min_distances = np.full(n_samples, np.inf)

        centroid_offset = self.n_clusters * model_idx

        # Remaining centroids: weighted by distance to nearest existing centroid
        for i in range(0, self.n_clusters):
            if i == 0:
                # First centroid: random selection
                first_idx = np.random.randint(n_samples)
                new_centroid = X[first_idx]
                self.centroids[centroid_offset + i, :] = new_centroid
            else:
                weights = min_distances / min_distances.sum()
                
                # Select next centroid
                next_idx = np.random.choice(n_samples, p=weights)
                new_centroid = X[next_idx]
                self.centroids[centroid_offset + i, :] = new_centroid
            
            distances = np.linalg.norm(X - new_centroid, axis=1) ** 2
            np.minimum(min_distances, distances, out=min_distances)

    def _kinit(self, X, offsets):
        num_centroids = self.n_models * self.n_clusters
        self.centroids = np.empty((num_centroids, X.shape[1]))

        for i in range(self.n_models):
            data = X[offsets[i]:offsets[i+1]]
            data_len = data.shape[0]
            if data_len > 1000:
                self._single_kinit(data[np.random.choice(data_len, 1000, replace=False)], i)
            else:
                self._single_kinit(data, i)

    def fit(self, X, model_idx_lookup, offsets):

        assert X.ndim == 2, "Input data must be a 2D array"
        assert X.shape[1] == 2, "Input data must have 2 features (2D points)"
        assert model_idx_lookup.shape[0] == X.shape[0], "model_idx_lookup should have the same length as the dataset"

        #start_time = time.time()
        self._kinit(X, offsets)
        #end_time = time.time()
        #print(f"Init took: {end_time - start_time}s")
        self.labels = []

        x_buffer = wp.array(X, dtype=wp.vec2f, device="cuda")
        centroids_buffer = wp.array(self.centroids, dtype=wp.vec2f, device="cuda")
        offsets_buffer = wp.array(offsets, dtype=wp.int32, device="cuda")

        wp.launch_tiled(kmean_iteration,
                        dim=self.n_models,
                        inputs=[x_buffer, centroids_buffer, offsets_buffer, self.n_clusters],
                        block_dim=BLOCK_SIZE)

        self.centroids = centroids_buffer.numpy()