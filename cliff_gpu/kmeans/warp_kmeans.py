import time

import numpy as np
import warp as wp
from warp.utils import array_sum, array_scan

@wp.kernel
def assign_labels(X: wp.array[wp.vec2f],
                centroids: wp.array[wp.vec2f],
                labels: wp.array[wp.int32],
                model_idx_lookup: wp.array[wp.int32],
                n_clusters: wp.int32):
    i = wp.tid()
    model_idx = model_idx_lookup[i]
    model_offset = model_idx * n_clusters
    if i < X.shape[0]:
        min_sq_dist = float(-1.0)
        label = int(-1)
        for j in range(n_clusters):
            diff = X[i] - centroids[model_offset + j]
            sq_dist = wp.dot(diff, diff)
            if sq_dist < min_sq_dist or label == -1:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

@wp.kernel
def accumulate_centroids(X: wp.array[wp.vec2f],
                    labels: wp.array[wp.int32],
                    centroids: wp.array[wp.vec2f],
                    counts: wp.array[wp.int32],
                    model_idx_lookup: wp.array[wp.int32],
                    n_clusters: wp.int32):
    i = wp.tid()
    model_idx = model_idx_lookup[i]
    model_offset = model_idx * n_clusters
    if i < X.shape[0]:
        label = labels[i]
        wp.atomic_add(counts, model_offset + label, 1)
        wp.atomic_add(centroids, model_offset + label, X[i])

@wp.kernel
def normalize_centroids(centroids: wp.array[wp.vec2f],
                    counts: wp.array[wp.int32]):
    i = wp.tid()
    if i < centroids.shape[0]:
        if counts[i] > 0:
            centroids[i] = centroids[i] / wp.float32(counts[i])
        

class WarpKMeans:
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
        labels_buffer = wp.array(np.zeros(X.shape[0], dtype=wp.int32), dtype=wp.int32, device="cuda")
        counts_buffer = wp.array(np.zeros(self.n_models * self.n_clusters, dtype=wp.int32), dtype=wp.int32, device="cuda")
        model_idx_lookup_buffer = wp.array(model_idx_lookup, dtype=wp.int32, device="cuda")

        for i in range(self.max_iter):
            start_time = time.time()
            #print(f"Iteration: {i}")
            # Assign clusters based on closest centroid
            wp.launch(assign_labels, dim=X.shape[0], inputs=[x_buffer, centroids_buffer, labels_buffer, model_idx_lookup_buffer, self.n_clusters])

            centroids_buffer.zero_()
            counts_buffer.zero_()
            wp.launch(accumulate_centroids, dim=X.shape[0], inputs=[x_buffer, labels_buffer, centroids_buffer, counts_buffer, model_idx_lookup_buffer, self.n_clusters])
            
            #print(f"Centroids: {centroids_buffer.numpy()}")
            #print(f"Labels: {centroids_buffer.numpy()}")

            wp.launch(normalize_centroids, dim=self.centroids.shape[0], inputs=[centroids_buffer, counts_buffer])

            #print(f"Centroids: {centroids_buffer.numpy()}")
            #print(f"Labels: {centroids_buffer.numpy()}")

            new_centroids = centroids_buffer.numpy()
            end_time = time.time()
            #print(f"Compute Took: {end_time - start_time}")

            start_time = time.time()
            max_progress = -np.inf
            for i in range(self.n_models):
                centroids_start = i * self.n_clusters
                centroids_end = centroids_start + self.n_clusters
                progress = np.linalg.norm(new_centroids[centroids_start:centroids_end] - self.centroids[centroids_start:centroids_end])
                max_progress = max(progress, max_progress)
            end_time = time.time()
            #print(f"Measure Took: {end_time - start_time}")
            
            should_stop = max_progress < self.tol
            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1 or should_stop):
                print(f"Iteration: {i}: Max Progress: {max_progress}")

            if should_stop:
                break

            self.centroids = new_centroids