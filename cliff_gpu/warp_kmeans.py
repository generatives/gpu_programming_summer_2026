import time

import numpy as np
import warp as wp


@wp.kernel
def assign_labels(X: wp.array[wp.vec2f],
                centroids: wp.array[wp.vec2f],
                labels: wp.array[wp.int32]):
    i = wp.tid()
    if i < X.shape[0]:
        min_dist = float(-1.0)
        label = int(-1)
        for j in range(centroids.shape[0]):
            dist = wp.dot(X[i] - centroids[j], X[i] - centroids[j])
            if dist < min_dist or label == -1:
                min_dist = dist
                label = j
        labels[i] = label

@wp.kernel
def accumulate_centroids(X: wp.array[wp.vec2f],
                    labels: wp.array[wp.int32],
                    centroids: wp.array[wp.vec2f],
                    counts: wp.array[wp.int32]):
    i = wp.tid()
    if i < X.shape[0]:
        label = labels[i]
        wp.atomic_add(counts, label, 1)
        wp.atomic_add(centroids, label, X[i])

@wp.kernel
def normalize_centroids(centroids: wp.array[wp.vec2f],
                    counts: wp.array[wp.int32]):
    i = wp.tid()
    if i < centroids.shape[0]:
        if counts[i] > 0:
            centroids[i] = centroids[i] / wp.float32(counts[i])

class WarpKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None
        self.labels = None

    def _kinit(self, X):
        # Randomly initialize with well spaced selections
        n_samples = X.shape[0]
        centroids = []
        
        # First centroid: random selection
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        # Remaining centroids: weighted by distance to nearest existing centroid
        for _ in range(1, self.n_clusters):
            # Calculate min distance from each point to any existing centroid
            min_distances = np.min([
                np.linalg.norm(X - c, axis=1) 
                for c in centroids
            ], axis=0)
            
            # Weight probabilities by distance squared
            weights = min_distances ** 2
            weights /= weights.sum()
            
            # Select next centroid
            next_idx = np.random.choice(n_samples, p=weights)
            centroids.append(X[next_idx])
        
        self.centroids = np.array(centroids)

    def fit(self, X):

        assert X.ndim == 2, "Input data must be a 2D array"
        assert X.shape[1] == 2, "Input data must have 2 features (2D points)"

        start_time = time.time()
        if X.shape[0] > 1000:
            self._kinit(X[np.random.choice(X.shape[0], 1000, replace=False)])
        else:
            self._kinit(X)
        end_time = time.time()
        print(f"Init took: {end_time - start_time}s")
        self.labels = None

        x_buffer = wp.array(X, dtype=wp.vec2f, device="cuda")
        centroids_buffer = wp.array(self.centroids, dtype=wp.vec2f, device="cuda")
        labels_buffer = wp.array(np.zeros(X.shape[0], dtype=wp.int32), dtype=wp.int32, device="cuda")
        counts_buffer = wp.array(np.zeros(self.n_clusters, dtype=wp.int32), dtype=wp.int32, device="cuda")

        for i in range(self.max_iter):
            # Assign clusters based on closest centroid
            wp.launch(assign_labels, dim=X.shape[0], inputs=[x_buffer, centroids_buffer, labels_buffer])

            centroids_buffer.zero_()
            counts_buffer.zero_()
            wp.launch(accumulate_centroids, dim=X.shape[0], inputs=[x_buffer, labels_buffer, centroids_buffer, counts_buffer])
            wp.launch(normalize_centroids, dim=self.n_clusters, inputs=[centroids_buffer, counts_buffer])

            new_centroids = centroids_buffer.numpy()

            progress = np.linalg.norm(new_centroids - self.centroids)
            should_stop = progress < self.tol
            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1 or should_stop):
                print(f"Iteration: {i}: Progress: {progress}")

            if should_stop:
                break

            self.centroids = new_centroids

    def predict(self, X):
        x_buffer = wp.array(X, dtype=wp.vec2f, device="cuda")
        centroids_buffer = wp.array(self.centroids, dtype=wp.vec2f, device="cuda")
        labels_buffer = wp.array(np.zeros(X.shape[0], dtype=wp.int32), dtype=wp.int32, device="cuda")

        wp.launch(assign_labels, dim=X.shape[0], inputs=[x_buffer, centroids_buffer, labels_buffer])

        return labels_buffer.numpy()