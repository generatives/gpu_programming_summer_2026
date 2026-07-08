import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=False):
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
            distances = np.array([
                np.min(np.linalg.norm(X - c, axis=1)) 
                for c in centroids
            ])
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
        if X.shape[0] > 1000:
            self._kinit(X[np.random.choice(X.shape[0], 1000, replace=False)])
        else:
            self._kinit(X)
        self.labels = None

        for i in range(self.max_iter):
            # Assign clusters based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            progress = np.linalg.norm(new_centroids - self.centroids)
            should_stop = progress < self.tol
            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1 or should_stop):
                print(f"Iteration: {i}: Progress: {progress}")

            if should_stop:
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)