from time import time
import numpy as np
from warp_kmeans import WarpKMeans
from tiled_kmeans import TiledKMeans
from sklearn.cluster import KMeans

np.random.seed(42)

n_models = 1000
n_components = 3
n_gaussians = n_components * n_models
n_samples = 3000

def fit_kmeans_models(X, offsets):
    for i in range(n_models):
        section = X[offsets[i]:offsets[i+1]]
        kmeans_model = KMeans(n_clusters=n_components, max_iter=1000, verbose=False)
        kmeans_model.fit(section)

def fit_model(X, model_idx_lookup, offsets):
    #model = WarpKMeans(n_models=n_models, n_clusters=n_components, max_iter=1000)
    model = TiledKMeans(n_models=n_models, n_clusters=n_components, max_iter=1000)
    model.fit(X, model_idx_lookup, offsets)
    #fit_kmeans_models(X, offsets)

def main():
    true_means = np.random.rand(n_gaussians, 2) * 10
    true_sigmas = np.array([np.eye(2) * (0.1 + 0.9 * np.random.rand()) for _ in range(n_gaussians)])
    true_weights = np.hstack([np.random.dirichlet(np.ones(n_components), size=1)[0] for _ in range(n_models)])
    model_labels = np.repeat(np.arange(n_models), n_components)

    sections = []
    labels = []
    for i in range(len(true_weights)):
        X_i = np.random.multivariate_normal(mean=true_means[i], cov=true_sigmas[i], size=int(n_samples * true_weights[i]))
        sections.append(X_i)
        labels.append(np.full(X_i.shape[0], model_labels[i]))
    X = np.vstack(sections)
    labels = np.hstack(labels)

    offsets = [0]

    for label in range(n_models):
        points = X[labels == label]
        offsets.append(points.shape[0] + offsets[label])

    offsets = np.array(offsets)

    for i in range(3):
        fit_model(X, labels, offsets)

    times = []
    for i in range(10):
        start_time = time()
        fit_model(X, labels, offsets)
        end_time = time()
        times.append(end_time - start_time)
        print(f"Done at {times[i]}")

    print(f"Average time for fitting KMeans with {n_components} components and {n_samples} samples: {np.mean(times):.4f} seconds")

if __name__ == "__main__":
    main()