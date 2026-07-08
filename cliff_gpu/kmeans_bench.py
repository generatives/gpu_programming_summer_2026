from time import time
import numpy as np
from warp_kmeans import WarpKMeans
from kmeans import KMeans

np.random.seed(42)

n_components = 100
n_samples = 10_000

def fit_model(X):
    model = WarpKMeans(n_clusters=n_components, max_iter=1000)
    #model = KMeans(n_clusters=n_components, max_iter=1000)
    model.fit(X)

def main():
    true_means = np.random.rand(n_components, 2) * 10
    true_sigmas = np.array([np.eye(2) * (0.1 + 0.9 * np.random.rand()) for _ in range(n_components)])

    true_weights = np.random.dirichlet(np.ones(n_components), size=1)[0]

    X = np.zeros((0, 2))
    sections = []
    for i in range(len(true_weights)):
        X_i = np.random.multivariate_normal(mean=true_means[i], cov=true_sigmas[i], size=int(n_samples * true_weights[i]))
        sections.append(X_i)
    X = np.vstack(sections)

    for i in range(3):
        fit_model(X)

    times = []
    for i in range(10):
        start_time = time()
        fit_model(X)
        end_time = time()
        times.append(end_time - start_time)

    print(f"Average time for fitting KMeans with {n_components} components and {n_samples} samples: {np.mean(times):.4f} seconds")

if __name__ == "__main__":
    main()