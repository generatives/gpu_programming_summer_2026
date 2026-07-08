from concurrent.futures import ProcessPoolExecutor
from time import time
import numpy as np
from sklearn import mixture
#import torch
#import triton
#import triton.language as tl

np.random.seed(42)
#torch.manual_seed(42)

#DEVICE = triton.runtime.driver.active.get_active_torch_device()

n_models = 100
n_components = 3
n_gaussians = n_components * n_models
n_samples = 100

def fit_gmm(section):
    sklearn_gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
    sklearn_gmm.fit(section)
    return sklearn_gmm

def fit_models(sectioned_points):
    with ProcessPoolExecutor() as executor:
        results = executor.map(fit_gmm, sectioned_points)
        return list(results)

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

    sectioned_points = []
    for label in range(n_models):
        points = X[labels == label]
        sectioned_points.append(points.copy())

    for i in range(0):
        sklearn_gmms = fit_models(sectioned_points)

    times = []
    for i in range(1):
        start_time = time()
        sklearn_gmms = fit_models(sectioned_points)
        end_time = time()
        times.append(end_time - start_time)

    print(f"Average time for fitting {n_models} GMMs with {n_components} components and {n_samples} samples: {np.mean(times):.4f} seconds")

if __name__ == "__main__":
    main()