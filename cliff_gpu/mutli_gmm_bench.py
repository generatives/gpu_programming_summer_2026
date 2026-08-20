from time import time
import numpy as np
from sklearn import mixture
from gmm.tiled_iterative_gmm import TiledGaussianMixtureModel

np.random.seed(42)

n_models = 100
n_components = 3
n_gaussians = n_components * n_models
n_samples = 100

def fit_gmm_models(X, offsets):
    iterations = []
    for i in range(n_models):
        section = X[offsets[i]:offsets[i+1]]
        model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000).fit(section)
        iterations.append(model.n_iter_)
    print(f"Finished after {iterations} iterations")

def fit_models(X, offsets):
    #return fit_gmm_models(X, offsets)
    TiledGaussianMixtureModel(n_models=n_models, n_components=n_components).fit(X, offsets)

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

    for i in range(0):
        fit_models(X, offsets)

    times = []
    for i in range(1):
        start_time = time()
        fit_models(X, offsets)
        end_time = time()
        times.append(end_time - start_time)

    print(f"Average time for fitting {n_models} GMMs with {n_components} components and {n_samples} samples: {np.mean(times):.4f} seconds")

if __name__ == "__main__":
    main()