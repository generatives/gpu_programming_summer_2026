import numpy as np
from kmeans import KMeans

class MultiGaussianMixtureModel:
    def __init__(self, n_models, n_components,
                 max_iter=100, tol=1e-3,
                 init='kmeans',
                 verbose=False,
                 mu=None, sigma=None, pi=None):
        self.n_models = n_models
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.verbose = verbose
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def _multivariate_gaussian_pdf(self, X, mu, sigma):
        n = X.shape[1]
        p = np.zeros((X.shape[0], mu.shape[0]))
        for k in range(mu.shape[0]):
            diff = X - mu[k]
            exponent = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(sigma[k]) * diff, axis=1))
            coefficient = 1 / (np.sqrt((2 * np.pi) ** n * np.linalg.det(sigma[k])))
            p[:, k] = coefficient * exponent
        return p
    
    def _multivariate_gaussian_pdf_batch(self, X, mu, sigma):
        n = X.shape[1]  # D

        diff = X[np.newaxis, :, :] - mu[:, np.newaxis, :]   # (K, N, D)
        inv_sigma = np.linalg.inv(sigma)                     # (K, D, D)
        det_sigma = np.linalg.det(sigma)                     # (K,)

        # batched matmul: (K, N, D) @ (K, D, D) -> (K, N, D), dispatches to BLAS
        tmp = diff @ inv_sigma

        # elementwise multiply + sum, same diagonal-extraction trick as before, now batched
        exponent_term = np.sum(tmp * diff, axis=-1)          # (K, N)

        coefficient = 1 / np.sqrt((2 * np.pi) ** n * det_sigma)  # (K,)
        p = coefficient[:, np.newaxis] * np.exp(-0.5 * exponent_term)  # (K, N)
        return p.T  # (N, K)

    def _expectation_step(self, X, mu, sigma, pi):
        gamma = np.zeros((X.shape[0], mu.shape[0]))
        pi_p = pi * self._multivariate_gaussian_pdf_batch(X, mu, sigma)
        gamma = pi_p / np.sum(pi_p, axis=1, keepdims=True)
        log_likelihood = np.sum(np.log(np.sum(pi_p, axis=1)))
        return gamma, log_likelihood

    def _maximization_step(self, X, gamma):
        N_k = np.sum(gamma, axis=0)
        mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
        sigma = np.zeros((mu.shape[0], X.shape[1], X.shape[1]))
        for k in range(mu.shape[0]):
            diff = X - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k] + 1e-6 * np.eye(X.shape[1])
        pi = N_k / X.shape[0]
        return mu, sigma, pi

    def fit(self, X):

        if self.init == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, max_iter=100)
            kmeans.fit(X)
            gamma_init = np.eye(self.n_components)[kmeans.labels]
            init_mu, init_sigma, init_pi = self._maximization_step(X, gamma_init)
        else:
            init_mu = np.random.rand(self.n_components, X.shape[1]) 
            init_sigma = np.array([np.eye(X.shape[1]) for _ in range(self.n_components)]) if self.sigma is None else self.sigma
            init_pi = np.ones(self.n_components) / self.n_components if self.pi is None else self.pi

        self.mu = init_mu if self.mu is None else self.mu
        self.sigma = init_sigma if self.sigma is None else self.sigma
        self.pi = init_pi if self.pi is None else self.pi

        last_mean_log_likelihood = None
        for i in range(self.max_iter):  # Number of iterations
            gamma, log_likelihood = self._expectation_step(X, self.mu, self.sigma, self.pi)

            mean_log_likelihood = log_likelihood / X.shape[0]
            progress = np.abs(mean_log_likelihood - last_mean_log_likelihood) if last_mean_log_likelihood is not None else np.inf
            should_stop = progress < self.tol
            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1 or should_stop):
                print(f"Iteration: {i}: Log Likelihood: {log_likelihood}, Progress: {progress}")

            if should_stop:
                break

            last_mean_log_likelihood = mean_log_likelihood
            
            self.mu, self.sigma, self.pi = self._maximization_step(X, gamma)
