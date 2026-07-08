from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from kmeans import KMeans

class OptimGaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3, init='kmeans', verbose=False, mu=None, sigma=None, pi=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.verbose = verbose
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def _compute_precision_cholesky(self, sigma):
        """Call this once per M-step, not per density evaluation."""
        K, D, _ = sigma.shape
        precisions_chol = np.empty((K, D, D))
        log_det_chol = np.empty(K)
        for k in range(K):
            chol = np.linalg.cholesky(sigma[k])                     # sigma_k = chol @ chol.T
            precisions_chol[k] = solve_triangular(
                chol, np.eye(D), lower=True).T                       # cheap: triangular solve, not a full inverse
            log_det_chol[k] = np.sum(np.log(np.diagonal(chol)))      # log|sigma_k|^{1/2}, O(D), no det() call
        return precisions_chol, log_det_chol

    def _estimate_log_gaussian_prob(self, X, mu, precisions_chol, log_det_chol):
        """Returns log N(x_i | mu_k, sigma_k) for every (i,k) — shape (N, K)."""
        N, D = X.shape
        K = mu.shape[0]
        log_prob = np.empty((N, K))
        for k in range(K):
            y = X @ precisions_chol[k] - mu[k] @ precisions_chol[k]  # single GEMM, no solve/inverse here
            log_prob[:, k] = np.sum(y ** 2, axis=1)
        # standard multivariate normal log-density in terms of the precision-Cholesky
        return -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det_chol

    def _expectation_step(self, X, mu, precisions_chol, log_det_chol, pi):
        log_pi = np.log(pi)
        log_gauss = self._estimate_log_gaussian_prob(X, mu, precisions_chol, log_det_chol)  # (N, K)
        weighted_log_prob = log_gauss + log_pi                       # log(pi_k) + log N(x_i|mu_k,sigma_k)

        log_prob_norm: Any = logsumexp(weighted_log_prob, axis=1)         # log p(x_i), shape (N,) — this IS your per-point log-likelihood
        log_gamma = weighted_log_prob - log_prob_norm[:, np.newaxis] # log responsibilities
        gamma = np.exp(log_gamma)

        return gamma, log_prob_norm

    def _maximization_step(self, X, gamma):
        N_k = np.sum(gamma, axis=0)
        mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
        sigma = np.zeros((mu.shape[0], X.shape[1], X.shape[1]))
        K = mu.shape[0]
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k] + 1e-6 * np.eye(X.shape[1])
        pi = N_k / X.shape[0]

        self.chol = np.array([np.linalg.cholesky(sigma[k]) for k in range(K)])
        self.log_det = np.array([2*np.sum(np.log(np.diagonal(self.chol[k]))) for k in range(K)])

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
            precisions_chol, log_det_chol = self._compute_precision_cholesky(self.sigma)  # once per iteration
            gamma, log_prob_norm = self._expectation_step(X, self.mu, precisions_chol, log_det_chol, self.pi)

            mean_log_likelihood = np.mean(log_prob_norm)   # mean per-sample, matches sklearn's lower_bound_ convention
            progress = np.abs(mean_log_likelihood - last_mean_log_likelihood) if last_mean_log_likelihood is not None else np.inf
            should_stop = progress < self.tol

            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1 or should_stop):
                print(f"Iteration: {i}: Log Likelihood: {mean_log_likelihood}, Progress: {progress}")

            if should_stop:
                break

            last_mean_log_likelihood = mean_log_likelihood

            self.mu, self.sigma, self.pi = self._maximization_step(X, gamma)
