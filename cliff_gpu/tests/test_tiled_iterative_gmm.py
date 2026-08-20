"""Correctness regression test for gmm/tiled_iterative_gmm.py.

Runs the tiled GPU GMM and a NumPy float64 EM reference (same math as
gmm/gmm.py) from the *same* fixed initialization, then checks that every
model reaches the reference solution.

Because EM stops on a log-likelihood *change* < PROGRESS_TOLERANCE, individual
parameters can still be drifting when both runs halt, so float32 (GPU) vs
float64 (reference) may differ by ~1e-2 in mu/sigma. The invariant that IS
tight is the fit quality: the GPU's mean log-likelihood must match the
reference to within the convergence tolerance (and never be meaningfully
worse). That is what we assert on; parameter deltas are reported for context.

Run directly:   python tests/test_tiled_iterative_gmm.py
Or via pytest:   pytest tests/test_tiled_iterative_gmm.py
Requires a CUDA device (Warp).
"""
import os
import sys
from itertools import permutations

import numpy as np

# allow running as a plain script from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmm.tiled_iterative_gmm import (  # noqa: E402
    TiledGaussianMixtureModel,
    NUM_CLUSTERS,
    MAX_ITERATIONS,
    PROGRESS_TOLERANCE,
)

N_MODELS = 8
N_COMPONENTS = NUM_CLUSTERS
N_SAMPLES = 100

# GPU runs in float32; a converged log-likelihood should match the float64
# reference to within the stopping tolerance, with a little float32 slack.
LL_TOL = PROGRESS_TOLERANCE + 5e-4


# ---- NumPy reference EM (same math as gmm/gmm.py) ----
def _mvn_pdf(X, mu, sigma):
    n = X.shape[1]
    p = np.zeros((X.shape[0], mu.shape[0]))
    for k in range(mu.shape[0]):
        diff = X - mu[k]
        exponent = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(sigma[k]) * diff, axis=1))
        coeff = 1.0 / np.sqrt((2 * np.pi) ** n * np.linalg.det(sigma[k]))
        p[:, k] = coeff * exponent
    return p


def _mean_ll(X, mu, sigma, pi):
    return np.sum(np.log(np.sum(pi * _mvn_pdf(X, mu, sigma), axis=1) + 1e-6)) / X.shape[0]


def _ref_fit(X, mu, sigma, pi):
    mu, sigma, pi = mu.copy(), sigma.copy(), pi.copy()
    last = None
    for _ in range(MAX_ITERATIONS):
        pi_p = pi * _mvn_pdf(X, mu, sigma)
        gamma = pi_p / np.sum(pi_p, axis=1, keepdims=True)
        mll = np.sum(np.log(np.sum(pi_p, axis=1) + 1e-6)) / X.shape[0]
        if last is not None and np.abs(mll - last) < PROGRESS_TOLERANCE:
            break
        last = mll
        N_k = np.sum(gamma, axis=0)
        mu = np.dot(gamma.T, X) / N_k[:, None]
        new_sigma = np.zeros_like(sigma)
        for k in range(mu.shape[0]):
            diff = X - mu[k]
            new_sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k] + 1e-6 * np.eye(X.shape[1])
        sigma = new_sigma
        pi = N_k / X.shape[0]
    return mu, sigma, pi


def _build_dataset(seed=0):
    rng = np.random.RandomState(seed)
    n_g = N_MODELS * N_COMPONENTS
    true_means = rng.rand(n_g, 2) * 10
    true_sigmas = np.array([np.eye(2) * (0.1 + 0.9 * rng.rand()) for _ in range(n_g)])
    model_labels = np.repeat(np.arange(N_MODELS), N_COMPONENTS)

    sections, labels = [], []
    for i in range(n_g):
        Xi = rng.multivariate_normal(true_means[i], true_sigmas[i], size=N_SAMPLES // N_COMPONENTS)
        sections.append(Xi)
        labels.append(np.full(Xi.shape[0], model_labels[i]))
    X = np.vstack(sections).astype(np.float32)
    labels = np.hstack(labels)

    offsets = [0]
    for label in range(N_MODELS):
        offsets.append(int(np.sum(labels == label)) + offsets[label])
    return X, np.array(offsets, dtype=np.int32)


def _fixed_init(X, offsets, seed=123):
    rng = np.random.RandomState(seed)
    mu = np.empty((N_MODELS * N_COMPONENTS, 2), dtype=np.float32)
    sigma = np.empty((N_MODELS * N_COMPONENTS, 2, 2), dtype=np.float32)
    pi = np.empty((N_MODELS * N_COMPONENTS,), dtype=np.float32)
    for m in range(N_MODELS):
        sec = X[offsets[m]:offsets[m + 1]]
        idx = rng.choice(sec.shape[0], N_COMPONENTS, replace=False)
        o = m * N_COMPONENTS
        mu[o:o + N_COMPONENTS] = sec[idx]
        sigma[o:o + N_COMPONENTS] = np.eye(2)
        pi[o:o + N_COMPONENTS] = 1.0 / N_COMPONENTS
    return mu, sigma, pi


def test_tiled_iterative_gmm_matches_reference():
    X, offsets = _build_dataset()
    init_mu, init_sigma, init_pi = _fixed_init(X, offsets)

    model = TiledGaussianMixtureModel(
        n_models=N_MODELS, n_components=N_COMPONENTS,
        mu=init_mu.copy(), sigma=init_sigma.copy(), pi=init_pi.copy(),
    )
    model.fit(X, offsets)

    gpu_mu = model.mu.reshape(N_MODELS, N_COMPONENTS, 2)
    gpu_sigma = model.sigma.reshape(N_MODELS, N_COMPONENTS, 2, 2)
    gpu_pi = model.pi.reshape(N_MODELS, N_COMPONENTS)

    worst_ll = 0.0
    for m in range(N_MODELS):
        sec = X[offsets[m]:offsets[m + 1]].astype(np.float64)
        o = m * N_COMPONENTS
        r_mu, r_sigma, r_pi = _ref_fit(
            sec, init_mu[o:o + N_COMPONENTS].astype(np.float64),
            init_sigma[o:o + N_COMPONENTS].astype(np.float64),
            init_pi[o:o + N_COMPONENTS].astype(np.float64),
        )
        g_mu = gpu_mu[m].astype(np.float64)
        g_sigma = gpu_sigma[m].astype(np.float64)
        g_pi = gpu_pi[m].astype(np.float64)

        # best cluster permutation (labels are arbitrary) for reporting deltas
        perm = min(permutations(range(N_COMPONENTS)),
                   key=lambda p: np.abs(g_mu[list(p)] - r_mu).max())
        p = list(perm)
        d_mu = np.abs(g_mu[p] - r_mu).max()
        d_sigma = np.abs(g_sigma[p] - r_sigma).max()
        d_pi = np.abs(g_pi[p] - r_pi).max()

        ll_gpu = _mean_ll(sec, g_mu, g_sigma, g_pi)
        ll_ref = _mean_ll(sec, r_mu, r_sigma, r_pi)
        d_ll = abs(ll_gpu - ll_ref)
        worst_ll = max(worst_ll, d_ll)

        print(f"model {m}: perm={p} d_mu={d_mu:.3e} d_sigma={d_sigma:.3e} "
              f"d_pi={d_pi:.3e} ll_gpu={ll_gpu:.5f} ll_ref={ll_ref:.5f} d_ll={d_ll:.2e}")

        # fit quality must match the reference to within the stopping tolerance
        assert d_ll < LL_TOL, f"model {m}: log-likelihood off by {d_ll:.3e} (tol {LL_TOL:.3e})"
        # and the GPU must never converge to a meaningfully worse optimum
        assert ll_gpu >= ll_ref - LL_TOL, f"model {m}: GPU ll {ll_gpu} worse than ref {ll_ref}"

    print(f"\nOK: worst log-likelihood delta {worst_ll:.3e} < tol {LL_TOL:.3e}")


if __name__ == "__main__":
    test_tiled_iterative_gmm_matches_reference()
    print("PASSED")
