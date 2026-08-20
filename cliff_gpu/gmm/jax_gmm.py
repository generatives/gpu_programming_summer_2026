"""Batched multi-GMM fitting in JAX.

Mirrors gmm/tiled_iterative_gmm.py: fit many small independent GMMs at once,
one per "model". The per-model EM is written for a SINGLE model and then
`vmap`-ed across the model axis -- the idiomatic JAX analogue of the Warp
block-per-model kernel. A single fused `lax.while_loop` drives all models;
each model freezes once its own mean-log-likelihood change drops below `tol`
(same per-model early-stop as the Warp version).

Data is padded to a fixed width P with a validity mask, since vmap needs
static shapes (the Warp kernel does the same with MAX_DATA_SIZE + masking).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, vmap

TWO_PI = 2.0 * jnp.pi


# ---- single-model EM (operates on one padded model) ----
def _mvn_pdf(X, mu, sigma):
    # X:(P,2)  mu:(K,2)  sigma:(K,2,2)  ->  (P,K)
    diff = X[:, None, :] - mu[None, :, :]            # (P,K,2)
    inv = jnp.linalg.inv(sigma)                      # (K,2,2)
    det = jnp.linalg.det(sigma)                      # (K,)
    maha = jnp.einsum("pki,kij,pkj->pk", diff, inv, diff)
    coeff = 1.0 / jnp.sqrt(TWO_PI ** 2 * det)        # (K,)
    return coeff[None, :] * jnp.exp(-0.5 * maha)


def _e_step(mu, sigma, pi, X, mask):
    p = _mvn_pdf(X, mu, sigma)                       # (P,K)
    pi_p = (pi[None, :] * p) * mask[:, None]         # zero invalid rows
    denom = jnp.sum(pi_p, axis=1, keepdims=True)
    gamma = jnp.where(mask[:, None] > 0, pi_p / denom, 0.0)
    ll = jnp.sum(jnp.where(mask > 0, jnp.log(jnp.sum(pi_p, axis=1) + 1e-6), 0.0))
    return gamma, ll


def _m_step(X, gamma, count):
    N_k = jnp.sum(gamma, axis=0)                     # (K,)
    mu = (gamma.T @ X) / N_k[:, None]                # (K,2)
    diff = X[:, None, :] - mu[None, :, :]            # (P,K,2)
    # stable centered form: sum_p gamma_pk diff diff^T / N_k  (+ eps I)
    sigma = jnp.einsum("pk,pki,pkj->kij", gamma, diff, diff) / N_k[:, None, None]
    sigma = sigma + 1e-6 * jnp.eye(2)[None]
    pi = N_k / count
    return mu, sigma, pi


def _fit_one_step(mu, sigma, pi, X, mask, count):
    """One EM iteration for a single model. Returns new params + mean-ll (of
    the params it was given, before the update -- matches the Warp version)."""
    gamma, ll = _e_step(mu, sigma, pi, X, mask)
    mu_n, sigma_n, pi_n = _m_step(X, gamma, count)
    return mu_n, sigma_n, pi_n, ll / count


# vmap the single-model step across the model axis
_batched_step = vmap(_fit_one_step, in_axes=(0, 0, 0, 0, 0, 0))


@partial(jax.jit, static_argnames=("max_iter",))
def fit(mu, sigma, pi, X, mask, count, tol, max_iter=100):
    """Fit M GMMs jointly.

    mu:(M,K,2) sigma:(M,K,2,2) pi:(M,K) init params
    X:(M,P,2) padded data, mask:(M,P) 1/0 valid, count:(M,) real point counts
    Returns (mu, sigma, pi, iters) with iters:(M,) per-model iteration counts.
    """
    M = mu.shape[0]
    neg_inf = jnp.full((M,), -jnp.inf)
    done0 = jnp.zeros((M,), dtype=bool)
    it0 = jnp.zeros((M,), dtype=jnp.int32)

    def cond(state):
        it, mu, sigma, pi, last_ll, done = state
        return jnp.any((it < max_iter) & (~done))

    def body(state):
        it, mu, sigma, pi, last_ll, done = state
        mu_n, sigma_n, pi_n, mll = _batched_step(mu, sigma, pi, X, mask, count)
        progress = jnp.abs(mll - last_ll)
        newly_done = progress < tol
        # freeze models that were already done (vmap-while runs all lanes)
        keep = done[:, None, None]
        mu_o = jnp.where(keep, mu, mu_n)
        sigma_o = jnp.where(done[:, None, None, None], sigma, sigma_n)
        pi_o = jnp.where(done[:, None], pi, pi_n)
        ll_o = jnp.where(done, last_ll, mll)
        it_o = jnp.where(done, it, it + 1)
        return (it_o, mu_o, sigma_o, pi_o, ll_o, done | newly_done)

    it, mu, sigma, pi, _, _ = lax.while_loop(
        cond, body, (it0, mu, sigma, pi, neg_inf, done0)
    )
    return mu, sigma, pi, it


def pad_dataset(X, offsets, n_models, n_components):
    """Turn packed (X, offsets) into padded (M,P,2) + mask + counts for vmap."""
    import numpy as np
    counts = np.diff(offsets).astype(np.int32)
    P = int(counts.max())
    Xp = np.zeros((n_models, P, 2), dtype=np.float32)
    mask = np.zeros((n_models, P), dtype=np.float32)
    for m in range(n_models):
        s, e = int(offsets[m]), int(offsets[m + 1])
        Xp[m, : e - s] = X[s:e]
        mask[m, : e - s] = 1.0
    return Xp, mask, counts
