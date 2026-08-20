"""Compare the JAX vmap multi-GMM fit against the Warp tiled kernel.

Both are run from the SAME fixed per-model initialization on the SAME data, so
we can check (a) they agree with each other and with a float64 numpy reference,
and (b) their GPU wall-clock. Usage:  python compare_jax_warp.py [n_models] [n_samples]
"""
import os
import sys
import time
from itertools import permutations

import numpy as np

N_MODELS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 100

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm.tiled_iterative_gmm import (  # noqa: E402
    NUM_CLUSTERS, MAX_ITERATIONS, PROGRESS_TOLERANCE, fit as warp_fit, BLOCK_SIZE,
)

N_COMPONENTS = NUM_CLUSTERS


# ---------- data + fixed init ----------
def build_dataset(seed=0):
    rng = np.random.RandomState(seed)
    n_g = N_MODELS * N_COMPONENTS
    means = rng.rand(n_g, 2) * 10
    sig = np.array([np.eye(2) * (0.1 + 0.9 * rng.rand()) for _ in range(n_g)])
    mlabels = np.repeat(np.arange(N_MODELS), N_COMPONENTS)
    sections, labels = [], []
    for i in range(n_g):
        Xi = rng.multivariate_normal(means[i], sig[i], size=N_SAMPLES // N_COMPONENTS)
        sections.append(Xi)
        labels.append(np.full(Xi.shape[0], mlabels[i]))
    X = np.vstack(sections).astype(np.float32)
    labels = np.hstack(labels)
    offsets = [0]
    for lbl in range(N_MODELS):
        offsets.append(int(np.sum(labels == lbl)) + offsets[lbl])
    return X, np.array(offsets, dtype=np.int32)


def fixed_init(X, offsets, seed=123):
    rng = np.random.RandomState(seed)
    mu = np.empty((N_MODELS, N_COMPONENTS, 2), np.float32)
    sigma = np.empty((N_MODELS, N_COMPONENTS, 2, 2), np.float32)
    pi = np.empty((N_MODELS, N_COMPONENTS), np.float32)
    for m in range(N_MODELS):
        sec = X[offsets[m]:offsets[m + 1]]
        idx = rng.choice(sec.shape[0], N_COMPONENTS, replace=False)
        mu[m] = sec[idx]
        sigma[m] = np.eye(2)
        pi[m] = 1.0 / N_COMPONENTS
    return mu, sigma, pi


def mean_ll(X, mu, sigma, pi):
    n = X.shape[1]
    p = np.zeros((X.shape[0], mu.shape[0]))
    for k in range(mu.shape[0]):
        diff = X - mu[k]
        ex = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(sigma[k]) * diff, axis=1))
        p[:, k] = ex / np.sqrt((2 * np.pi) ** n * np.linalg.det(sigma[k]))
    return np.sum(np.log(np.sum(pi * p, axis=1) + 1e-6)) / X.shape[0]


def perm_delta(gmu, gsig, gpi, rmu, rsig, rpi):
    p = min(permutations(range(N_COMPONENTS)),
            key=lambda q: np.abs(gmu[list(q)] - rmu).max())
    p = list(p)
    return (np.abs(gmu[p] - rmu).max(), np.abs(gsig[p] - rsig).max(), np.abs(gpi[p] - rpi).max())


def main():
    X, offsets = build_dataset()
    imu, isig, ipi = fixed_init(X, offsets)
    print(f"n_models={N_MODELS}  n_samples/model~{N_SAMPLES}  K={N_COMPONENTS}  "
          f"max_iter={MAX_ITERATIONS}  tol={PROGRESS_TOLERANCE}")

    # ---------- Warp ----------
    import warp as wp
    x_buf = wp.array(X, dtype=wp.vec2f, device="cuda")
    off_buf = wp.array(offsets, dtype=wp.int32, device="cuda")

    def warp_run():
        mu_b = wp.array(imu.reshape(-1, 2), dtype=wp.vec2f, device="cuda")
        sig_b = wp.array(isig.reshape(-1, 2, 2), dtype=wp.mat22f, device="cuda")
        pi_b = wp.array(ipi.reshape(-1), dtype=wp.float32, device="cuda")
        it_b = wp.zeros(N_MODELS, dtype=wp.int32, device="cuda")
        wp.launch_tiled(warp_fit, dim=N_MODELS,
                        inputs=[x_buf, off_buf, mu_b, sig_b, pi_b, it_b],
                        block_dim=BLOCK_SIZE)
        wp.synchronize()
        return mu_b, sig_b, pi_b, it_b

    warp_run()  # warmup / compile
    t = time.perf_counter()
    for _ in range(5):
        mu_b, sig_b, pi_b, it_b = warp_run()
    warp_ms = (time.perf_counter() - t) / 5 * 1e3
    wmu = mu_b.numpy().reshape(N_MODELS, N_COMPONENTS, 2)
    wsig = sig_b.numpy().reshape(N_MODELS, N_COMPONENTS, 2, 2)
    wpi = pi_b.numpy().reshape(N_MODELS, N_COMPONENTS)
    print(f"\nWarp  kernel: {warp_ms:8.3f} ms   iters(min/med/max)="
          f"{it_b.numpy().min()}/{int(np.median(it_b.numpy()))}/{it_b.numpy().max()}")

    # ---------- JAX ----------
    import jax
    import jax.numpy as jnp
    from gmm.jax_gmm import fit as jax_fit, pad_dataset

    Xp, mask, counts = pad_dataset(X, offsets, N_MODELS, N_COMPONENTS)
    jmu0 = jnp.asarray(imu); jsig0 = jnp.asarray(isig); jpi0 = jnp.asarray(ipi)
    jXp = jnp.asarray(Xp); jmask = jnp.asarray(mask); jcount = jnp.asarray(counts.astype(np.float32))
    tol = jnp.float32(PROGRESS_TOLERANCE)

    def jax_run():
        r = jax_fit(jmu0, jsig0, jpi0, jXp, jmask, jcount, tol, max_iter=MAX_ITERATIONS)
        jax.block_until_ready(r)
        return r

    r = jax_run()  # warmup / compile
    t = time.perf_counter()
    for _ in range(5):
        r = jax_run()
    jax_ms = (time.perf_counter() - t) / 5 * 1e3
    jmu, jsig, jpi, jit = (np.asarray(a) for a in r)
    print(f"JAX   device: {jax_ms:8.3f} ms   iters(min/med/max)="
          f"{jit.min()}/{int(np.median(jit))}/{jit.max()}   backend={jax.default_backend()}")

    # ---------- accuracy vs numpy float64 reference ----------
    from tests.test_tiled_iterative_gmm import _ref_fit
    worst = {"warp_ll": 0.0, "jax_ll": 0.0, "wj_mu": 0.0, "wj_sig": 0.0}
    for m in range(N_MODELS):
        sec = X[offsets[m]:offsets[m + 1]].astype(np.float64)
        rmu, rsig, rpi = _ref_fit(sec, imu[m].astype(np.float64),
                                  isig[m].astype(np.float64), ipi[m].astype(np.float64))
        ll_ref = mean_ll(sec, rmu, rsig, rpi)
        worst["warp_ll"] = max(worst["warp_ll"], abs(mean_ll(sec, wmu[m], wsig[m], wpi[m]) - ll_ref))
        worst["jax_ll"] = max(worst["jax_ll"], abs(mean_ll(sec, jmu[m], jsig[m], jpi[m]) - ll_ref))
        dmu, dsig, _ = perm_delta(wmu[m], wsig[m], wpi[m], jmu[m], jsig[m], jpi[m])
        worst["wj_mu"] = max(worst["wj_mu"], dmu)
        worst["wj_sig"] = max(worst["wj_sig"], dsig)

    print("\n--- accuracy (worst over models) ---")
    print(f"Warp vs numpy-ref   d_meanLL = {worst['warp_ll']:.3e}")
    print(f"JAX  vs numpy-ref   d_meanLL = {worst['jax_ll']:.3e}")
    print(f"Warp vs JAX         d_mu = {worst['wj_mu']:.3e}   d_sigma = {worst['wj_sig']:.3e}")
    print(f"\n--- speed ---\nWarp {warp_ms:.3f} ms   JAX {jax_ms:.3f} ms   "
          f"ratio {jax_ms / warp_ms:.2f}x")


if __name__ == "__main__":
    main()
