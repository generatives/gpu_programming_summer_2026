import time

import numpy as np
from kmeans.kmeans import KMeans
import warp as wp

wp.config.enable_backward = False

BLOCK_SIZE = 512
MAX_DATA_SIZE = 4096
NUM_CLUSTERS = 3
MAX_ITERATIONS = 100
PROGRESS_TOLERANCE = 1e-4

@wp.func
def mahabolis_dist(diff: wp.vec2h, sigma: wp.mat22h):
    return wp.dot(diff * wp.inverse(sigma), diff)

@wp.func
def safe_log(a: wp.float16):
    return wp.log(a + wp.float16(1e-6))

@wp.func
def vec2h_float16_sub(a: wp.vec2h, b: wp.float16):
    return wp.vec2h(a[0] - b, a[1] - b)

@wp.func
def _multivariate_gaussian_pdf(thread_idx: wp.int32,
                               data: wp.tile[wp.vec2h, MAX_DATA_SIZE, 1],
                               actual_data_size: int,
                               mu: wp.tile[wp.vec2h, 1, NUM_CLUSTERS],
                               sigma: wp.tile[wp.mat22h, 1, NUM_CLUSTERS]):
    '''
    Returns a (MAX_DATA_SIZE, NUM_CLUSTERS) tile containing the probability density of each data point evaluated at each gaussian
    '''
    n = wp.float16(2.0)
    data = wp.tile_broadcast(data, (MAX_DATA_SIZE, NUM_CLUSTERS))
    mu = wp.tile_broadcast(mu, (MAX_DATA_SIZE, NUM_CLUSTERS))
    sigma = wp.tile_broadcast(sigma, (MAX_DATA_SIZE, NUM_CLUSTERS))

    diff = data - mu
    exponent = wp.tile_map(wp.exp, wp.float16(-0.5) * wp.tile_map(mahabolis_dist, diff, sigma))
    coefficient = wp.float16(1.0) / wp.tile_map(wp.sqrt, (wp.float16(2.0) * wp.float16(wp.pi)) ** n * wp.tile_map(wp.determinant, sigma))

    p = coefficient * exponent

    # mask out the probabilities for invalid data entries
    tile_data_idx = thread_idx
    while tile_data_idx < MAX_DATA_SIZE:
        for j in range(NUM_CLUSTERS):
            p[tile_data_idx, j] = p[tile_data_idx, j] if tile_data_idx < actual_data_size else wp.float16(0.0)

        tile_data_idx = tile_data_idx + BLOCK_SIZE

    return p

@wp.func
def _expectation_step(thread_idx: wp.int32,
                      data_tile: wp.tile[wp.vec2h, MAX_DATA_SIZE, 1],
                      actual_data_size: int,
                      mu: wp.tile[wp.vec2h, 1, NUM_CLUSTERS],
                      sigma: wp.tile[wp.mat22h, 1, NUM_CLUSTERS],
                      pi: wp.tile[wp.float16, 1, NUM_CLUSTERS]):
    
    pi = wp.tile_broadcast(pi, (MAX_DATA_SIZE, NUM_CLUSTERS))
    pi_p = pi * _multivariate_gaussian_pdf(thread_idx, data_tile, actual_data_size, mu, sigma)
    
    pi_p_per_datapoint = wp.tile_sum(pi_p, axis=1)
    pi_p_per_datapoint = wp.tile_broadcast(wp.tile_reshape(pi_p_per_datapoint, (MAX_DATA_SIZE,1)), (MAX_DATA_SIZE, NUM_CLUSTERS))
    gamma = pi_p / pi_p_per_datapoint
    
    # masking for smaller datasets!
    log_likelihood_by_datapoint = wp.tile_map(safe_log, wp.tile_sum(pi_p, axis=1))

    # mask out log likelihoods
    tile_data_idx = thread_idx
    while tile_data_idx < MAX_DATA_SIZE:
        log_likelihood_by_datapoint[tile_data_idx] = log_likelihood_by_datapoint[tile_data_idx] if tile_data_idx < actual_data_size else wp.float16(0.0)
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    sum_log_likelihood = wp.tile_sum(log_likelihood_by_datapoint)[0]
    return gamma, sum_log_likelihood

@wp.kernel
def expectation_step(X: wp.array[wp.vec2h],
                     offsets: wp.array[wp.int32],
                     mu: wp.array2d[wp.vec2h],
                     sigma: wp.array2d[wp.mat22h],
                     pi: wp.array2d[wp.float16],
                     gamma: wp.array2d[wp.float16],
                     mean_log_likelihood: wp.array[wp.float16]):
    
    block_id, i = wp.tid()
    model_idx = block_id

    data_start = offsets[model_idx]
    data_end = offsets[model_idx+1]
    actual_data_size = data_end - data_start

    model_offset = model_idx * NUM_CLUSTERS
    model_mu = wp.tile_load(mu, shape=(1,NUM_CLUSTERS), offset=(0,model_offset))
    model_sigma = wp.tile_load(sigma, shape=(1,NUM_CLUSTERS), offset=(0,model_offset))
    model_pi = wp.tile_load(pi, shape=(1,NUM_CLUSTERS), offset=(0,model_offset))

    data_tile = wp.tile_empty((MAX_DATA_SIZE,1), dtype=wp.vec2h)

    # load the needed data into shared memory
    tile_data_idx = i
    while tile_data_idx < MAX_DATA_SIZE:
        buffer_data_idx = tile_data_idx + data_start
        data_tile[tile_data_idx, 0] = X[buffer_data_idx] if buffer_data_idx < data_end else wp.vec2h(0.0, 0.0)
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    model_gamma, model_sum_log_likelihood = _expectation_step(i, data_tile, actual_data_size, model_mu, model_sigma, model_pi)

    wp.tile_store(gamma, model_gamma, offset=(data_start,0))
    mean_log_likelihood[model_idx] = model_sum_log_likelihood / wp.float16(actual_data_size)

@wp.func
def _maximization_step(thread_idx: wp.int32,
                      data_tile: wp.tile[wp.vec2h, MAX_DATA_SIZE, 1],
                      actual_data_size: int,
                      gamma: wp.tile[wp.float16, MAX_DATA_SIZE, NUM_CLUSTERS]):
    
    N_k = wp.tile_sum(gamma, axis=0)
    pi = N_k / wp.float16(actual_data_size)
    pi = wp.tile_broadcast(pi, (1,NUM_CLUSTERS))

    wide_data_tile = wp.tile_broadcast(data_tile, (MAX_DATA_SIZE, NUM_CLUSTERS))
    mu = wp.tile_sum(gamma * wide_data_tile, axis=0) / N_k
    mu = wp.tile_broadcast(mu, (1,NUM_CLUSTERS))
    
    sigma = wp.tile_zeros((1,NUM_CLUSTERS), dtype=wp.mat22h, storage="shared")
    for k in range(NUM_CLUSTERS):
        diff = wp.tile_sum(wp.tile_map(wp.sub, data_tile, mu[0,k]), axis=1)
        # mask out diffs
        tile_data_idx = thread_idx
        while tile_data_idx < MAX_DATA_SIZE:
            diff[tile_data_idx] = diff[tile_data_idx] if tile_data_idx < actual_data_size else wp.vec2h(0.0, 0.0)
            tile_data_idx = tile_data_idx + BLOCK_SIZE

        sigma[0,k] = (
                (wp.tile_sum(
                    wp.tile_transpose(gamma)[k] *
                    wp.tile_map(wp.outer, diff, diff)
                ) / N_k[k])[0] +
            wp.mat22h(1e-6, 0.0, 0.0, 1e-6)
        )
    return mu, sigma, pi

@wp.kernel
def maximization_step(X: wp.array2d[wp.vec2h],
                      offsets: wp.array[wp.int32],
                      gamma: wp.array2d[wp.float16],
                      mu: wp.array2d[wp.vec2h],
                      sigma: wp.array2d[wp.mat22h],
                      pi: wp.array2d[wp.float16],):
    
    block_id, i = wp.tid()
    model_idx = block_id

    data_start = offsets[model_idx]
    data_end = offsets[model_idx+1]
    actual_data_size = data_end - data_start

    model_gamma = wp.tile_load(gamma, shape=(MAX_DATA_SIZE,NUM_CLUSTERS), offset=(data_start,0))

    data_tile = wp.tile_empty((MAX_DATA_SIZE,1), dtype=wp.vec2h)

    # load the needed data into shared memory
    tile_data_idx = i
    while tile_data_idx < MAX_DATA_SIZE:
        buffer_data_idx = tile_data_idx + data_start
        data_tile[tile_data_idx, 0] = X[buffer_data_idx, 0] if buffer_data_idx < data_end else wp.vec2h(0.0, 0.0)
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    model_mu, model_sigma, model_pi = _maximization_step(i, data_tile, actual_data_size, model_gamma)

    model_offset = model_idx * NUM_CLUSTERS
    wp.tile_store(mu, model_mu, offset=(0,model_offset))
    wp.tile_store(sigma, model_sigma, offset=(0,model_offset))
    wp.tile_store(pi, model_pi, offset=(0,model_offset))

#@wp.kernel
def fit(X: wp.array[wp.vec2f],
        offsets: wp.array[wp.int32],
        mu: wp.array[wp.vec2f],
        sigma: wp.array[wp.mat22f],
        pi: wp.array[wp.float32],):
    block_id, i = wp.tid()
    model_idx = block_id

    data_start = offsets[model_idx]
    data_end = offsets[model_idx+1]
    actual_data_size = data_end - data_start

    model_offset = model_idx * NUM_CLUSTERS
    model_mu = wp.tile_load(mu, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")
    model_sigma = wp.tile_load(sigma, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")
    model_pi = wp.tile_load(pi, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")

    data_tile = wp.tile_empty((MAX_DATA_SIZE,), dtype=wp.vec2f, storage="shared")

    # load the needed data into shared memory
    tile_data_idx = i
    while tile_data_idx < MAX_DATA_SIZE:
        buffer_data_idx = tile_data_idx + data_start
        data_tile[tile_data_idx] = X[buffer_data_idx] if buffer_data_idx < data_end else wp.vec2f(0.0, 0.0)
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    iteration = int(0)
    last_mean_log_likelihood = float(-wp.inf)
    while iteration < MAX_ITERATIONS:
        gamma, log_likelihood = _expectation_step(i, data_tile, actual_data_size, model_mu, model_sigma, model_pi)

        mean_log_likelihood = log_likelihood / float(actual_data_size)
        progress = wp.abs(mean_log_likelihood - last_mean_log_likelihood)
        if progress < PROGRESS_TOLERANCE:
            break

        model_mu, model_sigma, model_pi = _maximization_step(i, data_tile, actual_data_size, gamma)
    
    wp.tile_store(mu, model_mu, offset=(model_offset,))
    wp.tile_store(sigma, model_sigma, offset=(model_offset,))
    wp.tile_store(pi, model_pi, offset=(model_offset,))

class TiledGaussianMixtureModel:
    def __init__(self, n_models, n_components, max_iter=100, tol=1e-3, init='kmeans', verbose=False, mu=None, sigma=None, pi=None):
        self.n_models = n_models
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.verbose = verbose
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def _numpy_maximization_step(self, X, gamma):
        N_k = np.sum(gamma, axis=0)
        mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
        sigma = np.zeros((mu.shape[0], X.shape[1], X.shape[1]))
        for k in range(mu.shape[0]):
            diff = X - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k] + 1e-6 * np.eye(X.shape[1])
        pi = N_k / X.shape[0]
        return mu, sigma, pi

    def fit(self, X, offsets):

        start_time = time.time()
        if self.init == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, max_iter=100)
            kmeans.fit(X)
            gamma_init = np.eye(self.n_components)[kmeans.labels]
            init_mu, init_sigma, init_pi = self._numpy_maximization_step(X, gamma_init)
        else:
            init_mu = np.random.rand(self.n_components, X.shape[1]) 
            init_sigma = np.array([np.eye(X.shape[1]) for _ in range(self.n_components)]) if self.sigma is None else self.sigma
            init_pi = np.ones(self.n_components) / self.n_components if self.pi is None else self.pi
        end_time = time.time()
        print(f"Init took: {end_time - start_time}")

        self.mu = init_mu if self.mu is None else self.mu
        self.sigma = init_sigma if self.sigma is None else self.sigma
        self.pi = init_pi if self.pi is None else self.pi

        x_buffer = wp.array(X, dtype=wp.vec2h, device="cuda")
        offsets_buffer = wp.array(offsets, dtype=wp.int32, device="cuda")
        mu_buffer = wp.array(self.mu, dtype=wp.vec2h, device="cuda")
        sigma_buffer = wp.array(self.sigma, dtype=wp.mat22h, device="cuda")
        pi_buffer = wp.array(self.pi, dtype=wp.float16, device="cuda")
        gamma_buffer = wp.zeros(shape=(self.n_models * MAX_DATA_SIZE,NUM_CLUSTERS), dtype=wp.float16, device="cuda")
        mean_log_likelihoods_buffer = wp.zeros(shape=(self.n_models,), dtype=wp.float16, device="cuda")

        #wp.launch_tiled(fit,
        #                dim=self.n_models,
        #                inputs=[x_buffer, offsets_buffer, mu_buffer, sigma_buffer, pi_buffer],
        #                block_dim=BLOCK_SIZE)

        iteration = 0
        last_mean_log_likelihood = None
        while iteration < MAX_ITERATIONS:
            wp.launch_tiled(
                expectation_step,
                dim=self.n_models,
                inputs=[x_buffer, offsets_buffer, mu_buffer, sigma_buffer, pi_buffer, gamma_buffer, mean_log_likelihoods_buffer],
                block_dim=BLOCK_SIZE)

            mean_log_likelihoods = mean_log_likelihoods_buffer.numpy()

            progress = np.max(np.abs(mean_log_likelihoods - last_mean_log_likelihood)) if last_mean_log_likelihood is not None else np.inf
            if progress < PROGRESS_TOLERANCE:
                break

            wp.launch_tiled(
                maximization_step,
                dim=self.n_models,
                inputs=[x_buffer, offsets_buffer, gamma_buffer, mu_buffer, sigma_buffer, pi_buffer],
                block_dim=BLOCK_SIZE)

        self.mu = mu_buffer.numpy()
        self.sigma = sigma_buffer.numpy()
        self.pi = pi_buffer.numpy()