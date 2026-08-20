import time

import numpy as np
from kmeans.kmeans import KMeans
import warp as wp

wp.config.enable_backward = False

BLOCK_SIZE = 128
BATCH_DATA_SIZE = 128
MAX_DATA_SIZE = 4096
NUM_BATCHES = int(MAX_DATA_SIZE / BATCH_DATA_SIZE)
assert (NUM_BATCHES * BATCH_DATA_SIZE) == MAX_DATA_SIZE
NUM_CLUSTERS = 3
MAX_ITERATIONS = 100
PROGRESS_TOLERANCE = 1e-3

@wp.func
def mahabolis_dist(diff: wp.vec2f, sigma: wp.mat22f):
    return wp.dot(diff * wp.inverse(sigma), diff)

@wp.func
def safe_log(a: wp.float32):
    return wp.log(a + 1e-6)

@wp.func
def mat22f_add(a: wp.mat22f, b: wp.mat22f, c: wp.mat22f):
    return a + b + c

@wp.func
def _multivariate_gaussian_pdf(thread_idx: wp.int32,
                               data: wp.tile[wp.vec2f, BATCH_DATA_SIZE],
                               actual_data_size: int,
                               mu: wp.tile[wp.vec2f, NUM_CLUSTERS],
                               sigma: wp.tile[wp.mat22f, NUM_CLUSTERS]):
    '''
    Returns a (BATCH_DATA_SIZE, NUM_CLUSTERS) tile containing the probability density of each data point evaluated at each gaussian
    '''
    n = 2.0
    data = wp.tile_broadcast(wp.tile_reshape(data, (BATCH_DATA_SIZE,1)), (BATCH_DATA_SIZE, NUM_CLUSTERS))
    mu = wp.tile_broadcast(wp.tile_reshape(mu, (1,NUM_CLUSTERS)), (BATCH_DATA_SIZE, NUM_CLUSTERS))
    sigma = wp.tile_broadcast(wp.tile_reshape(sigma, (1,NUM_CLUSTERS)), (BATCH_DATA_SIZE, NUM_CLUSTERS))

    diff = data - mu
    exponent = wp.tile_map(wp.exp, -0.5 * wp.tile_map(mahabolis_dist, diff, sigma))
    coefficient = 1.0 / wp.tile_map(wp.sqrt, (2.0 * wp.pi) ** n * wp.tile_map(wp.determinant, sigma))

    p = coefficient * exponent

    # mask out the probabilities for invalid data entries
    tile_data_idx = thread_idx
    while tile_data_idx < BATCH_DATA_SIZE:
        for j in range(NUM_CLUSTERS):
            p[tile_data_idx, j] = p[tile_data_idx, j] if tile_data_idx < actual_data_size else 0.0

        tile_data_idx = tile_data_idx + BLOCK_SIZE

    return p

@wp.func
def _expectation_step(thread_idx: wp.int32,
                      data_tile: wp.tile[wp.vec2f, BATCH_DATA_SIZE],
                      actual_data_size: int,
                      mu: wp.tile[wp.vec2f, NUM_CLUSTERS],
                      sigma: wp.tile[wp.mat22f, NUM_CLUSTERS],
                      pi: wp.tile[wp.float32, NUM_CLUSTERS]):
    
    pi = wp.tile_broadcast(wp.tile_reshape(pi, (1,NUM_CLUSTERS)), (BATCH_DATA_SIZE, NUM_CLUSTERS))
    pi_p = pi * _multivariate_gaussian_pdf(thread_idx, data_tile, actual_data_size, mu, sigma)
    
    pi_p_per_datapoint = wp.tile_sum(pi_p, axis=1)
    pi_p_per_datapoint = wp.tile_broadcast(wp.tile_reshape(pi_p_per_datapoint, (BATCH_DATA_SIZE,1)), (BATCH_DATA_SIZE, NUM_CLUSTERS))
    gamma = pi_p / pi_p_per_datapoint

    # mask out the NaN for invalid data entries
    tile_data_idx = thread_idx
    while tile_data_idx < BATCH_DATA_SIZE:
        for j in range(NUM_CLUSTERS):
            gamma[tile_data_idx, j] = gamma[tile_data_idx, j] if tile_data_idx < actual_data_size else 0.0

        tile_data_idx = tile_data_idx + BLOCK_SIZE
    
    # masking for smaller datasets!
    log_likelihood_by_datapoint = wp.tile_map(safe_log, wp.tile_sum(pi_p, axis=1))

    # mask out log likelihoods
    tile_data_idx = thread_idx
    while tile_data_idx < BATCH_DATA_SIZE:
        log_likelihood_by_datapoint[tile_data_idx] = log_likelihood_by_datapoint[tile_data_idx] if tile_data_idx < actual_data_size else 0.0
        tile_data_idx = tile_data_idx + BLOCK_SIZE

    sum_log_likelihood = wp.tile_sum(log_likelihood_by_datapoint)[0]
    return gamma, sum_log_likelihood

@wp.func
def _maximization_step_acc(data_tile: wp.tile[wp.vec2f, BATCH_DATA_SIZE],
                      gamma: wp.tile[wp.float32, BATCH_DATA_SIZE, NUM_CLUSTERS]):
    
    n_k_tile = wp.tile_sum(gamma, axis=0)

    wide_data_tile = wp.tile_broadcast(wp.tile_reshape(data_tile, (BATCH_DATA_SIZE,1)), (BATCH_DATA_SIZE, NUM_CLUSTERS))
    m = wp.tile_sum(gamma * wide_data_tile, axis=0)
    
    q = wp.tile_zeros((NUM_CLUSTERS,), dtype=wp.mat22f, storage="shared")
    for k in range(NUM_CLUSTERS):

        q[k] = wp.tile_sum(
            wp.tile_transpose(gamma)[k] *
            wp.tile_map(wp.outer, data_tile, data_tile)
        )[0]

    return m, q, n_k_tile

@wp.func
def _maximization_step_final(n_k_tile: wp.tile[wp.float32, NUM_CLUSTERS],
                      actual_data_size: int,
                      m: wp.tile[wp.vec2f, NUM_CLUSTERS],
                      q: wp.tile[wp.mat22f, NUM_CLUSTERS],
                      pi: wp.tile[wp.float32, NUM_CLUSTERS]):
    
    pi = n_k_tile / float(actual_data_size)

    mu = m / n_k_tile
    
    sigma = wp.tile_map(mat22f_add, q / n_k_tile, -wp.tile_map(wp.outer, mu, mu), wp.mat22f(1e-6, 0.0, 0.0, 1e-6))
    
    return mu, sigma, pi

@wp.kernel
def fit(X: wp.array[wp.vec2f],
        offsets: wp.array[wp.int32],
        mu: wp.array[wp.vec2f],
        sigma: wp.array[wp.mat22f],
        pi: wp.array[wp.float32],
        iterations: wp.array[wp.int32]):
    block_id, i = wp.tid()
    model_idx = block_id

    data_start = offsets[model_idx]
    data_end = offsets[model_idx+1]
    actual_data_size = data_end - data_start

    model_offset = model_idx * NUM_CLUSTERS
    model_mu = wp.tile_load(mu, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")
    model_sigma = wp.tile_load(sigma, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")
    model_pi = wp.tile_load(pi, shape=(NUM_CLUSTERS,), offset=(model_offset,), storage="shared")

    data_tile = wp.tile_empty((BATCH_DATA_SIZE,), dtype=wp.vec2f, storage="shared")

    iteration = int(0)
    last_mean_log_likelihood = float(-wp.inf)
    while iteration < MAX_ITERATIONS:

        sum_log_likelihood_acc = float(0)
        model_m_acc = wp.tile_zeros(shape=(NUM_CLUSTERS,), dtype=wp.vec2f, storage="shared")
        model_q_acc = wp.tile_zeros(shape=(NUM_CLUSTERS,), dtype=wp.mat22f, storage="shared")
        model_n_k_acc = wp.tile_zeros(shape=(NUM_CLUSTERS,), dtype=wp.float32, storage="shared")

        for batch_idx in range(NUM_BATCHES):
            # load the needed data into shared memory
            tile_data_idx = i
            buffer_data_offset = data_start + batch_idx * BATCH_DATA_SIZE
            while tile_data_idx < BATCH_DATA_SIZE:
                buffer_data_idx = tile_data_idx + buffer_data_offset
                data_tile[tile_data_idx] = X[buffer_data_idx] if buffer_data_idx < data_end else wp.vec2f(0.0, 0.0)
                tile_data_idx = tile_data_idx + BLOCK_SIZE

            actual_batch_data_size = min(data_end - buffer_data_offset, BATCH_DATA_SIZE)
            gamma, sum_log_likelihood = _expectation_step(i, data_tile, actual_batch_data_size, model_mu, model_sigma, model_pi)
            sum_log_likelihood_acc += sum_log_likelihood

            shared_gamma = wp.tile_empty(shape=(BATCH_DATA_SIZE,NUM_CLUSTERS), dtype=wp.float32, storage="shared")
            wp.tile_assign(shared_gamma, gamma, offset=(0,0))

            new_model_m_acc, new_model_q_acc, new_model_n_k_acc = _maximization_step_acc(data_tile, shared_gamma)
            model_m_acc = model_m_acc + new_model_m_acc
            model_q_acc = model_q_acc + new_model_q_acc
            model_n_k_acc = model_n_k_acc + new_model_n_k_acc

        model_mu, model_sigma, model_pi = _maximization_step_final(model_n_k_acc, actual_data_size, model_m_acc, model_q_acc, model_pi)
        
        mean_log_likelihood = sum_log_likelihood_acc / float(actual_data_size)
        progress = wp.abs(mean_log_likelihood - last_mean_log_likelihood)
        if progress < PROGRESS_TOLERANCE:
            break
        else:
            last_mean_log_likelihood = mean_log_likelihood
            iteration += 1
    
    wp.tile_store(mu, model_mu, offset=(model_offset,))
    wp.tile_store(sigma, model_sigma, offset=(model_offset,))
    wp.tile_store(pi, model_pi, offset=(model_offset,))
    iterations[model_idx] = iteration

class TiledGaussianMixtureModel:
    def __init__(self, n_models, n_components, max_iter=100, tol=1e-3, verbose=False, mu=None, sigma=None, pi=None):
        self.n_models = n_models
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
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
    
    def _kmeans_init(self, X, offsets):
        mu = np.empty((self.n_models * NUM_CLUSTERS, 2), dtype=np.float32)
        sigma = np.empty((self.n_models * NUM_CLUSTERS, 2, 2), dtype=np.float32)
        pi = np.empty((self.n_models * NUM_CLUSTERS,), dtype=np.float32)

        for i in range(self.n_models):
            section = X[offsets[i]:offsets[i+1]]
            kmeans = KMeans(n_clusters=self.n_components, max_iter=100)
            kmeans.fit(section)
            gamma_init = np.eye(self.n_components)[kmeans.labels]
            init_mu, init_sigma, init_pi = self._numpy_maximization_step(section, gamma_init)

            model_offset = i * NUM_CLUSTERS
            mu[model_offset:model_offset+NUM_CLUSTERS] = init_mu
            sigma[model_offset:model_offset+NUM_CLUSTERS] = init_sigma
            pi[model_offset:model_offset+NUM_CLUSTERS] = init_pi

        return mu, sigma, pi

    def fit(self, X, offsets):
        
        start_time = time.time()
        init_mu, init_sigma, init_pi = self._kmeans_init(X, offsets)
        end_time = time.time()
        print(f"Init took: {end_time - start_time}")

        self.mu = init_mu if self.mu is None else self.mu
        self.sigma = init_sigma if self.sigma is None else self.sigma
        self.pi = init_pi if self.pi is None else self.pi

        x_buffer = wp.array(X, dtype=wp.vec2f, device="cuda")
        offsets_buffer = wp.array(offsets, dtype=wp.int32, device="cuda")
        mu_buffer = wp.array(self.mu, dtype=wp.vec2f, device="cuda")
        sigma_buffer = wp.array(self.sigma, dtype=wp.mat22f, device="cuda")
        pi_buffer = wp.array(self.pi, dtype=wp.float32, device="cuda")
        iterations_buffer = wp.zeros(shape=(self.n_models,), dtype=wp.int32, device="cuda")

        wp.launch_tiled(fit,
                        dim=self.n_models,
                        inputs=[x_buffer, offsets_buffer, mu_buffer, sigma_buffer, pi_buffer, iterations_buffer],
                        block_dim=BLOCK_SIZE)

        self.mu = mu_buffer.numpy()
        self.sigma = sigma_buffer.numpy()
        self.pi = pi_buffer.numpy()

        print(f"Finished after: {iterations_buffer.numpy()} iterations")