#include <iostream>
#include <math.h>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include "buffers.h"

constexpr int WARP_SIZE = 32;
constexpr int CONFIG_SIZE = 8;
constexpr int THREADS_PER_CONFIG = 2;
constexpr int CONFIGS_PER_WARP = WARP_SIZE / THREADS_PER_CONFIG;
constexpr int K = 8;
constexpr int N = 30000;

struct __align__(16) Config {
  float data[CONFIG_SIZE];
};

__device__ float vector_distance(const float4 a, const float4 b) {
  float x_diff = a.x - b.x;
  float y_diff = a.y - b.y;
  float z_diff = a.z - b.z;
  float w_diff = a.w - b.w;

  float distance = (x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff) + (w_diff * w_diff);

  return distance;
}

__device__ void attempt_add(int (&nearest_idx)[K], float (&nearest_dist)[K], int idx, float dist) {
  int largest_closer_entry = -1;
  float largest_closer_dist = -1.0;

  for(int i = 0; i < K; i++) {
    bool is_largest_closer_dist = (dist < nearest_dist[i]) && (nearest_dist[i] > largest_closer_dist);
    if(is_largest_closer_dist) {
      largest_closer_dist = nearest_dist[i];
      largest_closer_entry = i;
    }
  }

  if(largest_closer_entry != -1) {
    nearest_idx[largest_closer_entry] = idx;
    nearest_dist[largest_closer_entry] = dist;
  }
}

__device__ bool _attempt_add_step(int pos, bool c, uint (&nearest_idx)[K], float (&nearest_dist)[K], uint idx, float dist) {
  if(c) {
    if (dist <= nearest_dist[pos - 1]) {
      nearest_idx[pos] = nearest_idx[pos - 1];
      nearest_dist[pos] = nearest_dist[pos - 1];
      return true;
    } else {
      nearest_idx[pos] = idx;
      nearest_dist[pos] = dist;
      return false;
    }
  } else {
    return false;
  }
}

__device__ void attempt_add_fast(uint (&nearest_idx)[K], float (&nearest_dist)[K], uint idx, float dist) {

  // Fast rejection
  if (dist >= nearest_dist[K - 1]) {
      return;
  }

  // Trying to make this whole sort compile time constant so the nearest_? arrays can be in registers
  bool c = true;
  c = _attempt_add_step(K - 1, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 2, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 3, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 4, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 5, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 6, c, nearest_idx, nearest_dist, idx, dist);
  c = _attempt_add_step(K - 7, c, nearest_idx, nearest_dist, idx, dist);
}

__device__ void attempt_add_unsorted(float& worst_dist, int& worst_entry, uint (&nearest_idx)[K], float (&nearest_dist)[K], uint idx, float dist) {
  if (dist < worst_dist) {
    nearest_dist[worst_entry] = dist;
    nearest_idx[worst_entry] = idx;

    // recompute worst among K=8
    worst_entry = 0;
    worst_dist = nearest_dist[0];

    #pragma unroll
    for (int i = 1; i < K; i++) {
      if (nearest_dist[i] > worst_dist) {
        worst_dist = nearest_dist[i];
        worst_entry = i;
      }
    }
  }
}

__global__
void brute_knn_search(
  const Config* __restrict__ configs,
  uint*__restrict__ knn)
{
  uint nearest_k_idx[K];
  float nearest_k_dist[K];
  int worst_entry = 0;
  float worst_dist = INFINITY;

  for (int i = 0; i < K; i++) {
      nearest_k_idx[i] = N;
      nearest_k_dist[i] = INFINITY;
  }

  const float4* config_floats = reinterpret_cast<const float4*>(configs);

  int query_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int thread_base_config_idx = threadIdx.x / THREADS_PER_CONFIG;
  int config_slot_number = threadIdx.x % THREADS_PER_CONFIG;
  bool active = query_idx < N;
  if(active) {
    const Config query = configs[query_idx];
    const float4* query_floats = reinterpret_cast<const float4*>(query.data);

    for(int i = thread_base_config_idx; i < N; i += CONFIGS_PER_WARP) {
      int candidate_slot_number = (i * THREADS_PER_CONFIG) + config_slot_number;
      float4 candidate_v = config_floats[candidate_slot_number];
      float4 query_v = query_floats[config_slot_number];
      float distance = vector_distance(candidate_v, query_v);

      unsigned mask = 0xffffffff;
      for (int offset = THREADS_PER_CONFIG / 2; offset > 0; offset /= 2) {
        float other_distance = __shfl_down_sync(mask, distance, offset);
        distance += other_distance;
      }

      if(config_slot_number == 0) {
        attempt_add_unsorted(
          worst_dist,
          worst_entry,
          nearest_k_idx,
          nearest_k_dist,
          i,
          distance
        );
      }
    }

    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > (THREADS_PER_CONFIG - 1); offset /= 2) {
      int other_idx[K];
      float other_dist[K];

      for (int i = 0; i < K; i++) {
        other_idx[i] = __shfl_down_sync(mask, nearest_k_idx[i], offset);
        other_dist[i] = __shfl_down_sync(mask, nearest_k_dist[i], offset);
      }

      __syncwarp();

      for (int i = 0; i < K; i++) {
        attempt_add_unsorted(
          worst_dist,
          worst_entry,
          nearest_k_idx,
          nearest_k_dist,
          other_idx[i],
          other_dist[i]
        );
      }
    }

    if(threadIdx.x == 0) {
      int knn_base_idx = query_idx * K;
      for(int i = 0; i < K; i++) {
        knn[knn_base_idx + i] = nearest_k_idx[i];
      }
    }
  }
}

bool all_less_than(uint *values, int start, int n, int max) {
  for(int i = start; i < start + n; i++) {
    if(values[i] >= max) {
      //std::cout << "Found: "<< values[i] << " too large\n";
      return false;
    }
  }
  return true;
}

bool top_k_present(uint *values, int start, int n, int k) {
  bool present[k] = {};
  bool return_val = true;
  for(int i = start; i < start + n; i++) {
    int idx = values[i];
    //std::cout << "Check entry: "<< i << " with idx: " << idx << "\n";
    if(idx < k) {
      if(present[idx]) {
        //std::cout << "Position: "<< idx << " duplicated\n";
        return_val = false;
      } else {
        present[idx] = true;
      }
    } else {
      //std::cout << "Position: "<< idx << " larger than " << k << "\n";
      return_val = false;
    }
  }

  for(int i = 0; i < k; i++) {
    if(!present[i]) {
      //std::cout << "Position: "<< i << " not present\n";
      return_val = false;
    }
  }

  return return_val;
}

int run_program() {

  std::vector<Config> configs_v;

  //Add the 8 that we will look at for the test
  for(int i = 0; i < K; i++) {
    configs_v.push_back(Config({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  }

  //Add the remaining
  for(int i = 0; i < N - K; i++) {
    configs_v.push_back(Config({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  }

  Buffers<Config> configs(configs_v);
  configs.copy_to_device();

  Buffers<uint> knn(N * K);
  std::fill_n(knn.host, knn.count, N);
  knn.copy_to_device();

  int warpSize = 32;
  dim3 block(warpSize, 256 / warpSize, 1);
  dim3 grid(1, (N + block.y - 1) / block.y, 1);

  auto start = std::chrono::steady_clock::now();

  brute_knn_search<<<grid, block>>>(configs.device, knn.device);

  knn.copy_to_host();

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  assert(all_less_than(knn.host, 0, N * K, N));

  for(int i = 0; i < K; i++) {
    //std::cout << "Checking: "<< i << "\n";
    assert(top_k_present(knn.host, i * K, K, K));
  }

  return duration.count();
}
 
int main(int argc, char* argv[])
{
  //int N = std::stoi(argv[1]);

  // warm up
  for(int i = 0; i < 0; i++) {
    run_program();
  }

  int totalMicroseconds = 0;
  int numRuns = 1;
  for(int i = 0; i < numRuns; i++) {
    totalMicroseconds += run_program();
  }

  float averageMicroseconds = (float)totalMicroseconds / numRuns;

  std::cout << "Time taken: "<< std::fixed << std::setprecision(2) << averageMicroseconds << " microseconds\n";

  return 0;
}