#include <iostream>
#include <math.h>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include "buffers.h"

const int CONFIG_SIZE = 8;
const int K = 8;
const int N = 30000;

struct __align__(16) Config {
  float data[CONFIG_SIZE];
};

__device__ float4 operator-(const float4 &a, const float4 &b) {
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ float config_distance(const Config a, const Config b) {
  float distance = 0;
  for(int i = 0; i < CONFIG_SIZE; i++) {
    float diff = a.data[i] - b.data[i];
    distance += diff * diff;
  }

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

__global__
void brute_knn_search(
  const Config* __restrict__ configs,
  uint*__restrict__ knn)
{
  int nearest_k_idx[K] = {};
  float nearest_k_dist[K] = {};
  int query_counter = 0;
  const int lookahead_size = 4;
  const int buffer_size = 32 * lookahead_size;
  __shared__ Config config_buffer[buffer_size];

  float4* config_buffer_floats = reinterpret_cast<float4*>(config_buffer);
  const float4* config_floats = reinterpret_cast<const float4*>(configs);

  int query_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int query_thread_number = threadIdx.x;
  bool active = query_idx < N;
  if(active) {
    Config query = configs[query_idx];

    for(int buffer_base = 0; buffer_base < N; buffer_base += buffer_size) {
      int buffer_base_float4s = buffer_base * (CONFIG_SIZE / 4);
      int shared_buffer_index = blockDim.x * threadIdx.y + threadIdx.x;
      int buffer_index = shared_buffer_index + buffer_base_float4s;
      if (shared_buffer_index < buffer_size * (CONFIG_SIZE / 4)) {
        config_buffer_floats[shared_buffer_index] = config_floats[buffer_index];
      }

      __syncthreads();

      for(int i = 0; i < lookahead_size; i++) {
        int local_candidate_idx = (i * warpSize) + query_thread_number;
        int candidate_idx = local_candidate_idx + buffer_base;
        if(candidate_idx < N) {
          Config candidate = config_buffer[local_candidate_idx];
          float distance = config_distance(query, candidate);
          if(query_counter < K) {
            nearest_k_idx[query_counter] = candidate_idx;
            nearest_k_dist[query_counter] = distance;
            query_counter += 1;
          }
          else {
            attempt_add(nearest_k_idx, nearest_k_dist, candidate_idx, distance);
          }
        }
      }
    }

    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      int other_idx[K];
      float other_dist[K];

      for (int i = 0; i < K; i++) {
        other_idx[i] = __shfl_down_sync(mask, nearest_k_idx[i], offset);
        other_dist[i] = __shfl_down_sync(mask, nearest_k_dist[i], offset);
      }

      __syncwarp();

      for (int i = 0; i < K; i++) {
        attempt_add(
          nearest_k_idx,
          nearest_k_dist,
          other_idx[i],
          other_dist[i]
        );
      }
    }

    if(query_thread_number == 0) {
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
      std::cout << "Fround: "<< values[i] << " too large\n";
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