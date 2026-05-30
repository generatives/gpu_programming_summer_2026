#include <iostream>
#include <math.h>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include "buffers.h"

const int CONFIG_SIZE = 8;
const int CONFIG_FLOAT4_COUNT = 2;
const int K = 8;
const int N = 30000;

struct Config {
  float4 data[CONFIG_FLOAT4_COUNT];
};

__device__ float4 operator-(const float4 &a, const float4 &b) {
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ bool config_distance(const Config a, const Config b) {
  float distance = 0;
  for(int i = 0; i < CONFIG_FLOAT4_COUNT; i++) {
    float4 diff = a.data[i] - b.data[i];
    float distanceSq = (diff.x) * (diff.x) +
      (diff.y) * (diff.y) +
      (diff.z) * (diff.z) +
      (diff.w) * (diff.w);
    distance += distanceSq;
  }

  return distance;
}

__device__ void attempt_add(int (&nearest_idx)[K], float (&nearest_dist)[K], int idx, float dist) {
  int largest_closer_idx = -1;
  int largest_closer_dist = -1;

  for(int i = 0; i < K; i++) {
    if(dist < nearest_dist[i]) {
      if(nearest_dist[i] > largest_closer_dist) {
        largest_closer_dist = nearest_dist[i];
        largest_closer_idx = i;
      }
    }
  }

  if(largest_closer_idx != -1) {
    nearest_idx[largest_closer_idx] = idx;
    nearest_dist[largest_closer_idx] = dist;
  }
}
 
__global__
void brute_knn_search(
  const Config* __restrict__ configs,
  uint*__restrict__ knn)
{
  int nearest_k_idx[K] = {};
  float nearest_k_dist[K] = {};

  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(query_idx < N) {
    Config query = configs[query_idx];

    for(int i = 0; i < N; i++) {
      int candidate_idx = i;
      Config candidate = configs[candidate_idx];
      float distance = config_distance(query, candidate);
      if(i < K) {
        nearest_k_idx[i] = i;
        nearest_k_dist[i] = distance;
      } else {
        attempt_add(nearest_k_idx, nearest_k_dist, candidate_idx, distance);
      }
    }
  }

  int knn_base_idx = query_idx * K;
  for(int i = 0; i < K; i++) {
    knn[knn_base_idx + i] = nearest_k_idx[i];
  }
}

bool all_less_than(uint *values, int start, int n, int max) {
  for(int i = start; i < start + n; i++) {
    if(values[i] >= max) {
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
    if(idx < k) {
      if(present[idx]) {
        //std::cout << "Position: "<< idx << " duplicated\n";
        return_val = false;
        break;
      } else {
        present[idx] = true;
      }
    } else {
      //std::cout << "Position: "<< idx << " larger than" << k << "\n";
      return_val = false;
      break;
    }
  }

  for(int i = 0; i < k; i++) {
    if(!present[i]) {
      //std::cout << "Position: "<< i << " not present\n";
      return_val = false;
      break;
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

  int block = 256;
  int grid = (N + block - 1) / block;

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