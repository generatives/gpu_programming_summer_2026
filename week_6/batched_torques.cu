#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "buffers.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

const int NUM_JOINTS = 7;
const int NUM_FORCES = 16;
const int BLOCK_DIM_Y = 64;

__host__ __device__ inline bool nearlyEqual(float a, float b, float eps = 1e-5f) {
    return fabsf(a - b) <= eps;
}

__host__ __device__ inline bool nearlyEqual(const float4& a, const float4& b, float eps = 1e-5f) {
    return nearlyEqual(a.x, b.x, eps) &&
           nearlyEqual(a.y, b.y, eps) &&
           nearlyEqual(a.z, b.z, eps) &&
           nearlyEqual(a.w, b.w, eps);
}

__device__ float2 operator-(const float2 &a, const float2 &b) {
  return make_float2(a.x-b.x, a.y-b.y);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __host__ bool operator==(const float4 &a, const float4 &b) {
  return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w;
}

__device__ __host__ bool operator!=(const float4 &a, const float4 &b) {
  return a.x!=b.x || a.y!=b.y || a.z!=b.z || a.w!=b.w;
}

std::random_device rd;
std::mt19937 gen(rd()); 
std::uniform_real_distribution<> distr(0.0, 5.0);

template<unsigned int D>
struct VecD {
    float vec[D];

  __device__ __host__ bool operator==(const VecD &a) {
    #pragma unroll
    for (int i = 0; i < D; i++) {
      if (vec[i] != a.vec[i]) {
        return false;
      }
    }
    return true;
  }
};

template<unsigned int D>
__host__ __device__ inline bool nearlyEqual(const VecD<D>& a, const VecD<D>& b, float eps = 1e-5f) {
    #pragma unroll
    for (int i = 0; i < D; i++) {
      if (!nearlyEqual(a.vec[i], b.vec[i])) {
        return false;
      }
    }
    return true;
}

struct Mat6Col {
  float4 vec[2];
};

template<unsigned int D>
struct Mat6xD {
    Mat6Col cols[D];

  __device__ __host__ bool operator==(const Mat6xD &a) {
    #pragma unroll
    for (int i = 0; i < D * 2; i++) {
      if (cols[i] != a.cols[i]) {
        return false;
      }
    }
    return true;
  }
};

template<unsigned int D>
VecD<D> sample_vecd() {
  VecD<D> output;

  #pragma unroll
  for (int i = 0; i < D; i++) {
    output.vec[i] = distr(gen);
  }

  return output;
}

template<unsigned int D>
Mat6xD<D> sample_mat6xd() {
  Mat6xD<D> output;

  #pragma unroll
  for (int i = 0; i < D; i++) {
    output.cols[i].vec[0].x = distr(gen);
    output.cols[i].vec[0].y = distr(gen);
    output.cols[i].vec[0].z = distr(gen);
    output.cols[i].vec[0].w = distr(gen);
    output.cols[i].vec[1].x = distr(gen);
    output.cols[i].vec[1].y = distr(gen);
    output.cols[i].vec[1].z = 0.0f;
    output.cols[i].vec[1].w = 0.0f;
  }

  return output;
}

template<unsigned int D>
VecD<D> build_vecd(std::array<float, D> data) {
  VecD<D> output;

  #pragma unroll
  for (int i = 0; i < D; i++) {
    output.vec[i] = data[i];
  }

  return output;
}

// Build a mat6xD from data in column first order
template<unsigned int D>
Mat6xD<D> build_mat6xd(std::array<float, 6 * D> data) {
  Mat6xD<D> output;

  #pragma unroll
  for (int i = 0; i < D; i++) {
    int data_idx = i * 6;
    output.cols[i].vec[0].x = data[data_idx + 0];
    output.cols[i].vec[0].y = data[data_idx + 1];
    output.cols[i].vec[0].z = data[data_idx + 2];
    output.cols[i].vec[0].w = data[data_idx + 3];
    output.cols[i].vec[1].x = data[data_idx + 4];
    output.cols[i].vec[1].y = data[data_idx + 5];
    output.cols[i].vec[1].z = 0.0f;
    output.cols[i].vec[1].w = 0.0f;
  }

  return output;
}

template<unsigned int D>
Mat6xD<D> scaled_identity(float scale) {
  Mat6xD<D> output;

  #pragma unroll
  for (int i = 0; i < D; i++) {
    output.cols[i].vec[0].x = 0.0f;
    output.cols[i].vec[0].y = 0.0f;
    output.cols[i].vec[0].z = 0.0f;
    output.cols[i].vec[0].w = 0.0f;
    output.cols[i].vec[1].x = 0.0f;
    output.cols[i].vec[1].y = 0.0f;
    output.cols[i].vec[1].z = 0.0f;
    output.cols[i].vec[1].w = 0.0f;
  }

  output.cols[0].vec[0].x = scale;
  output.cols[1].vec[0].y = scale;
  output.cols[2].vec[0].z = scale;
  output.cols[3].vec[0].w = scale;
  output.cols[4].vec[1].x = scale;
  output.cols[5].vec[1].y = scale;

  return output;
}

// Matrix representing a J value for the robot
using MatJ = Mat6xD<NUM_JOINTS>;
// NUM_JOINT dimensional vector representing the torques required to produce and end effector force
using VecT = VecD<NUM_JOINTS>;
// 6 dimensional vector representing an end effector force
using Vec6 = VecD<6>;

// Compute the torque for dimension i
__device__ __host__ float compute_torque_i(const Mat6Col col, const Vec6 f) {

  return col.vec[0].x * f.vec[0] + \
      col.vec[0].y * f.vec[1] + \
      col.vec[0].z * f.vec[2] + \
      col.vec[0].w * f.vec[3] + \
      col.vec[1].x * f.vec[4] + \
      col.vec[1].y * f.vec[5];
}
 
__global__
void batched_torques(
  int num_configs,
  const MatJ* __restrict__ j,
  const Vec6*__restrict__ f,
  VecT*__restrict__ t)
{
  int col = threadIdx.x;
  int index = blockIdx.y * blockDim.y + threadIdx.y;
  int stride = blockDim.y * gridDim.y;
  //int force_idx = threadIdx.z;
  
  for (int config_idx = index; config_idx < num_configs; config_idx += stride) {
    for (int force_idx = 0; force_idx < NUM_FORCES; force_idx++) {
      int result_id = config_idx * NUM_FORCES + force_idx;
      t[result_id].vec[col] = compute_torque_i(j[config_idx].cols[col], f[force_idx]);
    }
  }
}

void test_program_correct() {

  int num_matrices = 3;

  Buffers<MatJ> j(num_matrices);
  Buffers<Vec6> f(NUM_FORCES);
  Buffers<VecT> t(num_matrices * NUM_FORCES);

  for (int i = 0; i < num_matrices; i++) {
    j.host[i] = build_mat6xd<NUM_JOINTS>({
      -1,  3, -6, -7,  6,  2,
      2, -2, 10, 11, 13, -10,
      2, -3,  9,  6,  1,  -9,
      2,  2,  2, 12, -2,   6,
      3, 11,  4,  8, -2,   4,
      4,  1,  7,  9,  1,   6,
      1,  0,  0,  0,  0,   1,
    });
  }

  for (int i = 0; i < NUM_FORCES; i++) {
    f.host[i] = build_vecd<6>({static_cast<float>(1. + i), 2., 3., 4., 5., 6.});
  }

  j.copy_to_device();
  f.copy_to_device();
  t.copy_to_device();

  dim3 block(NUM_JOINTS, BLOCK_DIM_Y, 1);
  dim3 grid(1, (num_matrices + block.y - 1) / block.y, 1);
  batched_torques<<<grid, block>>>(
    num_matrices,
    j.device,
    f.device,
    t.device
  );

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess)
      std::cerr << "launch: " << cudaGetErrorString(launch_err) << "\n";
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess)
      std::cerr << "exec: " << cudaGetErrorString(sync_err) << "\n";

  t.copy_to_host();

  for (int config_idx = 0; config_idx < num_matrices; config_idx++) {
    for (int force_idx = 0; force_idx < NUM_FORCES; force_idx++) {
      int result_id = config_idx * NUM_FORCES + force_idx;
      //std::cout << "Fetching: " << result_id << "\n";
      VecT out = t.host[result_id];
      VecT expected = build_vecd<NUM_JOINTS>({
        static_cast<float>(1 + -1 * force_idx),
        static_cast<float>(77 + 2 * force_idx),
        static_cast<float>(-2 + 2 * force_idx),
        static_cast<float>(86 + 2 * force_idx),
        static_cast<float>(83 + 3 * force_idx),
        static_cast<float>(104 + 4 * force_idx),
        static_cast<float>(7 + 1 * force_idx),
      });

      //std::cout << "Actual: " << out.vec[0] << ", " << out.vec[1] << ", " << out.vec[2] << ", " << out.vec[3] << ", " << out.vec[4] << ", " << out.vec[5] << "\n";
      //std::cout << "Expected: " << expected.vec[0] << ", " << expected.vec[1] << ", " << expected.vec[2] << ", " << expected.vec[3] << ", " << expected.vec[4] << ", " << expected.vec[5] << "\n";
      
      assert(nearlyEqual(out, expected));
    }
  }
}

long run_program() {

  int num_matrices = 320000;

  Buffers<MatJ> j(num_matrices);
  Buffers<Vec6> f(NUM_FORCES);
  Buffers<VecT> t(num_matrices * NUM_FORCES);

  for (int i = 0; i < num_matrices; i++) {
    j.host[i] = sample_mat6xd<NUM_JOINTS>();
  }

  for (int i = 0; i < NUM_FORCES; i++) {
    f.host[i] = sample_vecd<6>();
  }

  j.copy_to_device();
  f.copy_to_device();
  t.copy_to_device();

  auto start = std::chrono::steady_clock::now();

  dim3 block(NUM_JOINTS, BLOCK_DIM_Y, 1);
  dim3 grid(1, (num_matrices + block.y - 1) / block.y, 1);
  batched_torques<<<grid, block>>>(
    num_matrices,
    j.device,
    f.device,
    t.device
  );

  t.copy_to_host();

  auto end = std::chrono::steady_clock::now();
  auto custom_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return custom_duration.count();
}
 
int main(int argc, char* argv[])
{
  //test_program_correct();
  //return 0;

  // warm up
  int numWarmups = 0;
  for(int i = 0; i < numWarmups; i++) {
    run_program();
  }

  long totalCustomMicroseconds = 0;
  int numRuns = 1;
  for(int i = 0; i < numRuns; i++) {
    long time = run_program();
    totalCustomMicroseconds += time;
  }

  double averageCustomMicroseconds = (double)totalCustomMicroseconds / numRuns;

  std::cout << "Custom time taken: "<< std::fixed << std::setprecision(2) << averageCustomMicroseconds << " microseconds\n";

  return 0;
}