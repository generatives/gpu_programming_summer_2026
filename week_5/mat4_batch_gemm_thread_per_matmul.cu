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

std::random_device rd;
std::mt19937 gen(rd()); 
std::uniform_real_distribution<> distr(0.0, 5.0); 

struct Mat4 {
    float4 cols[4];

  __device__ __host__ bool operator==(const Mat4 &a) {
    return cols[0] == a.cols[0] &&
      cols[1] == a.cols[1] &&
      cols[2] == a.cols[2] &&
      cols[3] == a.cols[3];
  }
};

__host__ __device__ inline bool nearlyEqual(const Mat4& a, const Mat4& b, float eps = 1e-5f) {
    return nearlyEqual(a.cols[0], b.cols[0], eps) &&
           nearlyEqual(a.cols[1], b.cols[1], eps) &&
           nearlyEqual(a.cols[2], b.cols[2], eps) &&
           nearlyEqual(a.cols[3], b.cols[3], eps);
}

Mat4 sample_mat4() {
  Mat4 output;

  output.cols[0].x = distr(gen);
  output.cols[0].y = distr(gen);
  output.cols[0].z = distr(gen);
  output.cols[0].w = distr(gen);

  output.cols[1].x = distr(gen);
  output.cols[1].y = distr(gen);
  output.cols[1].z = distr(gen);
  output.cols[1].w = distr(gen);

  output.cols[2].x = distr(gen);
  output.cols[2].y = distr(gen);
  output.cols[2].z = distr(gen);
  output.cols[2].w = distr(gen);

  output.cols[3].x = distr(gen);
  output.cols[3].y = distr(gen);
  output.cols[3].z = distr(gen);
  output.cols[3].w = distr(gen);

  return output;
}


Mat4 scaled_identity(float scale) {
  Mat4 output;

  output.cols[0].x = scale;
  output.cols[0].y = 0.0f;
  output.cols[0].z = 0.0f;
  output.cols[0].w = 0.0f;

  output.cols[1].x = 0.0f;
  output.cols[1].y = scale;
  output.cols[1].z = 0.0f;
  output.cols[1].w = 0.0f;

  output.cols[2].x = 0.0f;
  output.cols[2].y = 0.0f;
  output.cols[2].z = scale;
  output.cols[2].w = 0.0f;

  output.cols[3].x = 0.0f;
  output.cols[3].y = 0.0f;
  output.cols[3].z = 0.0f;
  output.cols[3].w = scale;

  return output;
}

__device__ __host__ Mat4 gemm(const Mat4 a, const Mat4 b, Mat4 c) {
  c.cols[0].x = a.cols[0].x * b.cols[0].x + a.cols[1].x * b.cols[0].y + a.cols[2].x * b.cols[0].z + a.cols[3].x * b.cols[0].w + c.cols[0].x;
  c.cols[0].y = a.cols[0].y * b.cols[0].x + a.cols[1].y * b.cols[0].y + a.cols[2].y * b.cols[0].z + a.cols[3].y * b.cols[0].w + c.cols[0].y;
  c.cols[0].z = a.cols[0].z * b.cols[0].x + a.cols[1].z * b.cols[0].y + a.cols[2].z * b.cols[0].z + a.cols[3].z * b.cols[0].w + c.cols[0].z;
  c.cols[0].w = a.cols[0].w * b.cols[0].x + a.cols[1].w * b.cols[0].y + a.cols[2].w * b.cols[0].z + a.cols[3].w * b.cols[0].w + c.cols[0].w;

  c.cols[1].x = a.cols[0].x * b.cols[1].x + a.cols[1].x * b.cols[1].y + a.cols[2].x * b.cols[1].z + a.cols[3].x * b.cols[1].w + c.cols[1].x;
  c.cols[1].y = a.cols[0].y * b.cols[1].x + a.cols[1].y * b.cols[1].y + a.cols[2].y * b.cols[1].z + a.cols[3].y * b.cols[1].w + c.cols[1].y;
  c.cols[1].z = a.cols[0].z * b.cols[1].x + a.cols[1].z * b.cols[1].y + a.cols[2].z * b.cols[1].z + a.cols[3].z * b.cols[1].w + c.cols[1].z;
  c.cols[1].w = a.cols[0].w * b.cols[1].x + a.cols[1].w * b.cols[1].y + a.cols[2].w * b.cols[1].z + a.cols[3].w * b.cols[1].w + c.cols[1].w;

  c.cols[2].x = a.cols[0].x * b.cols[2].x + a.cols[1].x * b.cols[2].y + a.cols[2].x * b.cols[2].z + a.cols[3].x * b.cols[2].w + c.cols[2].x;
  c.cols[2].y = a.cols[0].y * b.cols[2].x + a.cols[1].y * b.cols[2].y + a.cols[2].y * b.cols[2].z + a.cols[3].y * b.cols[2].w + c.cols[2].y;
  c.cols[2].z = a.cols[0].z * b.cols[2].x + a.cols[1].z * b.cols[2].y + a.cols[2].z * b.cols[2].z + a.cols[3].z * b.cols[2].w + c.cols[2].z;
  c.cols[2].w = a.cols[0].w * b.cols[2].x + a.cols[1].w * b.cols[2].y + a.cols[2].w * b.cols[2].z + a.cols[3].w * b.cols[2].w + c.cols[2].w;

  c.cols[3].x = a.cols[0].x * b.cols[3].x + a.cols[1].x * b.cols[3].y + a.cols[2].x * b.cols[3].z + a.cols[3].x * b.cols[3].w + c.cols[3].x;
  c.cols[3].y = a.cols[0].y * b.cols[3].x + a.cols[1].y * b.cols[3].y + a.cols[2].y * b.cols[3].z + a.cols[3].y * b.cols[3].w + c.cols[3].y;
  c.cols[3].z = a.cols[0].z * b.cols[3].x + a.cols[1].z * b.cols[3].y + a.cols[2].z * b.cols[3].z + a.cols[3].z * b.cols[3].w + c.cols[3].z;
  c.cols[3].w = a.cols[0].w * b.cols[3].x + a.cols[1].w * b.cols[3].y + a.cols[2].w * b.cols[3].z + a.cols[3].w * b.cols[3].w + c.cols[3].w;

  return c;
}
 
__global__
void mat4_batched_gemm(
  int n,
  const Mat4* __restrict__ a,
  const Mat4*__restrict__ b,
  Mat4*__restrict__ c)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = index; i < n; i += stride) {
    c[i] = gemm(a[i], b[i], c[i]);
  }
}

struct BatchContext {
  int batch_id;
};

void CUDART_CB on_batch_ready(void* userData) {
  BatchContext* ctx = static_cast<BatchContext*>(userData);
  std::cout << "Finished Batch: "<< ctx->batch_id << "\n";
};

int cuBLAS_mat4_batched_gemm(cublasHandle_t *handle, const Mat4 *a_mat, const Mat4 *b_mat, Mat4 *c_mat, int batch_size) {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  
  cublasStatus_t stat = cublasSgemmStridedBatched(*handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    4, 4, 4,
                    &alpha,
                    reinterpret_cast<const float*>(a_mat), 4, 16,
                    reinterpret_cast<const float*>(b_mat), 4, 16,
                    &beta,
                    reinterpret_cast<float*>(c_mat), 4, 16,
                    batch_size);

  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("cublasSgemmStridedBatched failed\n");
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::array<long, 2> run_program(cublasHandle_t *handle) {

  int num_matrices = 320000;

  Buffers<Mat4> a(num_matrices);
  Buffers<Mat4> b(num_matrices);
  Buffers<Mat4> c_1(num_matrices);
  Buffers<Mat4> c_2(num_matrices);

  for (int i = 0; i < num_matrices; i++) {
    a.host[i] = sample_mat4();
    b.host[i] = sample_mat4();
    c_1.host[i] = sample_mat4();
    c_2.host[i] = c_1.host[i];

    //a.host[i] = scaled_identity(2.0);
    //b.host[i] = scaled_identity(1.0);
    //c_1.host[i] = scaled_identity(1.0);
    //c_2.host[i] = c_1.host[i];
  }

  a.copy_to_device();
  b.copy_to_device();
  c_1.copy_to_device();
  c_2.copy_to_device();

  auto start = std::chrono::steady_clock::now();

  dim3 block(256, 1, 1);
  dim3 grid((num_matrices + block.x - 1) / block.x, 1, 1);
  mat4_batched_gemm<<<grid, block>>>(
    num_matrices,
    a.device,
    b.device,
    c_1.device
  );

  c_1.copy_to_host();

  auto end = std::chrono::steady_clock::now();
  auto custom_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  bool doCUBLAS = false;

  start = std::chrono::steady_clock::now();

  if (doCUBLAS) {

    cuBLAS_mat4_batched_gemm(handle, a.device, b.device, c_2.device, num_matrices);

    c_2.copy_to_host();

  }

  end = std::chrono::steady_clock::now();
  auto cublas_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  for (int i = 0; i < num_matrices; i++) {
    //std::cout << "c_1: " << i << " c1r1: " << c_1.host[i].cols[1].y << "\n";
    if (doCUBLAS) {
      //std::cout << "c_2: " << i << " c1r1: " << c_2.host[i].cols[1].y << "\n";
      assert(nearlyEqual(c_1.host[i], c_2.host[i], 1e-4f));
    }
  }

  return {custom_duration.count(), cublas_duration.count()};
}
 
int main(int argc, char* argv[])
{

  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }

  // warm up
  int numWarmups = 0;
  for(int i = 0; i < numWarmups; i++) {
    run_program(&handle);
  }

  long totalCustomMicroseconds = 0;
  long totalCUBLASMicroseconds = 0;
  int numRuns = 1;
  for(int i = 0; i < numRuns; i++) {
    std::array<long, 2> times = run_program(&handle);
    totalCustomMicroseconds += times[0];
    totalCUBLASMicroseconds += times[1];
  }

  double averageCustomMicroseconds = (double)totalCustomMicroseconds / numRuns;
  double averageCUBLASMicroseconds = (double)totalCUBLASMicroseconds / numRuns;

  std::cout << "Custom time taken: "<< std::fixed << std::setprecision(2) << averageCustomMicroseconds << " microseconds\n";
  std::cout << "cublas time taken: "<< std::fixed << std::setprecision(2) << averageCUBLASMicroseconds << " microseconds\n";

  cublasDestroy(handle);

  return 0;
}