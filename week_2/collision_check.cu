#include <iostream>
#include <math.h>
#include <chrono>

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
 
__global__
void distance(int n, int n_sqr, float3 *points, float *distances)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n_sqr; i += stride) {
    int row = i / n;
    int col = i % n;
    float3 row_vec = points[row];
    float3 col_vec = points[col];
    float3 diff = row_vec - col_vec;
    distances[i] = norm3df(diff.x, diff.y, diff.z);
  }
}

void run_program(int N) {
  int n_sqr = N * N;
  float3 *host_points = new float3[N];

  for (int i = 0; i < N; i++) {
    host_points[i] = make_float3(i + 1.0f, i + 2.0f, i + 3.0f);
  }

  float3* device_points;
  cudaMalloc(&device_points, N * sizeof(float3));

  cudaMemcpy(device_points,
            host_points,
            N * sizeof(float3),
            cudaMemcpyHostToDevice);


  float* device_distances;
  cudaMalloc(&device_distances, n_sqr * sizeof(float));

  int blockSize = 256;
  int gridSize = (int)std::ceil((float)N / blockSize);

  distance<<<gridSize, blockSize>>>(N, n_sqr, device_points, device_distances);

  float *host_distances = new float[n_sqr];
  cudaMemcpy(host_distances,
            device_distances,
            n_sqr * sizeof(float),
            cudaMemcpyDeviceToHost);

  float maxDistance = 0.0f;
  for (int i = 0; i < n_sqr; i++) {
    maxDistance = fmax(maxDistance, host_distances[i]);
  }
  //std::cout << "Max Distance: " << maxDistance << std::endl;

  // Free memory
  cudaFree(device_points);
  delete[] host_points;
  cudaFree(device_distances);
  delete[] host_distances;
}
 
int main(int argc, char* argv[])
{
  int N = std::stoi(argv[1]);

  // warm up
  for(int i = 0; i < 3; i++) {
    run_program(N);
  }

  auto start = std::chrono::steady_clock::now();
   
  int numRuns = 5;
  for(int i = 0; i < numRuns; i++) {
    run_program(N);
  }

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto averageMilliseconds = (float)duration.count() / (float)numRuns;

  std::cout << "Time taken: " << averageMilliseconds << " milliseconds";

  return 0;
}