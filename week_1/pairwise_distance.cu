#include <iostream>
#include <math.h>

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
 
__global__
void distance(int n, int n_sqr, float3 *points, float *distances)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n_sqr; i += stride) {
  //for (int i = index * stride; i < n_sqr; i += 1) {
    int row = i / n;
    int col = i % n;
    float3 row_vec = points[row];
    float3 col_vec = points[col];
    float3 diff = row_vec - col_vec;
    distances[i] = norm3df(diff.x, diff.y, diff.z);
  }
}
 
int main(int argc, char* argv[])
{
 int N = std::stoi(argv[1]);
 int n_sqr = N * N;
 float3 *host_points = new float3[N];
 
 // initialize x and y arrays on the host
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
 
 // Run kernel on 1M elements on the GPU
 distance<<<8, 256>>>(N, n_sqr, device_points, device_distances);

 float *host_distances = new float[n_sqr];
 cudaMemcpy(host_distances,
           device_distances,
           n_sqr * sizeof(float),
           cudaMemcpyDeviceToHost);
 
 // Check for errors (all values should be 3.0f)
 float maxDistance = 0.0f;
 for (int i = 0; i < n_sqr; i++) {
   maxDistance = fmax(maxDistance, host_distances[i]);
 }
 std::cout << "Max Distance: " << maxDistance << std::endl;
 
 // Free memory
 cudaFree(device_points);
 delete[] host_points;
 cudaFree(device_distances);
 delete[] host_distances;
  return 0;
}