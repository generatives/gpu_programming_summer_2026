#include <iostream>
#include <math.h>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "buffers.h"

const int SPHERE_BLOCK_DIM = 8;
const int ROBOT_BLOCK_DIM = 32;
const int OBSTACLES_PER_THREAD = 32;

struct Sphere {
    float4 data; // x, y, z, radius

    Sphere() = default;

    __host__ __device__
    Sphere(float radius, float3 position)
        : data{position.x, position.y, position.z, radius} {}

    __host__ __device__
    float radius() const { return data.w; }

    __host__ __device__
    float3 position() const {
        return make_float3(data.x, data.y, data.z);
    }
};

__device__ float2 operator-(const float2 &a, const float2 &b) {
  return make_float2(a.x-b.x, a.y-b.y);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ bool spheres_collide(Sphere a, Sphere b) {
  float boundary = a.radius() + b.radius();
  float boundarySq = boundary * boundary;

  float diff_x = a.data.x - b.data.x;
  float diff_y = a.data.y - b.data.y;
  float diff_z = a.data.z - b.data.z;

  float distanceSq = (diff_x) * (diff_x) +
    (diff_y) * (diff_y) +
    (diff_z) * (diff_z);

  return distanceSq < boundarySq;
}
 
__global__
void check_collisions(
  int b, int j, int e,
  const Sphere* __restrict__ robots,
  const Sphere*__restrict__ obstacles,
  uint8_t*__restrict__ collides)
{
  int sphere = blockIdx.x * blockDim.x + threadIdx.x;
  int localSphere = threadIdx.x;
  int robot = blockIdx.y * blockDim.y + threadIdx.y;
  int localRobot = threadIdx.y;
  int baseObstacle = (blockIdx.z * blockDim.z + threadIdx.z) * OBSTACLES_PER_THREAD;

  __shared__ Sphere shared_obstacles[OBSTACLES_PER_THREAD];

  int load_obstacle_idx = localRobot * SPHERE_BLOCK_DIM + localSphere;
  int global_load_obstacle_idx = baseObstacle + load_obstacle_idx;
  if (load_obstacle_idx < OBSTACLES_PER_THREAD && global_load_obstacle_idx < e) {
      shared_obstacles[load_obstacle_idx] = obstacles[global_load_obstacle_idx];
  }

  __syncthreads();

  if (robot < b && sphere < j && baseObstacle < e) {
    Sphere robot_sphere = robots[robot * j + sphere];

    int8_t local_collides = 0;
    for(int obstacle = 0; obstacle < OBSTACLES_PER_THREAD; obstacle++) {
      Sphere obstacle_sphere = shared_obstacles[obstacle];
      local_collides |= spheres_collide(robot_sphere, obstacle_sphere);
    }

    if (local_collides == 1) {
      collides[robot] = 1;
    }
  }
}

__global__
void score_paths(
  const int num_paths,
  const uint32_t* __restrict__ robot_path_ranges,
  const uint8_t* __restrict__ collides,
  float_t*__restrict__ scores)
{
  for (int w = 0; w < 250; w++) {
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_idx < num_paths) {
      int path_start = robot_path_ranges[path_idx];
      int path_end = robot_path_ranges[path_idx + 1];

      float score = 0;
      for (int body_idx = path_start; body_idx < path_end; body_idx++) {
        score += collides[body_idx] ? -1000.0 : 1.0;
      }

      scores[path_idx] = score;
    }
  }
}

void add_cube(float3 p, float r, std::vector<Sphere> &spheres) {
  spheres.push_back({r, make_float3(p.x - r, p.y - r, p.z - r)});
  spheres.push_back({r, make_float3(p.x + r, p.y - r, p.z - r)});
  spheres.push_back({r, make_float3(p.x - r, p.y + r, p.z - r)});
  spheres.push_back({r, make_float3(p.x + r, p.y + r, p.z - r)});
  spheres.push_back({r, make_float3(p.x - r, p.y - r, p.z + r)});
  spheres.push_back({r, make_float3(p.x + r, p.y - r, p.z + r)});
  spheres.push_back({r, make_float3(p.x - r, p.y + r, p.z + r)});
  spheres.push_back({r, make_float3(p.x + r, p.y + r, p.z + r)});
}

void draw_cubes_in_grid(int2 start_grid_idx, int2 end_grid_idx, std::vector<Sphere> &spheres) {
  int x_diff = end_grid_idx.x - start_grid_idx.x;
  int y_diff = end_grid_idx.y - start_grid_idx.y;
  int longer_distance = max(x_diff, y_diff);
  int num_spheres = longer_distance * 10;

  float radius = 0.25;

  for(int i = 0; i < num_spheres; i++) {
    float progress = (float)i / num_spheres;
    float x = start_grid_idx.x + progress * x_diff;
    float y = start_grid_idx.y + progress * y_diff;
    float3 position = make_float3(x, y, 0.0);
    add_cube(position, radius, spheres);
  }
}

bool check_for_true(uint8_t *values, int start, int end) {
  for(int i = start; i < end; i++) {
    if(values[i] == 1) {
      return true;
    }
  }
  return false;
}

struct BatchContext {
  int batch_id;
  Buffers<float_t>* scores;
};

void CUDART_CB on_batch_ready(void* userData) {
  BatchContext* ctx = static_cast<BatchContext*>(userData);
  //std::cout << "Finished Batch: "<< ctx->batch_id << "\n";

  float_t *scores = ctx->scores->host;
  assert(scores[0] < 0);
  assert(scores[1] > 0);
  assert(scores[2] < 0);
  assert(scores[3] < 0);
  assert(scores[4] > 0);
  assert(scores[5] < 0);
  //std::cout << "Assertions passed\n";
};

int run_program() {

  int map_dim = 9;
  uint8_t map[] = {
    1, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 1, 1, 1, 1,
  };

  //collides at the end
  int2 robot_path_1[] = {
    make_int2(1, 8),
    make_int2(1, 5),
    make_int2(4, 5),
    make_int2(7, 5),
  };

  //does not collide
  int2 robot_path_2[] = {
    make_int2(1, 8),
    make_int2(1, 5),
    make_int2(5, 5),
    make_int2(5, 0),
  };

  //collides in the middle
  int2 robot_path_3[] = {
    make_int2(1, 8),
    make_int2(1, 4),
    make_int2(8, 4),
  };

  int num_path_repeats = 100;
  int num_paths_templates = 3;
  int num_paths = num_paths_templates * num_path_repeats;

  int j = 8;
  Buffers<uint32_t> robot_path_range_buffer(num_paths + 1);
  robot_path_range_buffer.host[0] = 0;
  std::vector<Sphere> robot_spheres_v;
  for (int r = 0; r < num_path_repeats; r++) {
    for(int i = 0; i < 3; i++) {
      draw_cubes_in_grid(robot_path_1[i], robot_path_1[i + 1], robot_spheres_v);
    }
    robot_path_range_buffer.host[r * num_paths_templates + 1] = robot_spheres_v.size() / j;

    for(int i = 0; i < 3; i++) {
      draw_cubes_in_grid(robot_path_2[i], robot_path_2[i + 1], robot_spheres_v);
    }
    robot_path_range_buffer.host[r * num_paths_templates + 2] = robot_spheres_v.size() / j;

    for(int i = 0; i < 2; i++) {
      draw_cubes_in_grid(robot_path_3[i], robot_path_3[i + 1], robot_spheres_v);
    }
    robot_path_range_buffer.host[r * num_paths_templates + 3] = robot_spheres_v.size() / j;
  }
  
  int max_path_length = *std::max_element(robot_path_range_buffer.host, robot_path_range_buffer.host + robot_path_range_buffer.count);

  int num_spheres = robot_spheres_v.size();
  int b = num_spheres / j;
  
  Buffers<Sphere> robot_spheres(num_spheres);

  std::vector<Sphere> obstacle_spheres_v;
  for(int x = 0; x < map_dim; x++) {
    for(int y = 0; y < map_dim; y++) {
      int i = y * map_dim + x;
      if(map[i] == 1) {
        for(int z = 0; z < map_dim; z++) {
          for(int i = 0; i < 10; i++) {
            add_cube(make_float3(x, y, z), 0.25, obstacle_spheres_v);
          }
        }
      }
    }
  }
  int e = obstacle_spheres_v.size();

  //std::cout << "B: "<< b << " J: "<< j << " E: "<< e << "\n";

  Buffers<Sphere> obstacle_spheres(e);

  // Separate buffers for collision data so the collision and scoring stages can run in parallel
  // We need to copy from one to the other after collision ends/before scoring starts 
  Buffers<uint8_t> collides_out(b);
  Buffers<uint8_t> collides_in(b);

  Buffers<float_t> scores(num_paths);

  int num_batches = 20;

  cudaStream_t copy_in_stream, compute_collision_stream, copy_across_stream, compute_score_stream, copy_out_stream;
  cudaEvent_t copy_in_done[num_batches], compute_collision_done[num_batches], copy_across_done[num_batches],  compute_score_done[num_batches], copy_out_done[num_batches];

  auto start = std::chrono::steady_clock::now();

  cudaStreamCreate(&copy_in_stream);
  cudaStreamCreate(&compute_collision_stream);
  cudaStreamCreate(&copy_across_stream);
  cudaStreamCreate(&compute_score_stream);
  cudaStreamCreate(&copy_out_stream);

  for (int i = 0; i < num_batches; i++) {
    cudaEventCreate(&copy_in_done[i]);
    cudaEventCreate(&compute_collision_done[i]);
    cudaEventCreate(&copy_across_done[i]);
    cudaEventCreate(&compute_score_done[i]);
    cudaEventCreate(&copy_out_done[i]);
  }

  // Prime the pipeline: kick off copy of batch 0
  std::copy(robot_spheres_v.begin(), robot_spheres_v.end(), robot_spheres.host);
  robot_spheres.copy_to_device_async(copy_in_stream);

  std::copy(obstacle_spheres_v.begin(), obstacle_spheres_v.end(), obstacle_spheres.host);
  obstacle_spheres.copy_to_device_async(copy_in_stream);

  // This only needs to be done once for now
  robot_path_range_buffer.copy_to_device_async(copy_in_stream);
  
  cudaEventRecord(copy_in_done[0], copy_in_stream);

  BatchContext batches[num_batches];
  for (int n = 0; n < num_batches; n++) {
    // Check for collisions
    // Ensuring data is available and the output buffer is free
    cudaStreamWaitEvent(compute_collision_stream, copy_in_done[n], 0);
    if (n > 0) {
      cudaStreamWaitEvent(compute_collision_stream, copy_across_done[n-1], 0);
    }
    dim3 coll_block(SPHERE_BLOCK_DIM, ROBOT_BLOCK_DIM, 1);
    dim3 coll_grid(1, (b + coll_block.y - 1) / coll_block.y, (e + OBSTACLES_PER_THREAD - 1) / OBSTACLES_PER_THREAD);
    check_collisions<<<coll_grid, coll_block, 0, compute_collision_stream>>>(
      b, j, e,
      robot_spheres.device,
      obstacle_spheres.device,
      collides_out.device
    );
    cudaEventRecord(compute_collision_done[n], compute_collision_stream);

    // Copy across buffers to the next collision check can start
    // Make sure the inputs are ready and the output is free
    cudaStreamWaitEvent(copy_across_stream, compute_collision_done[n], 0);
    if (n > 0) {
      cudaStreamWaitEvent(copy_across_stream, compute_score_done[n-1], 0);
    }
    collides_out.copy_across_device_buffers_async(&collides_in, copy_across_stream);
    cudaEventRecord(copy_across_done[n], copy_across_stream);
    
    // Score each trajectory
    // We need to wait on our input data (collision states) and for the previous copy to complete so we can use the buffer
    cudaStreamWaitEvent(compute_score_stream, copy_across_done[n], 0);
    if (n > 0) {
      cudaStreamWaitEvent(compute_score_stream, copy_out_done[n-1], 0);
    }
    dim3 score_block(256, 1, 1);
    dim3 score_grid((num_paths + score_block.x - 1) / score_block.x, 1, 1);
    score_paths<<<score_grid, score_block, 0, compute_score_stream>>>(
      num_paths,
      robot_path_range_buffer.device,
      collides_in.device,
      scores.device
    );
    cudaEventRecord(compute_score_done[n], compute_score_stream);

    // Once collisions are done we can immedietly start copying in the next batch, if it exists
    if (n + 1 < num_batches) {
      cudaStreamWaitEvent(copy_in_stream, compute_collision_done[n], 0);
      std::copy(robot_spheres_v.begin(), robot_spheres_v.end(), robot_spheres.host);
      robot_spheres.copy_to_device_async(copy_in_stream);

      std::copy(obstacle_spheres_v.begin(), obstacle_spheres_v.end(), obstacle_spheres.host);
      obstacle_spheres.copy_to_device_async(copy_in_stream);
      cudaEventRecord(copy_in_done[n+1], copy_in_stream);
    }

    cudaStreamWaitEvent(copy_out_stream, compute_score_done[n], 0);
    scores.copy_to_host_async(copy_out_stream);
    batches[n] = BatchContext{n, &scores};
    cudaLaunchHostFunc(copy_out_stream, on_batch_ready, &(batches[n]));
    cudaEventRecord(copy_out_done[n], copy_out_stream);
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count();
}
 
int main(int argc, char* argv[])
{
  //int N = std::stoi(argv[1]);

  // warm up
  int numWarmups = 3;
  for(int i = 0; i < numWarmups; i++) {
    run_program();
  }

  int totalMicroseconds = 0;
  int numRuns = 10;
  for(int i = 0; i < numRuns; i++) {
    totalMicroseconds += run_program();
  }

  float averageMicroseconds = (float)totalMicroseconds / numRuns;

  std::cout << "Time taken: "<< std::fixed << std::setprecision(2) << averageMicroseconds << " microseconds\n";

  return 0;
}