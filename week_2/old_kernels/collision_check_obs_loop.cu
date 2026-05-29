#include <iostream>
#include <math.h>
#include <chrono>
#include <cassert>
#include <vector>
#include <iomanip>
#include "buffers.h"

const int OBSTACLES_PER_THREAD = 8;

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

  float3 pos_diff = a.position() - b.position();
  float distanceSq = (pos_diff.x) * (pos_diff.x) +
    (pos_diff.y) * (pos_diff.y) +
    (pos_diff.z) * (pos_diff.z);

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
  int robot = blockIdx.y * blockDim.y + threadIdx.y;
  int baseObstacle = (blockIdx.z * blockDim.z + threadIdx.z) * OBSTACLES_PER_THREAD;

  if (robot < b && sphere < j && baseObstacle < e) {
    Sphere robot_sphere = robots[robot * j + sphere];

    int8_t local_collides = 0;
    for(int obstacle = baseObstacle; obstacle < baseObstacle + OBSTACLES_PER_THREAD; obstacle++) {
      Sphere obstacle_sphere = obstacles[obstacle];
      local_collides |= spheres_collide(robot_sphere, obstacle_sphere);
    }

    if (local_collides == 1) {
      collides[robot] = 1;
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
  int num_spheres = longer_distance * 3;

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

  int j = 8;
  int robot_path_ranges[3];
  std::vector<Sphere> robot_spheres_v;
  for(int i = 0; i < 3; i++) {
    draw_cubes_in_grid(robot_path_1[i], robot_path_1[i + 1], robot_spheres_v);
  }
  robot_path_ranges[0] = robot_spheres_v.size() / j;

  for(int i = 0; i < 3; i++) {
    draw_cubes_in_grid(robot_path_2[i], robot_path_2[i + 1], robot_spheres_v);
  }
  robot_path_ranges[1] = robot_spheres_v.size() / j;

  for(int i = 0; i < 2; i++) {
    draw_cubes_in_grid(robot_path_3[i], robot_path_3[i + 1], robot_spheres_v);
  }
  robot_path_ranges[2] = robot_spheres_v.size() / j;

  int num_spheres = robot_spheres_v.size();
  int b = num_spheres / j;
  
  Buffers<Sphere> robot_spheres(num_spheres);
  std::copy(robot_spheres_v.begin(), robot_spheres_v.end(), robot_spheres.host);
  robot_spheres.copy_to_device();


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
  Buffers<Sphere> obstacle_spheres(e);
  std::copy(obstacle_spheres_v.begin(), obstacle_spheres_v.end(), obstacle_spheres.host);
  obstacle_spheres.copy_to_device();

  Buffers<uint8_t> collides(b);
  std::fill_n(collides.host, collides.count, 0);
  collides.copy_to_device();

  dim3 block(8, 32, 1);
  dim3 grid(1, (b + block.y - 1) / block.y, (e + OBSTACLES_PER_THREAD - 1) / OBSTACLES_PER_THREAD);

  auto start = std::chrono::steady_clock::now();

  check_collisions<<<grid, block>>>(
      b, j, e,
      robot_spheres.device,
      obstacle_spheres.device,
      collides.device
  );

  collides.copy_to_host();

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  assert(check_for_true(collides.host, 0, robot_path_ranges[0]));
  assert(!check_for_true(collides.host, robot_path_ranges[0], robot_path_ranges[1]));
  assert(check_for_true(collides.host, robot_path_ranges[1], robot_path_ranges[2]));

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