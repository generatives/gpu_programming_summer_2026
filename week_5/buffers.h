template <typename T>
struct Buffers {
  T* host;
  T* device;
  size_t count = 0;

  Buffers() = default;

  explicit Buffers(size_t n) : count(n) {
    cudaMalloc(&device, n * sizeof(T));
    cudaHostAlloc(&host, n * sizeof(T), cudaHostAllocDefault);
  }

  ~Buffers() {
    if (device) cudaFree(device);
    if (host) cudaFreeHost(host);
  }

  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(Buffers&& other) noexcept
    : host(other.host), device(other.device), count(other.count) {
    other.host = nullptr;
    other.device = nullptr;
    other.count = 0;
  }

  Buffers& operator=(Buffers&& other) noexcept {
    if (this != &other) {
      if (device) cudaFree(device);
      if (host) delete[] host;

      device = other.device;
      host = other.host;
      count = other.count;

      other.device = nullptr;
      other.host = nullptr;
      other.count = 0;
    }
    return *this;
  }

  void copy_to_device() {
    cudaMemcpy(device,
      host,
      count * sizeof(T),
      cudaMemcpyHostToDevice);
  }

  void copy_to_device_async(cudaStream_t stream = (cudaStream_t)0) {
    cudaMemcpyAsync(
      device,
      host,
      count * sizeof(T),
      cudaMemcpyHostToDevice,
      stream);
  }

  void copy_to_host() {
    cudaMemcpy(host,
      device,
      count * sizeof(T),
      cudaMemcpyDeviceToHost);
  }

  void copy_to_host_async(cudaStream_t stream = (cudaStream_t)0) {
    cudaMemcpyAsync(
      host,
      device,
      count * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream);
  }

  void copy_across_device_buffers(Buffers<T> *target_buffer) {
    cudaMemcpy(
      target_buffer->device,
      device,
      count * sizeof(T),
      cudaMemcpyDeviceToDevice);
  }

  void copy_across_device_buffers_async(Buffers<T> *target_buffer, cudaStream_t stream = (cudaStream_t)0) {
    cudaMemcpyAsync(
      target_buffer->device,
      device,
      count * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream);
  }
};