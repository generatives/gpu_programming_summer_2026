template <typename T>
struct Buffers {
  T* host;
  T* device;
  size_t count = 0;

  Buffers() = default;

  explicit Buffers(size_t n) : count(n) {
    cudaMalloc(&device, n * sizeof(T));
    host = new T[n];
  }

  ~Buffers() {
    if (device) cudaFree(device);
    if (host) delete[] host;
  }

  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(Buffers&& other) noexcept
    : host(other.host), device(other.device), count(other.count) {
    other.host = nullptr;
    other.device = nullptr;
    other.count = 0;
  }

  Buffers(std::vector<T> v) : Buffers(v.size()) {
    copy_vector(v);
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

  void copy_vector(std::vector<T> v) {
    std::copy(v.begin(), v.end(), host);
  }

  void copy_to_device() {
    cudaMemcpy(device,
      host,
      count * sizeof(T),
      cudaMemcpyHostToDevice);
  }

  void copy_to_host() {
    cudaMemcpy(host,
      device,
      count * sizeof(T),
      cudaMemcpyDeviceToHost);
  }
};