#include "nn/ops/gemm.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  const int runs = argc > 1 ? std::stoi(argv[1]) : 5;
  if (runs <= 0) return 2;
  constexpr int dimension = 1024;
  const size_t elements = static_cast<size_t>(dimension) * dimension;
  std::vector<float> a(elements);
  std::vector<float> b(elements);
  std::vector<float> c(elements);
  for (size_t index = 0; index < elements; ++index) {
    a[index] = static_cast<float>(static_cast<int>(index % 127) - 63) / 127.0F;
    b[index] = static_cast<float>(static_cast<int>(index % 113) - 56) / 113.0F;
  }

  nn::kernel::gemm(
      a.data(), b.data(), c.data(), dimension, dimension, dimension);
  double total_seconds = 0.0;
  double best_seconds = 1.0e100;
  for (int run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    nn::kernel::gemm(
        a.data(), b.data(), c.data(), dimension, dimension, dimension);
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - start).count();
    total_seconds += seconds;
    best_seconds = std::min(best_seconds, seconds);
  }
  const double operations =
      2.0 * dimension * dimension * static_cast<double>(dimension);
  const double mean_gflops = operations / (total_seconds / runs) / 1.0e9;
  const double best_gflops = operations / best_seconds / 1.0e9;
  const char* backend = nn::kernel::gemm_avx2_available()
                            ? "avx2_fma"
                            : (nn::kernel::gemm_avx_available() ? "avx" : "scalar");
  std::cout << std::fixed << std::setprecision(3)
            << "backend=" << backend << " runs=" << runs
            << " mean_gflops=" << mean_gflops
            << " best_gflops=" << best_gflops
            << " checksum=" << c[0] << '\n';
  return mean_gflops > 25.0 ? 0 : 1;
}