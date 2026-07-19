#include "nn/ops/gemm.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: gemm_dump M N K alpha beta transA transB\n";
    return 2;
  }
  const int M = std::stoi(argv[1]);
  const int N = std::stoi(argv[2]);
  const int K = std::stoi(argv[3]);
  const float alpha = std::stof(argv[4]);
  const float beta = std::stof(argv[5]);
  const bool transA = std::stoi(argv[6]) != 0;
  const bool transB = std::stoi(argv[7]) != 0;

  std::vector<float> a(static_cast<size_t>(M) * K);
  std::vector<float> b(static_cast<size_t>(K) * N);
  std::vector<float> c(static_cast<size_t>(M) * N);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 29) - 14) / 7.0f;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 31) - 15) / 9.0f;
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = static_cast<float>(static_cast<int>((i * 7 + 1) % 13) - 6) / 5.0f;

  nn::kernel::gemm_scalar(a.data(), b.data(), c.data(), M, N, K,
                          alpha, beta, transA, transB);
  std::cout << std::setprecision(9);
  for (const float value : c) std::cout << value << '\n';
}
