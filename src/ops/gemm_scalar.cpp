#include "nn/ops/gemm.hpp"

#include <stdexcept>

namespace nn::kernel {

void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 float alpha, float beta,
                 bool transA, bool transB) {
  if (M < 0 || N < 0 || K < 0) {
    throw std::invalid_argument("GEMM dimensions must be non-negative");
  }
  if (M == 0 || N == 0) return;
  if ((M != 0 && K != 0 && A == nullptr) ||
      (K != 0 && N != 0 && B == nullptr) ||
      (M != 0 && N != 0 && C == nullptr)) {
    throw std::invalid_argument("GEMM received a null matrix pointer");
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        const float a = transA ? A[k * M + m] : A[m * K + k];
        const float b = transB ? B[n * K + k] : B[k * N + n];
        sum += a * b;
      }
      const int index = m * N + n;
      C[index] = alpha * sum + beta * C[index];
    }
  }
}

}  // namespace nn::kernel
