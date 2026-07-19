#include "nn/ops/gemm.hpp"

#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace nn::kernel {

bool gemm_avx_available() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx");
#else
  return false;
#endif
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx")))
void gemm_avx(const float* A, const float* B, float* C,
              int M, int N, int K) {
  if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("GEMM dimensions must be non-negative");
  if (M == 0 || N == 0) return;
  if ((K != 0 && (A == nullptr || B == nullptr)) || C == nullptr) throw std::invalid_argument("AVX GEMM received a null matrix pointer");

  int n = 0;
  for (; n + 15 < N; n += 16) {
    int m = 0;
    for (; m + 5 < M; m += 6) {
      __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
      __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
      __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
      __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
      __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
      __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        const __m256 b0 = _mm256_loadu_ps(B + k * N + n);
        const __m256 b1 = _mm256_loadu_ps(B + k * N + n + 8);
#define NN_AVX_ACC(row, l, r) do { const __m256 av = _mm256_broadcast_ss(A + (m + row) * K + k); l = _mm256_add_ps(l, _mm256_mul_ps(av, b0)); r = _mm256_add_ps(r, _mm256_mul_ps(av, b1)); } while (false)
        NN_AVX_ACC(0, c00, c01); NN_AVX_ACC(1, c10, c11);
        NN_AVX_ACC(2, c20, c21); NN_AVX_ACC(3, c30, c31);
        NN_AVX_ACC(4, c40, c41); NN_AVX_ACC(5, c50, c51);
#undef NN_AVX_ACC
      }
#define NN_AVX_STORE(row, l, r) do { _mm256_storeu_ps(C + (m + row) * N + n, l); _mm256_storeu_ps(C + (m + row) * N + n + 8, r); } while (false)
      NN_AVX_STORE(0, c00, c01); NN_AVX_STORE(1, c10, c11);
      NN_AVX_STORE(2, c20, c21); NN_AVX_STORE(3, c30, c31);
      NN_AVX_STORE(4, c40, c41); NN_AVX_STORE(5, c50, c51);
#undef NN_AVX_STORE
    }
    for (; m < M; ++m) {
      __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        const __m256 av = _mm256_broadcast_ss(A + m * K + k);
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(av, _mm256_loadu_ps(B + k * N + n)));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(av, _mm256_loadu_ps(B + k * N + n + 8)));
      }
      _mm256_storeu_ps(C + m * N + n, c0);
      _mm256_storeu_ps(C + m * N + n + 8, c1);
    }
  }
  for (; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      float sum = 0.0F;
      for (int k = 0; k < K; ++k) sum += A[m * K + k] * B[k * N + n];
      C[m * N + n] = sum;
    }
  }
}
#else
void gemm_avx(const float*, const float*, float*, int, int, int) { throw std::runtime_error("AVX GEMM is unavailable on this architecture"); }
#endif

}  // namespace nn::kernel