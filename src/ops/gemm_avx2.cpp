#include "nn/ops/gemm.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace nn::kernel {

bool gemm_avx2_available() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K) {
  if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("GEMM dimensions must be non-negative");
  if (M == 0 || N == 0) return;
  if ((K != 0 && (A == nullptr || B == nullptr)) || C == nullptr) throw std::invalid_argument("AVX2 GEMM received a null matrix pointer");

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
#define NN_AVX2_ACC(row, l, r) do { const __m256 av = _mm256_broadcast_ss(A + (m + row) * K + k); l = _mm256_fmadd_ps(av, b0, l); r = _mm256_fmadd_ps(av, b1, r); } while (false)
        NN_AVX2_ACC(0, c00, c01); NN_AVX2_ACC(1, c10, c11);
        NN_AVX2_ACC(2, c20, c21); NN_AVX2_ACC(3, c30, c31);
        NN_AVX2_ACC(4, c40, c41); NN_AVX2_ACC(5, c50, c51);
#undef NN_AVX2_ACC
      }
#define NN_AVX2_STORE(row, l, r) do { _mm256_storeu_ps(C + (m + row) * N + n, l); _mm256_storeu_ps(C + (m + row) * N + n + 8, r); } while (false)
      NN_AVX2_STORE(0, c00, c01); NN_AVX2_STORE(1, c10, c11);
      NN_AVX2_STORE(2, c20, c21); NN_AVX2_STORE(3, c30, c31);
      NN_AVX2_STORE(4, c40, c41); NN_AVX2_STORE(5, c50, c51);
#undef NN_AVX2_STORE
    }
    for (; m < M; ++m) {
      __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        const __m256 av = _mm256_broadcast_ss(A + m * K + k);
        c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(B + k * N + n), c0);
        c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(B + k * N + n + 8), c1);
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
void gemm_avx2(const float*, const float*, float*, int, int, int) { throw std::runtime_error("AVX2 GEMM is unavailable on this architecture"); }
#endif

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, float alpha, float beta,
          bool transA, bool transB) {
  const int64_t work = static_cast<int64_t>(M) * static_cast<int64_t>(N) * static_cast<int64_t>(K);
  if (alpha == 1.0F && beta == 0.0F && !transA && !transB && work > 1000000) {
    const bool use_avx2 = gemm_avx2_available();
    const bool use_avx = use_avx2 || gemm_avx_available();
    if (use_avx) {
      unsigned thread_count = 1;
      if (work >= 100000000 && M >= 12) {
        thread_count = std::min(6U, std::max(1U, std::thread::hardware_concurrency()));
        if (const char* setting = std::getenv("NN_GEMM_THREADS")) {
          const int requested = std::atoi(setting);
          if (requested > 0) thread_count = static_cast<unsigned>(requested);
        }
        thread_count = std::min(thread_count, static_cast<unsigned>(M / 6));
      }
      if (thread_count > 1) {
        const int rows_per_thread =
            ((M + static_cast<int>(thread_count) - 1) /
             static_cast<int>(thread_count) + 5) / 6 * 6;
        std::vector<std::thread> workers;
        for (unsigned index = 0; index < thread_count; ++index) {
          const int begin = static_cast<int>(index) * rows_per_thread;
          const int end = std::min(M, begin + rows_per_thread);
          if (begin >= end) break;
          workers.emplace_back([=]() {
            if (use_avx2) {
              gemm_avx2(A + static_cast<int64_t>(begin) * K, B,
                        C + static_cast<int64_t>(begin) * N,
                        end - begin, N, K);
            } else {
              gemm_avx(A + static_cast<int64_t>(begin) * K, B,
                       C + static_cast<int64_t>(begin) * N,
                       end - begin, N, K);
            }
          });
        }
        for (auto& worker : workers) worker.join();
        return;
      }
      if (use_avx2) gemm_avx2(A, B, C, M, N, K);
      else gemm_avx(A, B, C, M, N, K);
      return;
    }
  }
  gemm_scalar(A, B, C, M, N, K, alpha, beta, transA, transB);
}

}  // namespace nn::kernel