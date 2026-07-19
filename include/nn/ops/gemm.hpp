#pragma once

#include "nn/graph.hpp"

#include <unordered_map>

namespace nn::kernel {

void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 float alpha = 1.0f, float beta = 0.0f,
                 bool transA = false, bool transB = false);

void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K);
bool gemm_avx2_available();
void gemm_avx(const float* A, const float* B, float* C,
              int M, int N, int K);
bool gemm_avx_available();

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          float alpha = 1.0f, float beta = 0.0f,
          bool transA = false, bool transB = false);

void gemm_op(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);

}  // namespace nn::kernel
