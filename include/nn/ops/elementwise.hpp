#pragma once

#include "nn/graph.hpp"

#include <functional>
#include <unordered_map>

namespace nn::kernel {

Tensor broadcast_binary_op(
    const Tensor& a, const Tensor& b,
    const std::function<float(float, float)>& fn);

void elementwise_add(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
void elementwise_mul(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
void elementwise_sub(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
void elementwise_div(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
void elementwise_pow(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
void sqrt_op(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);
void erf_op(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);
void relu(const OpNode& node,
          std::unordered_map<std::string, Tensor>& tensors);
void reshape_op(const OpNode& node,
                std::unordered_map<std::string, Tensor>& tensors);
void flatten(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);
void transpose_op(const OpNode& node,
                  std::unordered_map<std::string, Tensor>& tensors);

}  // namespace nn::kernel
