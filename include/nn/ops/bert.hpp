#pragma once

#include "nn/graph.hpp"

#include <unordered_map>

namespace nn::kernel {

void constant_op(const OpNode& node,
                 std::unordered_map<std::string, Tensor>& tensors);
void shape_op(const OpNode& node,
              std::unordered_map<std::string, Tensor>& tensors);
void gather(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);
void concat(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);
void unsqueeze(const OpNode& node,
               std::unordered_map<std::string, Tensor>& tensors);
void squeeze(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);
void cast_op(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);
void slice_op(const OpNode& node,
              std::unordered_map<std::string, Tensor>& tensors);
void expand(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);
void reduce_mean(const OpNode& node,
                 std::unordered_map<std::string, Tensor>& tensors);
void softmax(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors);
void matmul(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);
void layer_norm(const OpNode& node,
                std::unordered_map<std::string, Tensor>& tensors);
void where_op(const OpNode& node,
              std::unordered_map<std::string, Tensor>& tensors);

}  // namespace nn::kernel