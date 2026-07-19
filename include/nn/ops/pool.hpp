#pragma once

#include "nn/graph.hpp"

#include <unordered_map>

namespace nn::kernel {

void max_pool(const OpNode& node,
              std::unordered_map<std::string, Tensor>& tensors);
void global_avg_pool(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);

}  // namespace nn::kernel
