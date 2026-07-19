#pragma once

#include "nn/graph.hpp"

#include <unordered_map>

namespace nn::kernel {

void conv2d(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors);

}  // namespace nn::kernel
