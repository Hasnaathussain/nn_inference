#pragma once

#include "nn/graph.hpp"

#include <string>

namespace nn {

InferenceGraph load_onnx(const std::string& path);

}  // namespace nn
