#pragma once

#include "nn/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn {

struct AttributeValue {
  enum class Type { INT, FLOAT, STRING, INTS, FLOATS, TENSOR };
  Type type = Type::INT;
  int64_t i = 0;
  float f = 0.0f;
  std::string s;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  Tensor tensor;
};

using AttributeMap = std::unordered_map<std::string, AttributeValue>;

struct TensorInfo {
  std::string name;
  Tensor::DataType dtype = Tensor::DataType::FLOAT32;
  std::vector<int64_t> shape;
  std::vector<std::string> dim_params;
  bool shape_known = true;
};

struct OpNode {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  AttributeMap attrs;

  int64_t get_int(const std::string& key, int64_t default_val = 0) const;
  float get_float(const std::string& key, float default_val = 0.0f) const;
  std::vector<int64_t> get_ints(
      const std::string& key, std::vector<int64_t> default_val = {}) const;
};

struct InferenceGraph {
  std::vector<OpNode> nodes;
  std::unordered_map<std::string, TensorInfo> value_info;
  std::unordered_map<std::string, Tensor> initializers;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::string model_name;
  int64_t opset_version = 0;
};

void sort_topologically(InferenceGraph& graph);

}  // namespace nn
