#include "nn/onnx_parser.hpp"

#ifndef ONNX_API
#define ONNX_API
#endif
#include <onnx/onnx-ml.pb.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn {
namespace {

Tensor::DataType data_type(int32_t type) {
  switch (type) {
    case onnx::TensorProto_DataType_FLOAT: return Tensor::DataType::FLOAT32;
    case onnx::TensorProto_DataType_INT64: return Tensor::DataType::INT64;
    case onnx::TensorProto_DataType_INT32: return Tensor::DataType::INT32;
    case onnx::TensorProto_DataType_BOOL: return Tensor::DataType::BOOL;
    case onnx::TensorProto_DataType_UINT8: return Tensor::DataType::UINT8;
    default:
      throw std::runtime_error("Unsupported ONNX tensor data type: " +
                               std::to_string(type));
  }
}

std::vector<int64_t> tensor_dims(const onnx::TensorProto& proto) {
  return {proto.dims().begin(), proto.dims().end()};
}

Tensor parse_tensor(const onnx::TensorProto& proto) {
  if (proto.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    throw std::runtime_error("External ONNX tensor data is unsupported: " + proto.name());
  }

  Tensor result(tensor_dims(proto), data_type(proto.data_type()));
  const size_t bytes = static_cast<size_t>(result.nbytes());
  if (!proto.raw_data().empty()) {
    if (proto.raw_data().size() != bytes) {
      throw std::runtime_error("ONNX raw_data size mismatch for initializer '" +
                               proto.name() + "'");
    }
    result.fill_from_bytes(proto.raw_data().data(), bytes);
    return result;
  }

  switch (result.dtype()) {
    case Tensor::DataType::FLOAT32: {
      if (proto.float_data_size() != result.numel()) {
        throw std::runtime_error("ONNX float_data size mismatch for initializer '" +
                                 proto.name() + "'");
      }
      std::vector<float> values(proto.float_data().begin(), proto.float_data().end());
      result.fill_from_raw(values.data(), values.size());
      break;
    }
    case Tensor::DataType::INT64: {
      if (proto.int64_data_size() != result.numel()) {
        throw std::runtime_error("ONNX int64_data size mismatch for initializer '" +
                                 proto.name() + "'");
      }
      std::vector<int64_t> values(proto.int64_data().begin(), proto.int64_data().end());
      result.fill_from_raw(values.data(), values.size());
      break;
    }
    case Tensor::DataType::INT32: {
      if (proto.int32_data_size() != result.numel()) {
        throw std::runtime_error("ONNX int32_data size mismatch for initializer '" +
                                 proto.name() + "'");
      }
      std::vector<int32_t> values;
      values.reserve(static_cast<size_t>(proto.int32_data_size()));
      for (const auto value : proto.int32_data()) values.push_back(value);
      result.fill_from_bytes(values.data(), values.size() * sizeof(int32_t));
      break;
    }
    case Tensor::DataType::BOOL:
    case Tensor::DataType::UINT8: {
      if (proto.int32_data_size() != result.numel()) {
        throw std::runtime_error("ONNX byte tensor size mismatch for initializer '" +
                                 proto.name() + "'");
      }
      std::vector<uint8_t> values;
      values.reserve(static_cast<size_t>(proto.int32_data_size()));
      for (const auto value : proto.int32_data()) {
        if (result.dtype() == Tensor::DataType::BOOL && value != 0 && value != 1) {
          throw std::runtime_error("Invalid BOOL initializer value");
        }
        if (value < 0 || value > 255) {
          throw std::runtime_error("Byte initializer value is out of range");
        }
        values.push_back(static_cast<uint8_t>(value));
      }
      result.fill_from_bytes(values.data(), values.size());
      break;
    }
  }
  return result;
}

TensorInfo parse_value_info(const onnx::ValueInfoProto& value) {
  if (!value.type().has_tensor_type()) {
    throw std::runtime_error("Only ONNX tensor value_info is supported: " + value.name());
  }
  const auto& tensor_type = value.type().tensor_type();
  TensorInfo result;
  result.name = value.name();
  result.dtype = data_type(tensor_type.elem_type());
  result.shape_known = tensor_type.has_shape();
  if (tensor_type.has_shape()) {
    for (const auto& dimension : tensor_type.shape().dim()) {
      if (dimension.has_dim_value()) {
        result.shape.push_back(dimension.dim_value());
        result.dim_params.emplace_back();
      } else if (dimension.has_dim_param()) {
        result.shape.push_back(-1);
        result.dim_params.push_back(dimension.dim_param());
      } else {
        result.shape.push_back(-1);
        result.dim_params.emplace_back();
      }
    }
  }
  return result;
}

AttributeValue parse_attribute(const onnx::AttributeProto& attribute) {
  AttributeValue result;
  switch (attribute.type()) {
    case onnx::AttributeProto_AttributeType_INT:
      result.type = AttributeValue::Type::INT;
      result.i = attribute.i();
      break;
    case onnx::AttributeProto_AttributeType_FLOAT:
      result.type = AttributeValue::Type::FLOAT;
      result.f = attribute.f();
      break;
    case onnx::AttributeProto_AttributeType_STRING:
      result.type = AttributeValue::Type::STRING;
      result.s = attribute.s();
      break;
    case onnx::AttributeProto_AttributeType_INTS:
      result.type = AttributeValue::Type::INTS;
      result.ints.assign(attribute.ints().begin(), attribute.ints().end());
      break;
    case onnx::AttributeProto_AttributeType_FLOATS:
      result.type = AttributeValue::Type::FLOATS;
      result.floats.assign(attribute.floats().begin(), attribute.floats().end());
      break;
    case onnx::AttributeProto_AttributeType_TENSOR:
      result.type = AttributeValue::Type::TENSOR;
      result.tensor = parse_tensor(attribute.t());
      break;
    default:
      throw std::runtime_error("Unsupported ONNX attribute type on '" +
                               attribute.name() + "'");
  }
  return result;
}

void add_value_info(InferenceGraph& graph, const onnx::ValueInfoProto& value) {
  auto info = parse_value_info(value);
  graph.value_info[info.name] = std::move(info);
}

}  // namespace

InferenceGraph load_onnx(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Unable to open ONNX model: " + path);

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&stream)) {
    throw std::runtime_error("Unable to deserialize ONNX model: " + path);
  }

  InferenceGraph graph;
  const auto& source = model.graph();
  graph.model_name = source.name();
  for (const auto& import : model.opset_import()) {
    if (import.domain().empty() || import.domain() == "ai.onnx") {
      graph.opset_version = import.version();
      break;
    }
  }

  for (const auto& initializer : source.initializer()) {
    if (initializer.name().empty()) {
      throw std::runtime_error("ONNX initializer has an empty name");
    }
    graph.initializers.emplace(initializer.name(), parse_tensor(initializer));
  }

  for (const auto& value : source.input()) add_value_info(graph, value);
  for (const auto& value : source.value_info()) add_value_info(graph, value);
  for (const auto& value : source.output()) add_value_info(graph, value);

  for (const auto& input : source.input()) {
    if (graph.initializers.find(input.name()) == graph.initializers.end()) {
      graph.input_names.push_back(input.name());
    }
  }
  for (const auto& output : source.output()) graph.output_names.push_back(output.name());

  graph.nodes.reserve(static_cast<size_t>(source.node_size()));
  for (const auto& source_node : source.node()) {
    OpNode node;
    node.name = source_node.name();
    node.op_type = source_node.op_type();
    node.inputs.assign(source_node.input().begin(), source_node.input().end());
    node.outputs.assign(source_node.output().begin(), source_node.output().end());
    for (const auto& attribute : source_node.attribute()) {
      if (!node.attrs.emplace(attribute.name(), parse_attribute(attribute)).second) {
        throw std::runtime_error("Duplicate ONNX attribute '" + attribute.name() +
                                 "' on node '" + node.name + "'");
      }
    }
    graph.nodes.push_back(std::move(node));
  }

  sort_topologically(graph);
  return graph;
}

}  // namespace nn
