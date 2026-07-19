#include "nn/ops/elementwise.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn::kernel {
namespace {

const Tensor& required_tensor(
    const OpNode& node, const std::unordered_map<std::string, Tensor>& tensors,
    size_t input_index) {
  if (input_index >= node.inputs.size() || node.inputs[input_index].empty()) {
    throw std::runtime_error("Missing required input " +
                             std::to_string(input_index) + " for " + node.op_type);
  }
  const auto tensor = tensors.find(node.inputs[input_index]);
  if (tensor == tensors.end()) {
    throw std::runtime_error("Missing tensor '" + node.inputs[input_index] +
                             "' for " + node.op_type);
  }
  return tensor->second;
}

const std::string& required_output(const OpNode& node) {
  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error("Missing required output for " + node.op_type);
  }
  return node.outputs[0];
}

int64_t logical_offset(const Tensor& tensor, int64_t linear) {
  int64_t offset = 0;
  for (size_t axis = tensor.shape().size(); axis > 0; --axis) {
    const int64_t dim = tensor.shape()[axis - 1];
    const int64_t coordinate = linear % dim;
    linear /= dim;
    offset += coordinate * tensor.strides()[axis - 1];
  }
  return offset;
}

int64_t checked_product(
    const std::vector<int64_t>& shape, size_t begin, size_t end) {
  int64_t product = 1;
  for (size_t axis = begin; axis < end; ++axis) {
    const int64_t dim = shape[axis];
    if (dim < 0) throw std::invalid_argument("Shape dimensions must be non-negative");
    if (dim != 0 && product > std::numeric_limits<int64_t>::max() / dim) {
      throw std::overflow_error("Shape element count overflow");
    }
    product *= dim;
  }
  return product;
}

void binary_node(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors,
    const std::function<float(float, float)>& fn) {
  const Tensor& a = required_tensor(node, tensors, 0);
  const Tensor& b = required_tensor(node, tensors, 1);
  tensors.insert_or_assign(required_output(node), broadcast_binary_op(a, b, fn));
}

void unary_node(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors,
    const std::function<float(float)>& fn) {
  const Tensor& input = required_tensor(node, tensors, 0);
  if (input.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error(node.op_type + " requires a FLOAT32 tensor");
  }
  Tensor output(input.shape(), Tensor::DataType::FLOAT32);
  for (int64_t linear = 0; linear < input.numel(); ++linear) {
    output.data_f32()[linear] =
        fn(input.data_f32()[logical_offset(input, linear)]);
  }
  tensors.insert_or_assign(required_output(node), std::move(output));
}

}  // namespace

Tensor broadcast_binary_op(
    const Tensor& a, const Tensor& b,
    const std::function<float(float, float)>& fn) {
  if (a.dtype() != Tensor::DataType::FLOAT32 ||
      b.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error("Element-wise arithmetic requires FLOAT32 tensors");
  }

  const size_t rank = std::max(a.shape().size(), b.shape().size());
  std::vector<int64_t> output_shape(rank, 1);
  for (size_t output_axis = 0; output_axis < rank; ++output_axis) {
    const int64_t a_axis =
        static_cast<int64_t>(output_axis) -
        static_cast<int64_t>(rank - a.shape().size());
    const int64_t b_axis =
        static_cast<int64_t>(output_axis) -
        static_cast<int64_t>(rank - b.shape().size());
    const int64_t a_dim = a_axis < 0 ? 1 : a.shape()[static_cast<size_t>(a_axis)];
    const int64_t b_dim = b_axis < 0 ? 1 : b.shape()[static_cast<size_t>(b_axis)];
    if (a_dim == b_dim) {
      output_shape[output_axis] = a_dim;
    } else if (a_dim == 1) {
      output_shape[output_axis] = b_dim;
    } else if (b_dim == 1) {
      output_shape[output_axis] = a_dim;
    } else {
      throw std::runtime_error("Shape mismatch in broadcast binary op");
    }
  }

  Tensor output(output_shape, Tensor::DataType::FLOAT32);
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    int64_t remaining = linear;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    for (size_t output_axis = rank; output_axis > 0; --output_axis) {
      const size_t axis = output_axis - 1;
      const int64_t coordinate = remaining % output_shape[axis];
      remaining /= output_shape[axis];

      const int64_t a_axis =
          static_cast<int64_t>(axis) -
          static_cast<int64_t>(rank - a.shape().size());
      if (a_axis >= 0 && a.shape()[static_cast<size_t>(a_axis)] != 1) {
        a_offset += coordinate * a.strides()[static_cast<size_t>(a_axis)];
      }

      const int64_t b_axis =
          static_cast<int64_t>(axis) -
          static_cast<int64_t>(rank - b.shape().size());
      if (b_axis >= 0 && b.shape()[static_cast<size_t>(b_axis)] != 1) {
        b_offset += coordinate * b.strides()[static_cast<size_t>(b_axis)];
      }
    }
    output.data_f32()[linear] = fn(a.data_f32()[a_offset], b.data_f32()[b_offset]);
  }
  return output;
}

void elementwise_add(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  binary_node(node, tensors, [](float a, float b) { return a + b; });
}

void elementwise_mul(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  binary_node(node, tensors, [](float a, float b) { return a * b; });
}

void elementwise_sub(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  binary_node(node, tensors, [](float a, float b) { return a - b; });
}

void elementwise_div(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  binary_node(node, tensors, [](float a, float b) { return a / b; });
}

void elementwise_pow(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  binary_node(node, tensors, [](float a, float b) { return std::pow(a, b); });
}

void sqrt_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  unary_node(node, tensors, [](float value) { return std::sqrt(value); });
}

void erf_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  unary_node(node, tensors, [](float value) { return std::erf(value); });
}

void relu(const OpNode& node,
          std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = required_tensor(node, tensors, 0);
  if (input.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error("Relu requires a FLOAT32 tensor");
  }
  Tensor output(input.shape(), input.dtype());
  for (int64_t linear = 0; linear < input.numel(); ++linear) {
    output.data_f32()[linear] =
        std::max(0.0f, input.data_f32()[logical_offset(input, linear)]);
  }
  tensors.insert_or_assign(required_output(node), std::move(output));
}

void reshape_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = required_tensor(node, tensors, 0);
  const Tensor& shape_tensor = required_tensor(node, tensors, 1);
  if (shape_tensor.dtype() != Tensor::DataType::INT64 ||
      shape_tensor.ndim() != 1) {
    throw std::runtime_error("Reshape shape input must be a rank-1 INT64 tensor");
  }

  const bool allow_zero = node.get_int("allowzero", 0) != 0;
  std::vector<int64_t> output_shape(static_cast<size_t>(shape_tensor.numel()));
  int64_t inferred_axis = -1;
  for (int64_t axis = 0; axis < shape_tensor.numel(); ++axis) {
    int64_t dim =
        shape_tensor.data_i64()[logical_offset(shape_tensor, axis)];
    if (dim == -1) {
      if (inferred_axis >= 0) {
        throw std::runtime_error("Reshape shape contains multiple -1 dimensions");
      }
      inferred_axis = axis;
      output_shape[static_cast<size_t>(axis)] = 1;
    } else if (dim == 0 && !allow_zero) {
      if (axis >= input.ndim()) {
        throw std::runtime_error("Reshape zero dimension has no matching input axis");
      }
      output_shape[static_cast<size_t>(axis)] =
          input.shape()[static_cast<size_t>(axis)];
    } else if (dim < 0) {
      throw std::runtime_error("Reshape contains an invalid negative dimension");
    } else {
      output_shape[static_cast<size_t>(axis)] = dim;
    }
  }

  const int64_t known_product =
      checked_product(output_shape, 0, output_shape.size());
  if (inferred_axis >= 0) {
    if (known_product == 0 || input.numel() % known_product != 0) {
      throw std::runtime_error("Reshape cannot infer a valid dimension");
    }
    output_shape[static_cast<size_t>(inferred_axis)] =
        input.numel() / known_product;
  } else if (known_product != input.numel()) {
    throw std::runtime_error("Reshape element count mismatch");
  }

  tensors.insert_or_assign(required_output(node), input.reshape(std::move(output_shape)));
}

void flatten(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = required_tensor(node, tensors, 0);
  int64_t axis = node.get_int("axis", 1);
  if (axis < 0) axis += input.ndim();
  if (axis < 0 || axis > input.ndim()) {
    throw std::runtime_error("Flatten axis is out of range");
  }

  const int64_t first =
      checked_product(input.shape(), 0, static_cast<size_t>(axis));
  const int64_t second =
      checked_product(input.shape(), static_cast<size_t>(axis),
                      input.shape().size());
  tensors.insert_or_assign(required_output(node), input.reshape({first, second}));
}

void transpose_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = required_tensor(node, tensors, 0);
  std::vector<int64_t> permutation;
  const auto attribute = node.attrs.find("perm");
  if (attribute == node.attrs.end()) {
    permutation.resize(static_cast<size_t>(input.ndim()));
    for (int64_t axis = 0; axis < input.ndim(); ++axis) {
      permutation[static_cast<size_t>(axis)] = input.ndim() - axis - 1;
    }
  } else {
    permutation = node.get_ints("perm");
  }
  tensors.insert_or_assign(required_output(node),
                           input.transpose(std::move(permutation)));
}

}  // namespace nn::kernel
