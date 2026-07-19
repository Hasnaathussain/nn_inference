#include "nn/ops/bert.hpp"

#include "nn/ops/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn::kernel {
namespace {

const Tensor& input_at(
    const OpNode& node, const std::unordered_map<std::string, Tensor>& tensors,
    size_t index) {
  if (index >= node.inputs.size() || node.inputs[index].empty()) {
    throw std::runtime_error(
        "Missing required " + node.op_type + " input " + std::to_string(index));
  }
  const auto found = tensors.find(node.inputs[index]);
  if (found == tensors.end()) {
    throw std::runtime_error(
        "Missing " + node.op_type + " tensor: " + node.inputs[index]);
  }
  return found->second;
}

const std::string& output_at(const OpNode& node) {
  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error(node.op_type + " is missing its output name");
  }
  return node.outputs[0];
}

int64_t product(
    const std::vector<int64_t>& shape, size_t begin = 0,
    size_t end = std::numeric_limits<size_t>::max()) {
  end = std::min(end, shape.size());
  int64_t result = 1;
  for (size_t axis = begin; axis < end; ++axis) {
    if (shape[axis] < 0) throw std::runtime_error("Runtime shape is unresolved");
    if (shape[axis] != 0 &&
        result > std::numeric_limits<int64_t>::max() / shape[axis]) {
      throw std::overflow_error("Tensor shape product overflow");
    }
    result *= shape[axis];
  }
  return result;
}

std::vector<int64_t> coordinates(
    int64_t linear, const std::vector<int64_t>& shape) {
  std::vector<int64_t> result(shape.size(), 0);
  for (size_t axis = shape.size(); axis > 0; --axis) {
    const int64_t dimension = shape[axis - 1];
    if (dimension != 0) {
      result[axis - 1] = linear % dimension;
      linear /= dimension;
    }
  }
  return result;
}

int64_t tensor_offset(
    const Tensor& tensor, const std::vector<int64_t>& coords) {
  int64_t offset = 0;
  for (size_t axis = 0; axis < coords.size(); ++axis) {
    offset += coords[axis] * tensor.strides()[axis];
  }
  return offset;
}

int64_t logical_offset(const Tensor& tensor, int64_t linear) {
  return tensor_offset(tensor, coordinates(linear, tensor.shape()));
}

void copy_element(
    const Tensor& source, int64_t source_offset,
    Tensor& destination, int64_t destination_offset) {
  if (source.dtype() != destination.dtype()) {
    throw std::runtime_error("Tensor element copy dtype mismatch");
  }
  switch (source.dtype()) {
    case Tensor::DataType::FLOAT32:
      destination.data_f32()[destination_offset] =
          source.data_f32()[source_offset];
      return;
    case Tensor::DataType::INT64:
      destination.data_i64()[destination_offset] =
          source.data_i64()[source_offset];
      return;
    case Tensor::DataType::INT32:
      destination.data_i32()[destination_offset] =
          source.data_i32()[source_offset];
      return;
    case Tensor::DataType::BOOL:
    case Tensor::DataType::UINT8:
      destination.data_u8()[destination_offset] =
          source.data_u8()[source_offset];
      return;
  }
  throw std::runtime_error("Unsupported tensor dtype");
}

double read_number(const Tensor& tensor, int64_t offset) {
  switch (tensor.dtype()) {
    case Tensor::DataType::FLOAT32: return tensor.data_f32()[offset];
    case Tensor::DataType::INT64:
      return static_cast<double>(tensor.data_i64()[offset]);
    case Tensor::DataType::INT32:
      return static_cast<double>(tensor.data_i32()[offset]);
    case Tensor::DataType::BOOL:
    case Tensor::DataType::UINT8:
      return static_cast<double>(tensor.data_u8()[offset]);
  }
  throw std::runtime_error("Unsupported numeric tensor dtype");
}

void write_number(
    Tensor& tensor, int64_t offset, double value) {
  switch (tensor.dtype()) {
    case Tensor::DataType::FLOAT32:
      tensor.data_f32()[offset] = static_cast<float>(value);
      return;
    case Tensor::DataType::INT64:
      tensor.data_i64()[offset] = static_cast<int64_t>(value);
      return;
    case Tensor::DataType::INT32:
      tensor.data_i32()[offset] = static_cast<int32_t>(value);
      return;
    case Tensor::DataType::BOOL:
      tensor.data_u8()[offset] = value != 0.0;
      return;
    case Tensor::DataType::UINT8:
      tensor.data_u8()[offset] = static_cast<uint8_t>(value);
      return;
  }
  throw std::runtime_error("Unsupported numeric tensor dtype");
}

std::vector<int64_t> int64_values(const Tensor& tensor) {
  if (tensor.dtype() != Tensor::DataType::INT64 || tensor.ndim() != 1) {
    throw std::runtime_error("Shape/index input must be a rank-1 INT64 tensor");
  }
  std::vector<int64_t> result(static_cast<size_t>(tensor.numel()));
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    result[static_cast<size_t>(index)] =
        tensor.data_i64()[logical_offset(tensor, index)];
  }
  return result;
}

int64_t normalize_axis(int64_t axis, int64_t rank, bool allow_end = false) {
  if (axis < 0) axis += rank;
  const int64_t upper = allow_end ? rank : rank - 1;
  if (axis < 0 || axis > upper) {
    throw std::runtime_error("Operator axis is out of range");
  }
  return axis;
}

std::vector<int64_t> broadcast_shape(
    const std::vector<std::vector<int64_t>>& shapes) {
  size_t rank = 0;
  for (const auto& shape : shapes) rank = std::max(rank, shape.size());
  std::vector<int64_t> result(rank, 1);
  for (const auto& shape : shapes) {
    const size_t shift = rank - shape.size();
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      const int64_t dimension = shape[axis];
      int64_t& output_dimension = result[shift + axis];
      if (output_dimension == 1) {
        output_dimension = dimension;
      } else if (dimension != 1 && dimension != output_dimension) {
        throw std::runtime_error("Broadcast shape mismatch");
      }
    }
  }
  return result;
}

int64_t broadcast_offset(
    const Tensor& tensor, const std::vector<int64_t>& output_coords) {
  const size_t shift = output_coords.size() - tensor.shape().size();
  int64_t offset = 0;
  for (size_t axis = 0; axis < tensor.shape().size(); ++axis) {
    const int64_t coordinate =
        tensor.shape()[axis] == 1 ? 0 : output_coords[shift + axis];
    offset += coordinate * tensor.strides()[axis];
  }
  return offset;
}

Tensor::DataType onnx_dtype(int64_t value) {
  switch (value) {
    case 1: return Tensor::DataType::FLOAT32;
    case 2: return Tensor::DataType::UINT8;
    case 6: return Tensor::DataType::INT32;
    case 7: return Tensor::DataType::INT64;
    case 9: return Tensor::DataType::BOOL;
    default:
      throw std::runtime_error("Cast target dtype is unsupported");
  }
}

}  // namespace

void constant_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const auto value = node.attrs.find("value");
  if (value == node.attrs.end() ||
      value->second.type != AttributeValue::Type::TENSOR) {
    throw std::runtime_error("Constant requires a tensor value attribute");
  }
  tensors.insert_or_assign(output_at(node), value->second.tensor.clone_contiguous());
}

void shape_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  int64_t start = node.get_int("start", 0);
  int64_t end = node.get_int("end", input.ndim());
  if (start < 0) start += input.ndim();
  if (end < 0) end += input.ndim();
  start = std::clamp<int64_t>(start, 0, input.ndim());
  end = std::clamp<int64_t>(end, 0, input.ndim());
  if (end < start) end = start;
  Tensor output({end - start}, Tensor::DataType::INT64);
  for (int64_t axis = start; axis < end; ++axis) {
    output.data_i64()[axis - start] = input.shape()[static_cast<size_t>(axis)];
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void gather(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& data = input_at(node, tensors, 0);
  const Tensor& indices = input_at(node, tensors, 1);
  if (indices.dtype() != Tensor::DataType::INT64) {
    throw std::runtime_error("Gather indices must be INT64");
  }
  const int64_t axis = normalize_axis(
      node.get_int("axis", 0), data.ndim());

  std::vector<int64_t> output_shape;
  output_shape.insert(
      output_shape.end(), data.shape().begin(), data.shape().begin() + axis);
  output_shape.insert(
      output_shape.end(), indices.shape().begin(), indices.shape().end());
  output_shape.insert(
      output_shape.end(), data.shape().begin() + axis + 1, data.shape().end());
  Tensor output(output_shape, data.dtype());

  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    const auto out_coords = coordinates(linear, output_shape);
    std::vector<int64_t> index_coords(
        indices.shape().size(), 0);
    std::vector<int64_t> data_coords(
        data.shape().size(), 0);
    for (int64_t before = 0; before < axis; ++before) {
      data_coords[static_cast<size_t>(before)] =
          out_coords[static_cast<size_t>(before)];
    }
    for (size_t index_axis = 0; index_axis < indices.shape().size(); ++index_axis) {
      index_coords[index_axis] =
          out_coords[static_cast<size_t>(axis) + index_axis];
    }
    int64_t selected = indices.data_i64()[
        tensor_offset(indices, index_coords)];
    const int64_t axis_dimension = data.shape()[static_cast<size_t>(axis)];
    if (selected < 0) selected += axis_dimension;
    if (selected < 0 || selected >= axis_dimension) {
      throw std::runtime_error("Gather index is out of range");
    }
    data_coords[static_cast<size_t>(axis)] = selected;
    for (size_t after = static_cast<size_t>(axis + 1);
         after < data.shape().size(); ++after) {
      const size_t out_axis =
          static_cast<size_t>(axis) + indices.shape().size() +
          after - static_cast<size_t>(axis + 1);
      data_coords[after] = out_coords[out_axis];
    }
    copy_element(data, tensor_offset(data, data_coords), output, linear);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void concat(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  if (node.inputs.empty()) throw std::runtime_error("Concat requires inputs");
  std::vector<const Tensor*> inputs;
  for (size_t index = 0; index < node.inputs.size(); ++index) {
    inputs.push_back(&input_at(node, tensors, index));
  }
  const int64_t rank = inputs.front()->ndim();
  const int64_t axis = normalize_axis(node.get_int("axis"), rank);
  std::vector<int64_t> output_shape = inputs.front()->shape();
  output_shape[static_cast<size_t>(axis)] = 0;
  for (const Tensor* input : inputs) {
    if (input->ndim() != rank || input->dtype() != inputs.front()->dtype()) {
      throw std::runtime_error("Concat input rank/dtype mismatch");
    }
    for (int64_t current = 0; current < rank; ++current) {
      if (current != axis &&
          input->shape()[static_cast<size_t>(current)] !=
              output_shape[static_cast<size_t>(current)]) {
        throw std::runtime_error("Concat non-axis shape mismatch");
      }
    }
    output_shape[static_cast<size_t>(axis)] +=
        input->shape()[static_cast<size_t>(axis)];
  }

  Tensor output(output_shape, inputs.front()->dtype());
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    auto coords = coordinates(linear, output_shape);
    int64_t selected_axis = coords[static_cast<size_t>(axis)];
    const Tensor* source = nullptr;
    for (const Tensor* candidate : inputs) {
      const int64_t length = candidate->shape()[static_cast<size_t>(axis)];
      if (selected_axis < length) {
        source = candidate;
        break;
      }
      selected_axis -= length;
    }
    coords[static_cast<size_t>(axis)] = selected_axis;
    copy_element(*source, tensor_offset(*source, coords), output, linear);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void unsqueeze(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  std::vector<int64_t> axes =
      node.inputs.size() > 1 && !node.inputs[1].empty()
          ? int64_values(input_at(node, tensors, 1))
          : node.get_ints("axes");
  const int64_t output_rank =
      input.ndim() + static_cast<int64_t>(axes.size());
  std::unordered_set<int64_t> normalized;
  for (int64_t axis : axes) {
    if (axis < 0) axis += output_rank;
    if (axis < 0 || axis >= output_rank || !normalized.insert(axis).second) {
      throw std::runtime_error("Unsqueeze axes are invalid");
    }
  }
  std::vector<int64_t> output_shape;
  size_t input_axis = 0;
  for (int64_t axis = 0; axis < output_rank; ++axis) {
    if (normalized.count(axis) != 0) {
      output_shape.push_back(1);
    } else {
      output_shape.push_back(input.shape()[input_axis++]);
    }
  }
  tensors.insert_or_assign(output_at(node), input.reshape(std::move(output_shape)));
}

void squeeze(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  std::vector<int64_t> axes =
      node.inputs.size() > 1 && !node.inputs[1].empty()
          ? int64_values(input_at(node, tensors, 1))
          : node.get_ints("axes");
  std::unordered_set<int64_t> normalized;
  if (axes.empty()) {
    for (int64_t axis = 0; axis < input.ndim(); ++axis) {
      if (input.shape()[static_cast<size_t>(axis)] == 1) normalized.insert(axis);
    }
  } else {
    for (int64_t axis : axes) {
      axis = normalize_axis(axis, input.ndim());
      if (input.shape()[static_cast<size_t>(axis)] != 1 ||
          !normalized.insert(axis).second) {
        throw std::runtime_error("Squeeze axes are invalid");
      }
    }
  }
  std::vector<int64_t> output_shape;
  for (int64_t axis = 0; axis < input.ndim(); ++axis) {
    if (normalized.count(axis) == 0) {
      output_shape.push_back(input.shape()[static_cast<size_t>(axis)]);
    }
  }
  tensors.insert_or_assign(output_at(node), input.reshape(std::move(output_shape)));
}

void cast_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  Tensor output(input.shape(), onnx_dtype(node.get_int("to")));
  for (int64_t linear = 0; linear < input.numel(); ++linear) {
    write_number(
        output, linear, read_number(input, logical_offset(input, linear)));
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void slice_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  const auto starts = int64_values(input_at(node, tensors, 1));
  const auto ends = int64_values(input_at(node, tensors, 2));
  const auto axes =
      node.inputs.size() > 3 && !node.inputs[3].empty()
          ? int64_values(input_at(node, tensors, 3))
          : [&]() {
              std::vector<int64_t> result(starts.size());
              for (size_t index = 0; index < result.size(); ++index) {
                result[index] = static_cast<int64_t>(index);
              }
              return result;
            }();
  const auto steps =
      node.inputs.size() > 4 && !node.inputs[4].empty()
          ? int64_values(input_at(node, tensors, 4))
          : std::vector<int64_t>(starts.size(), 1);
  if (starts.size() != ends.size() || starts.size() != axes.size() ||
      starts.size() != steps.size()) {
    throw std::runtime_error("Slice parameter lengths do not match");
  }

  std::vector<int64_t> begin(input.shape().size(), 0);
  std::vector<int64_t> step(input.shape().size(), 1);
  std::vector<int64_t> output_shape = input.shape();
  for (size_t index = 0; index < starts.size(); ++index) {
    const int64_t axis = normalize_axis(axes[index], input.ndim());
    if (steps[index] <= 0) {
      throw std::runtime_error("Slice currently requires positive steps");
    }
    const int64_t dimension = input.shape()[static_cast<size_t>(axis)];
    int64_t start = starts[index];
    int64_t end = ends[index];
    if (start < 0) start += dimension;
    if (end < 0) end += dimension;
    start = std::clamp<int64_t>(start, 0, dimension);
    end = std::clamp<int64_t>(end, 0, dimension);
    begin[static_cast<size_t>(axis)] = start;
    step[static_cast<size_t>(axis)] = steps[index];
    output_shape[static_cast<size_t>(axis)] =
        end > start ? (end - start + steps[index] - 1) / steps[index] : 0;
  }

  Tensor output(output_shape, input.dtype());
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    auto source_coords = coordinates(linear, output_shape);
    for (size_t axis = 0; axis < source_coords.size(); ++axis) {
      source_coords[axis] =
          begin[axis] + source_coords[axis] * step[axis];
    }
    copy_element(
        input, tensor_offset(input, source_coords), output, linear);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void expand(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  const auto requested = int64_values(input_at(node, tensors, 1));
  const auto output_shape =
      broadcast_shape({input.shape(), requested});
  if (output_shape != requested) {
    throw std::runtime_error("Expand target shape is incompatible");
  }
  Tensor output(output_shape, input.dtype());
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    const auto coords = coordinates(linear, output_shape);
    copy_element(
        input, broadcast_offset(input, coords), output, linear);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void reduce_mean(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  if (input.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error("ReduceMean requires FLOAT32 input");
  }
  std::vector<int64_t> axes = node.get_ints("axes");
  if (axes.empty()) {
    for (int64_t axis = 0; axis < input.ndim(); ++axis) axes.push_back(axis);
  }
  std::unordered_set<int64_t> reduced;
  for (int64_t axis : axes) {
    reduced.insert(normalize_axis(axis, input.ndim()));
  }
  const bool keepdims = node.get_int("keepdims", 1) != 0;
  std::vector<int64_t> output_shape;
  for (int64_t axis = 0; axis < input.ndim(); ++axis) {
    if (reduced.count(axis) != 0) {
      if (keepdims) output_shape.push_back(1);
    } else {
      output_shape.push_back(input.shape()[static_cast<size_t>(axis)]);
    }
  }
  Tensor output(output_shape, Tensor::DataType::FLOAT32);
  output.fill_zeros();
  std::vector<int64_t> counts(static_cast<size_t>(output.numel()), 0);
  for (int64_t linear = 0; linear < input.numel(); ++linear) {
    const auto input_coords = coordinates(linear, input.shape());
    std::vector<int64_t> output_coords;
    for (int64_t axis = 0; axis < input.ndim(); ++axis) {
      if (reduced.count(axis) != 0) {
        if (keepdims) output_coords.push_back(0);
      } else {
        output_coords.push_back(input_coords[static_cast<size_t>(axis)]);
      }
    }
    int64_t output_linear = 0;
    for (size_t axis = 0; axis < output_coords.size(); ++axis) {
      output_linear =
          output_linear * output_shape[axis] + output_coords[axis];
    }
    output.data_f32()[output_linear] +=
        input.data_f32()[logical_offset(input, linear)];
    ++counts[static_cast<size_t>(output_linear)];
  }
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    output.data_f32()[linear] /=
        static_cast<float>(counts[static_cast<size_t>(linear)]);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void softmax(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  if (input.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error("Softmax requires FLOAT32 input");
  }
  const int64_t axis = normalize_axis(
      node.get_int("axis", -1), input.ndim());
  const int64_t outer =
      product(input.shape(), 0, static_cast<size_t>(axis));
  const int64_t dimension = input.shape()[static_cast<size_t>(axis)];
  const int64_t inner =
      product(input.shape(), static_cast<size_t>(axis + 1));
  Tensor output(input.shape(), Tensor::DataType::FLOAT32);
  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (int64_t current = 0; current < dimension; ++current) {
        const int64_t linear =
            (outer_index * dimension + current) * inner + inner_index;
        maximum = std::max(
            maximum, input.data_f32()[logical_offset(input, linear)]);
      }
      float sum = 0.0F;
      for (int64_t current = 0; current < dimension; ++current) {
        const int64_t linear =
            (outer_index * dimension + current) * inner + inner_index;
        const float value = std::exp(
            input.data_f32()[logical_offset(input, linear)] - maximum);
        output.data_f32()[linear] = value;
        sum += value;
      }
      for (int64_t current = 0; current < dimension; ++current) {
        const int64_t linear =
            (outer_index * dimension + current) * inner + inner_index;
        output.data_f32()[linear] /= sum;
      }
    }
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void matmul(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& a = input_at(node, tensors, 0);
  const Tensor& b = input_at(node, tensors, 1);
  if (a.dtype() != Tensor::DataType::FLOAT32 ||
      b.dtype() != Tensor::DataType::FLOAT32 ||
      a.ndim() < 2 || b.ndim() < 2) {
    throw std::runtime_error("MatMul requires FLOAT32 tensors of rank at least 2");
  }
  const int64_t m = a.shape()[a.shape().size() - 2];
  const int64_t k = a.shape().back();
  const int64_t b_k = b.shape()[b.shape().size() - 2];
  const int64_t n = b.shape().back();
  if (k != b_k) throw std::runtime_error("MatMul inner dimensions do not match");
  if (m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    throw std::runtime_error("MatMul dimensions exceed kernel limits");
  }

  std::vector<int64_t> a_batch(a.shape().begin(), a.shape().end() - 2);
  std::vector<int64_t> b_batch(b.shape().begin(), b.shape().end() - 2);
  const auto batch_shape = broadcast_shape({a_batch, b_batch});
  std::vector<int64_t> output_shape = batch_shape;
  output_shape.push_back(m);
  output_shape.push_back(n);
  Tensor output(output_shape, Tensor::DataType::FLOAT32);

  const int64_t batch_count = product(batch_shape);
  std::vector<float> packed_a(static_cast<size_t>(m * k));
  std::vector<float> packed_b(static_cast<size_t>(k * n));
  std::vector<float> result(static_cast<size_t>(m * n));
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    const auto batch_coords = coordinates(batch, batch_shape);
    const size_t a_shift = batch_shape.size() - a_batch.size();
    const size_t b_shift = batch_shape.size() - b_batch.size();
    int64_t a_base = 0;
    int64_t b_base = 0;
    for (size_t axis = 0; axis < a_batch.size(); ++axis) {
      const int64_t coordinate =
          a_batch[axis] == 1 ? 0 : batch_coords[a_shift + axis];
      a_base += coordinate * a.strides()[axis];
    }
    for (size_t axis = 0; axis < b_batch.size(); ++axis) {
      const int64_t coordinate =
          b_batch[axis] == 1 ? 0 : batch_coords[b_shift + axis];
      b_base += coordinate * b.strides()[axis];
    }
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t inner = 0; inner < k; ++inner) {
        packed_a[static_cast<size_t>(row * k + inner)] =
            a.data_f32()[
                a_base + row * a.strides()[a.shape().size() - 2] +
                inner * a.strides().back()];
      }
    }
    for (int64_t inner = 0; inner < k; ++inner) {
      for (int64_t column = 0; column < n; ++column) {
        packed_b[static_cast<size_t>(inner * n + column)] =
            b.data_f32()[
                b_base + inner * b.strides()[b.shape().size() - 2] +
                column * b.strides().back()];
      }
    }
    gemm(
        packed_a.data(), packed_b.data(), result.data(),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    std::copy(
        result.begin(), result.end(),
        output.data_f32() + batch * m * n);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void layer_norm(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  const Tensor& scale = input_at(node, tensors, 1);
  const Tensor& bias = input_at(node, tensors, 2);
  if (input.dtype() != Tensor::DataType::FLOAT32 ||
      scale.dtype() != Tensor::DataType::FLOAT32 ||
      bias.dtype() != Tensor::DataType::FLOAT32 ||
      scale.ndim() != 1 || bias.shape() != scale.shape() ||
      input.shape().empty() || input.shape().back() != scale.shape()[0]) {
    throw std::runtime_error("LayerNormalization input/parameter shapes are invalid");
  }
  const int64_t width = input.shape().back();
  const int64_t rows = input.numel() / width;
  const float epsilon = node.get_float("epsilon", 1.0e-5F);
  Tensor output(input.shape(), Tensor::DataType::FLOAT32);
  for (int64_t row = 0; row < rows; ++row) {
    double mean = 0.0;
    double m2 = 0.0;
    for (int64_t column = 0; column < width; ++column) {
      const double value =
          input.data_f32()[logical_offset(input, row * width + column)];
      const double delta = value - mean;
      mean += delta / static_cast<double>(column + 1);
      m2 += delta * (value - mean);
    }
    const double variance = m2 / static_cast<double>(width);
    const double denominator = std::sqrt(variance + epsilon);
    for (int64_t column = 0; column < width; ++column) {
      const float value =
          input.data_f32()[logical_offset(input, row * width + column)];
      output.data_f32()[row * width + column] =
          scale.data_f32()[column * scale.strides()[0]] *
              static_cast<float>((value - mean) / denominator) +
          bias.data_f32()[column * bias.strides()[0]];
    }
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

void where_op(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& condition = input_at(node, tensors, 0);
  const Tensor& x = input_at(node, tensors, 1);
  const Tensor& y = input_at(node, tensors, 2);
  if (condition.dtype() != Tensor::DataType::BOOL || x.dtype() != y.dtype()) {
    throw std::runtime_error("Where condition/dtype inputs are invalid");
  }
  const auto output_shape =
      broadcast_shape({condition.shape(), x.shape(), y.shape()});
  Tensor output(output_shape, x.dtype());
  for (int64_t linear = 0; linear < output.numel(); ++linear) {
    const auto coords = coordinates(linear, output_shape);
    const bool select_x =
        condition.data_u8()[broadcast_offset(condition, coords)] != 0;
    const Tensor& source = select_x ? x : y;
    copy_element(
        source, broadcast_offset(source, coords), output, linear);
  }
  tensors.insert_or_assign(output_at(node), std::move(output));
}

}  // namespace nn::kernel