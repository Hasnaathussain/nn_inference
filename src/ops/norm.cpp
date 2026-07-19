#include "nn/ops/norm.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace nn::kernel {
namespace {

const Tensor& input_at(
    const OpNode& node, const std::unordered_map<std::string, Tensor>& tensors,
    size_t index) {
  if (index >= node.inputs.size() || node.inputs[index].empty()) {
    throw std::runtime_error(
        "Missing required BatchNormalization input " + std::to_string(index));
  }
  const auto found = tensors.find(node.inputs[index]);
  if (found == tensors.end()) {
    throw std::runtime_error(
        "Missing BatchNormalization tensor: " + node.inputs[index]);
  }
  return found->second;
}

}  // namespace

void batch_norm(const OpNode& node,
                std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  const Tensor& scale = input_at(node, tensors, 1);
  const Tensor& bias = input_at(node, tensors, 2);
  const Tensor& mean = input_at(node, tensors, 3);
  const Tensor& variance = input_at(node, tensors, 4);

  if (input.dtype() != Tensor::DataType::FLOAT32 || input.ndim() != 4) {
    throw std::runtime_error(
        "BatchNormalization input must be FLOAT32 NCHW rank 4");
  }
  const int64_t channels = input.shape()[1];
  for (const Tensor* parameter : {&scale, &bias, &mean, &variance}) {
    if (parameter->dtype() != Tensor::DataType::FLOAT32 ||
        parameter->ndim() != 1 || parameter->shape()[0] != channels) {
      throw std::runtime_error(
          "BatchNormalization parameters must be FLOAT32 vectors of C");
    }
  }

  const float epsilon = node.get_float("epsilon", 1e-5f);
  if (epsilon < 0.0f) {
    throw std::runtime_error("BatchNormalization epsilon must be non-negative");
  }

  Tensor output(input.shape(), Tensor::DataType::FLOAT32);
  const int64_t batch = input.shape()[0];
  const int64_t height = input.shape()[2];
  const int64_t width = input.shape()[3];
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      const float gamma = scale.data_f32()[c * scale.strides()[0]];
      const float beta = bias.data_f32()[c * bias.strides()[0]];
      const float running_mean = mean.data_f32()[c * mean.strides()[0]];
      const float running_variance =
          variance.data_f32()[c * variance.strides()[0]];
      if (running_variance + epsilon <= 0.0f) {
        throw std::runtime_error(
            "BatchNormalization variance plus epsilon must be positive");
      }
      const float factor = gamma / std::sqrt(running_variance + epsilon);
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const int64_t input_offset =
              n * input.strides()[0] + c * input.strides()[1] +
              h * input.strides()[2] + w * input.strides()[3];
          const int64_t output_offset =
              ((n * channels + c) * height + h) * width + w;
          output.data_f32()[output_offset] =
              (input.data_f32()[input_offset] - running_mean) * factor + beta;
        }
      }
    }
  }

  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error("BatchNormalization is missing its output name");
  }
  tensors.insert_or_assign(node.outputs[0], std::move(output));
}

}  // namespace nn::kernel
