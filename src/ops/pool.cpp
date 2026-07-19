#include "nn/ops/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn::kernel {
namespace {

const Tensor& input_tensor(
    const OpNode& node, const std::unordered_map<std::string, Tensor>& tensors) {
  if (node.inputs.empty() || node.inputs[0].empty()) {
    throw std::runtime_error("Pooling operator is missing its input");
  }
  const auto found = tensors.find(node.inputs[0]);
  if (found == tensors.end()) {
    throw std::runtime_error("Missing pooling tensor: " + node.inputs[0]);
  }
  if (found->second.dtype() != Tensor::DataType::FLOAT32 ||
      found->second.ndim() != 4) {
    throw std::runtime_error("Pooling requires a FLOAT32 rank-4 NCHW tensor");
  }
  return found->second;
}

const std::string& output_name(const OpNode& node) {
  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error("Pooling operator is missing its output");
  }
  return node.outputs[0];
}

void require_positive_pair(
    const std::vector<int64_t>& values, const char* attribute) {
  if (values.size() != 2 || values[0] <= 0 || values[1] <= 0) {
    throw std::runtime_error(std::string("MaxPool ") + attribute +
                             " must contain two positive values");
  }
}

int64_t output_size(
    int64_t input, int64_t pad_begin, int64_t pad_end,
    int64_t effective_kernel, int64_t stride, bool ceil_mode) {
  const int64_t numerator = input + pad_begin + pad_end - effective_kernel;
  if (numerator < 0) return 0;
  int64_t output =
      (ceil_mode ? (numerator + stride - 1) / stride : numerator / stride) + 1;
  if (ceil_mode && output > 0 &&
      (output - 1) * stride >= input + pad_begin) {
    --output;
  }
  return output;
}

}  // namespace

void max_pool(const OpNode& node,
              std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_tensor(node, tensors);
  const auto kernel = node.get_ints("kernel_shape");
  const auto strides = node.get_ints("strides", {1, 1});
  const auto dilations = node.get_ints("dilations", {1, 1});
  const auto pads = node.get_ints("pads", {0, 0, 0, 0});
  require_positive_pair(kernel, "kernel_shape");
  require_positive_pair(strides, "strides");
  require_positive_pair(dilations, "dilations");
  if (pads.size() != 4 ||
      std::any_of(pads.begin(), pads.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error(
        "MaxPool pads must contain four non-negative values");
  }
  if (node.get_int("storage_order", 0) != 0) {
    throw std::runtime_error("MaxPool storage_order=1 is unsupported");
  }
  if (node.outputs.size() > 1 && !node.outputs[1].empty()) {
    throw std::runtime_error("MaxPool indices output is unsupported");
  }

  const bool ceil_mode = node.get_int("ceil_mode", 0) != 0;
  const int64_t batch = input.shape()[0];
  const int64_t channels = input.shape()[1];
  const int64_t height = input.shape()[2];
  const int64_t width = input.shape()[3];
  const int64_t effective_h = dilations[0] * (kernel[0] - 1) + 1;
  const int64_t effective_w = dilations[1] * (kernel[1] - 1) + 1;
  const int64_t output_h =
      output_size(height, pads[0], pads[2], effective_h, strides[0], ceil_mode);
  const int64_t output_w =
      output_size(width, pads[1], pads[3], effective_w, strides[1], ceil_mode);

  Tensor output(
      {batch, channels, output_h, output_w}, Tensor::DataType::FLOAT32);
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t oh = 0; oh < output_h; ++oh) {
        for (int64_t ow = 0; ow < output_w; ++ow) {
          float maximum = -std::numeric_limits<float>::infinity();
          for (int64_t kh = 0; kh < kernel[0]; ++kh) {
            const int64_t ih =
                oh * strides[0] - pads[0] + kh * dilations[0];
            if (ih < 0 || ih >= height) continue;
            for (int64_t kw = 0; kw < kernel[1]; ++kw) {
              const int64_t iw =
                  ow * strides[1] - pads[1] + kw * dilations[1];
              if (iw < 0 || iw >= width) continue;
              const int64_t input_offset =
                  n * input.strides()[0] + c * input.strides()[1] +
                  ih * input.strides()[2] + iw * input.strides()[3];
              maximum = std::max(maximum, input.data_f32()[input_offset]);
            }
          }
          const int64_t output_offset =
              ((n * channels + c) * output_h + oh) * output_w + ow;
          output.data_f32()[output_offset] = maximum;
        }
      }
    }
  }
  tensors.insert_or_assign(output_name(node), std::move(output));
}

void global_avg_pool(
    const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_tensor(node, tensors);
  const int64_t batch = input.shape()[0];
  const int64_t channels = input.shape()[1];
  const int64_t height = input.shape()[2];
  const int64_t width = input.shape()[3];
  if (height == 0 || width == 0) {
    throw std::runtime_error(
        "GlobalAveragePool requires non-empty spatial dimensions");
  }

  Tensor output({batch, channels, 1, 1}, Tensor::DataType::FLOAT32);
  const float divisor = static_cast<float>(height * width);
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      float sum = 0.0f;
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const int64_t input_offset =
              n * input.strides()[0] + c * input.strides()[1] +
              h * input.strides()[2] + w * input.strides()[3];
          sum += input.data_f32()[input_offset];
        }
      }
      output.data_f32()[n * channels + c] = sum / divisor;
    }
  }
  tensors.insert_or_assign(output_name(node), std::move(output));
}

}  // namespace nn::kernel
