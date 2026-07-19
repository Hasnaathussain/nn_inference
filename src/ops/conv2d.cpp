#include "nn/ops/conv2d.hpp"

#include "nn/ops/gemm.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn::kernel {
namespace {

const Tensor& input_at(
    const OpNode& node, const std::unordered_map<std::string, Tensor>& tensors,
    size_t index) {
  if (index >= node.inputs.size() || node.inputs[index].empty()) {
    throw std::runtime_error("Missing required Conv input " + std::to_string(index));
  }
  const auto found = tensors.find(node.inputs[index]);
  if (found == tensors.end()) {
    throw std::runtime_error("Missing Conv tensor: " + node.inputs[index]);
  }
  return found->second;
}

void require_pair(const std::vector<int64_t>& values, const char* name) {
  if (values.size() != 2 || values[0] <= 0 || values[1] <= 0) {
    throw std::runtime_error(std::string("Conv ") + name +
                             " must contain two positive values");
  }
}

}  // namespace

void conv2d(const OpNode& node,
            std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& input = input_at(node, tensors, 0);
  const Tensor& weights = input_at(node, tensors, 1);
  if (input.dtype() != Tensor::DataType::FLOAT32 ||
      weights.dtype() != Tensor::DataType::FLOAT32) {
    throw std::runtime_error("Conv requires FLOAT32 input and weights");
  }
  if (input.ndim() != 4 || weights.ndim() != 4) {
    throw std::runtime_error("Conv requires rank-4 NCHW input and OIHW weights");
  }

  const int64_t batch = input.shape()[0];
  const int64_t channels = input.shape()[1];
  const int64_t height = input.shape()[2];
  const int64_t width = input.shape()[3];
  const int64_t output_channels = weights.shape()[0];
  const int64_t kernel_channels = weights.shape()[1];
  const int64_t kernel_h = weights.shape()[2];
  const int64_t kernel_w = weights.shape()[3];

  const int64_t groups = node.get_int("group", 1);
  if (groups <= 0 || channels % groups != 0 ||
      output_channels % groups != 0 ||
      kernel_channels != channels / groups) {
    throw std::runtime_error("Conv group/channel configuration is invalid");
  }

  const auto kernel_shape =
      node.get_ints("kernel_shape", {kernel_h, kernel_w});
  require_pair(kernel_shape, "kernel_shape");
  if (kernel_shape[0] != kernel_h || kernel_shape[1] != kernel_w) {
    throw std::runtime_error("Conv kernel_shape does not match weight shape");
  }
  const auto dilations = node.get_ints("dilations", {1, 1});
  const auto strides = node.get_ints("strides", {1, 1});
  require_pair(dilations, "dilations");
  require_pair(strides, "strides");
  const auto pads = node.get_ints("pads", {0, 0, 0, 0});
  if (pads.size() != 4 ||
      std::any_of(pads.begin(), pads.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error("Conv pads must contain four non-negative values");
  }

  const int64_t effective_h = dilations[0] * (kernel_h - 1) + 1;
  const int64_t effective_w = dilations[1] * (kernel_w - 1) + 1;
  const int64_t padded_h = height + pads[0] + pads[2];
  const int64_t padded_w = width + pads[1] + pads[3];
  const int64_t output_h =
      padded_h < effective_h ? 0 : (padded_h - effective_h) / strides[0] + 1;
  const int64_t output_w =
      padded_w < effective_w ? 0 : (padded_w - effective_w) / strides[1] + 1;

  const Tensor* bias = nullptr;
  if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
    bias = &input_at(node, tensors, 2);
    if (bias->dtype() != Tensor::DataType::FLOAT32 || bias->ndim() != 1 ||
        bias->shape()[0] != output_channels) {
      throw std::runtime_error("Conv bias must be FLOAT32 with shape [output_channels]");
    }
  }

  Tensor output(
      {batch, output_channels, output_h, output_w},
      Tensor::DataType::FLOAT32);
  output.fill_zeros();

  const int64_t channels_per_group = channels / groups;
  const int64_t outputs_per_group = output_channels / groups;
  const int64_t patch_size = channels_per_group * kernel_h * kernel_w;
  const int64_t rows = batch * output_h * output_w;
  if (rows > std::numeric_limits<int>::max() ||
      outputs_per_group > std::numeric_limits<int>::max() ||
      patch_size > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Conv dimensions exceed scalar GEMM limits");
  }

  std::vector<float> column(static_cast<size_t>(rows * patch_size));
  std::vector<float> packed_weights(
      static_cast<size_t>(outputs_per_group * patch_size));
  std::vector<float> product(static_cast<size_t>(rows * outputs_per_group));

  for (int64_t group = 0; group < groups; ++group) {
    for (int64_t local_oc = 0; local_oc < outputs_per_group; ++local_oc) {
      const int64_t oc = group * outputs_per_group + local_oc;
      for (int64_t local_ic = 0; local_ic < channels_per_group; ++local_ic) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
          for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t patch =
                (local_ic * kernel_h + kh) * kernel_w + kw;
            const int64_t weight_offset =
                oc * weights.strides()[0] +
                local_ic * weights.strides()[1] +
                kh * weights.strides()[2] +
                kw * weights.strides()[3];
            packed_weights[static_cast<size_t>(
                patch * outputs_per_group + local_oc)] =
                weights.data_f32()[weight_offset];
          }
        }
      }
    }

    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t oh = 0; oh < output_h; ++oh) {
        for (int64_t ow = 0; ow < output_w; ++ow) {
          const int64_t row = (n * output_h + oh) * output_w + ow;
          for (int64_t local_ic = 0; local_ic < channels_per_group; ++local_ic) {
            const int64_t ic = group * channels_per_group + local_ic;
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
              const int64_t ih = oh * strides[0] - pads[0] + kh * dilations[0];
              for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = ow * strides[1] - pads[1] + kw * dilations[1];
                const int64_t patch =
                    (local_ic * kernel_h + kh) * kernel_w + kw;
                float value = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const int64_t input_offset =
                      n * input.strides()[0] + ic * input.strides()[1] +
                      ih * input.strides()[2] + iw * input.strides()[3];
                  value = input.data_f32()[input_offset];
                }
                column[static_cast<size_t>(row * patch_size + patch)] = value;
              }
            }
          }
        }
      }
    }

    gemm(column.data(), packed_weights.data(), product.data(),
         static_cast<int>(rows), static_cast<int>(outputs_per_group),
         static_cast<int>(patch_size), 1.0f, 0.0f, false, false);

    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t oh = 0; oh < output_h; ++oh) {
        for (int64_t ow = 0; ow < output_w; ++ow) {
          const int64_t row = (n * output_h + oh) * output_w + ow;
          for (int64_t local_oc = 0; local_oc < outputs_per_group; ++local_oc) {
            const int64_t oc = group * outputs_per_group + local_oc;
            const int64_t output_offset =
                ((n * output_channels + oc) * output_h + oh) * output_w + ow;
            const float bias_value =
                bias == nullptr
                    ? 0.0f
                    : bias->data_f32()[oc * bias->strides()[0]];
            output.data_f32()[output_offset] =
                product[static_cast<size_t>(row * outputs_per_group + local_oc)] +
                bias_value;
          }
        }
      }
    }
  }

  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error("Conv is missing its output name");
  }
  tensors.insert_or_assign(node.outputs[0], std::move(output));
}

}  // namespace nn::kernel
