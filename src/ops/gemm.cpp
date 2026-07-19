#include "nn/ops/gemm.hpp"

#include "nn/ops/elementwise.hpp"

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
    throw std::runtime_error("Missing required Gemm input " + std::to_string(index));
  }
  const auto found = tensors.find(node.inputs[index]);
  if (found == tensors.end()) {
    throw std::runtime_error("Missing Gemm tensor: " + node.inputs[index]);
  }
  return found->second;
}

}  // namespace

void gemm_op(const OpNode& node,
             std::unordered_map<std::string, Tensor>& tensors) {
  const Tensor& a = input_at(node, tensors, 0);
  const Tensor& b = input_at(node, tensors, 1);
  if (a.dtype() != Tensor::DataType::FLOAT32 ||
      b.dtype() != Tensor::DataType::FLOAT32 ||
      a.ndim() != 2 || b.ndim() != 2) {
    throw std::runtime_error("Gemm A and B must be FLOAT32 rank-2 tensors");
  }

  const bool trans_a = node.get_int("transA", 0) != 0;
  const bool trans_b = node.get_int("transB", 0) != 0;
  const int64_t m = trans_a ? a.shape()[1] : a.shape()[0];
  const int64_t k = trans_a ? a.shape()[0] : a.shape()[1];
  const int64_t b_k = trans_b ? b.shape()[1] : b.shape()[0];
  const int64_t n = trans_b ? b.shape()[0] : b.shape()[1];
  if (k != b_k) throw std::runtime_error("Gemm inner dimensions do not match");
  if (m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Gemm dimensions exceed scalar kernel limits");
  }

  Tensor output({m, n}, Tensor::DataType::FLOAT32);
  output.fill_zeros();
  if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
    const Tensor& c = input_at(node, tensors, 2);
    Tensor broadcast = broadcast_binary_op(
        output, c, [](float, float value) { return value; });
    if (broadcast.shape() != output.shape()) {
      throw std::runtime_error("Gemm C cannot broadcast to [M, N]");
    }
    output = std::move(broadcast);
  }

  Tensor packed_a = a.reshape({a.numel()});
  Tensor packed_b = b.reshape({b.numel()});
  gemm(
      packed_a.data_f32(), packed_b.data_f32(), output.data_f32(),
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      node.get_float("alpha", 1.0f), node.get_float("beta", 1.0f),
      trans_a, trans_b);

  if (node.outputs.empty() || node.outputs[0].empty()) {
    throw std::runtime_error("Gemm is missing its output name");
  }
  tensors.insert_or_assign(node.outputs[0], std::move(output));
}

}  // namespace nn::kernel
