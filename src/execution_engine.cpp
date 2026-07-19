#include "nn/execution_engine.hpp"

#include "nn/onnx_parser.hpp"
#include "nn/ops/bert.hpp"
#include "nn/ops/conv2d.hpp"
#include "nn/ops/elementwise.hpp"
#include "nn/ops/gemm.hpp"
#include "nn/ops/norm.hpp"
#include "nn/ops/pool.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nn {

ExecutionEngine::ExecutionEngine(InferenceGraph graph)
    : graph_(std::move(graph)) {}

ExecutionEngine::ExecutionEngine(const std::string& model_path)
    : ExecutionEngine(load_onnx(model_path)) {}

std::unordered_map<std::string, Tensor> ExecutionEngine::run(
    const std::unordered_map<std::string, Tensor>& inputs) {
  tensor_map_ = graph_.initializers;
  op_times_ms_.clear();

  for (const auto& name : graph_.input_names) {
    const auto input = inputs.find(name);
    if (input == inputs.end()) {
      throw std::runtime_error("Missing required model input: " + name);
    }

    const auto info = graph_.value_info.find(name);
    if (info != graph_.value_info.end()) {
      if (input->second.dtype() != info->second.dtype) {
        throw std::runtime_error("Input dtype mismatch for tensor: " + name);
      }
      const auto& expected = info->second.shape;
      const auto& actual = input->second.shape();
      if (expected.size() != actual.size()) {
        throw std::runtime_error("Input rank mismatch for tensor: " + name);
      }
      for (size_t axis = 0; axis < expected.size(); ++axis) {
        if (expected[axis] >= 0 && expected[axis] != actual[axis]) {
          throw std::runtime_error("Input shape mismatch for tensor: " + name);
        }
      }
    }
    tensor_map_.insert_or_assign(name, input->second);
  }

  prepare_memory_plan(inputs);
  for (const auto& node : graph_.nodes) {
    const auto begin = std::chrono::steady_clock::now();
    dispatch(node);
    if (memory_planner_enabled_) materialize_node_outputs(node);
    if (profiling_) {
      const auto end = std::chrono::steady_clock::now();
      const double milliseconds =
          std::chrono::duration<double, std::milli>(end - begin).count();
      const std::string key = node.name.empty() ? node.op_type : node.name;
      op_times_ms_[key] += milliseconds;
    }
  }

  std::unordered_map<std::string, Tensor> outputs;
  for (const auto& name : graph_.output_names) {
    outputs.emplace(name, get_tensor(name).clone_contiguous());
  }
  return outputs;
}

void ExecutionEngine::set_profiling(bool enable) { profiling_ = enable; }

std::unordered_map<std::string, double>
ExecutionEngine::get_op_times_ms() const {
  return op_times_ms_;
}

void ExecutionEngine::set_memory_planner(bool enable) {
  memory_planner_enabled_ = enable;
}

size_t ExecutionEngine::get_planned_activation_bytes() const {
  return memory_plan_.total_bytes;
}

size_t ExecutionEngine::get_naive_activation_bytes() const {
  return naive_activation_bytes_;
}

Tensor ExecutionEngine::get_tensor(const std::string& name) const {
  const auto tensor = tensor_map_.find(name);
  if (tensor == tensor_map_.end()) {
    throw std::runtime_error("Tensor is unavailable: " + name);
  }
  return tensor->second;
}

void ExecutionEngine::prepare_memory_plan(
    const std::unordered_map<std::string, Tensor>& inputs) {
  InferenceGraph resolved = graph_;
  std::unordered_map<std::string, int64_t> symbols;

  for (const auto& name : graph_.input_names) {
    const auto input = inputs.find(name);
    const auto info = graph_.value_info.find(name);
    if (input == inputs.end() || info == graph_.value_info.end()) continue;
    if (info->second.dim_params.size() != info->second.shape.size()) continue;
    for (size_t axis = 0; axis < info->second.shape.size(); ++axis) {
      const std::string& symbol = info->second.dim_params[axis];
      if (symbol.empty()) continue;
      const int64_t value = input->second.shape()[axis];
      const auto [entry, inserted] = symbols.emplace(symbol, value);
      if (!inserted && entry->second != value) {
        throw std::runtime_error(
            "Conflicting runtime values for symbolic dimension: " + symbol);
      }
    }
  }

  for (auto& [name, info] : resolved.value_info) {
    (void)name;
    if (info.dim_params.size() != info.shape.size()) continue;
    for (size_t axis = 0; axis < info.shape.size(); ++axis) {
      if (info.shape[axis] >= 0 || info.dim_params[axis].empty()) continue;
      const auto value = symbols.find(info.dim_params[axis]);
      if (value != symbols.end()) info.shape[axis] = value->second;
    }
  }

  naive_activation_bytes_ = 0;
  for (const auto& lifetime : tensor_lifetimes(resolved)) {
    naive_activation_bytes_ += lifetime.nbytes;
  }

  if (!memory_planner_enabled_) {
    memory_plan_ = {};
    std::vector<uint8_t>().swap(arena_);
    arena_base_offset_ = 0;
    return;
  }

  memory_plan_ = plan_memory(resolved);
  if (memory_plan_.total_bytes == 0) {
    std::vector<uint8_t>().swap(arena_);
    arena_base_offset_ = 0;
    return;
  }

  arena_.resize(memory_plan_.total_bytes + 63);
  const uintptr_t raw = reinterpret_cast<uintptr_t>(arena_.data());
  const uintptr_t aligned = (raw + 63U) & ~uintptr_t{63U};
  arena_base_offset_ = static_cast<size_t>(aligned - raw);
}

uint8_t* ExecutionEngine::arena_base() {
  return arena_.empty() ? nullptr : arena_.data() + arena_base_offset_;
}

const uint8_t* ExecutionEngine::arena_base() const {
  return arena_.empty() ? nullptr : arena_.data() + arena_base_offset_;
}

void ExecutionEngine::materialize_node_outputs(const OpNode& node) {
  uint8_t* base = arena_base();
  for (const auto& name : node.outputs) {
    const auto slot = memory_plan_.tensor_slots.find(name);
    if (name.empty() || slot == memory_plan_.tensor_slots.end()) continue;
    const auto produced = tensor_map_.find(name);
    if (produced == tensor_map_.end()) {
      throw std::runtime_error(
          "Operator did not produce planned tensor: " + name);
    }
    if (static_cast<size_t>(produced->second.nbytes()) != slot->second.size) {
      throw std::runtime_error(
          "Runtime tensor size differs from memory plan for: " + name);
    }

    Tensor source = produced->second;
    const uintptr_t source_address =
        reinterpret_cast<uintptr_t>(source.data_raw());
    const uintptr_t arena_address = reinterpret_cast<uintptr_t>(base);
    const bool source_in_arena =
        base != nullptr && source_address >= arena_address &&
        source_address < arena_address + memory_plan_.total_bytes;
    if (source_in_arena) source = source.clone_contiguous();

    void* destination = base + slot->second.offset;
    source.copy_to_bytes(destination, slot->second.size);
    produced->second =
        Tensor(source.shape(), source.dtype(), destination);
  }
}

void ExecutionEngine::dispatch(const OpNode& node) {
  if (node.op_type == "Add") {
    kernel::elementwise_add(node, tensor_map_);
  } else if (node.op_type == "Mul") {
    kernel::elementwise_mul(node, tensor_map_);
  } else if (node.op_type == "Relu") {
    kernel::relu(node, tensor_map_);
  } else if (node.op_type == "Reshape") {
    kernel::reshape_op(node, tensor_map_);
  } else if (node.op_type == "Flatten") {
    kernel::flatten(node, tensor_map_);
  } else if (node.op_type == "Transpose") {
    kernel::transpose_op(node, tensor_map_);
  } else if (node.op_type == "Conv") {
    kernel::conv2d(node, tensor_map_);
  } else if (node.op_type == "BatchNormalization") {
    kernel::batch_norm(node, tensor_map_);
  } else if (node.op_type == "MaxPool") {
    kernel::max_pool(node, tensor_map_);
  } else if (node.op_type == "GlobalAveragePool") {
    kernel::global_avg_pool(node, tensor_map_);
  } else if (node.op_type == "Gemm") {
    kernel::gemm_op(node, tensor_map_);
  } else if (node.op_type == "Constant") {
    kernel::constant_op(node, tensor_map_);
  } else if (node.op_type == "Shape") {
    kernel::shape_op(node, tensor_map_);
  } else if (node.op_type == "Gather") {
    kernel::gather(node, tensor_map_);
  } else if (node.op_type == "Concat") {
    kernel::concat(node, tensor_map_);
  } else if (node.op_type == "Unsqueeze") {
    kernel::unsqueeze(node, tensor_map_);
  } else if (node.op_type == "Squeeze") {
    kernel::squeeze(node, tensor_map_);
  } else if (node.op_type == "Cast") {
    kernel::cast_op(node, tensor_map_);
  } else if (node.op_type == "Slice") {
    kernel::slice_op(node, tensor_map_);
  } else if (node.op_type == "Expand") {
    kernel::expand(node, tensor_map_);
  } else if (node.op_type == "ReduceMean") {
    kernel::reduce_mean(node, tensor_map_);
  } else if (node.op_type == "Softmax") {
    kernel::softmax(node, tensor_map_);
  } else if (node.op_type == "MatMul") {
    kernel::matmul(node, tensor_map_);
  } else if (node.op_type == "LayerNormalization") {
    kernel::layer_norm(node, tensor_map_);
  } else if (node.op_type == "Where") {
    kernel::where_op(node, tensor_map_);
  } else if (node.op_type == "Sub") {
    kernel::elementwise_sub(node, tensor_map_);
  } else if (node.op_type == "Div") {
    kernel::elementwise_div(node, tensor_map_);
  } else if (node.op_type == "Pow") {
    kernel::elementwise_pow(node, tensor_map_);
  } else if (node.op_type == "Sqrt") {
    kernel::sqrt_op(node, tensor_map_);
  } else if (node.op_type == "Erf") {
    kernel::erf_op(node, tensor_map_);
  } else {
    throw std::runtime_error("Unsupported op: " + node.op_type);
  }
}

}  // namespace nn