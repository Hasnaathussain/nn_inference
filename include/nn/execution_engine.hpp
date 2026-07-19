#pragma once

#include "nn/graph.hpp"
#include "nn/memory_planner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn {

class ExecutionEngine {
public:
  explicit ExecutionEngine(InferenceGraph graph);
  explicit ExecutionEngine(const std::string& model_path);

  std::unordered_map<std::string, Tensor> run(
      const std::unordered_map<std::string, Tensor>& inputs);

  void set_profiling(bool enable);
  std::unordered_map<std::string, double> get_op_times_ms() const;

  void set_memory_planner(bool enable);
  size_t get_planned_activation_bytes() const;
  size_t get_naive_activation_bytes() const;

private:
  InferenceGraph graph_;
  MemoryPlan memory_plan_;
  std::vector<uint8_t> arena_;
  size_t arena_base_offset_ = 0;
  size_t naive_activation_bytes_ = 0;
  bool memory_planner_enabled_ = true;
  bool profiling_ = false;
  std::unordered_map<std::string, double> op_times_ms_;
  std::unordered_map<std::string, Tensor> tensor_map_;

  Tensor get_tensor(const std::string& name) const;
  void prepare_memory_plan(
      const std::unordered_map<std::string, Tensor>& inputs);
  void materialize_node_outputs(const OpNode& node);
  uint8_t* arena_base();
  const uint8_t* arena_base() const;
  void dispatch(const OpNode& node);
};

}  // namespace nn