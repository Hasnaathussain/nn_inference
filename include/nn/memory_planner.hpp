#pragma once

#include "nn/graph.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn {

struct TensorLifetime {
  std::string name;
  int first_use = 0;
  int last_use = 0;
  size_t nbytes = 0;
};

struct MemoryPlan {
  struct Slot {
    size_t offset = 0;
    size_t size = 0;
  };

  std::unordered_map<std::string, Slot> tensor_slots;
  size_t total_bytes = 0;
};

MemoryPlan plan_memory(const InferenceGraph& graph);
std::vector<TensorLifetime> tensor_lifetimes(const InferenceGraph& graph);

}  // namespace nn
