#include "nn/memory_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn {
namespace {

size_t dtype_size(Tensor::DataType dtype) {
  switch (dtype) {
    case Tensor::DataType::FLOAT32: return sizeof(float);
    case Tensor::DataType::INT64: return sizeof(int64_t);
    case Tensor::DataType::INT32: return sizeof(int32_t);
    case Tensor::DataType::BOOL: return sizeof(bool);
    case Tensor::DataType::UINT8: return sizeof(uint8_t);
  }
  throw std::runtime_error("Unknown tensor dtype in memory planner");
}

size_t tensor_bytes(const TensorInfo& info) {
  if (!info.shape_known) return 0;
  size_t elements = 1;
  for (const int64_t dimension : info.shape) {
    if (dimension < 0) return 0;
    const size_t value = static_cast<size_t>(dimension);
    if (value != 0 &&
        elements > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error("Tensor size overflow in memory planner");
    }
    elements *= value;
  }
  const size_t element_bytes = dtype_size(info.dtype);
  if (elements > std::numeric_limits<size_t>::max() / element_bytes) {
    throw std::overflow_error("Tensor byte size overflow in memory planner");
  }
  return elements * element_bytes;
}

size_t align_up(size_t value, size_t alignment) {
  if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("Memory arena alignment overflow");
  }
  return (value + alignment - 1) / alignment * alignment;
}

bool overlaps(const TensorLifetime& a, const TensorLifetime& b) {
  return !(a.last_use <= b.first_use || b.last_use <= a.first_use);
}

struct PhysicalSlot {
  size_t offset = 0;
  size_t size = 0;
  std::vector<TensorLifetime> assignments;
};

}  // namespace

std::vector<TensorLifetime> tensor_lifetimes(const InferenceGraph& graph) {
  std::unordered_map<std::string, TensorLifetime> lifetimes;
  const int node_count = static_cast<int>(graph.nodes.size());

  for (int index = 0; index < node_count; ++index) {
    for (const auto& output : graph.nodes[static_cast<size_t>(index)].outputs) {
      if (output.empty() || graph.initializers.count(output) != 0) continue;
      const auto info = graph.value_info.find(output);
      if (info == graph.value_info.end()) continue;
      const size_t bytes = tensor_bytes(info->second);
      if (bytes == 0) continue;
      auto [entry, inserted] = lifetimes.emplace(
          output, TensorLifetime{output, index, index, bytes});
      if (!inserted) {
        throw std::runtime_error(
            "Multiple producers found while planning tensor '" + output + "'");
      }
    }
  }

  for (int index = 0; index < node_count; ++index) {
    for (const auto& input : graph.nodes[static_cast<size_t>(index)].inputs) {
      const auto lifetime = lifetimes.find(input);
      if (lifetime != lifetimes.end()) {
        lifetime->second.last_use =
            std::max(lifetime->second.last_use, index);
      }
    }
  }

  for (const auto& output : graph.output_names) {
    const auto lifetime = lifetimes.find(output);
    if (lifetime != lifetimes.end()) {
      lifetime->second.last_use = node_count;
    }
  }

  std::vector<TensorLifetime> result;
  result.reserve(lifetimes.size());
  for (auto& [name, lifetime] : lifetimes) {
    (void)name;
    result.push_back(std::move(lifetime));
  }
  std::sort(
      result.begin(), result.end(),
      [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.first_use != b.first_use) return a.first_use < b.first_use;
        return a.name < b.name;
      });
  return result;
}

MemoryPlan plan_memory(const InferenceGraph& graph) {
  auto lifetimes = tensor_lifetimes(graph);
  std::sort(
      lifetimes.begin(), lifetimes.end(),
      [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
        if (a.first_use != b.first_use) return a.first_use < b.first_use;
        return a.name < b.name;
      });

  MemoryPlan plan;
  std::vector<PhysicalSlot> slots;
  for (const auto& lifetime : lifetimes) {
    size_t best = slots.size();
    for (size_t index = 0; index < slots.size(); ++index) {
      if (slots[index].size < lifetime.nbytes) continue;
      const bool compatible = std::none_of(
          slots[index].assignments.begin(), slots[index].assignments.end(),
          [&lifetime](const TensorLifetime& assigned) {
            return overlaps(lifetime, assigned);
          });
      if (!compatible) continue;
      if (best == slots.size() || slots[index].size < slots[best].size) {
        best = index;
      }
    }

    if (best == slots.size()) {
      const size_t offset = align_up(plan.total_bytes, 64);
      slots.push_back(PhysicalSlot{offset, lifetime.nbytes, {lifetime}});
      best = slots.size() - 1;
      plan.total_bytes = offset + lifetime.nbytes;
    } else {
      slots[best].assignments.push_back(lifetime);
    }
    plan.tensor_slots.emplace(
        lifetime.name,
        MemoryPlan::Slot{slots[best].offset, lifetime.nbytes});
  }

  return plan;
}

}  // namespace nn
