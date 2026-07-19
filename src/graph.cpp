#include "nn/graph.hpp"

#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace nn {

int64_t OpNode::get_int(const std::string& key, int64_t default_val) const {
  const auto it = attrs.find(key);
  if (it == attrs.end()) return default_val;
  if (it->second.type != AttributeValue::Type::INT) {
    throw std::runtime_error("Attribute '" + key + "' is not an integer");
  }
  return it->second.i;
}

float OpNode::get_float(const std::string& key, float default_val) const {
  const auto it = attrs.find(key);
  if (it == attrs.end()) return default_val;
  if (it->second.type != AttributeValue::Type::FLOAT) {
    throw std::runtime_error("Attribute '" + key + "' is not a float");
  }
  return it->second.f;
}

std::vector<int64_t> OpNode::get_ints(
    const std::string& key, std::vector<int64_t> default_val) const {
  const auto it = attrs.find(key);
  if (it == attrs.end()) return default_val;
  if (it->second.type != AttributeValue::Type::INTS) {
    throw std::runtime_error("Attribute '" + key + "' is not an integer list");
  }
  return it->second.ints;
}

void sort_topologically(InferenceGraph& graph) {
  const size_t count = graph.nodes.size();
  std::unordered_map<std::string, size_t> producer;
  for (size_t index = 0; index < count; ++index) {
    for (const auto& output : graph.nodes[index].outputs) {
      if (output.empty()) continue;
      if (!producer.emplace(output, index).second) {
        throw std::runtime_error("Multiple nodes produce tensor '" + output + "'");
      }
    }
  }

  std::vector<size_t> indegree(count, 0);
  std::vector<std::vector<size_t>> consumers(count);
  for (size_t consumer = 0; consumer < count; ++consumer) {
    std::unordered_set<size_t> dependencies;
    for (const auto& input : graph.nodes[consumer].inputs) {
      if (input.empty()) continue;
      const auto it = producer.find(input);
      if (it != producer.end() && it->second != consumer) {
        dependencies.insert(it->second);
      }
    }
    indegree[consumer] = dependencies.size();
    for (const size_t dependency : dependencies) {
      consumers[dependency].push_back(consumer);
    }
  }

  std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
  for (size_t index = 0; index < count; ++index) {
    if (indegree[index] == 0) ready.push(index);
  }

  std::vector<OpNode> sorted;
  sorted.reserve(count);
  while (!ready.empty()) {
    const size_t index = ready.top();
    ready.pop();
    sorted.push_back(std::move(graph.nodes[index]));
    for (const size_t consumer : consumers[index]) {
      if (--indegree[consumer] == 0) ready.push(consumer);
    }
  }

  if (sorted.size() != count) {
    throw std::runtime_error("ONNX graph contains a dependency cycle");
  }
  graph.nodes = std::move(sorted);
}

}  // namespace nn
