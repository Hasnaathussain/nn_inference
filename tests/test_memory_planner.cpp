#include "nn/memory_planner.hpp"
#include "nn/execution_engine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nn;

namespace {

TensorInfo float_info(const std::string& name, std::vector<int64_t> shape) {
  return TensorInfo{
      name, Tensor::DataType::FLOAT32, std::move(shape), {}, true};
}

OpNode node(
    const std::string& name, std::vector<std::string> inputs,
    const std::string& output) {
  OpNode result;
  result.name = name;
  result.op_type = "Relu";
  result.inputs = std::move(inputs);
  result.outputs = {output};
  return result;
}

}  // namespace

TEST_CASE("Memory planner reuses sequential activation storage") {
  InferenceGraph graph;
  graph.nodes = {
      node("n0", {"input"}, "a"),
      node("n1", {"a"}, "b"),
      node("n2", {"b"}, "c"),
      node("n3", {"c"}, "output"),
  };
  graph.output_names = {"output"};
  for (const auto& name : {"a", "b", "c", "output"}) {
    graph.value_info[name] = float_info(name, {256});
  }

  const MemoryPlan plan = plan_memory(graph);
  REQUIRE(plan.tensor_slots.size() == 4);
  REQUIRE(plan.total_bytes == 1024);
  REQUIRE(plan.tensor_slots.at("a").offset ==
          plan.tensor_slots.at("b").offset);
  REQUIRE(plan.tensor_slots.at("b").offset ==
          plan.tensor_slots.at("c").offset);
  REQUIRE(plan.tensor_slots.at("c").offset ==
          plan.tensor_slots.at("output").offset);

  const size_t naive_bytes = 4 * 256 * sizeof(float);
  REQUIRE(plan.total_bytes <= naive_bytes * 8 / 10);
}

TEST_CASE("Memory planner keeps overlapping and graph-output lifetimes separate") {
  InferenceGraph graph;
  graph.nodes = {
      node("make_a", {"input"}, "a"),
      node("make_b", {"input"}, "b"),
      node("use_b", {"b"}, "c"),
      node("join", {"a", "c"}, "output"),
  };
  graph.output_names = {"a", "output"};
  for (const auto& name : {"a", "b", "c", "output"}) {
    graph.value_info[name] = float_info(name, {100});
  }

  const MemoryPlan plan = plan_memory(graph);
  REQUIRE(plan.tensor_slots.at("a").offset !=
          plan.tensor_slots.at("b").offset);
  REQUIRE(plan.tensor_slots.at("a").offset !=
          plan.tensor_slots.at("output").offset);
  for (const auto& [name, slot] : plan.tensor_slots) {
    (void)name;
    REQUIRE(slot.offset % 64 == 0);
  }
}

TEST_CASE("Memory planner excludes inputs initializers and unresolved tensors") {
  InferenceGraph graph;
  graph.input_names = {"input"};
  graph.initializers.emplace(
      "weight", Tensor({4}, Tensor::DataType::FLOAT32));
  graph.nodes = {
      node("dynamic", {"input", "weight"}, "dynamic_output"),
      node("static", {"dynamic_output"}, "static_output"),
  };
  graph.output_names = {"static_output"};
  graph.value_info["input"] = float_info("input", {1, 4});
  graph.value_info["weight"] = float_info("weight", {4});
  graph.value_info["dynamic_output"] =
      float_info("dynamic_output", {-1, 4});
  graph.value_info["static_output"] =
      float_info("static_output", {1, 4});

  const MemoryPlan plan = plan_memory(graph);
  REQUIRE(plan.tensor_slots.count("input") == 0);
  REQUIRE(plan.tensor_slots.count("weight") == 0);
  REQUIRE(plan.tensor_slots.count("dynamic_output") == 0);
  REQUIRE(plan.tensor_slots.count("static_output") == 1);
  REQUIRE(plan.total_bytes == 16);
}

TEST_CASE("Execution engine uses arena safely and can disable it") {
  InferenceGraph graph;
  graph.input_names = {"input"};
  graph.output_names = {"output"};
  graph.nodes = {
      node("relu_a", {"input"}, "a"),
      node("relu_b", {"a"}, "b"),
      node("relu_output", {"b"}, "output"),
  };
  for (const auto& name : {"input", "a", "b", "output"}) {
    graph.value_info[name] = float_info(name, {4});
  }

  Tensor first_input({4}, Tensor::DataType::FLOAT32);
  const float first_values[] = {-1.0F, 2.0F, -3.0F, 4.0F};
  first_input.fill_from_raw(first_values, 4);
  ExecutionEngine engine(graph);
  const auto first = engine.run({{"input", first_input}}).at("output");
  REQUIRE(engine.get_planned_activation_bytes() == 16);
  REQUIRE(engine.get_naive_activation_bytes() == 48);
  REQUIRE(engine.get_planned_activation_bytes() <=
          engine.get_naive_activation_bytes() * 8 / 10);
  REQUIRE(first.data_f32()[0] == 0.0F);
  REQUIRE(first.data_f32()[1] == 2.0F);
  REQUIRE(first.data_f32()[2] == 0.0F);
  REQUIRE(first.data_f32()[3] == 4.0F);

  Tensor second_input({4}, Tensor::DataType::FLOAT32);
  const float second_values[] = {5.0F, -6.0F, 7.0F, -8.0F};
  second_input.fill_from_raw(second_values, 4);
  const auto second = engine.run({{"input", second_input}}).at("output");
  REQUIRE(second.data_f32()[0] == 5.0F);
  REQUIRE(second.data_f32()[1] == 0.0F);
  REQUIRE(first.data_f32()[1] == 2.0F);

  engine.set_memory_planner(false);
  const auto unplanned = engine.run({{"input", first_input}}).at("output");
  REQUIRE(engine.get_planned_activation_bytes() == 0);
  REQUIRE(engine.get_naive_activation_bytes() == 48);
  REQUIRE(unplanned.data_f32()[3] == 4.0F);
}