#include "nn/execution_engine.hpp"
#include "nn/ops/elementwise.hpp"

#include <catch2/catch_test_macros.hpp>

#include <unordered_map>
#include <vector>

using namespace nn;

TEST_CASE("Broadcast arithmetic handles scalars and channel broadcasting") {
  Tensor input({2, 3, 2, 2}, Tensor::DataType::FLOAT32);
  std::vector<float> values(static_cast<size_t>(input.numel()));
  for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
  input.fill_from_raw(values.data(), values.size());

  Tensor channel({1, 3, 1, 1}, Tensor::DataType::FLOAT32);
  const float channel_values[] = {10.0f, 20.0f, 30.0f};
  channel.fill_from_raw(channel_values, 3);

  Tensor output = kernel::broadcast_binary_op(
      input, channel, [](float a, float b) { return a + b; });
  REQUIRE(output.shape() == input.shape());
  REQUIRE(output.data_f32()[0] == 10.0f);
  REQUIRE(output.data_f32()[4] == 24.0f);
  REQUIRE(output.data_f32()[8] == 38.0f);
  REQUIRE(output.data_f32()[20] == 50.0f);

  Tensor scalar({}, Tensor::DataType::FLOAT32);
  const float scalar_value = 2.0f;
  scalar.fill_from_raw(&scalar_value, 1);
  Tensor multiplied = kernel::broadcast_binary_op(
      output, scalar, [](float a, float b) { return a * b; });
  REQUIRE(multiplied.data_f32()[20] == 100.0f);
}

TEST_CASE("Relu materializes non-contiguous input in logical order") {
  Tensor input({2, 3}, Tensor::DataType::FLOAT32);
  const float values[] = {-1, 2, -3, 4, -5, 6};
  input.fill_from_raw(values, 6);
  std::unordered_map<std::string, Tensor> tensors{
      {"x", input.transpose({1, 0})}};

  OpNode node;
  node.op_type = "Relu";
  node.inputs = {"x"};
  node.outputs = {"y"};
  kernel::relu(node, tensors);

  REQUIRE(tensors.at("y").shape() == std::vector<int64_t>{3, 2});
  const float expected[] = {0, 4, 2, 0, 0, 6};
  for (int i = 0; i < 6; ++i) {
    REQUIRE(tensors.at("y").data_f32()[i] == expected[i]);
  }
}

TEST_CASE("ExecutionEngine runs a graph and records profiling") {
  InferenceGraph graph;
  graph.input_names = {"x"};
  graph.output_names = {"y"};
  graph.value_info["x"] = {
      "x", Tensor::DataType::FLOAT32, {2, 2}, {}, true};

  Tensor bias({}, Tensor::DataType::FLOAT32);
  const float bias_value = 1.5f;
  bias.fill_from_raw(&bias_value, 1);
  graph.initializers["bias"] = bias;

  OpNode add;
  add.name = "add_node";
  add.op_type = "Add";
  add.inputs = {"x", "bias"};
  add.outputs = {"sum"};

  OpNode relu;
  relu.name = "relu_node";
  relu.op_type = "Relu";
  relu.inputs = {"sum"};
  relu.outputs = {"y"};
  graph.nodes = {add, relu};

  Tensor input({2, 2}, Tensor::DataType::FLOAT32);
  const float values[] = {-3.0f, -1.0f, 0.0f, 2.0f};
  input.fill_from_raw(values, 4);

  ExecutionEngine engine(std::move(graph));
  engine.set_profiling(true);
  const auto output = engine.run({{"x", input}}).at("y");
  const float expected[] = {0.0f, 0.5f, 1.5f, 3.5f};
  for (int i = 0; i < 4; ++i) REQUIRE(output.data_f32()[i] == expected[i]);

  const auto times = engine.get_op_times_ms();
  REQUIRE(times.count("add_node") == 1);
  REQUIRE(times.count("relu_node") == 1);
  REQUIRE(times.at("add_node") >= 0.0);
  REQUIRE(times.at("relu_node") >= 0.0);
  REQUIRE_THROWS(engine.run({}));
}

TEST_CASE("ExecutionEngine rejects unsupported operators") {
  InferenceGraph graph;
  graph.output_names = {"y"};
  OpNode node;
  node.op_type = "DefinitelyUnsupported";
  node.outputs = {"y"};
  graph.nodes = {node};
  ExecutionEngine engine(std::move(graph));
  REQUIRE_THROWS(engine.run({}));
}
