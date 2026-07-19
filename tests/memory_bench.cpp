#include "nn/execution_engine.hpp"

#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 || (std::string(argv[1]) != "planned" &&
                    std::string(argv[1]) != "naive")) {
    std::cerr << "usage: memory_bench planned|naive\n";
    return 2;
  }

  constexpr int64_t elements = 4 * 1024 * 1024;
  constexpr int node_count = 16;
  nn::InferenceGraph graph;
  graph.input_names = {"input"};
  graph.value_info["input"] = {
      "input", nn::Tensor::DataType::FLOAT32, {elements}, {}};

  std::string previous = "input";
  for (int index = 0; index < node_count; ++index) {
    const std::string output =
        index + 1 == node_count ? "output" : "activation_" + std::to_string(index);
    nn::OpNode node;
    node.name = "relu_" + std::to_string(index);
    node.op_type = "Relu";
    node.inputs = {previous};
    node.outputs = {output};
    graph.nodes.push_back(std::move(node));
    graph.value_info[output] = {
        output, nn::Tensor::DataType::FLOAT32, {elements}, {}};
    previous = output;
  }
  graph.output_names = {"output"};

  nn::Tensor input({elements}, nn::Tensor::DataType::FLOAT32);
  input.fill_zeros();
  nn::ExecutionEngine engine(std::move(graph));
  engine.set_memory_planner(std::string(argv[1]) == "planned");
  const auto outputs = engine.run({{"input", input}});
  if (outputs.at("output").numel() != elements) return 3;

  std::cout << "planned_bytes=" << engine.get_planned_activation_bytes()
            << " naive_bytes=" << engine.get_naive_activation_bytes() << '\n';
  return 0;
}