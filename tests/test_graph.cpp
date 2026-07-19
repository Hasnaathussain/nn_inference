#include "nn/graph.hpp"
#include "nn/onnx_parser.hpp"

#ifndef ONNX_API
#define ONNX_API
#endif
#include <onnx/onnx-ml.pb.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

namespace {

onnx::ValueInfoProto* add_value(
    onnx::GraphProto* graph, const std::string& name, bool output = false) {
  auto* value = output ? graph->add_output() : graph->add_input();
  value->set_name(name);
  auto* tensor_type = value->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
  tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  tensor_type->mutable_shape()->add_dim()->set_dim_param("features");
  return value;
}

}  // namespace

TEST_CASE("OpNode attribute helpers return values and defaults") {
  OpNode node;
  AttributeValue axis;
  axis.type = AttributeValue::Type::INT;
  axis.i = -1;
  node.attrs["axis"] = axis;

  AttributeValue epsilon;
  epsilon.type = AttributeValue::Type::FLOAT;
  epsilon.f = 1e-5f;
  node.attrs["epsilon"] = epsilon;

  AttributeValue pads;
  pads.type = AttributeValue::Type::INTS;
  pads.ints = {1, 2, 3, 4};
  node.attrs["pads"] = pads;

  REQUIRE(node.get_int("axis") == -1);
  REQUIRE(node.get_int("missing", 7) == 7);
  REQUIRE(node.get_float("epsilon") == 1e-5f);
  REQUIRE(node.get_ints("pads") == std::vector<int64_t>{1, 2, 3, 4});
  REQUIRE_THROWS(node.get_float("axis"));
}

TEST_CASE("Graph topological sort is stable and follows tensor dependencies") {
  InferenceGraph graph;

  OpNode relu;
  relu.name = "relu";
  relu.op_type = "Relu";
  relu.inputs = {"sum"};
  relu.outputs = {"output"};

  OpNode add;
  add.name = "add";
  add.op_type = "Add";
  add.inputs = {"input", "bias"};
  add.outputs = {"sum"};

  OpNode independent;
  independent.name = "constant";
  independent.op_type = "Constant";
  independent.outputs = {"unused"};

  graph.nodes = {relu, add, independent};
  sort_topologically(graph);

  REQUIRE(graph.nodes[0].name == "add");
  REQUIRE(graph.nodes[1].name == "relu");
  REQUIRE(graph.nodes[2].name == "constant");
}

TEST_CASE("Graph topological sort rejects cycles and duplicate producers") {
  InferenceGraph cycle;
  OpNode first;
  first.inputs = {"b"};
  first.outputs = {"a"};
  OpNode second;
  second.inputs = {"a"};
  second.outputs = {"b"};
  cycle.nodes = {first, second};
  REQUIRE_THROWS(sort_topologically(cycle));

  InferenceGraph duplicate;
  first.inputs.clear();
  second.inputs.clear();
  first.outputs = {"same"};
  second.outputs = {"same"};
  duplicate.nodes = {first, second};
  REQUIRE_THROWS(sort_topologically(duplicate));
}

TEST_CASE("ONNX parser extracts metadata initializers attributes and sorts nodes") {
  onnx::ModelProto model;
  model.set_ir_version(8);
  model.set_producer_name("nn_inference_test");
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* source = model.mutable_graph();
  source->set_name("tiny_unsorted");
  add_value(source, "input");
  add_value(source, "bias");
  add_value(source, "output", true);

  auto* intermediate = source->add_value_info();
  intermediate->set_name("sum");
  auto* intermediate_type = intermediate->mutable_type()->mutable_tensor_type();
  intermediate_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
  intermediate_type->mutable_shape()->add_dim()->set_dim_value(1);
  intermediate_type->mutable_shape()->add_dim()->set_dim_value(2);

  const float bias_values[] = {0.25f, -0.5f};
  auto* bias = source->add_initializer();
  bias->set_name("bias");
  bias->set_data_type(onnx::TensorProto_DataType_FLOAT);
  bias->add_dims(2);
  bias->set_raw_data(
      std::string(reinterpret_cast<const char*>(bias_values), sizeof(bias_values)));

  auto* relu = source->add_node();
  relu->set_name("relu");
  relu->set_op_type("Relu");
  relu->add_input("sum");
  relu->add_output("output");
  auto* axis = relu->add_attribute();
  axis->set_name("axis");
  axis->set_type(onnx::AttributeProto_AttributeType_INT);
  axis->set_i(-1);

  auto* add = source->add_node();
  add->set_name("add");
  add->set_op_type("Add");
  add->add_input("input");
  add->add_input("bias");
  add->add_output("sum");

  const auto path = std::filesystem::temp_directory_path() /
                    "nn_inference_parser_test.onnx";
  {
    std::ofstream output(path, std::ios::binary);
    REQUIRE(model.SerializeToOstream(&output));
  }

  const InferenceGraph graph = load_onnx(path.string());
  std::filesystem::remove(path);

  REQUIRE(graph.model_name == "tiny_unsorted");
  REQUIRE(graph.opset_version == 13);
  REQUIRE(graph.input_names == std::vector<std::string>{"input"});
  REQUIRE(graph.output_names == std::vector<std::string>{"output"});
  REQUIRE(graph.nodes.size() == 2);
  REQUIRE(graph.nodes[0].name == "add");
  REQUIRE(graph.nodes[1].name == "relu");
  REQUIRE(graph.nodes[1].get_int("axis") == -1);
  REQUIRE(graph.value_info.at("input").shape == std::vector<int64_t>{1, -1});
  REQUIRE(graph.value_info.at("sum").shape == std::vector<int64_t>{1, 2});
  const Tensor& parsed_bias = graph.initializers.at("bias");
  REQUIRE(parsed_bias.shape() == std::vector<int64_t>{2});
  REQUIRE(parsed_bias.data_f32()[0] == 0.25f);
  REQUIRE(parsed_bias.data_f32()[1] == -0.5f);
}

TEST_CASE("ONNX parser reports missing and malformed files") {
  REQUIRE_THROWS(load_onnx("/definitely/missing/model.onnx"));

  const auto path = std::filesystem::temp_directory_path() /
                    "nn_inference_malformed.onnx";
  {
    std::ofstream output(path, std::ios::binary);
    output << "not an onnx model";
  }
  REQUIRE_THROWS(load_onnx(path.string()));
  std::filesystem::remove(path);
}
