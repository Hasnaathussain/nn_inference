#include "nn/execution_engine.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<int64_t> parse_shape(const std::string& text) {
  std::vector<int64_t> shape;
  if (text.empty()) return shape;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ',')) shape.push_back(std::stoll(part));
  return shape;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: model_runner model input_name shape input.bin "
                   "output_name output.bin\n";
      return 2;
    }

    const std::vector<int64_t> shape = parse_shape(argv[3]);
    nn::Tensor input(shape, nn::Tensor::DataType::FLOAT32);
    std::ifstream input_file(argv[4], std::ios::binary);
    if (!input_file) throw std::runtime_error("Unable to open input fixture");
    input_file.read(reinterpret_cast<char*>(input.data_f32()), input.nbytes());
    if (input_file.gcount() != input.nbytes()) {
      throw std::runtime_error("Input fixture byte count mismatch");
    }

    nn::ExecutionEngine engine(argv[1]);
    auto outputs = engine.run({{argv[2], input}});
    const auto found = outputs.find(argv[5]);
    if (found == outputs.end()) throw std::runtime_error("Requested output is absent");
    if (found->second.dtype() != nn::Tensor::DataType::FLOAT32) {
      throw std::runtime_error("Runner only supports FLOAT32 outputs");
    }

    nn::Tensor contiguous = found->second.reshape({found->second.numel()});
    std::ofstream output_file(argv[6], std::ios::binary);
    if (!output_file) throw std::runtime_error("Unable to open output fixture");
    output_file.write(reinterpret_cast<const char*>(contiguous.data_f32()),
                      contiguous.nbytes());
    if (!output_file) throw std::runtime_error("Unable to write output fixture");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
