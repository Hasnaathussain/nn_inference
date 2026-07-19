#include "nn/execution_engine.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<int64_t> tensor_shape(const py::buffer_info& info) {
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(info.ndim));
  for (const py::ssize_t dimension : info.shape) {
    if (dimension < 0) throw std::invalid_argument("NumPy shape is invalid");
    shape.push_back(static_cast<int64_t>(dimension));
  }
  return shape;
}

nn::Tensor tensor_from_numpy(const py::handle& value) {
  py::array array = py::array::ensure(value, py::array::c_style);
  if (!array) throw std::invalid_argument("Model inputs must be NumPy arrays");
  const py::buffer_info info = array.request();
  const auto shape = tensor_shape(info);

  if (array.dtype().is(py::dtype::of<float>())) {
    nn::Tensor tensor(shape, nn::Tensor::DataType::FLOAT32);
    tensor.fill_from_raw(static_cast<const float*>(info.ptr),
                         static_cast<size_t>(tensor.numel()));
    return tensor;
  }
  if (array.dtype().is(py::dtype::of<int64_t>())) {
    nn::Tensor tensor(shape, nn::Tensor::DataType::INT64);
    tensor.fill_from_raw(static_cast<const int64_t*>(info.ptr),
                         static_cast<size_t>(tensor.numel()));
    return tensor;
  }
  if (array.dtype().is(py::dtype::of<int32_t>())) {
    nn::Tensor tensor(shape, nn::Tensor::DataType::INT32);
    tensor.fill_from_bytes(info.ptr, static_cast<size_t>(tensor.nbytes()));
    return tensor;
  }
  if (array.dtype().is(py::dtype::of<bool>())) {
    nn::Tensor tensor(shape, nn::Tensor::DataType::BOOL);
    tensor.fill_from_bytes(info.ptr, static_cast<size_t>(tensor.nbytes()));
    return tensor;
  }
  if (array.dtype().is(py::dtype::of<uint8_t>())) {
    nn::Tensor tensor(shape, nn::Tensor::DataType::UINT8);
    tensor.fill_from_bytes(info.ptr, static_cast<size_t>(tensor.nbytes()));
    return tensor;
  }
  throw std::invalid_argument(
      "Model inputs must use float32, int64, int32, bool, or uint8 dtype");
}

int64_t logical_offset(const nn::Tensor& tensor, int64_t linear) {
  int64_t offset = 0;
  for (size_t axis = tensor.shape().size(); axis > 0; --axis) {
    const int64_t dimension = tensor.shape()[axis - 1];
    const int64_t coordinate = linear % dimension;
    linear /= dimension;
    offset += coordinate * tensor.strides()[axis - 1];
  }
  return offset;
}

std::vector<py::ssize_t> numpy_shape(const nn::Tensor& tensor) {
  std::vector<py::ssize_t> shape;
  shape.reserve(tensor.shape().size());
  for (const int64_t dimension : tensor.shape()) {
    shape.push_back(static_cast<py::ssize_t>(dimension));
  }
  return shape;
}

py::array tensor_to_numpy(const nn::Tensor& tensor) {
  const auto shape = numpy_shape(tensor);
  if (tensor.dtype() == nn::Tensor::DataType::FLOAT32) {
    py::array_t<float> result(shape);
    auto* destination = static_cast<float*>(result.request().ptr);
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      destination[linear] =
          tensor.data_f32()[logical_offset(tensor, linear)];
    }
    return result;
  }
  if (tensor.dtype() == nn::Tensor::DataType::INT64) {
    py::array_t<int64_t> result(shape);
    auto* destination = static_cast<int64_t*>(result.request().ptr);
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      destination[linear] =
          tensor.data_i64()[logical_offset(tensor, linear)];
    }
    return result;
  }
  if (tensor.dtype() == nn::Tensor::DataType::INT32) {
    py::array_t<int32_t> result(shape);
    auto* destination = static_cast<int32_t*>(result.request().ptr);
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      destination[linear] =
          tensor.data_i32()[logical_offset(tensor, linear)];
    }
    return result;
  }
  if (tensor.dtype() == nn::Tensor::DataType::BOOL) {
    py::array_t<bool> result(shape);
    auto* destination = static_cast<bool*>(result.request().ptr);
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      destination[linear] =
          tensor.data_u8()[logical_offset(tensor, linear)] != 0;
    }
    return result;
  }
  if (tensor.dtype() == nn::Tensor::DataType::UINT8) {
    py::array_t<uint8_t> result(shape);
    auto* destination = static_cast<uint8_t*>(result.request().ptr);
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      destination[linear] =
          tensor.data_u8()[logical_offset(tensor, linear)];
    }
    return result;
  }
  throw std::runtime_error("Python output conversion encountered unsupported dtype");
}

}  // namespace

PYBIND11_MODULE(nn_inference, module) {
  module.doc() = "Neural network inference engine";

  py::class_<nn::ExecutionEngine>(module, "ExecutionEngine")
      .def(py::init<const std::string&>(), py::arg("model_path"))
      .def(
          "run",
          [](nn::ExecutionEngine& engine, const py::dict& python_inputs) {
            std::unordered_map<std::string, nn::Tensor> inputs;
            for (const auto item : python_inputs) {
              const std::string name = py::cast<std::string>(item.first);
              inputs.emplace(name, tensor_from_numpy(item.second));
            }

            std::unordered_map<std::string, nn::Tensor> outputs;
            {
              py::gil_scoped_release release;
              outputs = engine.run(inputs);
            }

            py::dict result;
            for (const auto& [name, tensor] : outputs) {
              result[py::str(name)] = tensor_to_numpy(tensor);
            }
            return result;
          },
          py::arg("inputs"))
      .def("set_profiling", &nn::ExecutionEngine::set_profiling,
           py::arg("enable"))
      .def("get_op_times_ms", &nn::ExecutionEngine::get_op_times_ms)
      .def("set_memory_planner", &nn::ExecutionEngine::set_memory_planner,
           py::arg("enable"))
      .def("get_planned_activation_bytes",
           &nn::ExecutionEngine::get_planned_activation_bytes)
      .def("get_naive_activation_bytes",
           &nn::ExecutionEngine::get_naive_activation_bytes);
}
