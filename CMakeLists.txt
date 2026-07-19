cmake_minimum_required(VERSION 3.20)

project(nn_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(NN_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" ON)

include(FetchContent)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOBUF_BINARIES ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v21.12
  SOURCE_SUBDIR cmake
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(protobuf)

set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG v1.15.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(onnx)

set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(nn_inference_smoke src/main.cpp)

add_library(nn_inference_core
  src/tensor.cpp
  src/graph.cpp
  src/onnx_parser.cpp
  src/memory_planner.cpp
  src/execution_engine.cpp
  src/ops/elementwise.cpp
  src/ops/bert.cpp
  src/ops/conv2d.cpp
  src/ops/norm.cpp
  src/ops/pool.cpp
  src/ops/gemm.cpp
  src/ops/gemm_scalar.cpp
  src/ops/gemm_avx.cpp
  src/ops/gemm_avx2.cpp
)
target_include_directories(nn_inference_core PUBLIC include)
target_link_libraries(nn_inference_core PUBLIC onnx_proto)
set_target_properties(nn_inference_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE nn_inference_core Catch2::Catch2WithMain)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE nn_inference_core Catch2::Catch2WithMain)
add_test(NAME gemm COMMAND test_gemm)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph PRIVATE nn_inference_core Catch2::Catch2WithMain)
add_test(NAME graph COMMAND test_graph)

add_executable(test_memory_planner tests/test_memory_planner.cpp)
target_link_libraries(test_memory_planner PRIVATE
  nn_inference_core Catch2::Catch2WithMain
)
add_test(NAME memory_planner COMMAND test_memory_planner)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE nn_inference_core Catch2::Catch2WithMain)
add_test(NAME ops COMMAND test_ops)

add_executable(gemm_dump tests/gemm_dump.cpp)
target_link_libraries(gemm_dump PRIVATE nn_inference_core)

add_executable(gemm_bench tests/gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE nn_inference_core)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME gemm_numpy
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gemm_numpy.py
          $<TARGET_FILE:gemm_dump>)

add_executable(model_runner tests/model_runner.cpp)
target_link_libraries(model_runner PRIVATE nn_inference_core)

add_executable(memory_bench tests/memory_bench.cpp)
target_link_libraries(memory_bench PRIVATE nn_inference_core)

pybind11_add_module(nn_inference python/bindings.cpp)
target_link_libraries(nn_inference PRIVATE nn_inference_core)
set_target_properties(nn_inference PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

set(NN_VALIDATION_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
if(EXISTS "${NN_VALIDATION_PYTHON}")
  add_test(NAME ops_onnxruntime
    COMMAND ${NN_VALIDATION_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_ops_ort.py
            $<TARGET_FILE:model_runner>)
  add_test(NAME python_bindings
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      "ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0"
      ${NN_VALIDATION_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bindings.py)
  add_test(NAME bert_ops_onnxruntime
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python"
      "ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0"
      ${NN_VALIDATION_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bert_ops_ort.py)
  add_test(NAME bert_model_onnxruntime
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python"
      "ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0"
      ${NN_VALIDATION_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bert_model.py
      ${CMAKE_CURRENT_SOURCE_DIR}/models/bert-tiny-opset13.onnx)
  add_test(NAME bert_mlm_validation
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python"
      "ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0"
      ${NN_VALIDATION_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_bert.py
      ${CMAKE_CURRENT_SOURCE_DIR}/models/bert-tiny-mlm-opset13.onnx
      ${CMAKE_CURRENT_SOURCE_DIR}/data/bert-vocab.txt --n 50)
endif()

if(NN_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target nn_inference_smoke nn_inference_core test_tensor test_gemm
                 test_graph test_memory_planner test_ops gemm_dump gemm_bench model_runner memory_bench
                 nn_inference)
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -fsanitize=address,undefined
    )
    target_link_options(${target} PRIVATE
      -fsanitize=address,undefined
    )
  endforeach()
endif()

enable_testing()
