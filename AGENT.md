SYSTEM CONTEXT - Flagship Project - ONNX Inference Runtime in C++ with SIMD-optimised GEMM, Memory Planner, and Python validation layer
==============
You are an expert systems programming agent building a neural network inference
engine from scratch in C++17. You have shell access for compiling, running tests,
and inspecting outputs. The target machine is Linux Mint x86_64 with AVX2+FMA
support, GCC 12, Python 3.10+, CMake 3.20+, and 8GB RAM. Build with -j2 to
stay within memory limits. Always verify your changes compile before declaring
a task complete.

PROJECT SPECIFICATION
=====================
Build: nn_inference/ — a minimal but correct ONNX CPU inference engine in C++17.

GOALS (in priority order):
  1. Correctly load and execute ResNet-18 (ONNX opset 13) with < 0.5% mean
     absolute error vs ONNXRuntime on 100 ImageNet validation images.
  2. Correctly load and execute BERT-tiny (ONNX opset 13) for masked language
     modelling with < 1.0% logit MSE vs ONNXRuntime.
  3. SIMD-accelerated GEMM using AVX2 intrinsics that achieves > 25 GFLOPS/s
     sustained throughput on a 1024×1024×1024 matrix multiply.
  4. Memory planner that reduces peak activation memory by ≥ 20% vs naive
     allocation (verified via peak RSS difference).
  5. Python pybind11 bridge enabling model.run(inputs_dict) → outputs_dict.

NON-GOALS (do not implement):
  - GPU/CUDA support
  - Multi-threading (OpenMP can be added last if time permits)
  - Quantisation (INT8)
  - Training (inference only)
  - Any operator not required by ResNet-18 or BERT-tiny

REQUIRED ONNX OPERATORS
========================
ResNet-18 operators (implement first):
  Conv (attrs: dilations, group, kernel_shape, pads, strides)
  BatchNormalization (attrs: epsilon, momentum; inference mode only — fuse scale)
  Relu
  MaxPool (attrs: kernel_shape, pads, strides)
  GlobalAveragePool
  Flatten (attrs: axis)
  Gemm (attrs: alpha, beta, transA, transB)
  Add
  Reshape
  Shape
  Gather

BERT-tiny additional operators:
  MatMul
  LayerNorm (as "LayerNormalization" op or manual decomposition)
  Softmax (attrs: axis)
  Transpose (attrs: perm)
  Squeeze / Unsqueeze
  Concat (attrs: axis)
  Cast (attrs: to)
  Slice
  Expand
  Mul, Div, Sub, Pow, Sqrt, Erf
  Gelu (= x * 0.5 * (1 + erf(x/sqrt(2))))
  Where

ARCHITECTURE SPECIFICATION
===========================
Project layout:
  nn_inference/
  ├── CMakeLists.txt                 # Root CMake, FetchContent for deps
  ├── include/
  │   ├── nn/tensor.hpp              # Tensor class
  │   ├── nn/graph.hpp               # InferenceGraph, OpNode, TensorInfo
  │   ├── nn/onnx_parser.hpp         # ONNX → InferenceGraph
  │   ├── nn/execution_engine.hpp    # Engine class
  │   ├── nn/memory_planner.hpp      # MemoryPlan + planner
  │   └── nn/ops/                    # One header per operator group
  │       ├── gemm.hpp
  │       ├── conv2d.hpp
  │       ├── attention.hpp
  │       ├── norm.hpp
  │       ├── pool.hpp
  │       └── elementwise.hpp
  ├── src/                           # Corresponding .cpp files
  ├── python/
  │   ├── bindings.cpp               # pybind11 module "nn_inference"
  │   ├── model_export.py            # Export ResNet-18 / BERT to ONNX
  │   ├── preprocess.py              # ImageNet preprocessing
  │   ├── validate.py                # Compare to ONNXRuntime
  │   └── bench.py                   # Latency + throughput benchmark
  └── tests/
      ├── test_tensor.cpp
      ├── test_ops.cpp               # Per-operator numpy comparisons
      ├── test_graph.cpp
      └── test_memory_planner.cpp

TENSOR CLASS SPECIFICATION
===========================
File: include/nn/tensor.hpp + src/tensor.cpp

class Tensor {
public:
  enum class DataType { FLOAT32, INT64, INT32, BOOL, UINT8 };
  
  // Constructors
  Tensor();
  Tensor(std::vector<int64_t> shape, DataType dtype);
  Tensor(std::vector<int64_t> shape, DataType dtype, void* external_data); // no-copy view
  
  // Data access
  float*   data_f32();
  const float* data_f32() const;
  int64_t* data_i64();
  
  // Properties
  const std::vector<int64_t>& shape() const;
  const std::vector<int64_t>& strides() const;  // row-major strides in elements
  int64_t ndim() const;
  int64_t numel() const;
  int64_t nbytes() const;
  DataType dtype() const;
  
  // Operations (return new Tensor, do not modify in place)
  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor transpose(std::vector<int64_t> perm) const;  // returns a view
  Tensor slice(int axis, int64_t start, int64_t end, int64_t step = 1) const;
  
  // Fill helpers
  void fill_zeros();
  void fill_from_raw(const float* src, size_t n);
  void fill_from_raw(const int64_t* src, size_t n);
};

IMPLEMENTATION RULES:
- Allocate data_ with posix_memalign(64) for AVX-512 readiness
- Strides are in ELEMENTS, not bytes. stride[i] = product(shape[i+1:])
- reshape() must check numel() is conserved; throw std::invalid_argument otherwise
- transpose() creates a view with reordered strides; does NOT copy data
- All tensor operations must work on non-contiguous tensors (stride != contiguous)

GRAPH IR SPECIFICATION
=======================
File: include/nn/graph.hpp

struct AttributeValue {
  enum class Type { INT, FLOAT, STRING, INTS, FLOATS, TENSOR };
  Type type;
  int64_t i;
  float f;
  std::string s;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  Tensor tensor;
};
using AttributeMap = std::unordered_map<std::string, AttributeValue>;

struct TensorInfo {
  std::string name;
  Tensor::DataType dtype;
  std::vector<int64_t> shape;   // -1 for dynamic dims
};

struct OpNode {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;   // tensor names (can be empty string for optional)
  std::vector<std::string> outputs;
  AttributeMap attrs;
  
  // Attribute helpers
  int64_t get_int(const std::string& key, int64_t default_val = 0) const;
  float get_float(const std::string& key, float default_val = 0.0f) const;
  std::vector<int64_t> get_ints(const std::string& key, 
                                 std::vector<int64_t> default_val = {}) const;
};

struct InferenceGraph {
  std::vector<OpNode> nodes;                                 // topologically sorted
  std::unordered_map<std::string, TensorInfo> value_info;   // all tensor shapes
  std::unordered_map<std::string, Tensor> initializers;      // model weights
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::string model_name;
  int64_t opset_version;
};

ONNX PARSER SPECIFICATION
===========================
File: include/nn/onnx_parser.hpp + src/onnx_parser.cpp

InferenceGraph load_onnx(const std::string& path);

IMPLEMENTATION STEPS:
1. Use CMake FetchContent to pull onnx/onnx at tag v1.15.0 (has pregenerated
   protobuf files in onnx/onnx.proto3 — use those, do NOT run protoc yourself)
   Alternative: use the `onnx_proto` target from the onnx CMake build.
2. Deserialize with onnx::ModelProto model; model.ParseFromString(...)
3. Extract graph.node[] → OpNode (op_type, inputs, outputs, attributes)
4. Extract graph.initializer[] → Tensor (name, dims, float_data / raw_data)
   Handle raw_data: cast bytes to float32 array
5. Extract graph.value_info[] + graph.input[] + graph.output[] → TensorInfo
6. Perform topological sort using Kahn's algorithm:
   - Build in-degree map from tensor producer → consumer
   - Queue nodes with all inputs already produced (initializers count as produced)
   - Emit nodes in BFS order
7. Return populated InferenceGraph

MEMORY PLANNER SPECIFICATION
==============================
File: include/nn/memory_planner.hpp + src/memory_planner.cpp

struct TensorLifetime {
  std::string name;
  int first_use;   // index into sorted node list
  int last_use;    // last node index that uses this tensor
  size_t nbytes;
};

struct MemoryPlan {
  struct Slot {
    size_t offset;
    size_t size;
  };
  std::unordered_map<std::string, Slot> tensor_slots;
  size_t total_bytes;
};

MemoryPlan plan_memory(const InferenceGraph& graph);

ALGORITHM:
1. For each tensor (non-initializer):
   first_use = index of the node that produces it (output)
   last_use = max index of all nodes that consume it (input)
2. Sort tensors by nbytes descending (greedy: large first)
3. Maintain list of "free slots" (offset, size, free_at_node_index)
4. For each tensor in sorted order:
   - Find smallest free slot where: slot.size >= tensor.nbytes AND
     slot.free_at_node_index <= tensor.first_use
   - If found: assign tensor → slot (update slot.free_at_node_index = tensor.last_use)
   - Else: append new slot at current arena end
5. total_bytes = max offset + size across all slots

EXECUTION ENGINE SPECIFICATION
================================
File: include/nn/execution_engine.hpp + src/execution_engine.cpp

class ExecutionEngine {
public:
  explicit ExecutionEngine(InferenceGraph graph);
  
  // Run inference. inputs: {name → Tensor}. Returns {name → Tensor}
  std::unordered_map<std::string, Tensor> run(
      const std::unordered_map<std::string, Tensor>& inputs);
  
  // Enable per-op timing
  void set_profiling(bool enable);
  std::unordered_map<std::string, double> get_op_times_ms() const;
  
private:
  InferenceGraph graph_;
  MemoryPlan memory_plan_;
  std::vector<uint8_t> arena_;         // activation arena
  std::unordered_map<std::string, Tensor> tensor_map_;  // name → tensor view
  
  Tensor get_tensor(const std::string& name) const;
  void dispatch(const OpNode& node);
};

DISPATCH TABLE (implement as switch or unordered_map<string, function>):
  "Conv"                → kernel::conv2d(node, tensor_map_)
  "BatchNormalization"  → kernel::batch_norm(node, tensor_map_)
  "Relu"                → kernel::relu(node, tensor_map_)
  "MaxPool"             → kernel::max_pool(node, tensor_map_)
  "GlobalAveragePool"   → kernel::global_avg_pool(node, tensor_map_)
  "Flatten"             → kernel::flatten(node, tensor_map_)
  "Gemm"                → kernel::gemm_op(node, tensor_map_)
  "Add"                 → kernel::elementwise_add(node, tensor_map_)
  "Mul"                 → kernel::elementwise_mul(node, tensor_map_)
  "MatMul"              → kernel::matmul(node, tensor_map_)
  "Reshape"             → kernel::reshape_op(node, tensor_map_)
  "Transpose"           → kernel::transpose_op(node, tensor_map_)
  "Softmax"             → kernel::softmax(node, tensor_map_)
  "LayerNormalization"  → kernel::layer_norm(node, tensor_map_)
  "Gather"              → kernel::gather(node, tensor_map_)
  "Concat"              → kernel::concat(node, tensor_map_)
  "Cast"                → kernel::cast_op(node, tensor_map_)
  "Slice"               → kernel::slice_op(node, tensor_map_)
  ... (add as needed)
  default: throw std::runtime_error("Unsupported op: " + node.op_type)

OPERATOR IMPLEMENTATION RULES
================================
1. Every operator MUST be validated against numpy BEFORE integration.
   Write the numpy reference in tests/test_ops.cpp (compare to onnxruntime output).
   Run the test BEFORE submitting the implementation.

2. Operator signature convention:
   void kernel::relu(const OpNode& node,
                     std::unordered_map<std::string, Tensor>& tensors);
   Input tensors: read via tensors.at(node.inputs[i])
   Output tensor: write to tensors[node.outputs[0]] (create if not exists)

3. For all element-wise ops: handle broadcasting via numpy-style rules.
   Implement a helper: Tensor broadcast_binary_op(const Tensor& a, const Tensor& b,
                                                    std::function<float(float,float)> fn)
   This helper must handle: scalar-tensor, (1,C,1,1)-tensor, and general cases.

GEMM KERNEL SPECIFICATION
===========================
File: include/nn/ops/gemm.hpp + src/ops/gemm_avx2.cpp

// Scalar fallback (implement first, test for correctness):
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 float alpha, float beta,
                 bool transA, bool transB);

// AVX2 optimised (implement after scalar passes tests):
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K);
// Note: gemm_avx2 assumes alpha=1, beta=0, no transpose, row-major, 64-byte aligned

// Dispatch: use gemm_avx2 when alpha==1, beta==0, !transA, !transB, M*N*K > 1M
//           else use gemm_scalar

AVX2 MICRO-KERNEL IMPLEMENTATION GUIDE:
- Include <immintrin.h>
- Add -mavx2 -mfma to compiler flags
- Use __attribute__((target("avx2,fma"))) on the kernel function
- Tile sizes: Mr=6, Nr=16 (= 2 AVX2 registers × 8 floats)
- Allocate packed_A and packed_B on stack (small) or via arena
- Inner loop: _mm256_fmadd_ps for fused multiply-add
- Handle M % Mr and N % Nr remainders with scalar cleanup loop
- Verify: compare output to gemm_scalar on random [100,200]×[200,300] matrices
          Max abs diff must be < 1e-4

CONV2D SPECIFICATION (im2col approach)
========================================
File: src/ops/conv2d.cpp

void conv2d(const OpNode& node, std::unordered_map<std::string, Tensor>& tensors) {
  // Extract inputs: X [N,C,H,W], W [OC,IC,KH,KW], B [OC] (optional)
  // Extract attrs: dilations, group, kernel_shape, pads [top,left,bot,right], strides
  
  // Step 1: im2col
  //   output_h = (H + pad_top + pad_bot - dil_h*(KH-1) - 1) / stride_h + 1
  //   col shape: [N*OH*OW, IC*KH*KW]
  //   For each (n, oh, ow): extract patch starting at
  //     (oh*stride_h - pad_top, ow*stride_w - pad_left) with dilation
  //   Pad with 0.0f for out-of-bounds positions
  
  // Step 2: GEMM
  //   W_mat shape: [OC, IC*KH*KW]
  //   output_mat = W_mat @ col.T  shape: [OC, N*OH*OW]
  
  // Step 3: Add bias (broadcast OC across N*OH*OW)
  
  // Step 4: Reshape → [N, OC, OH, OW]
}

NORMALIZATION SPEC
===================
LayerNorm (Welford's stable algorithm):
  - Input: x [*, D] (normalise over last axis)
  - Two-pass: first compute mean, then variance
  - For numerical stability, use Welford's online formula OR
    variance = E[x²] - E[x]² (check for negative before sqrt)
  - Apply scale (gamma) and shift (beta): output = gamma * (x - mean) / (std + eps) + beta
  - eps default = 1e-5

BatchNorm (inference mode):
  - x_norm = (x - running_mean) / sqrt(running_var + eps)
  - output = gamma * x_norm + beta
  - Input shape: [N, C, H, W] — normalise per channel

ATTENTION SPEC
===============
Implement as a sequence of primitive ops (MatMul, Div, Add, Softmax, Reshape, Transpose).
Do NOT implement as a single fused kernel. BERT-tiny uses standard ONNX ops:
  1. Linear projections (MatMul + Add)
  2. Reshape to [B, H, S, D/H]
  3. Transpose K for scores: [B, H, D/H, S]
  4. Scores = Q @ K_T / sqrt(D/H)  [B, H, S, S]
  5. Masked Softmax
  6. Context = Softmax_scores @ V  [B, H, S, D/H]
  7. Reshape + linear projection

This means if your MatMul, Reshape, Transpose, Softmax ops are correct,
Attention comes for free. Verify this rather than implementing a fused op.

PYTHON BINDINGS SPECIFICATION
==============================
File: python/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Wrap: Tensor ↔ numpy array (float32, int64)
// Wrap: ExecutionEngine

PYBIND11_MODULE(nn_inference, m) {
  m.doc() = "Neural network inference engine";
  
  py::class_<ExecutionEngine>(m, "ExecutionEngine")
    .def(py::init<const std::string&>())   // path to .onnx file
    .def("run", [](ExecutionEngine& e, py::dict inputs) {
      // Convert numpy arrays → Tensor
      // Call e.run(...)
      // Convert outputs Tensor → numpy array
      // Return py::dict
    })
    .def("set_profiling", &ExecutionEngine::set_profiling)
    .def("get_op_times_ms", &ExecutionEngine::get_op_times_ms);
}

VALIDATION SPECIFICATION
=========================
File: python/validate.py

import onnxruntime as ort
import numpy as np
import nn_inference   # your pybind11 module

def validate_resnet18(model_path, images_dir, n=100, tol=0.005):
    ort_session = ort.InferenceSession(model_path)
    engine = nn_inference.ExecutionEngine(model_path)
    
    errors = []
    for img in load_images(images_dir, n):
        x = preprocess(img)   # [1, 3, 224, 224] float32
        
        ort_out = ort_session.run(None, {"input": x})[0]
        our_out = engine.run({"input": x})["output"]
        
        err = np.mean(np.abs(ort_out - our_out))
        errors.append(err)
    
    mean_err = np.mean(errors)
    print(f"Mean abs error vs ONNXRuntime: {mean_err:.6f}")
    assert mean_err < tol, f"Accuracy check failed: {mean_err} > {tol}"
    return mean_err

BENCHMARK SPECIFICATION
========================
File: python/bench.py

def benchmark(engine, input_dict, n_warmup=10, n_runs=100):
    for _ in range(n_warmup):
        engine.run(input_dict)
    
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        engine.run(input_dict)
        times.append((time.perf_counter() - t0) * 1000)
    
    return {
        "mean_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "throughput_ips": 1000 / np.mean(times)
    }

Compare against:
  ort_session = ort.InferenceSession(model_path,
                    providers=["CPUExecutionProvider"],
                    sess_options=ort.SessionOptions()  # single thread)

BUILD ORDER (IMPLEMENT IN THIS EXACT ORDER)
============================================

  1. CMakeLists.txt with FetchContent (protobuf, onnx, pybind11, Catch2)
  2. Verify build: cmake + ninja, empty main.cpp
  3. Tensor class (shape, strides, data) + test_tensor.cpp
  4. gemm_scalar() + test that C = A @ B is correct vs numpy
  

  5. ONNX parser: load model proto, extract nodes + initializers
  6. Graph IR builder + topological sort (test on small manually-built graph)
  7. Execution engine skeleton (dispatch table, tensor_map, no memory planner yet)
  8. Implement: Add, Mul, ReLU, Reshape, Flatten, Transpose
  9. Test each op vs onnxruntime output on random inputs


  10. Conv2D (im2col + gemm_scalar) — test vs onnxruntime Conv on [1,3,7,7] input
  11. BatchNorm (inference mode)
  12. MaxPool, GlobalAveragePool
  13. Gemm operator wrapper
  14. Run ResNet-18 end-to-end (expect crashes → fix operator by operator)


  15. Python pybind11 bindings
  16. validate.py vs ONNXRuntime (goal: <0.5% error)
  17. Memory planner + arena allocator
  18. Integrate memory planner into execution engine
  19. Verify ResNet-18 still passes validation after memory planner


  20. AVX2 GEMM micro-kernel (scalar fallback already tested)
  21. Verify GEMM correctness vs scalar (max diff < 1e-4)
  22. Re-run ResNet-18 validation (should still pass)
  23. bench.py — compare ResNet-18 latency vs ONNXRuntime


  24. MatMul, Softmax, LayerNorm, Gather, Concat, Slice, Cast, Expand, Mul(broadcast)
  25. Run BERT-tiny end-to-end
  26. BERT-tiny validation vs ONNXRuntime
  27. bench.py BERT-tiny benchmark
  28. README + benchmark table + architecture diagram

TESTING PROTOCOL (NON-NEGOTIABLE)
===================================
For EVERY operator, before considering it done:
  1. Generate random input tensors in Python (numpy)
  2. Export to ONNX as a single-node model (or use onnxruntime directly)
  3. Run through your engine
  4. Assert np.allclose(your_output, ort_output, atol=1e-4, rtol=1e-4)
  5. Test at least 3 input shapes (including edge cases: batch=1, C=1, H=1)
  6. Test with and without optional inputs (e.g. Conv bias=None)

CONSTRAINTS AND GUARDRAILS
============================
- DO NOT add multi-threading until ResNet-18 validates correctly
- DO NOT implement AVX2 kernel until scalar GEMM validates correctly  
- DO NOT run BERT-tiny until ResNet-18 passes validation
- DO NOT skip the per-operator test for any operator
- Pin ONNX opset to 13 for all exported models
- Use float32 only (no mixed precision until everything else works)
- All operator implementations must handle batch_size=1 AND batch_size>1
- Use -fsanitize=address,undefined during development; disable for benchmarks

ERROR HANDLING
===============
- Throw std::runtime_error with descriptive messages for:
  - Unsupported op_type
  - Shape mismatch in binary ops
  - Missing required input
  - Unsupported attribute value
- Never silently produce wrong outputs; it's better to crash with a clear message

SUCCESS CRITERIA (Definition of Done)
=======================================
✓ ResNet-18 ImageNet Top-1 accuracy matches ONNXRuntime within 0.5% on 100 images
✓ BERT-tiny logit MSE vs ONNXRuntime < 1e-3 on 50 test sentences
✓ GEMM kernel achieves > 25 GFLOPS/s on [1024,1024] × [1024,1024] float32
✓ Memory planner reduces peak RSS by ≥ 20% vs baseline (verified with /usr/bin/time)
✓ Python API: engine = nn_inference.ExecutionEngine("resnet18.onnx"); engine.run({...})
✓ README contains: architecture diagram, benchmark table, build instructions, sample output
✓ All operator unit tests pass (Catch2 test suite with 0 failures)