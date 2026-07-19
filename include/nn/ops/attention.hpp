#pragma once

// Attention is intentionally composed from primitive ONNX operators.
// This compatibility group header exposes MatMul, Softmax, reshape/transpose,
// and shape/index kernels declared by bert.hpp rather than a fused attention op.
#include "nn/ops/bert.hpp"