#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


RNG = np.random.default_rng(20260715)


def make_model(op_type, x_shape, attributes=None, initializers=None):
    attributes = attributes or {}
    initializers = initializers or []
    node_inputs = ["x"] + [initializer.name for initializer in initializers]
    node = helper.make_node(
        op_type, node_inputs, ["y"], name=f"{op_type.lower()}_under_test",
        **attributes
    )
    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_single_node",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def run_case(runner, directory, label, model, x):
    model_path = directory / f"{label}.onnx"
    input_path = directory / f"{label}.input.bin"
    output_path = directory / f"{label}.output.bin"
    onnx.save(model, model_path)
    np.ascontiguousarray(x, dtype=np.float32).tofile(input_path)

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    expected = session.run(None, {"x": x.astype(np.float32, copy=False)})[0]

    shape_text = ",".join(str(dim) for dim in x.shape)
    environment = dict(os.environ)
    environment.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    subprocess.run(
        [
            runner, str(model_path), "x", shape_text, str(input_path),
            "y", str(output_path),
        ],
        check=True,
        env=environment,
    )
    actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def binary_cases(runner, directory, op_type):
    cases = [
        ((1,), np.array(0.25, dtype=np.float32)),
        ((1, 1, 1, 1), np.array([[[[-1.5]]]], dtype=np.float32)),
        ((2, 3, 4, 5), RNG.normal(size=(1, 3, 1, 1)).astype(np.float32)),
    ]
    for index, (shape, rhs) in enumerate(cases):
        x = RNG.normal(size=shape).astype(np.float32)
        initializer = numpy_helper.from_array(rhs, name="rhs")
        model = make_model(op_type, shape, initializers=[initializer])
        run_case(runner, directory, f"{op_type.lower()}_{index}", model, x)


def relu_cases(runner, directory):
    for index, shape in enumerate([(1,), (1, 1, 1, 1), (2, 3, 4)]):
        x = RNG.normal(size=shape).astype(np.float32)
        model = make_model("Relu", shape)
        run_case(runner, directory, f"relu_{index}", model, x)


def reshape_cases(runner, directory):
    cases = [
        ((1,), np.array([1, 1], dtype=np.int64)),
        ((1, 1, 1, 1), np.array([1], dtype=np.int64)),
        ((2, 3, 4), np.array([0, -1], dtype=np.int64)),
    ]
    for index, (shape, target) in enumerate(cases):
        x = RNG.normal(size=shape).astype(np.float32)
        initializer = numpy_helper.from_array(target, name="shape")
        model = make_model("Reshape", shape, initializers=[initializer])
        run_case(runner, directory, f"reshape_{index}", model, x)


def flatten_cases(runner, directory):
    cases = [
        ((1,), 0),
        ((1, 1, 1, 1), 2),
        ((2, 3, 4), -1),
    ]
    for index, (shape, axis) in enumerate(cases):
        x = RNG.normal(size=shape).astype(np.float32)
        model = make_model("Flatten", shape, attributes={"axis": axis})
        run_case(runner, directory, f"flatten_{index}", model, x)


def transpose_cases(runner, directory):
    cases = [
        ((1,), {"perm": [0]}),
        ((1, 1, 1, 1), {}),
        ((2, 3, 4), {"perm": [2, 0, 1]}),
    ]
    for index, (shape, attributes) in enumerate(cases):
        x = RNG.normal(size=shape).astype(np.float32)
        model = make_model("Transpose", shape, attributes=attributes)
        run_case(runner, directory, f"transpose_{index}", model, x)


def conv_cases(runner, directory):
    configurations = [
        ((1, 1, 1, 1), (1, 1, 1, 1), {}),
        (
            (1, 3, 7, 7),
            (4, 3, 3, 3),
            {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
             "strides": [2, 2]},
        ),
        (
            (2, 4, 6, 5),
            (6, 2, 2, 3),
            {"kernel_shape": [2, 3], "group": 2,
             "pads": [1, 0, 0, 1], "strides": [2, 1],
             "dilations": [1, 2]},
        ),
    ]
    for index, (input_shape, weight_shape, attributes) in enumerate(configurations):
        x = RNG.normal(size=input_shape).astype(np.float32)
        weights = RNG.normal(size=weight_shape).astype(np.float32)
        for with_bias in (False, True):
            initializers = [numpy_helper.from_array(weights, name="weights")]
            if with_bias:
                bias = RNG.normal(size=(weight_shape[0],)).astype(np.float32)
                initializers.append(numpy_helper.from_array(bias, name="bias"))
            model = make_model(
                "Conv", input_shape, attributes=attributes,
                initializers=initializers
            )
            suffix = "bias" if with_bias else "no_bias"
            run_case(
                runner, directory, f"conv_{index}_{suffix}", model, x
            )


def batch_norm_cases(runner, directory):
    for index, shape in enumerate(
        [(1, 1, 1, 1), (1, 3, 7, 7), (2, 4, 3, 5)]
    ):
        x = RNG.normal(size=shape).astype(np.float32)
        channels = shape[1]
        scale = RNG.normal(size=(channels,)).astype(np.float32)
        bias = RNG.normal(size=(channels,)).astype(np.float32)
        mean = RNG.normal(size=(channels,)).astype(np.float32)
        variance = (
            np.abs(RNG.normal(size=(channels,))).astype(np.float32) + 0.5
        )
        initializers = [
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(bias, name="bias"),
            numpy_helper.from_array(mean, name="mean"),
            numpy_helper.from_array(variance, name="variance"),
        ]
        model = make_model(
            "BatchNormalization",
            shape,
            attributes={"epsilon": 1.3e-4, "momentum": 0.9},
            initializers=initializers,
        )
        run_case(runner, directory, f"batch_norm_{index}", model, x)


def pooling_cases(runner, directory):
    max_pool_configurations = [
        ((1, 1, 1, 1), {"kernel_shape": [1, 1]}),
        (
            (1, 3, 7, 7),
            {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
             "strides": [2, 2]},
        ),
        (
            (2, 2, 5, 6),
            {"kernel_shape": [2, 3], "pads": [1, 0, 0, 1],
             "strides": [2, 1], "dilations": [1, 2],
             "ceil_mode": 1},
        ),
    ]
    for index, (shape, attributes) in enumerate(max_pool_configurations):
        x = RNG.normal(size=shape).astype(np.float32)
        model = make_model("MaxPool", shape, attributes=attributes)
        run_case(runner, directory, f"max_pool_{index}", model, x)

    for index, shape in enumerate(
        [(1, 1, 1, 1), (1, 3, 7, 7), (2, 4, 3, 5)]
    ):
        x = RNG.normal(size=shape).astype(np.float32)
        model = make_model("GlobalAveragePool", shape)
        run_case(runner, directory, f"global_avg_pool_{index}", model, x)


def gemm_cases(runner, directory):
    configurations = [
        ((1, 1), (1, 1), (), {}),
        (
            (2, 3),
            (4, 3),
            (4,),
            {"transB": 1, "alpha": 0.75, "beta": -0.5},
        ),
        (
            (5, 3),
            (5, 2),
            (3, 1),
            {"transA": 1, "alpha": 1.25, "beta": 0.25},
        ),
    ]
    for index, (a_shape, b_shape, c_shape, attributes) in enumerate(
        configurations
    ):
        a = RNG.normal(size=a_shape).astype(np.float32)
        b = RNG.normal(size=b_shape).astype(np.float32)
        for with_c in (False, True):
            initializers = [numpy_helper.from_array(b, name="b")]
            if with_c:
                c = RNG.normal(size=c_shape).astype(np.float32)
                initializers.append(numpy_helper.from_array(c, name="c"))
            model = make_model(
                "Gemm", a_shape, attributes=attributes,
                initializers=initializers
            )
            suffix = "with_c" if with_c else "without_c"
            run_case(runner, directory, f"gemm_{index}_{suffix}", model, a)


def main():
    runner = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="nn_inference_ops_") as temporary:
        directory = pathlib.Path(temporary)
        binary_cases(runner, directory, "Add")
        binary_cases(runner, directory, "Mul")
        relu_cases(runner, directory)
        reshape_cases(runner, directory)
        flatten_cases(runner, directory)
        transpose_cases(runner, directory)
        conv_cases(runner, directory)
        batch_norm_cases(runner, directory)
        pooling_cases(runner, directory)
        gemm_cases(runner, directory)
    print("ONNXRuntime validation passed for 39 operator cases")


if __name__ == "__main__":
    main()
