#!/usr/bin/env python3
import pathlib
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import nn_inference


def save_model(path, nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(
        nodes, path.stem, inputs, outputs, initializer=list(initializers)
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_float_and_strided_output(directory):
    model_path = directory / "add_transpose.onnx"
    bias = numpy_helper.from_array(np.array(1.25, dtype=np.float32), "bias")
    nodes = [
        helper.make_node("Add", ["x", "bias"], ["sum"], name="add"),
        helper.make_node(
            "Transpose", ["sum"], ["y"], name="transpose", perm=[1, 0]
        ),
    ]
    save_model(
        model_path,
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
        [bias],
    )

    engine = nn_inference.ExecutionEngine(str(model_path))
    engine.set_profiling(True)
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    non_contiguous = base[:, ::-1]
    assert not non_contiguous.flags.c_contiguous
    actual = engine.run({"x": non_contiguous})["y"]
    expected = (non_contiguous + np.float32(1.25)).T
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.flags.c_contiguous

    times = engine.get_op_times_ms()
    assert set(times) == {"add", "transpose"}
    assert all(value >= 0.0 for value in times.values())

    try:
        engine.run({"x": base.astype(np.float64)})
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("float64 input should be rejected")


def test_int64_input(directory):
    model_path = directory / "runtime_reshape.onnx"
    node = helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape")
    save_model(
        model_path,
        [node],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
    )

    engine = nn_inference.ExecutionEngine(str(model_path))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = engine.run(
        {"x": x, "shape": np.array([3, 2], dtype=np.int64)}
    )
    np.testing.assert_array_equal(result["y"], x.reshape(3, 2))


def main():
    with tempfile.TemporaryDirectory(prefix="nn_inference_bindings_") as temporary:
        directory = pathlib.Path(temporary)
        test_float_and_strided_output(directory)
        test_int64_input(directory)
    print("Python bindings validation passed")


if __name__ == "__main__":
    main()
