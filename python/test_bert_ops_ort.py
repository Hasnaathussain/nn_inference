#!/usr/bin/env python3
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

import nn_inference


DTYPES = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.bool_): TensorProto.BOOL,
    np.dtype(np.uint8): TensorProto.UINT8,
}


def run_case(
    directory,
    name,
    op_type,
    inputs,
    output_dtype,
    *,
    attrs=None,
    opset=13,
    output_name="y",
    node_inputs=None,
):
    attrs = attrs or {}
    node_inputs = node_inputs or list(inputs)
    input_infos = [
        helper.make_tensor_value_info(key, DTYPES[value.dtype], list(value.shape))
        for key, value in inputs.items()
    ]
    output_info = helper.make_tensor_value_info(output_name, output_dtype, None)
    node = helper.make_node(
        op_type, node_inputs, [output_name], name=name, **attrs
    )
    graph = helper.make_graph([node], name, input_infos, [output_info])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=9,
    )
    model = shape_inference.infer_shapes(model)

    path = directory / f"{name}.onnx"
    onnx.save(model, path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    expected = session.run(None, inputs)[0]
    engine = nn_inference.ExecutionEngine(str(path))
    engine.set_memory_planner(False)
    actual = engine.run(inputs)[output_name]
    if expected.dtype.kind in "f":
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        np.testing.assert_array_equal(actual, expected)


def constant_case(directory, name, array):
    tensor = numpy_helper.from_array(array, name="constant_value")
    output_info = helper.make_tensor_value_info(
        "y", DTYPES[array.dtype], list(array.shape)
    )
    node = helper.make_node("Constant", [], ["y"], name=name, value=tensor)
    model = helper.make_model(
        helper.make_graph([node], name, [], [output_info]),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    path = directory / f"{name}.onnx"
    onnx.save(model, path)
    expected = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {})[0]
    actual = nn_inference.ExecutionEngine(str(path)).run({})["y"]
    np.testing.assert_array_equal(actual, expected)


def main():
    rng = np.random.default_rng(20260715)
    cases = 0
    with tempfile.TemporaryDirectory(prefix="nn_bert_ops_") as temporary:
        directory = Path(temporary)

        matmul_shapes = [
            ((2, 3), (3, 4)),
            ((1, 2, 3), (3, 1)),
            ((2, 1, 3, 4), (1, 5, 4, 2)),
        ]
        for index, (a_shape, b_shape) in enumerate(matmul_shapes):
            run_case(
                directory,
                f"matmul_{index}",
                "MatMul",
                {
                    "a": rng.normal(size=a_shape).astype(np.float32),
                    "b": rng.normal(size=b_shape).astype(np.float32),
                },
                TensorProto.FLOAT,
            )
            cases += 1

        for index, (shape, axis) in enumerate(
            [((1, 1), -1), ((2, 3), 0), ((2, 3, 4), 1)]
        ):
            run_case(
                directory,
                f"softmax_{index}",
                "Softmax",
                {"x": rng.normal(size=shape).astype(np.float32)},
                TensorProto.FLOAT,
                attrs={"axis": axis},
            )
            cases += 1

        for index, (shape, axes, keepdims) in enumerate(
            [((1,), [0], 1), ((2, 3), [1], 0), ((2, 3, 4), [0, 2], 1)]
        ):
            run_case(
                directory,
                f"reduce_mean_{index}",
                "ReduceMean",
                {"x": rng.normal(size=shape).astype(np.float32)},
                TensorProto.FLOAT,
                attrs={"axes": axes, "keepdims": keepdims},
            )
            cases += 1

        for index, shape in enumerate([(1, 4), (2, 5), (2, 3, 8)]):
            width = shape[-1]
            run_case(
                directory,
                f"layer_norm_{index}",
                "LayerNormalization",
                {
                    "x": rng.normal(size=shape).astype(np.float32),
                    "scale": rng.normal(size=(width,)).astype(np.float32),
                    "bias": rng.normal(size=(width,)).astype(np.float32),
                },
                TensorProto.FLOAT,
                attrs={"axis": -1, "epsilon": 1.0e-5},
                opset=17,
            )
            cases += 1

        gather_cases = [
            (
                np.arange(6, dtype=np.float32).reshape(3, 2),
                np.array(1, dtype=np.int64),
                0,
            ),
            (
                np.arange(12, dtype=np.float32).reshape(3, 4),
                np.array([3, 0], dtype=np.int64),
                1,
            ),
            (
                np.arange(24, dtype=np.int64).reshape(2, 3, 4),
                np.array([[0, -1], [1, 1]], dtype=np.int64),
                -2,
            ),
        ]
        for index, (data, indices, axis) in enumerate(gather_cases):
            run_case(
                directory,
                f"gather_{index}",
                "Gather",
                {"data": data, "indices": indices},
                DTYPES[data.dtype],
                attrs={"axis": axis},
            )
            cases += 1

        concat_cases = [
            (
                np.ones((1,), np.int64),
                np.zeros((2,), np.int64),
                0,
            ),
            (
                np.ones((1, 2), np.float32),
                np.zeros((2, 2), np.float32),
                0,
            ),
            (
                np.ones((2, 1, 3), np.float32),
                np.zeros((2, 2, 3), np.float32),
                1,
            ),
        ]
        for index, (a, b, axis) in enumerate(concat_cases):
            run_case(
                directory,
                f"concat_{index}",
                "Concat",
                {"a": a, "b": b},
                DTYPES[a.dtype],
                attrs={"axis": axis},
            )
            cases += 1

        slice_cases = [
            ((5,), [1], [5], [0], [2]),
            ((2, 5), [0], [4], [1], [1]),
            ((2, 4, 5), [1, 0], [4, 5], [1, 2], [2, 2]),
        ]
        for index, (shape, starts, ends, axes, steps) in enumerate(slice_cases):
            run_case(
                directory,
                f"slice_{index}",
                "Slice",
                {
                    "data": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                    "starts": np.array(starts, dtype=np.int64),
                    "ends": np.array(ends, dtype=np.int64),
                    "axes": np.array(axes, dtype=np.int64),
                    "steps": np.array(steps, dtype=np.int64),
                },
                TensorProto.FLOAT,
            )
            cases += 1

        cast_cases = [
            (np.array([-2, 0, 3], np.int64), TensorProto.FLOAT),
            (np.array([[-1.8, 2.2]], np.float32), TensorProto.INT64),
            (np.array([[True, False]], np.bool_), TensorProto.FLOAT),
        ]
        for index, (value, target) in enumerate(cast_cases):
            run_case(
                directory,
                f"cast_{index}",
                "Cast",
                {"x": value},
                target,
                attrs={"to": target},
            )
            cases += 1

        expand_cases = [
            (np.array(2.0, np.float32), (2, 3)),
            (np.arange(3, dtype=np.float32).reshape(1, 3), (2, 3)),
            (np.arange(2, dtype=np.int64).reshape(2, 1, 1), (2, 3, 4)),
        ]
        for index, (value, target_shape) in enumerate(expand_cases):
            run_case(
                directory,
                f"expand_{index}",
                "Expand",
                {
                    "x": value,
                    "shape": np.array(target_shape, dtype=np.int64),
                },
                DTYPES[value.dtype],
            )
            cases += 1

        for index, (shape, axes) in enumerate(
            [((3,), [0]), ((2, 3), [1]), ((2, 3, 4), [0, 3])]
        ):
            run_case(
                directory,
                f"unsqueeze_{index}",
                "Unsqueeze",
                {
                    "x": np.arange(np.prod(shape), dtype=np.int64).reshape(shape),
                    "axes": np.array(axes, dtype=np.int64),
                },
                TensorProto.INT64,
            )
            cases += 1

        for index, (shape, axes) in enumerate(
            [((1,), [0]), ((2, 1, 3), [1]), ((1, 2, 1, 3), [0, 2])]
        ):
            run_case(
                directory,
                f"squeeze_{index}",
                "Squeeze",
                {
                    "x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                    "axes": np.array(axes, dtype=np.int64),
                },
                TensorProto.FLOAT,
            )
            cases += 1

        for index, shape in enumerate([(1,), (2, 3), (2, 1, 4)]):
            run_case(
                directory,
                f"shape_{index}",
                "Shape",
                {"x": np.zeros(shape, np.float32)},
                TensorProto.INT64,
            )
            cases += 1

        binary_ops = {
            "Mul": lambda shape: (
                rng.normal(size=shape).astype(np.float32),
                rng.normal(size=(shape[-1],)).astype(np.float32),
            ),
            "Sub": lambda shape: (
                rng.normal(size=shape).astype(np.float32),
                rng.normal(size=(shape[-1],)).astype(np.float32),
            ),
            "Div": lambda shape: (
                rng.normal(size=shape).astype(np.float32),
                (np.abs(rng.normal(size=(shape[-1],))) + 0.5).astype(np.float32),
            ),
            "Pow": lambda shape: (
                (np.abs(rng.normal(size=shape)) + 0.1).astype(np.float32),
                np.array(2.0, np.float32),
            ),
        }
        for op_type, values in binary_ops.items():
            for index, shape in enumerate([(1,), (2, 3), (2, 1, 4)]):
                a, b = values(shape)
                run_case(
                    directory,
                    f"{op_type.lower()}_{index}",
                    op_type,
                    {"a": a, "b": b},
                    TensorProto.FLOAT,
                )
                cases += 1

        for op_type in ("Sqrt", "Erf"):
            for index, shape in enumerate([(1,), (2, 3), (2, 1, 4)]):
                value = rng.normal(size=shape).astype(np.float32)
                if op_type == "Sqrt":
                    value = np.abs(value)
                run_case(
                    directory,
                    f"{op_type.lower()}_{index}",
                    op_type,
                    {"x": value},
                    TensorProto.FLOAT,
                )
                cases += 1

        where_cases = [
            (
                np.array(True),
                np.array(1.0, np.float32),
                np.array(2.0, np.float32),
            ),
            (
                np.array([[True], [False]]),
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.array([10, 20, 30], np.float32),
            ),
            (
                np.array([[[True], [False]]]),
                np.ones((2, 2, 3), np.int64),
                np.zeros((1, 2, 1), np.int64),
            ),
        ]
        for index, (condition, x, y) in enumerate(where_cases):
            run_case(
                directory,
                f"where_{index}",
                "Where",
                {"condition": condition, "x": x, "y": y},
                DTYPES[x.dtype],
                output_name="out",
            )
            cases += 1

        for index, value in enumerate(
            [
                np.array(3.5, np.float32),
                np.array([1, 2, 3], np.int64),
                np.arange(6, dtype=np.float32).reshape(2, 3),
            ]
        ):
            constant_case(directory, f"constant_{index}", value)
            cases += 1

    print(f"BERT operator ONNX Runtime validation passed for {cases} cases")


if __name__ == "__main__":
    main()