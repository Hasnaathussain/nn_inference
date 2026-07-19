#!/usr/bin/env python3
import collections
import sys

import onnx

model = onnx.load(sys.argv[1], load_external_data=False)
print("opsets", [(item.domain, item.version) for item in model.opset_import])
print(
    "inputs",
    [
        (
            value.name,
            [
                dimension.dim_value or dimension.dim_param
                for dimension in value.type.tensor_type.shape.dim
            ],
            value.type.tensor_type.elem_type,
        )
        for value in model.graph.input
    ],
)
print("outputs", [value.name for value in model.graph.output])
print("ops", collections.Counter(node.op_type for node in model.graph.node))