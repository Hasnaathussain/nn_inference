#!/usr/bin/env python3
import sys

import onnx

model = onnx.load(sys.argv[1])
target = sys.argv[2]
for value in list(model.graph.value_info) + list(model.graph.output):
    if value.name == target:
        print(
            value.name,
            [
                (dimension.dim_value, dimension.dim_param)
                for dimension in value.type.tensor_type.shape.dim
            ],
        )
for node in model.graph.node:
    if target in node.input or target in node.output:
        print(node.op_type, list(node.input), list(node.output))