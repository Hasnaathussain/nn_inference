#!/usr/bin/env python3
import collections
import sys

import onnx

model = onnx.load(sys.argv[1])
types = {}
for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
    if value.type.HasField("tensor_type"):
        types[value.name] = value.type.tensor_type.elem_type
for value in model.graph.initializer:
    types[value.name] = value.data_type
for node in model.graph.node:
    signature = (
        tuple(types.get(name, 0) for name in node.input if name),
        tuple(types.get(name, 0) for name in node.output if name),
    )
    print(node.op_type, signature, list(node.input), list(node.output))