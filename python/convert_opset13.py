#!/usr/bin/env python3
import sys

import onnx
from onnx import shape_inference, version_converter

source, destination = sys.argv[1:3]
model = onnx.load(source)
converted = version_converter.convert_version(model, 13)
converted = shape_inference.infer_shapes(converted)
onnx.checker.check_model(converted)
onnx.save(converted, destination)
print(f"converted {source} to opset 13 at {destination}")