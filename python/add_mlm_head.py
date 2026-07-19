#!/usr/bin/env python3
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

source, destination = sys.argv[1:3]
model = onnx.load(source)
embedding = next(
    initializer
    for initializer in model.graph.initializer
    if initializer.name.endswith("word_embeddings.weight")
)
embedding_array = numpy_helper.to_array(embedding).astype(np.float32, copy=False)
decoder = numpy_helper.from_array(
    np.ascontiguousarray(embedding_array.T), name="mlm_decoder_weight"
)
model.graph.initializer.append(decoder)
model.graph.node.append(
    helper.make_node(
        "MatMul",
        [model.graph.output[0].name, decoder.name],
        ["logits"],
        name="tied_embedding_mlm_projection",
    )
)
input_shape = model.graph.input[0].type.tensor_type.shape
batch = input_shape.dim[0].dim_param or input_shape.dim[0].dim_value
sequence = input_shape.dim[1].dim_param or input_shape.dim[1].dim_value
logits = helper.make_tensor_value_info(
    "logits", TensorProto.FLOAT, [batch, sequence, embedding_array.shape[0]]
)
del model.graph.output[:]
model.graph.output.append(logits)
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, destination)
print(
    f"wrote {destination}: hidden={embedding_array.shape[1]} "
    f"vocab={embedding_array.shape[0]} output=logits"
)