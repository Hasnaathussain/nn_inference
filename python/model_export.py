#!/usr/bin/env python3
"""Prepare opset-13 ONNX models used by nn_inference.

This utility intentionally avoids requiring Torch. It converts an existing ONNX
model to opset 13 and can attach the tied-embedding BERT masked-token logit head
used by the project validation suite.
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference, version_converter


def convert_opset13(model):
    converted = version_converter.convert_version(model, 13)
    converted = shape_inference.infer_shapes(converted)
    onnx.checker.check_model(converted)
    return converted


def add_tied_bert_mlm_head(model):
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
    hidden_output = model.graph.output[0].name
    model.graph.node.append(
        helper.make_node(
            "MatMul",
            [hidden_output, decoder.name],
            ["logits"],
            name="tied_embedding_mlm_projection",
        )
    )
    input_shape = model.graph.input[0].type.tensor_type.shape
    batch = input_shape.dim[0].dim_param or input_shape.dim[0].dim_value
    sequence = input_shape.dim[1].dim_param or input_shape.dim[1].dim_value
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT,
            [batch, sequence, embedding_array.shape[0]],
        )
    )
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--bert-mlm-head",
        action="store_true",
        help="attach the tied word-embedding vocabulary projection",
    )
    args = parser.parse_args()

    model = convert_opset13(onnx.load(args.source))
    if args.bert_mlm_head:
        model = add_tied_bert_mlm_head(model)
    onnx.save(model, args.destination)
    print(f"wrote {args.destination}")


if __name__ == "__main__":
    main()