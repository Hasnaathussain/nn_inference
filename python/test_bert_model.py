#!/usr/bin/env python3
import sys

import numpy as np
import onnxruntime as ort

import nn_inference


def main():
    model_path = sys.argv[1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    engine = nn_inference.ExecutionEngine(model_path)
    rng = np.random.default_rng(20260715)
    for batch, sequence in [(1, 1), (1, 4), (2, 8)]:
        inputs = {
            "input_ids": rng.integers(
                0, 30522, size=(batch, sequence), dtype=np.int64
            ),
            "attention_mask": np.ones((batch, sequence), dtype=np.int64),
            "token_type_ids": np.zeros((batch, sequence), dtype=np.int64),
        }
        expected = session.run(None, inputs)[0]
        actual = engine.run(inputs)["last_hidden_state"]
        difference = np.abs(expected - actual)
        print(
            {
                "shape": (batch, sequence),
                "mse": float(np.mean((expected - actual) ** 2)),
                "mean_abs_error": float(difference.mean()),
                "max_abs_error": float(difference.max()),
                "planned_activation_bytes": engine.get_planned_activation_bytes(),
                "naive_activation_bytes": engine.get_naive_activation_bytes(),
            },
            flush=True,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    main()