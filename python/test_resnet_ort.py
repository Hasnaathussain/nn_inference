#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort


def main():
    runner = sys.argv[1]
    model_path = pathlib.Path(sys.argv[2])
    rng = np.random.default_rng(20260715)
    x = rng.normal(size=(1, 3, 224, 224)).astype(np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start = time.perf_counter()
    expected = session.run(None, {input_name: x})[0]
    ort_seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="nn_inference_resnet_") as temporary:
        temporary = pathlib.Path(temporary)
        input_path = temporary / "input.bin"
        output_path = temporary / "output.bin"
        x.tofile(input_path)
        environment = dict(os.environ)
        environment.setdefault("ASAN_OPTIONS", "detect_leaks=0")
        start = time.perf_counter()
        subprocess.run(
            [
                runner,
                str(model_path),
                input_name,
                "1,3,224,224",
                str(input_path),
                output_name,
                str(output_path),
            ],
            check=True,
            env=environment,
        )
        engine_seconds = time.perf_counter() - start
        actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)

    absolute = np.abs(actual - expected)
    mean_error = float(absolute.mean())
    max_error = float(absolute.max())
    print(
        f"ResNet-18 opset13: mean_abs_error={mean_error:.8f}, "
        f"max_abs_error={max_error:.8f}, "
        f"engine_seconds={engine_seconds:.3f}, ort_seconds={ort_seconds:.3f}"
    )
    if mean_error >= 0.005:
        raise AssertionError(
            f"ResNet-18 mean absolute error {mean_error} exceeds 0.005"
        )


if __name__ == "__main__":
    main()
