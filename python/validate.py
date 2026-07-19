#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import nn_inference
from preprocess import load_images


def load_synset_indices(path):
    indices = {}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if stripped:
            indices[stripped.split()[0]] = index
    return indices


def validate_resnet18(
    model_path,
    images_dir,
    n=100,
    tol=0.005,
    synset_path=None,
    memory_planner=True,
):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    engine = nn_inference.ExecutionEngine(str(model_path))
    engine.set_memory_planner(memory_planner)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    synsets = load_synset_indices(synset_path) if synset_path else None

    errors = []
    max_errors = []
    matching_predictions = 0
    ort_correct = 0
    engine_correct = 0
    engine_times = []
    ort_times = []

    for index, (path, x) in enumerate(load_images(images_dir, n), start=1):
        start = time.perf_counter()
        expected = session.run(None, {input_name: x})[0]
        ort_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        actual = engine.run({input_name: x})[output_name]
        engine_times.append(time.perf_counter() - start)

        difference = np.abs(expected - actual)
        errors.append(float(difference.mean()))
        max_errors.append(float(difference.max()))
        ort_prediction = int(expected.argmax(axis=1)[0])
        engine_prediction = int(actual.argmax(axis=1)[0])
        matching_predictions += int(ort_prediction == engine_prediction)

        if synsets is not None:
            label = synsets.get(path.parent.name)
            if label is None:
                raise ValueError(
                    f"Image directory {path.parent.name!r} is absent from synset file"
                )
            ort_correct += int(ort_prediction == label)
            engine_correct += int(engine_prediction == label)

        if index % 10 == 0 or index == n:
            print(
                f"validated {index}/{n}: running_mean_abs_error="
                f"{np.mean(errors):.8f}",
                flush=True,
            )

    mean_error = float(np.mean(errors))
    top1_agreement = matching_predictions / n
    result = {
        "images": n,
        "mean_abs_error": mean_error,
        "max_abs_error": float(np.max(max_errors)),
        "top1_agreement": top1_agreement,
        "engine_mean_ms": float(np.mean(engine_times) * 1000.0),
        "ort_mean_ms": float(np.mean(ort_times) * 1000.0),
        "planned_activation_bytes": engine.get_planned_activation_bytes(),
        "naive_activation_bytes": engine.get_naive_activation_bytes(),
    }
    if synsets is not None:
        ort_accuracy = ort_correct / n
        engine_accuracy = engine_correct / n
        result.update(
            {
                "ort_top1_accuracy": ort_accuracy,
                "engine_top1_accuracy": engine_accuracy,
                "top1_accuracy_difference": abs(engine_accuracy - ort_accuracy),
            }
        )
        if abs(engine_accuracy - ort_accuracy) > 0.005:
            raise AssertionError(
                "Top-1 accuracy differs from ONNXRuntime by more than 0.5%"
            )

    print("ResNet-18 validation:", result)
    if mean_error >= tol:
        raise AssertionError(
            f"Mean absolute error {mean_error} exceeds tolerance {tol}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("images_dir")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument("--synset-file")
    parser.add_argument("--no-memory-planner", action="store_true")
    args = parser.parse_args()
    validate_resnet18(
        args.model_path,
        args.images_dir,
        n=args.n,
        tol=args.tol,
        synset_path=args.synset_file,
        memory_planner=not args.no_memory_planner,
    )


if __name__ == "__main__":
    main()
