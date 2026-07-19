#!/usr/bin/env python3
import argparse
import time

import numpy as np
import onnxruntime as ort

import nn_inference
from preprocess import image_paths, preprocess_image
from bert_tokenizer import BertTokenizer


def benchmark(run, input_dict, n_warmup=10, n_runs=100):
    if n_warmup < 0 or n_runs <= 0:
        raise ValueError("n_warmup must be non-negative and n_runs must be positive")
    for _ in range(n_warmup):
        run(input_dict)

    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run(input_dict)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_ips": float(1000.0 / np.mean(times_ms)),
    }


def benchmark_resnet(model_path, images_dir, n_warmup=10, n_runs=100):
    paths = image_paths(images_dir, 1)
    inputs = preprocess_image(paths[0])

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    engine = nn_inference.ExecutionEngine(str(model_path))
    input_dict = {input_name: inputs}

    engine_result = benchmark(
        engine.run, input_dict, n_warmup=n_warmup, n_runs=n_runs
    )
    ort_result = benchmark(
        lambda values: session.run(None, values),
        input_dict,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )
    result = {
        "model": str(model_path),
        "image": str(paths[0]),
        "warmup_runs": n_warmup,
        "measured_runs": n_runs,
        "engine": engine_result,
        "onnxruntime_single_thread": ort_result,
        "ort_speedup_over_engine": (
            engine_result["mean_ms"] / ort_result["mean_ms"]
        ),
        "planned_activation_bytes": engine.get_planned_activation_bytes(),
        "naive_activation_bytes": engine.get_naive_activation_bytes(),
    }
    print("ResNet-18 benchmark:", result)
    return result


def benchmark_bert(model_path, vocab_path, n_warmup=10, n_runs=100):
    tokenizer = BertTokenizer(vocab_path)
    input_dict = tokenizer.encode("The weather is [MASK] today.", max_length=16)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    engine = nn_inference.ExecutionEngine(str(model_path))
    engine_result = benchmark(
        engine.run, input_dict, n_warmup=n_warmup, n_runs=n_runs
    )
    ort_result = benchmark(
        lambda values: session.run(None, values),
        input_dict,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )
    result = {
        "model": str(model_path),
        "warmup_runs": n_warmup,
        "measured_runs": n_runs,
        "engine": engine_result,
        "onnxruntime_single_thread": ort_result,
        "ort_speedup_over_engine": (
            engine_result["mean_ms"] / ort_result["mean_ms"]
        ),
        "planned_activation_bytes": engine.get_planned_activation_bytes(),
        "naive_activation_bytes": engine.get_naive_activation_bytes(),
    }
    print("BERT-tiny benchmark:", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("input_path")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--bert", action="store_true")
    args = parser.parse_args()
    if args.bert:
        benchmark_bert(
            args.model_path,
            args.input_path,
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
    else:
        benchmark_resnet(
            args.model_path,
            args.input_path,
            n_warmup=args.warmup,
            n_runs=args.runs,
        )


if __name__ == "__main__":
    main()