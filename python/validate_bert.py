#!/usr/bin/env python3
import argparse
import time

import numpy as np
import onnxruntime as ort

import nn_inference
from bert_tokenizer import BertTokenizer


SENTENCES = [
    f"The {subject} is [MASK] today."
    for subject in (
        "weather", "sky", "ocean", "garden", "city",
        "music", "movie", "book", "coffee", "house",
    )
] + [
    f"I want to [MASK] the {object_name}."
    for object_name in (
        "door", "window", "letter", "question", "problem",
        "story", "picture", "computer", "car", "bottle",
    )
] + [
    f"She went to the [MASK] after work for {reason}."
    for reason in (
        "food", "exercise", "school", "music", "medicine",
        "shopping", "dinner", "books", "coffee", "flowers",
    )
] + [
    f"The {animal} sat on the [MASK] near the house."
    for animal in (
        "cat", "dog", "bird", "horse", "rabbit",
        "mouse", "fox", "duck", "goat", "cow",
    )
] + [
    f"They will [MASK] to {place} tomorrow morning."
    for place in (
        "school", "london", "paris", "work", "home",
        "canada", "france", "germany", "india", "china",
    )
]


def validate(model_path, vocab_path, n=50, max_length=16):
    if n <= 0 or n > len(SENTENCES):
        raise ValueError(f"n must be between 1 and {len(SENTENCES)}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    engine = nn_inference.ExecutionEngine(model_path)
    tokenizer = BertTokenizer(vocab_path)
    mse_values = []
    mean_errors = []
    max_errors = []
    engine_times = []
    ort_times = []
    for index, sentence in enumerate(SENTENCES[:n], start=1):
        inputs = tokenizer.encode(sentence, max_length=max_length)
        start = time.perf_counter()
        expected = session.run(None, inputs)[0]
        ort_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        actual = engine.run(inputs)["logits"]
        engine_times.append(time.perf_counter() - start)
        difference = expected - actual
        mse_values.append(float(np.mean(difference * difference)))
        absolute = np.abs(difference)
        mean_errors.append(float(absolute.mean()))
        max_errors.append(float(absolute.max()))
        if index % 10 == 0 or index == n:
            print(
                f"validated {index}/{n}: running_logit_mse={np.mean(mse_values):.10g}",
                flush=True,
            )
    result = {
        "sentences": n,
        "logit_mse": float(np.mean(mse_values)),
        "max_sentence_mse": float(np.max(mse_values)),
        "mean_abs_error": float(np.mean(mean_errors)),
        "max_abs_error": float(np.max(max_errors)),
        "engine_mean_ms": float(np.mean(engine_times) * 1000.0),
        "ort_mean_ms": float(np.mean(ort_times) * 1000.0),
        "planned_activation_bytes": engine.get_planned_activation_bytes(),
        "naive_activation_bytes": engine.get_naive_activation_bytes(),
    }
    print("BERT-tiny MLM validation:", result)
    if result["logit_mse"] >= 1.0e-3:
        raise AssertionError("BERT-tiny logit MSE exceeds 1e-3")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("vocab_path")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=16)
    args = parser.parse_args()
    validate(args.model_path, args.vocab_path, args.n, args.max_length)


if __name__ == "__main__":
    main()