#!/usr/bin/env python3
import argparse

import nn_inference
from preprocess import load_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("images_dir")
    parser.add_argument("--no-memory-planner", action="store_true")
    args = parser.parse_args()

    engine = nn_inference.ExecutionEngine(args.model_path)
    engine.set_memory_planner(not args.no_memory_planner)
    _, inputs = next(load_images(args.images_dir, 1))
    outputs = engine.run({"data": inputs})
    print(
        {
            "output_names": sorted(outputs),
            "planned_activation_bytes": engine.get_planned_activation_bytes(),
            "naive_activation_bytes": engine.get_naive_activation_bytes(),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()