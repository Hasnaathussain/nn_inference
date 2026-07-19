#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(images_dir, n=None):
    paths = sorted(
        path
        for path in Path(images_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if n is not None:
        if len(paths) < n:
            raise ValueError(
                f"Requested {n} images but found only {len(paths)} in {images_dir}"
            )
        paths = paths[:n]
    return paths


def preprocess_image(path, resize_size=256, crop_size=224):
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width <= height:
            resized_width = resize_size
            resized_height = round(height * resize_size / width)
        else:
            resized_height = resize_size
            resized_width = round(width * resize_size / height)
        image = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        left = (resized_width - crop_size) // 2
        top = (resized_height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)

    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def load_images(images_dir, n=None):
    for path in image_paths(images_dir, n):
        yield path, preprocess_image(path)
