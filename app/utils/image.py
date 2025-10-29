"""Image preprocessing utilities for detection and embedding pipelines."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import transform

# Canonical facial landmark targets for 112x112 aligned output (ArcFace style).
_DEFAULT_ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class DetectionBatchInputs:
    """Prepared tensors required by the detection model."""

    batched_images: np.ndarray
    scales: np.ndarray
    centers: np.ndarray
    original_sizes: list[tuple[int, int]]


def square_crop(image: np.ndarray, input_size: Tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize the image to fit inside a square canvas while preserving aspect ratio.

    Returns the padded image and the scale factor applied along the shortest side.
    """
    target_width, target_height = input_size
    if image.ndim != 3:
        raise ValueError("Expected image with 3 channels (H, W, C)")

    image_height, image_width = image.shape[:2]
    image_ratio = float(image_height) / float(image_width)
    target_ratio = float(target_height) / float(target_width)

    if image_ratio > target_ratio:
        new_height = target_height
        new_width = max(1, int(new_height / image_ratio))
    else:
        new_width = target_width
        new_height = max(1, int(new_width * image_ratio))

    scale = float(new_height) / float(image_height)
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized

    return canvas, scale


def prepare_detection_inputs(
    images: Sequence[np.ndarray], input_size: Tuple[int, int]
) -> DetectionBatchInputs:
    """Convert raw images into tensors accepted by the detection model."""
    if not images:
        raise ValueError("At least one image is required for detection preprocessing")

    prepared_images = []
    scales = []
    centers = []
    original_sizes: list[tuple[int, int]] = []

    for image in images:
        if image.ndim != 3:
            raise ValueError("Detection expects images with shape (H, W, C)")

        height, width = image.shape[:2]
        cropped, scale = square_crop(image, input_size)

        prepared_images.append(cropped)
        scales.append((scale, scale))
        centers.append((height / 2.0, width / 2.0))
        original_sizes.append((height, width))

    return DetectionBatchInputs(
        batched_images=np.stack(prepared_images, axis=0),
        scales=np.asarray(scales, dtype=np.float32),
        centers=np.asarray(centers, dtype=np.float32),
        original_sizes=original_sizes,
    )


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    *,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align a face crop using five landmark points."""
    if landmarks.shape != (5, 2):
        raise ValueError("Expected landmarks with shape (5, 2)")

    template = _DEFAULT_ALIGNMENT_TEMPLATE.copy()
    template[:, 0] = template[:, 0] * (output_size[0] / 112.0)
    template[:, 1] = template[:, 1] * (output_size[1] / 112.0)

    transform_estimator = transform.SimilarityTransform()
    if not transform_estimator.estimate(landmarks, template):
        raise RuntimeError("Failed to estimate alignment transform from landmarks")

    matrix = transform_estimator.params[0:2, :]
    aligned = cv2.warpAffine(image, matrix, output_size, borderValue=0)
    return aligned


def normalize_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp bounding box coordinates to the image bounds."""
    if box.shape[0] < 5:
        raise ValueError("Bounding box must include confidence as the fifth value")

    xmin = max(0.0, float(box[0]))
    ymin = max(0.0, float(box[1]))
    xmax = min(float(width), float(box[2]))
    ymax = min(float(height), float(box[3]))
    confidence = float(box[4])
    return np.array([xmin, ymin, xmax, ymax, confidence], dtype=np.float32)


def bytes_to_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR numpy image."""
    image = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def normalize_embeddings(vectors: np.ndarray, *, axis: int = 1) -> np.ndarray:
    """L2-normalize embedding vectors along the specified axis."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    # Avoid division by zero by falling back to ones.
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
