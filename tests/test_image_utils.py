import base64

import numpy as np
import pytest

from app.utils.image import (
    DetectionBatchInputs,
    decode_base64_image,
    decode_base64_to_bytes,
    image_to_base64,
    normalize_embeddings,
    normalize_box,
    prepare_detection_inputs,
    square_crop,
)


def test_square_crop_returns_canvas_and_scale() -> None:
    image = np.ones((100, 50, 3), dtype=np.uint8) * 255
    canvas, scale = square_crop(image, (200, 200))

    assert canvas.shape == (200, 200, 3)
    assert scale == pytest.approx(2.0)
    assert np.all(canvas[:, 100:, :] == 0)


def test_prepare_detection_inputs_batches_images() -> None:
    image_one = np.random.randint(0, 255, size=(120, 80, 3), dtype=np.uint8)
    image_two = np.random.randint(0, 255, size=(80, 120, 3), dtype=np.uint8)

    prepared = prepare_detection_inputs([image_one, image_two], (160, 160))

    assert isinstance(prepared, DetectionBatchInputs)
    assert prepared.batched_images.shape == (2, 160, 160, 3)
    assert prepared.scales.shape == (2, 2)
    assert prepared.centers.shape == (2, 2)
    assert prepared.original_sizes == [(120, 80), (80, 120)]


def test_prepare_detection_inputs_requires_three_channel_images() -> None:
    single_channel = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_detection_inputs([single_channel], (32, 32))


def test_normalize_embeddings_handles_zero_vectors() -> None:
    embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(embeddings)

    assert normalized.shape == embeddings.shape
    assert normalized[0, 0] == pytest.approx(0.6)
    assert normalized[0, 1] == pytest.approx(0.8)
    assert np.all(normalized[1] == 0.0)


def test_decode_base64_round_trip_png() -> None:
    original = np.zeros((12, 12, 3), dtype=np.uint8)
    original[:, :] = (10, 20, 30)

    encoded = image_to_base64(original, format="PNG")
    decoded = decode_base64_image(encoded)

    assert decoded.shape == original.shape
    assert np.array_equal(decoded, original)


def test_decode_base64_to_bytes_handles_data_uri_prefix() -> None:
    payload = base64.b64encode(b"sample-bytes").decode()
    data_uri = f"data:image/jpeg;base64,{payload}"

    raw = decode_base64_to_bytes(data_uri)
    assert raw == b"sample-bytes"


def test_normalize_box_clamps_within_bounds() -> None:
    box = np.array([-5.0, -10.0, 150.0, 200.0, 0.7], dtype=np.float32)
    normalized = normalize_box(box, width=100, height=100)

    assert np.array_equal(
        normalized,
        np.array([0.0, 0.0, 100.0, 100.0, 0.7], dtype=np.float32),
    )
