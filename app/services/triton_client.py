"""Triton inference client abstraction for detection and embedding models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
from loguru import logger

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError as exc:  # pragma: no cover - import side effect
    grpcclient = None
    InferenceServerException = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from app.utils.image import align_face, normalize_box, normalize_embeddings, prepare_detection_inputs

_TRITON_TO_NUMPY_DTYPE = {
    "BOOL": np.bool_,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
}


class TritonClientError(RuntimeError):
    """Errors raised by the Triton client wrapper."""


@dataclass(slots=True)
class ModelIOConfig:
    """Configuration describing the Triton model inputs and outputs."""

    name: str
    input_names: Tuple[str, ...]
    input_dtypes: Tuple[str, ...]
    output_names: Tuple[str, ...]
    max_batch_size: Optional[int] = None

    @classmethod
    def detection_defaults(cls) -> "ModelIOConfig":
        return cls(
            name="detection",
            input_names=("input_0", "input_1", "input_2"),
            input_dtypes=("UINT8", "FP32", "FP32"),
            output_names=("output_0", "output_1", "output_2"),
        )

    @classmethod
    def extraction_defaults(cls) -> "ModelIOConfig":
        return cls(
            name="extraction",
            input_names=("input_0",),
            input_dtypes=("UINT8",),
            output_names=("output_0",),
        )


@dataclass(slots=True)
class DetectionResult:
    """Detected face with optional embedding vector."""

    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray
    face_size: float
    aligned_face: np.ndarray
    embedding: Optional[np.ndarray] = None


class TritonClientProtocol(Protocol):
    """Protocol implemented by Triton client services."""

    async def warmup(self) -> None:  # pragma: no cover - interface definition
        ...

    def run_detection(
        self, images: Sequence[np.ndarray], *, input_size: Tuple[int, int] | None = None
    ) -> list[list[DetectionResult]]:  # pragma: no cover - interface definition
        ...

    def run_extraction(
        self, faces: Sequence[np.ndarray], *, normalize: bool = True
    ) -> np.ndarray:  # pragma: no cover - interface definition
        ...

    def detect_and_embed(
        self,
        images: Sequence[np.ndarray],
        *,
        input_size: Tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> list[list[DetectionResult]]:  # pragma: no cover - interface definition
        ...

    def close(self) -> None:  # pragma: no cover - interface definition
        ...


class TritonClient(TritonClientProtocol):
    """Concrete Triton inference client."""

    def __init__(
        self,
        url: str,
        *,
        detection_model: Optional[ModelIOConfig] = None,
        extraction_model: Optional[ModelIOConfig] = None,
        detection_input_size: Tuple[int, int] = (640, 640),
        request_timeout: float = 30.0,
        healthcheck: bool = True,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "tritonclient package is required but not installed"
            ) from _IMPORT_ERROR

        self._url = url
        self._request_timeout = request_timeout
        self._detection_input_size = detection_input_size

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - network init
            raise TritonClientError(f"Failed to create Triton client for {url}") from exc

        if healthcheck:
            self._assert_server_live()

        self._detection_model = detection_model or ModelIOConfig.detection_defaults()
        self._extraction_model = extraction_model or ModelIOConfig.extraction_defaults()

        self._cache_model_metadata(self._detection_model)
        self._cache_model_metadata(self._extraction_model)

    async def warmup(self) -> None:
        """Asynchronously verify the server is ready to accept requests."""
        await asyncio.to_thread(self._assert_server_live)

    def close(self) -> None:
        """Close the underlying gRPC channel if supported."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()  # pragma: no cover - cleanup path

    def run_detection(
        self, images: Sequence[np.ndarray], *, input_size: Tuple[int, int] | None = None
    ) -> list[list[DetectionResult]]:
        """Run face detection for a batch of images."""
        if not images:
            return []

        effective_input_size = input_size or self._detection_input_size
        try:
            prepared = prepare_detection_inputs(images, effective_input_size)
        except ValueError as exc:
            raise TritonClientError(f"Invalid detection input: {exc}") from exc

        outputs = self._infer(
            self._detection_model,
            (
                prepared.batched_images,
                prepared.scales,
                prepared.centers,
            ),
        )

        boxes_batch, landmarks_batch, counts_batch = outputs
        return self._parse_detection_results(
            images, prepared.original_sizes, boxes_batch, landmarks_batch, counts_batch
        )

    def run_extraction(
        self, faces: Sequence[np.ndarray], *, normalize: bool = True
    ) -> np.ndarray:
        """Extract embeddings for aligned face crops."""
        if not faces:
            raise TritonClientError("At least one face is required for embedding extraction")

        batched_faces = np.stack(
            [np.asarray(face, dtype=np.uint8) for face in faces], axis=0
        )

        outputs = self._infer(self._extraction_model, (batched_faces,))

        embeddings = outputs[0].astype(np.float32, copy=False)
        if normalize:
            embeddings = normalize_embeddings(embeddings, axis=1)
        return embeddings

    def detect_and_embed(
        self,
        images: Sequence[np.ndarray],
        *,
        input_size: Tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> list[list[DetectionResult]]:
        """Run detection followed by embedding extraction for each detected face."""
        detections = self.run_detection(images, input_size=input_size)

        faces: list[np.ndarray] = []
        index_map: list[Tuple[int, int]] = []

        for image_idx, face_list in enumerate(detections):
            for face_idx, detection in enumerate(face_list):
                faces.append(detection.aligned_face)
                index_map.append((image_idx, face_idx))

        if not faces:
            return detections

        embeddings = self.run_extraction(faces, normalize=normalize)
        for (image_idx, face_idx), embedding in zip(index_map, embeddings):
            detections[image_idx][face_idx].embedding = embedding

        return detections

    def _assert_server_live(self) -> None:
        """Raise if the Triton server is not responding."""
        try:
            if not self._client.is_server_live():
                raise TritonClientError(f"Triton server at {self._url} is not live")
        except (InferenceServerException, Exception) as exc:  # pragma: no cover - network path
            raise TritonClientError(f"Triton health check failed: {exc}") from exc

    def _cache_model_metadata(self, model: ModelIOConfig) -> None:
        """Fetch and cache model metadata from Triton."""
        try:
            metadata = self._client.get_model_metadata(model_name=model.name)
        except Exception as exc:
            logger.warning("Unable to fetch metadata for model {}: {}", model.name, exc)
            return

        try:
            model.input_names = tuple(inp.name for inp in metadata.inputs)
            model.input_dtypes = tuple(inp.datatype for inp in metadata.inputs)
            model.output_names = tuple(out.name for out in metadata.outputs)
        except AttributeError:
            # Fallback for dictionary-style metadata responses.
            model.input_names = tuple(inp["name"] for inp in metadata["inputs"])
            model.input_dtypes = tuple(inp["datatype"] for inp in metadata["inputs"])
            model.output_names = tuple(out["name"] for out in metadata["outputs"])

        try:
            config = self._client.get_model_config(model_name=model.name)
        except Exception:
            return

        max_batch = getattr(getattr(config, "config", None), "max_batch_size", None)
        if max_batch is None and isinstance(config, dict):
            max_batch = config.get("config", {}).get("max_batch_size")
        model.max_batch_size = max_batch

    def _infer(
        self,
        model: ModelIOConfig,
        input_tensors: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference against a Triton model."""
        if len(input_tensors) != len(model.input_names):
            raise TritonClientError(
                f"Model {model.name} expects {len(model.input_names)} inputs, "
                f"received {len(input_tensors)}"
            )

        infer_inputs = []
        for name, dtype, array in zip(model.input_names, model.input_dtypes, input_tensors):
            numpy_dtype = _TRITON_TO_NUMPY_DTYPE.get(dtype)
            if numpy_dtype is None:
                raise TritonClientError(
                    f"Unsupported Triton datatype '{dtype}' for input '{name}'"
                )
            tensor = np.asarray(array, dtype=numpy_dtype)
            tensor = np.ascontiguousarray(tensor)
            infer_input = grpcclient.InferInput(name, tensor.shape, dtype)
            infer_input.set_data_from_numpy(tensor)
            infer_inputs.append(infer_input)

        infer_outputs = [
            grpcclient.InferRequestedOutput(name) for name in model.output_names
        ]

        try:
            response = self._client.infer(
                model_name=model.name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                client_timeout=self._request_timeout,
            )
        except InferenceServerException as exc:
            raise TritonClientError(
                f"Inference call to model {model.name} failed: {exc}"
            ) from exc

        return [response.as_numpy(name) for name in model.output_names]

    def _parse_detection_results(
        self,
        images: Sequence[np.ndarray],
        original_sizes: Sequence[tuple[int, int]],
        boxes_batch: np.ndarray,
        landmarks_batch: np.ndarray,
        counts_batch: np.ndarray,
    ) -> list[list[DetectionResult]]:
        """Convert Triton detection outputs into structured results."""
        all_detections: list[list[DetectionResult]] = []
        offset = 0

        for index, (image, size) in enumerate(zip(images, original_sizes)):
            num_faces = self._extract_face_count(counts_batch, index)
            if num_faces == 0:
                all_detections.append([])
                continue

            height, width = size
            end = offset + num_faces
            boxes = boxes_batch[offset:end]
            landmarks = landmarks_batch[offset:end]
            detections: list[DetectionResult] = []

            for box, landmark in zip(boxes, landmarks):
                landmark_points = np.asarray(landmark, dtype=np.float32).reshape(5, 2)
                normalized_box = normalize_box(np.asarray(box, dtype=np.float32), width, height)
                aligned_face = align_face(image, landmark_points)
                face_area = float(
                    (normalized_box[2] - normalized_box[0]) * (normalized_box[3] - normalized_box[1])
                )

                detections.append(
                    DetectionResult(
                        bbox=normalized_box[:4],
                        confidence=float(normalized_box[4]),
                        landmarks=landmark_points,
                        face_size=face_area,
                        aligned_face=aligned_face,
                    )
                )

            all_detections.append(detections)
            offset = end

        return all_detections

    @staticmethod
    def _extract_face_count(counts_batch: np.ndarray, index: int) -> int:
        """Safely extract number of faces for an image index."""
        try:
            value = counts_batch[index]
        except IndexError:
            return 0

        if np.isscalar(value):
            return int(value)

        return int(value[0])
