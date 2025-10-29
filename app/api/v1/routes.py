from __future__ import annotations

import asyncio
import time
import uuid
from typing import Sequence

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from app.core.config import Settings, get_settings
from app.schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    FaceEmbeddingPayload,
    ImageEmbeddingsResponse,
    RerankRequest,
    RerankResponse,
    RerankScorePayload,
)
from app.services.rerank import RerankServiceProtocol
from app.services.triton_client import DetectionResult, TritonClientError, TritonClientProtocol
from app.utils.image import bytes_to_image, decode_base64_to_bytes, image_to_base64

router = APIRouter()


def _get_runtime_settings(request: Request) -> Settings:
    """Prefer application-scoped settings if available, fallback to global cache."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def _get_triton_client(request: Request) -> TritonClientProtocol:
    """Retrieve Triton client from FastAPI application state."""
    client = getattr(request.app.state, "triton_client", None)
    if client is None:
        logger.error("Triton client requested before initialization")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client is not ready",
        )
    return client


def _get_rerank_service(request: Request) -> RerankServiceProtocol:
    """Retrieve rerank service instance from FastAPI application state."""
    service = getattr(request.app.state, "rerank_service", None)
    if service is None:
        logger.error("Rerank service requested before initialization")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rerank service is not ready",
        )
    return service


def _resolve_request_id(request: Request) -> str:
    """Extract or generate a request identifier for logging."""
    header_value = request.headers.get("x-request-id")
    if header_value:
        return header_value
    return uuid.uuid4().hex


@router.get("/healthz")
async def read_health(
    request: Request, settings: Settings = Depends(_get_runtime_settings)
) -> dict[str, object]:
    """Health check indicating readiness of backend dependencies."""
    triton_ready = getattr(request.app.state, "triton_client", None) is not None
    rerank_ready = getattr(request.app.state, "rerank_service", None) is not None
    status_value = "ok" if triton_ready and rerank_ready else "degraded"
    return {
        "status": status_value,
        "environment": settings.env,
        "triton_ready": triton_ready,
        "rerank_ready": rerank_ready,
    }


@router.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    status_code=status.HTTP_200_OK,
)
async def create_embeddings(
    payload: EmbeddingsRequest,
    settings: Settings = Depends(_get_runtime_settings),
    triton_client: TritonClientProtocol = Depends(_get_triton_client),
    request: Request,
) -> EmbeddingsResponse:
    """Generate face embeddings for one or more images."""
    total_start = time.perf_counter()
    decoded_images: list[np.ndarray] = []
    image_ids: list[str] = []
    request_id = _resolve_request_id(request)

    decode_start = time.perf_counter()
    for index, image_payload in enumerate(payload.images):
        identifier = image_payload.id or f"image_{index + 1}"
        try:
            raw_bytes = decode_base64_to_bytes(image_payload.data)
        except ValueError as exc:
            logger.warning("Base64 decode failed for image {}: {}", identifier, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 data for image '{identifier}'",
            ) from exc

        try:
            image = bytes_to_image(raw_bytes)
        except Exception as exc:
            logger.warning("Image decode failed for image {}: {}", identifier, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to decode image bytes for '{identifier}'",
            ) from exc

        if image.ndim != 3 or image.shape[2] != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image '{identifier}' must be RGB or BGR with 3 channels",
            )

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image '{identifier}' has invalid dimensions ({width}x{height})",
            )

        decoded_images.append(image)
        image_ids.append(identifier)

    timings_ms: dict[str, float] = {
        "decode_ms": (time.perf_counter() - decode_start) * 1000.0,
    }

    if len(decoded_images) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(decoded_images)} exceeds configured limit "
                f"{settings.max_batch_size}"
            ),
        )

    normalize_embeddings = not (
        payload.skip_embedding_normalization
        if payload.skip_embedding_normalization is not None
        else settings.skip_embedding_normalization
    )
    include_aligned = (
        payload.return_aligned_faces
        if payload.return_aligned_faces is not None
        else settings.return_aligned
    )

    try:
        inference_start = time.perf_counter()
        detections = await asyncio.to_thread(
            triton_client.detect_and_embed,
            decoded_images,
            input_size=settings.detection_input_size,
            normalize=normalize_embeddings,
        )
        timings_ms["inference_ms"] = (time.perf_counter() - inference_start) * 1000.0
    except TritonClientError as exc:
        logger.error("Triton inference failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Triton inference failed",
        ) from exc

    if not isinstance(detections, Sequence):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected Triton response payload",
        )

    results: list[ImageEmbeddingsResponse] = []
    for image_id, face_detections in zip(image_ids, detections):
        faces_payload = [
            _serialize_detection(face, include_aligned=include_aligned)
            for face in face_detections
        ]
        results.append(
            ImageEmbeddingsResponse(
                image_id=image_id,
                faces=faces_payload,
                num_faces=len(faces_payload),
            )
        )

    timings_ms["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    rounded_timings = {key: round(value, 3) for key, value in timings_ms.items()}
    total_faces = sum(result.num_faces for result in results)

    logger.bind(request_id=request_id).info(
        "embeddings_request images={} faces={} normalized={} aligned={} timings_ms={}",
        len(results),
        total_faces,
        normalize_embeddings,
        include_aligned,
        rounded_timings,
    )

    return EmbeddingsResponse(
        results=results,
        timings_ms=rounded_timings,
        normalized_embeddings=normalize_embeddings,
        included_aligned_faces=include_aligned,
    )


@router.post(
    "/rerank",
    response_model=RerankResponse,
    status_code=status.HTTP_200_OK,
)
async def rerank_embeddings(
    payload: RerankRequest,
    rerank_service: RerankServiceProtocol = Depends(_get_rerank_service),
    request: Request,
) -> RerankResponse:
    """Compute rerank scores for candidate embeddings."""
    start_time = time.perf_counter()
    request_id = _resolve_request_id(request)
    try:
        query = np.asarray(payload.query_embedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid query embedding payload: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query embedding contains non-numeric values",
        ) from exc

    candidate_ids = [candidate.id for candidate in payload.candidates]
    try:
        candidate_embeddings = np.asarray(
            [candidate.embedding for candidate in payload.candidates],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid candidate embeddings payload: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Candidate embeddings contain non-numeric values",
        ) from exc

    effective_threshold = (
        payload.threshold
        if payload.threshold is not None
        else rerank_service.default_threshold
    )

    try:
        scores = await asyncio.to_thread(
            rerank_service.compute_scores,
            query,
            candidate_embeddings,
            threshold=effective_threshold,
        )
    except ValueError as exc:
        logger.warning("Rerank validation failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Rerank computation failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Rerank computation failed",
        ) from exc

    if scores.shape[0] != len(candidate_ids):
        logger.error(
            "Rerank returned mismatched score count ({} vs {} candidates)",
            scores.shape[0],
            len(candidate_ids),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rerank returned mismatched score count",
        )

    total_ms = (time.perf_counter() - start_time) * 1000.0
    timings_ms = {"total_ms": round(total_ms, 3)}

    score_payloads = [
        RerankScorePayload(id=candidate_id, score=float(score))
        for candidate_id, score in zip(candidate_ids, scores)
    ]

    logger.bind(request_id=request_id).info(
        "rerank_request candidates={} threshold={} timings_ms={}",
        len(score_payloads),
        float(effective_threshold),
        timings_ms,
    )

    return RerankResponse(
        scores=score_payloads,
        threshold_applied=float(effective_threshold),
        candidate_count=len(score_payloads),
        timings_ms=timings_ms,
    )


def _serialize_detection(
    detection: DetectionResult,
    *,
    include_aligned: bool,
) -> FaceEmbeddingPayload:
    """Convert a DetectionResult into a FaceEmbeddingPayload."""
    if detection.embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection result missing embedding vector",
        )

    aligned_face = None
    if include_aligned:
        try:
            aligned_face = image_to_base64(detection.aligned_face)
        except ValueError as exc:
            logger.warning("Failed to encode aligned face to base64: {}", exc)

    return FaceEmbeddingPayload(
        bbox=[float(v) for v in detection.bbox.tolist()],
        confidence=float(detection.confidence),
        landmarks=detection.landmarks.astype(float).tolist(),
        face_size=float(detection.face_size),
        embedding=detection.embedding.astype(float).tolist(),
        aligned_face=aligned_face,
    )
