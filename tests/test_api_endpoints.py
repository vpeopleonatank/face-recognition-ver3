from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes import router
from app.core.config import Settings
from app.services.triton_client import DetectionResult, TritonClientError
from app.utils.image import image_to_base64

TEST_API_KEY = "test-api-key"


class DummyTritonClient:
    def __init__(self, *, response: list[list[DetectionResult]] | None = None, error: Exception | None = None) -> None:
        self._response = response or []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def detect_and_embed(
        self,
        images: list[np.ndarray],
        *,
        input_size: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> list[list[DetectionResult]]:
        if self._error:
            raise self._error
        self.calls.append(
            {"count": len(images), "input_size": input_size, "normalize": normalize}
        )
        return self._response


class DummyRerankService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.default_threshold = 0.6
        self.calls: list[dict[str, Any]] = []

    @property
    def embedding_dimension(self) -> int:
        return 4

    def compute_scores(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        if self._error:
            raise self._error
        scores = np.sum(candidate_embeddings * query_embedding, axis=1)
        self.calls.append(
            {
                "query": query_embedding.copy(),
                "candidates": candidate_embeddings.copy(),
                "threshold": threshold,
                "scores": scores.copy(),
            }
        )
        return scores


@pytest.fixture()
def make_test_app() -> callable:
    def _make(
        *,
        triton_client: DummyTritonClient,
        rerank_service: DummyRerankService,
    ) -> FastAPI:
        settings = Settings(
            environment="local",
            triton_healthcheck=False,
            rerank_library_path="face_v3/libs/librerank_compute.so",
            detection_input_width=160,
            detection_input_height=160,
            api_key=TEST_API_KEY,
        )
        app = FastAPI()
        app.state.settings = settings
        app.state.triton_client = triton_client
        app.state.rerank_service = rerank_service
        app.include_router(router, prefix=settings.api_prefix)
        return app

    return _make


def _make_test_image_base64() -> str:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    return image_to_base64(image, format="PNG")


def test_embeddings_endpoint_success(make_test_app) -> None:
    detection = DetectionResult(
        bbox=np.array([1.0, 2.0, 50.0, 60.0], dtype=np.float32),
        confidence=0.95,
        landmarks=np.ones((5, 2), dtype=np.float32),
        face_size=48.0,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    triton_client = DummyTritonClient(response=[[detection]])
    rerank_service = DummyRerankService()
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "images": [{"id": "img1", "data": _make_test_image_base64()}],
        "return_aligned_faces": True,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["normalized_embeddings"] is True
    assert body["included_aligned_faces"] is True
    assert body["results"][0]["image_id"] == "img1"
    assert body["results"][0]["faces"][0]["aligned_face"] is not None
    assert triton_client.calls[0]["normalize"] is True


def test_embeddings_endpoint_invalid_base64(make_test_app) -> None:
    triton_client = DummyTritonClient(response=[])
    rerank_service = DummyRerankService()
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    invalid_base64 = "!!!!####$$$$%%%%"  # length >= 16 but contains invalid base64 chars
    payload = {
        "images": [{"id": "img1", "data": invalid_base64}],
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 400


def test_embeddings_endpoint_handles_triton_failure(make_test_app) -> None:
    triton_client = DummyTritonClient(error=TritonClientError("boom"))
    rerank_service = DummyRerankService()
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "images": [{"id": "img1", "data": _make_test_image_base64()}],
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 502


def test_rerank_endpoint_success(make_test_app) -> None:
    triton_client = DummyTritonClient(response=[])
    rerank_service = DummyRerankService()
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "query_embedding": [0.1, 0.2, 0.3, 0.4],
        "candidates": [
            {"id": "a", "embedding": [0.2, 0.2, 0.2, 0.2]},
            {"id": "b", "embedding": [0.5, 0.4, 0.3, 0.2]},
        ],
        "threshold": 0.9,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rerank",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["candidate_count"] == 2
    assert [score["id"] for score in body["scores"]] == ["a", "b"]
    assert rerank_service.calls[0]["threshold"] == pytest.approx(0.9)


def test_rerank_endpoint_handles_validation_error(make_test_app) -> None:
    triton_client = DummyTritonClient(response=[])
    rerank_service = DummyRerankService(error=ValueError("bad input"))
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "query_embedding": [0.1, 0.2, 0.3, 0.4],
        "candidates": [
            {"id": "a", "embedding": [0.2, 0.2, 0.2, 0.2]},
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rerank",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 400


def test_embeddings_endpoint_rejects_missing_api_key(make_test_app) -> None:
    detection = DetectionResult(
        bbox=np.array([1.0, 2.0, 50.0, 60.0], dtype=np.float32),
        confidence=0.95,
        landmarks=np.ones((5, 2), dtype=np.float32),
        face_size=48.0,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    triton_client = DummyTritonClient(response=[[detection]])
    rerank_service = DummyRerankService()
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "images": [{"id": "img1", "data": _make_test_image_base64()}],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/embeddings", json=payload)

    assert response.status_code == 401


def test_rerank_endpoint_handles_runtime_error(make_test_app) -> None:
    triton_client = DummyTritonClient(response=[])
    rerank_service = DummyRerankService(error=RuntimeError("failure"))
    app = make_test_app(triton_client=triton_client, rerank_service=rerank_service)

    payload = {
        "query_embedding": [0.1, 0.2, 0.3, 0.4],
        "candidates": [
            {"id": "a", "embedding": [0.2, 0.2, 0.2, 0.2]},
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rerank",
            json=payload,
            headers={"X-API-KEY": TEST_API_KEY},
        )

    assert response.status_code == 502
