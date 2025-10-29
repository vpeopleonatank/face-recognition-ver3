from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from app.services.rerank import RerankService


class FakeCFunc:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.last_threshold = None
        self.argtypes = None
        self.restype = None

    def __call__(
        self,
        query_ptr: ctypes.POINTER(ctypes.c_double),
        candidate_ptr: ctypes.POINTER(ctypes.c_double),
        num_candidates: ctypes.c_int,
        threshold: ctypes.c_double,
        scores_ptr: ctypes.POINTER(ctypes.c_double),
    ) -> int:
        count = int(getattr(num_candidates, "value", num_candidates))
        thresh = float(getattr(threshold, "value", threshold))
        self.last_threshold = thresh

        query = np.ctypeslib.as_array(query_ptr, shape=(self.embedding_dim,))
        candidates = np.ctypeslib.as_array(
            candidate_ptr, shape=(count * self.embedding_dim,)
        ).reshape(count, self.embedding_dim)
        scores = np.ctypeslib.as_array(scores_ptr, shape=(count,))

        if count:
            scores[:] = candidates @ query
        return 0


@pytest.fixture()
def fake_rerank_service(monkeypatch, tmp_path) -> RerankService:
    embedding_dim = 4
    fake_func = FakeCFunc(embedding_dim)

    class FakeLib:
        def __init__(self) -> None:
            self.compute_rerank_scores_from_embeddings = fake_func

    fake_library_path = tmp_path / "librerank_compute.so"
    fake_library_path.write_bytes(b"fake")

    monkeypatch.setattr(
        RerankService,
        "_resolve_library_path",
        staticmethod(lambda path: Path(fake_library_path)),
    )
    monkeypatch.setattr("app.services.rerank.ctypes.CDLL", lambda path: FakeLib())

    return RerankService(
        library_path=str(fake_library_path),
        embedding_dimension=embedding_dim,
        default_threshold=0.25,
    )


def test_rerank_service_compute_scores(fake_rerank_service: RerankService) -> None:
    query = np.array([1.0, 0.0, -1.0, 2.0], dtype=np.float64)
    candidates = np.array(
        [
            [0.5, 0.0, -0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    scores = fake_rerank_service.compute_scores(query, candidates)

    expected = candidates @ query
    assert np.allclose(scores, expected)


def test_rerank_service_handles_empty_candidates(fake_rerank_service: RerankService) -> None:
    query = np.ones(fake_rerank_service.embedding_dimension, dtype=np.float64)
    scores = fake_rerank_service.compute_scores(query, [])
    assert scores.shape == (0,)


def test_rerank_service_validates_dimensions(fake_rerank_service: RerankService) -> None:
    query = np.ones(fake_rerank_service.embedding_dimension - 1, dtype=np.float64)
    with pytest.raises(ValueError):
        fake_rerank_service.compute_scores(query, [])


def test_rerank_service_validates_candidate_shape(fake_rerank_service: RerankService) -> None:
    dim = fake_rerank_service.embedding_dimension
    query = np.ones(dim, dtype=np.float64)
    bad_candidates = np.ones((2, dim + 1), dtype=np.float64)
    with pytest.raises(ValueError):
        fake_rerank_service.compute_scores(query, bad_candidates)
