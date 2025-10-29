"""Rerank service backed by the C++ shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from loguru import logger


class RerankServiceProtocol(Protocol):
    """Interface for the rerank scoring service."""

    @property
    def default_threshold(self) -> float:  # pragma: no cover - interface definition
        """Default rerank threshold applied when requests do not override it."""
        ...

    @property
    def embedding_dimension(self) -> int:  # pragma: no cover - interface definition
        """Embedding dimension expected by the underlying implementation."""
        ...

    def compute_scores(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Sequence[np.ndarray] | np.ndarray,
        *,
        threshold: float | None = None,
    ) -> Sequence[float]:  # pragma: no cover - interface definition
        """Return rerank score for each candidate embedding."""
        ...


class RerankService(RerankServiceProtocol):
    """Python wrapper around the `librerank_compute.so` shared library."""

    _FUNC_NAME = "compute_rerank_scores_from_embeddings"

    def __init__(
        self,
        library_path: str,
        *,
        embedding_dimension: int = 1024,
        default_threshold: float = 0.5,
    ) -> None:
        self._embedding_dim = int(embedding_dimension)
        self._default_threshold = float(default_threshold)
        self._library_path = self._resolve_library_path(library_path)
        logger.info("Loading rerank library from {}", self._library_path)

        try:
            self._lib = ctypes.CDLL(str(self._library_path))
        except OSError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"Failed to load rerank library: {self._library_path}") from exc

        self._compute = getattr(self._lib, self._FUNC_NAME, None)
        if self._compute is None:
            raise AttributeError(
                f"Shared library missing expected symbol '{self._FUNC_NAME}'"
            )

        self._compute.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # query_embedding
            ctypes.POINTER(ctypes.c_double),  # candidate_embeddings
            ctypes.c_int,  # num_candidates
            ctypes.c_double,  # threshold
            ctypes.POINTER(ctypes.c_double),  # rerank_scores output
        ]
        self._compute.restype = ctypes.c_int

    @property
    def embedding_dimension(self) -> int:
        """Embedding dimension expected by the C++ implementation."""
        return self._embedding_dim

    @property
    def default_threshold(self) -> float:
        """Default rerank threshold applied when requests do not override it."""
        return self._default_threshold

    def compute_scores(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Sequence[np.ndarray] | np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Compute rerank scores using the C++ shared library."""
        if self._compute is None:
            raise RuntimeError("Rerank library is not initialized")

        query = self._validate_vector(query_embedding, name="query_embedding")
        candidates = self._prepare_candidates(candidate_embeddings)
        num_candidates = candidates.shape[0]

        if num_candidates == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros(num_candidates, dtype=np.float64)

        query_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        candidates_ptr = candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        applied_threshold = float(self._default_threshold if threshold is None else threshold)

        result_code = self._compute(
            query_ptr,
            candidates_ptr,
            ctypes.c_int(num_candidates),
            ctypes.c_double(applied_threshold),
            scores_ptr,
        )

        if result_code != 0:
            raise RuntimeError(f"Rerank computation failed with status {result_code}")

        return scores

    def _prepare_candidates(
        self, candidate_embeddings: Sequence[np.ndarray] | np.ndarray
    ) -> np.ndarray:
        """Validate and stack candidate embeddings into contiguous float64 array."""
        array = np.asarray(candidate_embeddings)
        if array.size == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float64)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError("Candidate embeddings must form a 2D array")

        if array.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Expected candidate embeddings with dimension {self._embedding_dim}, "
                f"received {array.shape[1]}"
            )

        return np.ascontiguousarray(array, dtype=np.float64)

    def _validate_vector(self, vector: np.ndarray, *, name: str) -> np.ndarray:
        """Ensure the provided vector matches expected dimensionality."""
        array = np.asarray(vector)
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        if array.shape[0] != self._embedding_dim:
            raise ValueError(
                f"{name} must have dimension {self._embedding_dim}, "
                f"received {array.shape[0]}"
            )
        return np.ascontiguousarray(array, dtype=np.float64)

    @staticmethod
    def _resolve_library_path(library_path: str) -> Path:
        """Resolve the shared library path, supporting relative paths."""
        candidates: list[Path] = []
        provided = Path(library_path).expanduser()
        candidates.append(provided)

        if not provided.is_absolute():
            root = Path(__file__).resolve().parents[2]
            candidates.extend(
                [
                    (root / provided).resolve(),
                    (Path.cwd() / provided).resolve(),
                ]
            )
            module_dir = Path(__file__).resolve().parent
            candidates.append((module_dir / provided.name).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Unable to locate rerank shared library. Checked: {', '.join(str(c) for c in candidates)}"
        )
