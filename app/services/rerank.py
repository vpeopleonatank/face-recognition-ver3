"""Rerank service placeholder.

In later phases this module will provide a thin wrapper around the
`librerank_compute.so` shared library. For now we only define the interface
used by the FastAPI dependency wiring.
"""

from typing import Protocol, Sequence

import numpy as np


class RerankServiceProtocol(Protocol):
    """Interface for the rerank scoring service."""

    def compute_scores(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Sequence[np.ndarray],
    ) -> Sequence[float]:  # pragma: no cover - placeholder
        """Return scores for each candidate embedding."""
        ...

