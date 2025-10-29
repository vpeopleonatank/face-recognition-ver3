"""Triton client service placeholder.

This module will house the gRPC client wrappers for detection and embedding
models provided by the Triton inference server. It is intentionally left
minimal for Phase 1 scaffolding.
"""

from typing import Protocol


class TritonClientProtocol(Protocol):
    """Interface for future Triton client implementations."""

    async def warmup(self) -> None:  # pragma: no cover - placeholder
        """Optional warmup hook to prime connections."""
        ...

