"""Schemas for the /rerank endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RerankCandidatePayload(BaseModel):
    """Candidate embedding supplied for reranking."""

    id: str = Field(..., description="Caller-supplied identifier for the candidate.")
    embedding: list[float] = Field(
        ...,
        description="Embedding vector associated with the candidate.",
        min_length=1,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        """Ensure candidate embedding is not empty."""
        if not value:
            msg = "Candidate embedding must contain at least one value"
            raise ValueError(msg)
        return value


class RerankRequest(BaseModel):
    """Request payload for rerank score computation."""

    query_embedding: list[float] = Field(
        ...,
        description="Query embedding vector.",
        min_length=1,
    )
    candidates: list[RerankCandidatePayload] = Field(
        ...,
        description="List of candidate embeddings to score.",
        min_length=1,
    )
    threshold: float | None = Field(
        default=None,
        description="Optional override for the rerank threshold.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("query_embedding")
    @classmethod
    def validate_query(cls, value: list[float]) -> list[float]:
        """Ensure query embedding is not empty."""
        if not value:
            msg = "Query embedding must contain at least one value"
            raise ValueError(msg)
        return value


class RerankScorePayload(BaseModel):
    """Scored candidate entry in the response."""

    id: str = Field(..., description="Identifier of the candidate.")
    score: float = Field(..., ge=0.0, le=1.0, description="Rerank score in range [0, 1].")

    model_config = ConfigDict(extra="forbid")


class RerankResponse(BaseModel):
    """Response payload for rerank computation."""

    scores: list[RerankScorePayload] = Field(
        ...,
        description="Scores aligned with the order of supplied candidates.",
    )
    threshold_applied: float = Field(
        ...,
        description="Threshold used for the rerank computation.",
    )
    candidate_count: int = Field(
        ..., description="Total number of candidates evaluated."
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics for rerank processing.",
    )

    model_config = ConfigDict(extra="forbid")

