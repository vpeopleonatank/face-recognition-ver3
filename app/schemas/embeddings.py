"""Pydantic schemas for the /embeddings endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingImagePayload(BaseModel):
    """Request payload describing a single image to embed."""

    id: str | None = Field(
        default=None,
        description="Optional caller-supplied identifier for the image.",
    )
    data: str = Field(
        ...,
        description="Base64-encoded image bytes. Data URI prefixes are accepted.",
        min_length=16,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("data")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Ensure the provided string resembles base64 without decoding yet."""
        if "," in value:
            value = value.split(",", 1)[1]
        stripped = value.strip()
        if len(stripped) % 4 != 0:
            msg = "Base64 image data length must be a multiple of four"
            raise ValueError(msg)
        return value


class EmbeddingsRequest(BaseModel):
    """Request body for the /embeddings endpoint."""

    images: list[EmbeddingImagePayload] = Field(
        ...,
        description="Collection of base64-encoded images to process.",
        min_length=1,
    )
    return_aligned_faces: bool | None = Field(
        default=None,
        description="Override to include aligned face crops in the response.",
    )
    skip_embedding_normalization: bool | None = Field(
        default=None,
        description="Override to skip L2-normalization on the embedding vectors.",
    )

    model_config = ConfigDict(extra="forbid")


class FaceEmbeddingPayload(BaseModel):
    """Response payload describing a detected face embedding."""

    bbox: list[float] = Field(
        ...,
        description="Bounding box [xmin, ymin, xmax, ymax] in source image pixels.",
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(
        ..., description="Detection confidence score returned by the model."
    )
    landmarks: list[list[float]] = Field(
        ...,
        description="Facial keypoints [[x, y] ...] corresponding to the detection.",
        min_length=5,
        max_length=5,
    )
    face_size: float = Field(
        ..., description="Approximate face area in pixel^2 derived from the bounding box."
    )
    embedding: list[float] = Field(
        ...,
        description="1024-dimensional face embedding vector.",
        min_length=1,
    )
    aligned_face: str | None = Field(
        default=None,
        description="Base64-encoded aligned face crop, included when requested.",
    )

    model_config = ConfigDict(extra="forbid")


class ImageEmbeddingsResponse(BaseModel):
    """Per-image embeddings response."""

    image_id: str = Field(
        ...,
        description="Identifier for the processed image (caller-supplied or generated).",
    )
    faces: list[FaceEmbeddingPayload] = Field(
        ...,
        description="Faces detected within the image along with embeddings.",
    )
    num_faces: int = Field(
        ..., description="Total number of faces detected in the image."
    )

    model_config = ConfigDict(extra="forbid")


class EmbeddingsResponse(BaseModel):
    """Top-level response for embedding generation."""

    results: list[ImageEmbeddingsResponse] = Field(
        ..., description="Embedding results grouped by source image."
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate timing metrics (decode, inference, total).",
    )
    normalized_embeddings: bool = Field(
        ...,
        description="Whether embeddings returned were L2-normalized.",
    )
    included_aligned_faces: bool = Field(
        ...,
        description="Whether aligned face crops were included in the response.",
    )

    model_config = ConfigDict(extra="forbid")
