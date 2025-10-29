"""Application configuration powered by pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven app configuration."""

    env: Literal["local", "staging", "production"] = Field(
        default="local", alias="environment"
    )
    app_name: str = "Face Search Backend"
    api_prefix: str = "/api/v1"

    triton_url: str = "localhost:8001"
    max_batch_size: int = 8
    request_timeout_seconds: float = 30.0

    rerank_threshold: float = 0.5
    return_aligned: bool = False
    skip_embedding_normalization: bool = False

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="FACE_V3_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @computed_field
    @property
    def debug(self) -> bool:
        """Whether the application should run in debug mode."""
        return self.env == "local"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()  # type: ignore[call-arg]

