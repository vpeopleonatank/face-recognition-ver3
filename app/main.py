"""FastAPI application factory for the face search backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request
from loguru import logger

from app.api.v1.routes import router
from app.core.config import Settings, get_settings
from app.services.rerank import RerankServiceProtocol
from app.services.triton_client import ModelIOConfig, TritonClient, TritonClientProtocol


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    settings = get_settings()
    lifespan = _lifespan_context(settings=settings)
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    app.include_router(router, prefix=settings.api_prefix)
    return app


def _lifespan_context(*, settings: Settings):
    """Create lifespan context manager to manage startup/shutdown resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting {}", settings.app_name)

        app.state.settings = settings
        triton_client = _create_triton_client(settings)
        if triton_client:
            await triton_client.warmup()
        app.state.triton_client = triton_client
        app.state.rerank_service = None  # placeholder for future rerank service

        yield

        triton_client = getattr(app.state, "triton_client", None)
        if hasattr(triton_client, "close"):
            try:
                triton_client.close()  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - best effort shutdown
                logger.warning("Failed to close Triton client cleanly: {}", exc)

        logger.info("Shutdown complete for {}", settings.app_name)

    return lifespan


def _create_triton_client(settings: Settings) -> TritonClient | None:
    """Instantiate the Triton client using application settings."""
    try:
        detection_model = ModelIOConfig.detection_defaults()
        detection_model.name = settings.detection_model_name

        extraction_model = ModelIOConfig.extraction_defaults()
        extraction_model.name = settings.extraction_model_name

        logger.info(
            "Connecting to Triton server at {} for models det='{}' ext='{}'",
            settings.triton_url,
            detection_model.name,
            extraction_model.name,
        )

        return TritonClient(
            url=settings.triton_url,
            detection_model=detection_model,
            extraction_model=extraction_model,
            detection_input_size=settings.detection_input_size,
            request_timeout=settings.request_timeout_seconds,
            healthcheck=settings.triton_healthcheck,
        )
    except Exception as exc:  # pragma: no cover - startup failure
        logger.error("Failed to initialize Triton client: {}", exc)
        raise


def get_app_settings(request: Request) -> Settings:
    """Provide Settings instance stored on the FastAPI application state."""
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):  # pragma: no cover - defensive
        raise RuntimeError("Application settings are not initialized")
    return settings


def get_triton_client(
    request: Request, _: Settings = Depends(get_app_settings)
) -> TritonClientProtocol:
    """Dependency placeholder for the Triton client."""
    client: Optional[TritonClientProtocol] = getattr(request.app.state, "triton_client", None)
    if client is None:
        raise RuntimeError("Triton client has not been initialized")
    return client


def get_rerank_service(
    request: Request, _: Settings = Depends(get_app_settings)
) -> RerankServiceProtocol:
    """Dependency placeholder for the rerank service."""
    service: Optional[RerankServiceProtocol] = getattr(
        request.app.state, "rerank_service", None
    )
    if service is None:
        raise RuntimeError("Rerank service has not been initialized")
    return service


app = create_app()
