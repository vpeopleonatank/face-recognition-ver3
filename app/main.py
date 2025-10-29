"""FastAPI application factory for the face search backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request
from loguru import logger

from app.api.v1.routes import router
from app.core.config import Settings, get_settings
from app.services.rerank import RerankServiceProtocol
from app.services.triton_client import TritonClientProtocol


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
        app.state.triton_client = None  # placeholder for future Triton client
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
