from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings

router = APIRouter()


@router.get("/healthz")
async def read_health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Lightweight health check placeholder."""
    return {"status": "ok", "environment": settings.env}

