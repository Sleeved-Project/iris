from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/")
def get_root():
    return {
        "name": "iris",
        "description": settings.PROJECT_DESCRIPTION,
        "versions": [
            {"version": "v1", "url": "/api/v1", "status": "current"},
        ],
        "status": "WIP",
        "documentation": "https://sleeved.atlassian.net/wiki/x/A4BP",
    }
