from datetime import datetime

from app.core.config import settings


def get_api_info():
    return {
        "name": "iris",
        "description": "Image recognition microservice for the Sleeved ecosystem",
        "version": settings.PROJECT_VERSION,
        "timestamp": datetime.now().isoformat(),
        "status": "WIP",
    }
