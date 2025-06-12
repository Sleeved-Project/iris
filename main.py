import logging
import time

from fastapi import FastAPI

from app.core.config import settings
from app.router.routes import include_routes

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
)

# Include all routes
include_routes(app)
