from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings


def check_database_connection(db: Session):
    try:
        db.execute(text("SELECT 1"))
        return {
            "is_healthy": True,
            "data": {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat(),
                "version": settings.PROJECT_VERSION,
            },
        }
    except Exception as e:
        return {
            "is_healthy": False,
            "data": {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        }
