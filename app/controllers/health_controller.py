from fastapi import Response, status
from sqlalchemy.orm import Session

from app.services.health_service import check_database_connection


def check_health(db: Session):
    health_status = check_database_connection(db)

    if health_status["is_healthy"]:
        return health_status["data"]

    return Response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=str(health_status["data"]),
    )
