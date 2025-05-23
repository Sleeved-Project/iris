from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.controllers import api_info_controller, health_controller, root_controller
from app.db.session import get_db

# Create routers
root_router = APIRouter(tags=["root"])
health_router = APIRouter(tags=["health"])
api_v1_router = APIRouter(prefix="/api/v1", tags=["api"])


@root_router.get("/")
def root():
    return root_controller.get_root()


@health_router.get("/health")
def health(db: Session = Depends(get_db)):
    return health_controller.check_health(db)


@api_v1_router.get("/")
def api_info():
    return api_info_controller.get_api_info()


def include_routes(app):
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(api_v1_router)
