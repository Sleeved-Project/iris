from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session

from app.controllers import (
    api_info_controller,
    hash_controller,
    health_controller,
    root_controller,
    scan_controller,
    analysis_controller,
    analysis_controller_v3,
)

from app.db.session import get_db

from app.dependencies.image_request_validators import (
    validate_image_url,
    validate_image_upload,
    ValidationResult,
)

from app.dependencies.scan_request_validators import validate_scan_image_upload
from app.dependencies.analysis_request_validators import validate_analysis_image_upload

from app.schemas.image_schemas import ImageHashResponse, ImageHashRequest
from app.schemas.scan_schemas import ScanResponse
from app.schemas.analysis_schemas import AnalysisResponse


# Routers par fonction
root_router = APIRouter(tags=["root"])
health_router = APIRouter(tags=["health"])
api_v1_router = APIRouter(prefix="/api/v1", tags=["api"])
images_router = APIRouter(prefix="/api/v1/images", tags=["images"])
images_router_v3 = APIRouter(prefix="/api/v3/images", tags=["images"])


# ------------------- Root ------------------- #
@root_router.get("/")
def root():
    return root_controller.get_root()


# ------------------- Health ------------------- #
@health_router.get("/health")
def health(db: Session = Depends(get_db)):
    return health_controller.check_health(db)


# ------------------- API Info ------------------- #
@api_v1_router.get("/")
def api_info():
    return api_info_controller.get_api_info()


# ------------------- Scan ------------------- #
@images_router.post("/scan/detect", response_model=ScanResponse)
async def scan_image_file(file: UploadFile = File(...), debug: bool = False):
    """
    Detect and extract card-like objects from an uploaded image file.
    """
    validated_input = await validate_scan_image_upload(file)
    return await scan_controller.detect_card(
        validated_input=validated_input, debug=debug
    )


# ------------------- Image Hashing ------------------- #
@images_router.post("/hash/url", response_model=ImageHashResponse)
async def hash_image_url(request: ImageHashRequest):
    """
    Calculate perceptual hashes for an image from a URL.
    """
    url = await validate_image_url(str(request.url))
    validated_input = ValidationResult(url=url)
    return await hash_controller.hash_image(validated_input=validated_input)


@images_router.post("/hash/file", response_model=ImageHashResponse)
async def hash_image_file(file: UploadFile = File(...)):
    """
    Calculate perceptual hashes for an uploaded image file.
    """
    validated_file = await validate_image_upload(file)
    validated_input = ValidationResult(file=validated_file)
    return await hash_controller.hash_image(validated_input=validated_input)


# ------------------- Card Analysis ------------------- #
@images_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image_file(
    file: UploadFile = File(...),
    debug: bool = False,
    db: Session = Depends(get_db),
):
    """
    Detects card-like objects in an uploaded image, extracts them,
    and analyzes them using perceptual hashing.
    """
    validated_input = await validate_analysis_image_upload(file)
    return await analysis_controller.analyze_image(
        validated_input=validated_input, debug=debug, db=db
    )


# ------------------- Card Analysis V3 ------------------- #
@images_router_v3.post("/analyze", response_model=AnalysisResponse)
async def analyze_image_file_v3(
    file: UploadFile = File(...),
    debug: bool = False,
    db: Session = Depends(get_db),
):
    """
    Detects card-like objects in an uploaded image, extracts them,
    and analyzes them using perceptual hashing.
    """
    validated_input = await validate_analysis_image_upload(file)
    return await analysis_controller_v3.analyze_image(
        validated_input=validated_input, debug=debug, db=db
    )


# ------------------- Register Routers ------------------- #
def include_routes(app):
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(api_v1_router)
    app.include_router(images_router)
    app.include_router(images_router_v3)  # <--- AJOUT ESSENTIEL
