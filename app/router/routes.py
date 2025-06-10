from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session

from app.controllers import (
    api_info_controller,
    hash_controller,
    health_controller,
    root_controller,
    scan_controller,
)
from app.db.session import get_db

from app.dependencies.image_request_validators import (
    validate_image_url,
    validate_image_upload,
    ValidationResult as ImageValidationResult,
)

from app.dependencies.scan_request_validators import (
    validate_scan_image_upload,
)

from app.schemas.image_schemas import ImageHashResponse, ImageHashRequest
from app.schemas.scan_schemas import ScanResponse


root_router = APIRouter(tags=["root"])
health_router = APIRouter(tags=["health"])
api_v1_router = APIRouter(prefix="/api/v1", tags=["api"])
images_router = APIRouter(prefix="/api/v1/images", tags=["images"])


@root_router.get("/")
def root():
    return root_controller.get_root()


@health_router.get("/health")
def health(db: Session = Depends(get_db)):
    return health_controller.check_health(db)


@api_v1_router.get("/")
def api_info():
    return api_info_controller.get_api_info()


@images_router.post("/scan/file", response_model=ScanResponse)
async def scan_image_file(file: UploadFile = File(...), debug: bool = False):
    """
    Detect and extract card-like objects from an uploaded image file.

    Upload a file using multipart/form-data.

    Args:
        file: Image file (JPG, PNG, ...)
        debug: Optional flag to save debug images (default False)
    """
    validated_input = await validate_scan_image_upload(file)
    return await scan_controller.scan_image(
        validated_input=validated_input, debug=debug
    )


@images_router.post("/hash/url", response_model=ImageHashResponse)
async def hash_image_url(request: ImageHashRequest):
    """
    Calculate perceptual hashes for an image from URL.

    Send a JSON object with the URL:
    ```json
    {
        "url": "https://example.com/image.jpg"
    }
    ```

    The URL must use the HTTPS protocol.
    """
    # Validate the URL (raises HTTPException if invalid)
    url = await validate_image_url(str(request.url))
    validated_input = ImageValidationResult(url=url)
    return await hash_controller.hash_image(validated_input=validated_input)


@images_router.post("/hash/file", response_model=ImageHashResponse)
async def hash_image_file(file: UploadFile = File(...)):
    """
    Calculate perceptual hashes for an uploaded image file.

    Upload a file using multipart/form-data.

    Max file size: 5MB
    Supported formats: JPG, PNG, GIF, WEBP
    """
    validated_file = await validate_image_upload(file)
    validated_input = ImageValidationResult(file=validated_file)
    return await hash_controller.hash_image(validated_input=validated_input)


def include_routes(app):
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(api_v1_router)
    app.include_router(images_router)
