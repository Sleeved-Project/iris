from typing import Optional
from fastapi import File, UploadFile, HTTPException

from app.core.constants import SUPPORTED_FORMATS, MAX_FILE_SIZE
from app.utils.validators.image_validators import validate_image_mime_type
from app.utils.validators.file_validators import validate_file_size


class ScanValidationResult:
    """Container for validated scan image file."""

    def __init__(self, file: UploadFile, filename: str):
        self.file = file
        self.filename = filename


def raise_scan_validation_error(status_code: int, message: str) -> None:
    """Raises an HTTPException for scan validation errors."""
    raise HTTPException(status_code=status_code, detail=message)


async def validate_scan_image_format(mime_type: str, filename: Optional[str]) -> None:
    """Validates file format and extension."""
    if not mime_type:
        raise_scan_validation_error(400, "Missing content type")

    if not validate_image_mime_type(mime_type):
        raise_scan_validation_error(400, "Invalid file type. Only images are accepted.")

    content_format = mime_type.split("/")[-1].lower()
    filename_format = filename.split(".")[-1].lower() if filename else ""

    if not (
        content_format in SUPPORTED_FORMATS or filename_format in SUPPORTED_FORMATS
    ):
        raise_scan_validation_error(
            400,
            f"Unsupported format. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )


async def validate_scan_image_upload(
    file: Optional[UploadFile] = File(None),
) -> ScanValidationResult:
    """
    Validates uploaded file and returns a ScanValidationResult.

    Args:
        file: UploadFile from FastAPI

    Returns:
        ScanValidationResult

    Raises:
        HTTPException if file is invalid
    """
    if not file:
        raise_scan_validation_error(400, "No file uploaded")

    await validate_scan_image_format(file.content_type, file.filename)

    total_size = 0
    chunk_size = 8192

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_size += len(chunk)
        if not validate_file_size(total_size, MAX_FILE_SIZE):
            await file.seek(0)
            raise_scan_validation_error(
                413, f"Image too large (max {MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
            )

    await file.seek(0)

    return ScanValidationResult(file=file, filename=file.filename)
