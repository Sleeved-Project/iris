from typing import Optional
from fastapi import Depends, File, HTTPException, Query, UploadFile

from app.core.constants import SUPPORTED_FORMATS, MAX_FILE_SIZE
from app.utils.validators.url_validators import validate_url_scheme
from app.utils.validators.image_validators import validate_image_mime_type
from app.utils.validators.file_validators import validate_file_size


class ValidationResult:
    """Container for validated image source."""

    def __init__(self, url: Optional[str] = None, file: Optional[UploadFile] = None):
        self.url = url
        self.file = file


def raise_validation_error(status_code: int, message: str) -> None:
    """Centralized validation error handler."""
    raise HTTPException(status_code=status_code, detail=message)


async def validate_image_format(mime_type: str, filename: Optional[str]) -> None:
    """Validates image format based on mime type and filename."""
    if not mime_type:
        raise_validation_error(400, "Missing content type")

    if not validate_image_mime_type(mime_type):
        raise_validation_error(400, "Invalid file type. Only images are accepted.")

    format_from_content = mime_type.split("/")[-1].lower()
    format_from_filename = filename.split(".")[-1].lower() if filename else ""

    if not (
        format_from_content in SUPPORTED_FORMATS
        or format_from_filename in SUPPORTED_FORMATS
    ):
        raise_validation_error(
            400,
            (
                "Unsupported image format. Supported formats: "
                f"{', '.join(SUPPORTED_FORMATS)}"
            ),
        )


async def validate_image_upload(
    file: Optional[UploadFile] = File(None),
) -> Optional[UploadFile]:
    """
    Validates an uploaded image file.

    Args:
        file: The uploaded file to validate

    Returns:
        The validated file or None

    Raises:
        HTTPException: If the file is invalid
    """
    if not file:
        return None

    # Validate file format
    await validate_image_format(file.content_type, file.filename)

    # Check if we can get size from headers (more efficient)
    if hasattr(file, "size") and file.size is not None:
        if not validate_file_size(file.size, MAX_FILE_SIZE):
            raise_validation_error(
                413,
                f"Image too large (max {MAX_FILE_SIZE/1024/1024:.1f}MB)",
            )
    else:
        # FastAPI's UploadFile doesn't provide size info directly
        # Read the file in chunks to check size without loading entire file
        total_size = 0
        chunk_size = 8192  # 8KB chunks

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break

            total_size += len(chunk)

            # Check size after each chunk
            if not validate_file_size(total_size, MAX_FILE_SIZE):
                await file.seek(0)  # Reset position
                raise_validation_error(
                    413,
                    f"Image too large (max {MAX_FILE_SIZE/1024/1024:.1f}MB)",
                )

        # Reset file position after reading
        await file.seek(0)

    return file


async def validate_image_url(url: Optional[str] = Query(None)) -> Optional[str]:
    """
    Validates an image URL.

    Args:
        url: The URL to validate

    Returns:
        The validated URL or None

    Raises:
        HTTPException: If the URL is invalid
    """
    if not url:
        return None

    if not validate_url_scheme(url):
        raise_validation_error(400, "Invalid URL scheme. Only HTTPS is supported.")

    return url


async def validate_image_request(
    url: Optional[str] = Depends(validate_image_url),
    file: Optional[UploadFile] = Depends(validate_image_upload),
) -> ValidationResult:
    """
    Validates and returns an image source (URL or file).

    Args:
        url: The image URL to validate
        file: The uploaded file to validate

    Returns:
        ValidationResult containing either the URL or file

    Raises:
        HTTPException: If validation fails
    """
    if url and file:
        raise_validation_error(
            400,
            "Ambiguous request: provide either image URL or file, not both",
        )

    if not url and not file:
        raise_validation_error(
            400,
            "Either image URL or file must be provided",
        )

    return ValidationResult(url=url, file=file)
