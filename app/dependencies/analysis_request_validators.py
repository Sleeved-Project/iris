from fastapi import UploadFile
from app.dependencies.image_request_validators import (
    validate_image_upload,
    ValidationResult,
)


async def validate_analysis_image_upload(file: UploadFile) -> ValidationResult:
    """
    Validates an uploaded image file for analysis.
    Reuses image_request_validators for common image validations (size, format).
    """
    # This leverages the existing image upload validation
    validated_file = await validate_image_upload(file)
    return ValidationResult(file=validated_file)
