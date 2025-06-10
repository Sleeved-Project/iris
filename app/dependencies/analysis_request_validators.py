# app/dependencies/analysis_request_validators.py
from fastapi import HTTPException, status, UploadFile
from typing import Union
from app.dependencies.image_request_validators import validate_image_upload, ValidationResult

async def validate_analysis_image_upload(file: UploadFile) -> ValidationResult:
    """
    Validates an uploaded image file for analysis.
    Reuses image_request_validators for common image validations (size, format).
    """
    # This leverages the existing image upload validation
    validated_file = await validate_image_upload(file)
    return ValidationResult(file=validated_file)