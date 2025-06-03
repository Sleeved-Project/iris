from fastapi import HTTPException

from app.services.image_hash_service import image_hash_service
from app.schemas.image_schemas import ImageHashResponse
from app.dependencies.image_request_validators import ValidationResult


async def hash_image(validated_input: ValidationResult) -> ImageHashResponse:
    """
    Calculate perceptual hashes for a validated image source.

    Args:
        validated_input: Container with validated URL or file

    Returns:
        ImageHashResponse with hash value

    Raises:
        HTTPException: If processing fails
    """
    try:
        if validated_input.file:
            hash_value = await image_hash_service.hash_from_file(validated_input.file)
        else:
            hash_value = await image_hash_service.hash_from_url(validated_input.url)

        return ImageHashResponse(hash=hash_value)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
