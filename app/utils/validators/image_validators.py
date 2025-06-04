from app.core.constants import SUPPORTED_FORMATS


def validate_image_mime_type(content_type: str) -> bool:
    """
    Validates that the MIME type corresponds to an image.

    Args:
        content_type: The MIME type to validate

    Returns:
        bool: True if the MIME type is that of an image, False otherwise
    """
    # Check that content_type exists and is not empty
    if not content_type:
        return False

    # Check that it's a valid image MIME type and has a subtype after the /
    parts = content_type.split("/")
    return len(parts) == 2 and parts[0] == "image" and bool(parts[1])


def validate_image_format(format: str) -> bool:
    """
    Validates that an image format is supported.

    Args:
        format: The image format to validate

    Returns:
        bool: True if the format is supported, False otherwise
    """
    return format.lower() in SUPPORTED_FORMATS
