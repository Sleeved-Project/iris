import pytest
from app.utils.validators.image_validators import (
    validate_image_mime_type,
    validate_image_format,
)
from app.core.constants import SUPPORTED_FORMATS


@pytest.mark.parametrize(
    "mime_type, expected",
    [
        # Valid image MIME types
        ("image/jpeg", True),
        ("image/png", True),
        ("image/gif", True),
        ("image/webp", True),
        # Invalid MIME types
        ("text/plain", False),
        ("application/pdf", False),
        ("video/mp4", False),
        # Edge cases
        ("", False),
        (None, False),
        ("image", False),
        ("image/", False),
    ],
)
def test_validate_image_mime_type(mime_type, expected):
    assert validate_image_mime_type(mime_type) is expected


@pytest.mark.parametrize(
    "format, expected",
    [
        # Valid formats - test a few directly
        ("jpg", True),
        ("png", True),
        ("webp", True),
        ("JPG", True),  # Test case insensitivity
        # Invalid formats
        ("tiff", False),
        ("raw", False),
        ("pdf", False),
    ],
)
def test_validate_image_format(format, expected):
    assert validate_image_format(format) is expected


@pytest.mark.parametrize("format", SUPPORTED_FORMATS)
def test_all_supported_formats(format):
    assert validate_image_format(format) is True
    assert validate_image_format(format.upper()) is True  # Case insensitivity


def test_validate_image_format_none():
    with pytest.raises(AttributeError):
        validate_image_format(None)
