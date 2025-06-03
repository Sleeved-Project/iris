from unittest.mock import MagicMock, patch
import pytest
from fastapi import UploadFile
from PIL import Image

from app.services.image_hash_service import image_hash_service


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def mock_file():
    mock = MagicMock(spec=UploadFile)
    mock.content_type = "image/jpeg"
    mock.filename = "test.jpg"
    return mock


def test_calculate_hashes(sample_image):
    result = image_hash_service.calculate_hashes(sample_image)

    assert isinstance(result, str)
    assert len(result) == 128


@pytest.mark.asyncio
@patch(
    "app.services.image_extraction_service.image_extraction_service.get_image_from_url"
)
async def test_hash_from_url(mock_get_image, sample_image):
    mock_get_image.return_value = sample_image
    url = "https://example.com/image.jpg"

    result = await image_hash_service.hash_from_url(url)

    assert isinstance(result, str)
    assert len(result) == 128
    mock_get_image.assert_called_once_with(url)


@pytest.mark.asyncio
@patch(
    "app.services.image_extraction_service.image_extraction_service.get_image_from_file"
)
async def test_hash_from_file(mock_get_image, mock_file, sample_image):
    mock_get_image.return_value = sample_image

    result = await image_hash_service.hash_from_file(mock_file)

    assert isinstance(result, str)
    assert len(result) == 128
    mock_get_image.assert_called_once_with(mock_file)
