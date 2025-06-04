import io
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.core.constants import MAX_FILE_SIZE
from app.services.image_extraction_service import (
    ImageExtractionService,
    image_extraction_service,
)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def mock_url_response():
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/jpeg", "content-length": "100"}
    mock_response.content = b"fake_image_content"
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
def mock_file():
    mock = MagicMock(spec=UploadFile)
    mock.content_type = "image/jpeg"
    mock.filename = "test.jpg"
    mock.read = AsyncMock(return_value=b"fake_image_content")
    mock.seek = AsyncMock()
    return mock


def test_load_image_from_bytes():
    img = Image.new("RGB", (10, 10), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    loaded_img = ImageExtractionService._load_image_from_bytes(img_bytes)
    assert isinstance(loaded_img, Image.Image)


@pytest.mark.asyncio
@patch("app.services.image_extraction_service.requests.get")
async def test_get_image_from_url_success(mock_get, mock_url_response, sample_image):
    mock_get.return_value = mock_url_response
    test_url = "https://images.pokemontcg.io/base1/1.png"

    with patch(
        "app.services.image_extraction_service.Image.open", return_value=sample_image
    ):
        image = await image_extraction_service.get_image_from_url(test_url)

    assert isinstance(image, Image.Image)
    mock_get.assert_called_once_with(test_url, stream=True, timeout=10)


@pytest.mark.asyncio
@patch("app.services.image_extraction_service.requests.get")
async def test_get_image_from_url_too_large(mock_get):
    mock_response = MagicMock()
    mock_response.headers = {
        "content-type": "image/jpeg",
        "content-length": str(MAX_FILE_SIZE + 1),
    }
    mock_get.return_value = mock_response

    with pytest.raises(HTTPException) as excinfo:
        await image_extraction_service.get_image_from_url(
            "https://example.com/large.jpg"
        )

    assert excinfo.value.status_code == 413
    assert "Image too large" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_get_image_from_file_success(mock_file, sample_image):
    with patch(
        "app.services.image_extraction_service.Image.open", return_value=sample_image
    ):
        image = await image_extraction_service.get_image_from_file(mock_file)

    assert isinstance(image, Image.Image)
    mock_file.read.assert_called_once()
    mock_file.seek.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_get_image_from_file_too_large(mock_file):
    mock_file.read = AsyncMock(return_value=b"x" * (MAX_FILE_SIZE + 1))

    with pytest.raises(HTTPException) as excinfo:
        await image_extraction_service.get_image_from_file(mock_file)

    assert excinfo.value.status_code == 413
    assert "Image too large" in str(excinfo.value.detail)
