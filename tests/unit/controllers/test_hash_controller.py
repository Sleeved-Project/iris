from unittest.mock import AsyncMock, patch
import pytest

from app.controllers.hash_controller import hash_image
from app.dependencies.image_request_validators import ValidationResult
from app.schemas.image_schemas import ImageHashResponse


@pytest.fixture
def url_validation_result():
    result = ValidationResult()
    result.url = "https://example.com/image.jpg"
    result.file = None
    return result


@pytest.fixture
def file_validation_result():
    result = ValidationResult()
    result.url = None
    result.file = "mock_file_object"
    return result


@pytest.mark.asyncio
@patch("app.controllers.hash_controller.image_hash_service")
async def test_hash_image_with_url(mock_service, url_validation_result):
    mock_service.hash_from_url = AsyncMock(return_value="fake_hash_1234")

    response = await hash_image(validated_input=url_validation_result)

    assert isinstance(response, ImageHashResponse)
    assert response.hash == "fake_hash_1234"
    mock_service.hash_from_url.assert_called_once_with(url_validation_result.url)

    mock_service.hash_from_file.assert_not_called()
