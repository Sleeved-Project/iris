from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.constants import MAX_FILE_SIZE

client = TestClient(app)


def test_hash_image_with_public_url():
    test_url = "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"

    response = client.post("/api/v1/images/hash/url", json={"url": test_url})

    assert response.status_code == 200
    result = response.json()
    assert "hash" in result
    assert result["hash"] == (
        "0000000010000052118096573d533cd319d7194b91ab10100128100000000000"
        "bf81d8c6d07e27292fa4d8d41f83390ed07e26b92f8cd9461ac336c9c53c6738"
    )


def test_hash_image_with_real_file():
    assets_dir = Path(__file__).parent.parent.parent / "assets"
    test_image_path = assets_dir / "test_card.png"

    if not test_image_path.exists():
        pytest.skip(f"Test image not found at {test_image_path}")

    with open(test_image_path, "rb") as f:
        file_content = f.read()

    response = client.post(
        "/api/v1/images/hash/file",
        files={"file": ("test_card.png", file_content, "image/png")},
    )

    assert response.status_code == 200
    result = response.json()
    assert "hash" in result
    assert result["hash"] == (
        "25992d9b0cc736931f172d4b18c334c73bc35c0713c723c360e3346733271327e"
        "af57e31c0fbf13bc30a934cdd08cdc89dc24e428e37d839e7413275e41e701c"
    )


def test_hash_image_with_invalid_url_scheme():
    test_url = "http://example.com/image.jpg"

    # Use the new /hash/url endpoint
    response = client.post("/api/v1/images/hash/url", json={"url": test_url})

    assert response.status_code == 400
    assert "Invalid URL scheme" in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint, payload, expected_status, expected_message",
    [
        ("/api/v1/images/hash/url", {"json": {}}, 400, "URL"),
        ("/api/v1/images/hash/file", {}, 422, "Field required"),
    ],
)
def test_hash_image_with_missing_inputs(
    endpoint, payload, expected_status, expected_message
):
    """Test behavior when required inputs are missing."""
    response = client.post(endpoint, **payload)
    assert response.status_code == expected_status

    if expected_status == 422:
        assert expected_message in str(response.json())
    else:
        assert expected_message in response.json().get("detail", "")


def test_hash_image_with_invalid_file_type():
    text_content = b"This is not an image file"

    response = client.post(
        "/api/v1/images/hash/file",
        files={"file": ("test.txt", text_content, "text/plain")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_hash_image_with_unsupported_format():
    fake_tiff = b"II*\x00\x08\x00\x00\x00"

    response = client.post(
        "/api/v1/images/hash/file",
        files={"file": ("test.tiff", fake_tiff, "image/tiff")},
    )

    assert response.status_code == 400
    assert (
        "Unsupported image format" in response.json()["detail"]
        or "Invalid image format" in response.json()["detail"]
    )


def test_hash_image_with_large_file():
    large_content = b"x" * (MAX_FILE_SIZE + 1024)

    response = client.post(
        "/api/v1/images/hash/file",
        files={"file": ("large.jpg", large_content, "image/jpeg")},
    )

    assert response.status_code == 413
    assert "Image too large" in response.json()["detail"]
