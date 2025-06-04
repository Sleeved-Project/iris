import io

import requests
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.constants import MAX_FILE_SIZE


class ImageExtractionService:
    """Service for extracting images from various sources."""

    @staticmethod
    async def get_image_from_url(url: str) -> Image.Image:
        """
        Fetch an image from a URL.

        Args:
            url: The validated URL of the image to fetch.

        Returns:
            PIL.Image: The loaded image.

        Raises:
            HTTPException: If the image couldn't be processed.
        """
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            # Check content size
            content_length = int(response.headers.get("content-length", 0))
            if content_length > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large (max {MAX_FILE_SIZE/1024/1024}MB)",
                )

            # Load the image
            return ImageExtractionService._load_image_from_bytes(response.content)

        except requests.RequestException as e:
            raise HTTPException(
                status_code=400, detail=f"Error fetching image: {str(e)}"
            )
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image format")

    @staticmethod
    async def get_image_from_file(file: UploadFile) -> Image.Image:
        """
        Process a validated uploaded image file.

        Args:
            file: The UploadFile object containing the image.

        Returns:
            PIL.Image: The loaded image.

        Raises:
            HTTPException: If the image is too large or invalid.
        """
        try:
            contents = await file.read()

            # Check file size
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large (max {MAX_FILE_SIZE/1024/1024}MB)",
                )

            return ImageExtractionService._load_image_from_bytes(contents)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image format")
        finally:
            await file.seek(0)

    @staticmethod
    def _load_image_from_bytes(data: bytes) -> Image.Image:
        """
        Load an image from bytes.

        Args:
            data: The binary image data.

        Returns:
            PIL.Image: The loaded image.
        """
        image_data = io.BytesIO(data)
        return Image.open(image_data)


image_extraction_service = ImageExtractionService()
