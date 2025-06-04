import imagehash
from fastapi import UploadFile
from PIL import Image

from app.core.constants import HASH_SIZE
from app.services.image_extraction_service import image_extraction_service


class ImageHashService:
    """Service for calculating perceptual hashes of images."""

    @staticmethod
    async def hash_from_url(url: str) -> str:
        """
        Calculate perceptual hashes for an image from a URL.

        Args:
            url: The validated URL of the image to hash.

        Returns:
            String containing perceptual hash.
        """
        image = await image_extraction_service.get_image_from_url(url)
        return ImageHashService.calculate_hashes(image)

    @staticmethod
    async def hash_from_file(file: UploadFile) -> str:
        """
        Calculate perceptual hashes for an uploaded image file.

        Args:
            file: The UploadFile object containing the image.

        Returns:
            String containing perceptual hash.
        """
        image = await image_extraction_service.get_image_from_file(file)
        return ImageHashService.calculate_hashes(image)

    @staticmethod
    def calculate_hashes(image: Image.Image) -> str:
        """
        Calculate perceptual hashes for an image.

        Args:
            image: The PIL image to hash.

        Returns:
            String containing concatenated dhash and phash.
        """
        img = image.convert("RGB")
        dhash = str(imagehash.dhash(img, HASH_SIZE))
        phash = str(imagehash.phash(img, HASH_SIZE))

        return f"{dhash}{phash}"


# Singleton instance
image_hash_service = ImageHashService()
