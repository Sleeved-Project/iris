import imagehash
from fastapi import UploadFile
from PIL import Image

from app.core.constants import (
    HASH_SIZE,
)
from app.services.image_extraction_service import image_extraction_service


class ImageHashService:
    """Service for calculating perceptual hashes
    of images and comparing them."""

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

    @staticmethod
    def hamming_distance(hash1_str: str, hash2_str: str) -> int:
        """
        Calculates the Hamming distance between two hash strings.
        Assumes hashes are of equal length.
        """
        if len(hash1_str) != len(hash2_str):
            raise ValueError(
                "Hash lengths are not equal for Hamming distance calculation."
            )
        return sum(ch1 != ch2 for ch1, ch2 in zip(hash1_str, hash2_str))

    @staticmethod
    def calculate_similarity_metrics(hash1_str: str, hash2_str: str) -> (int, float):
        """
        Calculates the Hamming distance and similarity percentage
        between two concatenated hash strings.
        Returns (total_hamming_distance, similarity_percentage).
        """
        total_distance = ImageHashService.hamming_distance(hash1_str, hash2_str)

        max_possible_distance = (
            len(hash1_str) * 4
        )
        if max_possible_distance == 0:
            return total_distance, 0.0

        similarity_percentage = 100 - (total_distance / max_possible_distance) * 100
        return total_distance, max(0.0, min(100.0, similarity_percentage))


# Singleton instance
image_hash_service = ImageHashService()
