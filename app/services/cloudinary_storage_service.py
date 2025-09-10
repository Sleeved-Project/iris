import os
import cloudinary
import cloudinary.uploader
from typing import Optional

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


class CloudinaryStorageService:
    @staticmethod
    def upload_image(
        image_np,
        folder: str = "tmp_extracted_cards",
        public_id: Optional[str] = None,
        resource_type: str = "image",
        format: str = "jpg",
        ttl_minutes: int = 1,
    ):
        """
        Upload an image (as a numpy array) to Cloudinary with an optional TTL
        (in minutes).
        """
        import cv2
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp:
            cv2.imwrite(temp.name, image_np)
            temp_filename = temp.name

        try:
            options = {
                "folder": folder,
                "resource_type": resource_type,
                "invalidate": True,
            }

            if public_id:
                options["public_id"] = public_id

            if ttl_minutes > 0:
                import time

                current_time = int(time.time())
                ttl_seconds = ttl_minutes * 60
                expiration = current_time + ttl_seconds

                options["access_mode"] = "public"
                options["context"] = f"expiration={expiration}"

            upload_result = cloudinary.uploader.upload(temp_filename, **options)

            return upload_result["secure_url"]

        except Exception as e:
            print(f"Error occured when upload on cloudinary: {e}")
            raise e

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    @staticmethod
    def delete_image(public_id: str, folder: str = "tmp_extracted_cards"):
        """
        Delete an image from Cloudinary.
        """
        full_public_id = f"{folder}/{public_id}"
        result = cloudinary.uploader.destroy(full_public_id)
        return result


cloudinary_storage = CloudinaryStorageService()
