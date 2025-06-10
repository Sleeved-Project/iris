from fastapi import HTTPException
from app.services.contour_detection_service import detect_cards
from app.schemas.scan_schemas import ScanResponse
from app.dependencies.scan_request_validators import ScanValidationResult

import uuid
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def detect_card(
    validated_input: ScanValidationResult, debug: bool = False
) -> ScanResponse:
    try:
        if not validated_input.file:
            raise HTTPException(
                status_code=400, detail="Only files are supported for contour detection"
            )

        # Read the content as bytes
        file_bytes = await validated_input.file.read()

        image_id = str(uuid.uuid4())
        file_ext = os.path.splitext(validated_input.filename)[1]
        input_path = os.path.join(OUTPUT_DIR, f"{image_id}{file_ext}")

        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # Reset the file cursor to the beginning (if needed later)
        await validated_input.file.seek(0)

        warped_images, _ = detect_cards(
            image_path=input_path, debug=debug, output_dir=OUTPUT_DIR if debug else None
        )

        # Do not save cropped card images, just count how many were detected
        cards_count = len(warped_images)
        cards_detected = cards_count > 0

        # Optionally delete the temporary input file
        os.remove(input_path)

        return ScanResponse(
            message=f"{cards_count} cards detected", cards_detected=cards_detected
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
