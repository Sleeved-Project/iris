import os
from app.dependencies.image_request_validators import ValidationResult
from sqlalchemy.orm import Session
from app.services.image_analysis_service import ImageAnalysisService


async def analyze_image(
    validated_input: ValidationResult,
    db: Session,
    debug: bool = False,
):
    temp_image_path = None
    if validated_input.file:
        contents = await validated_input.file.read()
        temp_image_path = f"/tmp/{validated_input.file.filename}"
        with open(temp_image_path, "wb") as f:
            f.write(contents)
    elif validated_input.url:
        raise NotImplementedError("Analysis from URL is not yet implemented.")

    if not temp_image_path or not os.path.exists(temp_image_path):
        raise ValueError("Image file not found for analysis.")

    try:
        analysis_service = ImageAnalysisService(db=db, debug=debug)
        return await analysis_service.analyze_image(temp_image_path)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
