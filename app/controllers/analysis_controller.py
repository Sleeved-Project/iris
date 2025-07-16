import os
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies.image_request_validators import ValidationResult
from app.db.session import get_db
from app.services.image_analysis_service import analyze_image_logic


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
):
    temp_image_path = None
    try:
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("Analysis from URL not implemented.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Image file not found for analysis.")

        return analyze_image_logic(temp_image_path, db, debug=debug)

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
