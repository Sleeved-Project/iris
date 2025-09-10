from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies.image_request_validators import ValidationResult
from app.db.session import get_db
from app.schemas.analysis_schemas import AnalysisResponse
from app.services.image_analysis_service_v3 import analyze_image_logic
import os


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    temp_image_path = None
    output_dir = "final_output"
    output_dir_low = "final_output2"

    if debug:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_low, exist_ok=True)

    try:
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("Analysis from URL is not yet implemented.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Image file not found for analysis.")

        return analyze_image_logic(
            image_path=temp_image_path,
            db=db,
            debug=debug,
            output_dir=output_dir,
            output_dir_low=output_dir_low,
        )

    except Exception as e:
        print(f"Erreur durant l'analyse : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
