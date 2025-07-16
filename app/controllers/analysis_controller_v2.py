from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies.image_request_validators import ValidationResult
from app.db.session import get_db
from app.services.image_analysis_service_v2 import analyze_image_logic
from app.schemas.analysis_schemas import AnalysisResponse
import os


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    temp_image_path = None
    output_dir = "v2_result"

    try:
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("Analyse à partir d'URL non implémentée.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Fichier image introuvable pour l'analyse.")

        response = analyze_image_logic(
            temp_image_path, db, debug=debug, output_dir=output_dir
        )
        return response

    except Exception as e:
        print(f"Erreur durant l'analyse : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
