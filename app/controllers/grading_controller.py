import os
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies.image_request_validators import ValidationResult
from app.db.session import get_db
from app.services.card_grading_service import grade_card
from app.schemas.grading_schemas import GradingResponse, GradedCard, MatchedDefectDetail


async def analyze_grading(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> GradingResponse:
    temp_image_path = None

    try:
        # 🔹 Sauvegarde temporaire du fichier image
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError(
                "Analyse à partir d'URL non implémentée pour grading."
            )

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Fichier image introuvable pour l'analyse de grading.")

        # 🔹 Analyse via le service
        grading_result = grade_card(temp_image_path)

        # 🔹 Création de l'objet GradedCard avec description
        graded_card = GradedCard(
            card_hash="N/A",
            card_index=0,
            similarity_percentage=0.0,
            is_similar=False,
            matched_card_id="N/A",
            matched_card_name=grading_result.get("score", "unknown"),
            description=grading_result.get("description", ""),
            top_n_matches=[
                MatchedDefectDetail(
                    card_id="N/A",
                    card_name=d.get("type", "defect"),
                    similarity_percentage=round(d.get("confidence", 0) * 100, 2),
                    hamming_distance=0,
                )
                for d in grading_result.get("defects", [])
            ],
        )

        return GradingResponse(
            message=(f"Gradation complétée. Score: {grading_result.get('score')}"),
            cards=[graded_card],
        )

    except Exception as e:
        print(f"Erreur durant l'analyse de grading : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
