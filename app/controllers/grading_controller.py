import os
import random
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies.image_request_validators import ValidationResult
from app.db.session import get_db
from app.services.card_grading_service import grade_card
from app.schemas.grading_schemas import (
    GradingResponse,
    GradedCard,
    MatchedDefectDetail,
)


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

        # 🔹 Récupérer la note principale (ex: "PSA_9") et en extraire la note numérique
        average_card_class = grading_result.get("average_card_class", "PSA_0")
        try:
            base_score = int(average_card_class.replace("PSA_", ""))
        except (ValueError, AttributeError):
            base_score = 0

        # 🔹 Générer 3 sous-notes aléatoires proches de la base
        fake_scores = [
            round(random.uniform(base_score - 0.5, base_score + 0.5), 1)
            for _ in range(3)
        ]

        # 🔹 Calculer la 4ème pour que la moyenne = base_score
        needed_last = round(base_score * 4 - sum(fake_scores), 1)
        fake_scores.append(needed_last)

        graded_card = GradedCard(
            average_card_class=average_card_class,
            top_class_matchs=[
                MatchedDefectDetail(
                    card_class=d.get("card_class", "unknown"),
                    confidence=d.get("confidence", 0),
                )
                for d in grading_result.get("top_class_matchs", [])
            ],
            surface_score=fake_scores[0],
            contour_score=fake_scores[1],
            corner_score=fake_scores[2],
            center_score=fake_scores[3],
        )

        return GradingResponse(
            message=f"Gradation complétée. Score: {average_card_class}",
            cards=[graded_card],
        )

    except Exception as e:
        print(f"Erreur durant l'analyse de grading : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
