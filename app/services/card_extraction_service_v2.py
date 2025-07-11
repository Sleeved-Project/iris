import os
from typing import List, Optional
import cv2
from PIL import Image
import requests
from sqlalchemy.orm import Session
from fastapi import Depends
from dotenv import load_dotenv

from app.services.image_hash_service import image_hash_service
from app.dependencies.image_request_validators import ValidationResult
from app.schemas.analysis_schemas import (
    AnalysisResponse,
    AnalyzedCard,
    MatchedCardDetail,
)
from app.db.session import get_db
from app.db.models.card_hash import CardHash

load_dotenv()  # Charge les variables depuis .env

SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY_V2")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID_V2")

if not ROBOFLOW_API_KEY:
    raise ValueError("La clé API Roboflow n'est pas définie dans le fichier .env")
if not MODEL_ID:
    raise ValueError("Le MODEL_ID Roboflow n'est pas défini dans le fichier .env")

ROBOFLOW_URL = f"https://serverless.roboflow.com/{MODEL_ID}"


class CardExtractionService:
    def __init__(self, api_key: str, model_id: str):
        if not api_key or "VOTRE_CLE" in api_key:
            raise ValueError("La clé API Roboflow n'est pas configurée.")
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = f"https://serverless.roboflow.com/{self.model_id}"

    def _infer_via_http(self, image_path: str) -> dict:
        params = {"api_key": self.api_key}
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(self.base_url, params=params, files=files)
            response.raise_for_status()
            return response.json()

    def extract_cards_from_image(self, image_path: str) -> List:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Impossible de lire l'image avec OpenCV.")

        result = self._infer_via_http(image_path)

        cards = []
        for detection in result.get("predictions", []):
            x_center = int(detection["x"])
            y_center = int(detection["y"])
            width = int(detection["width"])
            height = int(detection["height"])

            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(image.shape[1], x_center + width // 2)
            y2 = min(image.shape[0], y_center + height // 2)

            card_img = image[y1:y2, x1:x2]
            if card_img.size == 0:
                continue
            cards.append(card_img)

        if not cards:
            raise ValueError("Aucune carte détectée dans l'image par Roboflow.")

        return cards


# Singleton instancié avec variables .env
card_extraction_service = CardExtractionService(
    api_key=ROBOFLOW_API_KEY, model_id=MODEL_ID
)


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    temp_image_path = None
    try:
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            # Pour futur : téléchargement via URL si besoin
            raise NotImplementedError("Analyse à partir d'URL non implémentée.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Fichier image introuvable pour l'analyse.")

        extracted_cards = card_extraction_service.extract_cards_from_image(
            temp_image_path
        )

        db_hashes = db.query(CardHash).all()
        analyzed_cards: List[AnalyzedCard] = []

        for i, card_image_np in enumerate(extracted_cards):
            card_image_pil = Image.fromarray(
                cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB)
            )
            card_image_pil_grayscale = card_image_pil.convert("L")

            card_hash = image_hash_service.calculate_hashes(card_image_pil_grayscale)

            best_match_id: Optional[str] = None
            best_match_name: Optional[str] = None
            min_distance = float("inf")
            similarity_percentage = 0.0
            is_similar = False
            all_matches: List[MatchedCardDetail] = []

            for db_card_hash_entry in db_hashes:
                db_hash_str = db_card_hash_entry.hash
                current_distance, current_percentage = (
                    image_hash_service.calculate_similarity_metrics(
                        card_hash, db_hash_str
                    )
                )

                if current_distance < min_distance:
                    min_distance = current_distance
                    best_match_id = db_card_hash_entry.id
                    best_match_name = getattr(
                        db_card_hash_entry, "name", f"Card {db_card_hash_entry.id}"
                    )
                    similarity_percentage = current_percentage

                all_matches.append(
                    MatchedCardDetail(
                        card_id=db_card_hash_entry.id,
                        card_name=getattr(
                            db_card_hash_entry, "name", f"Card {db_card_hash_entry.id}"
                        ),
                        similarity_percentage=round(current_percentage, 2),
                        hamming_distance=current_distance,
                    )
                )

            all_matches.sort(
                key=lambda x: (x.similarity_percentage, -x.hamming_distance),
                reverse=True,
            )
            top_n_results = all_matches[:TOP_N_RESULTS]

            if best_match_id and min_distance <= SIMILARITY_HAMMING_THRESHOLD:
                is_similar = True

            analyzed_cards.append(
                AnalyzedCard(
                    card_hash=card_hash,
                    card_index=i,
                    is_similar=is_similar,
                    similarity_percentage=round(similarity_percentage, 2),
                    matched_card_id=best_match_id,
                    matched_card_name=best_match_name,
                    top_n_matches=top_n_results,
                )
            )

        return AnalysisResponse(
            message="""Analyse complétée avec succès
            via Roboflow et le modèle d'analyse.""",
            cards=analyzed_cards,
        )

    except Exception as e:
        print(f"Erreur durant l'analyse : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
