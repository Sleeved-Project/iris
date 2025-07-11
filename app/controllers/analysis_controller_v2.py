import os
import requests
from typing import List, Optional
import cv2
from PIL import Image
from sqlalchemy.orm import Session
from fastapi import Depends

from app.services.image_hash_service import image_hash_service
from app.dependencies.image_request_validators import ValidationResult
from app.schemas.analysis_schemas import (
    AnalysisResponse,
    AnalyzedCard,
    MatchedCardDetail,
)
from app.db.session import get_db
from app.db.models.card_hash import CardHash

from app.services.card_extraction_service_v2 import (
    card_extraction_service,
)  # Roboflow extraction
from app.services.contour_detection_service import (
    detect_cards as detect_cards_with_contours,
)  # contour extraction

HASH_SIZE_DIMENSION = 16
BITS_PER_HASH = HASH_SIZE_DIMENSION * HASH_SIZE_DIMENSION
TOTAL_HASH_BITS = BITS_PER_HASH * 2

SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    temp_image_path = None
    output_dir = "v2_result"
    matched_cards_dir = os.path.join(output_dir, "matched_cards")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(matched_cards_dir, exist_ok=True)

    try:
        # Sauvegarde temporaire de l'image envoyée
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("Analyse à partir d'URL non implémentée.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Fichier image introuvable pour l'analyse.")

        # 1. Extraction avec Roboflow (card_extraction_service)
        extracted_cards_rf = card_extraction_service.extract_cards_from_image(
            temp_image_path
        )

        # Enregistrement des cartes extraites par Roboflow
        for i, card_img_np in enumerate(extracted_cards_rf):
            card_path = os.path.join(output_dir, f"roboflow_card_{i}.png")
            cv2.imwrite(card_path, card_img_np)

        # 2. Extraction avec contour_detection_service
        extracted_cards_v2, contours = detect_cards_with_contours(
            image_path=temp_image_path,
            debug=debug,
            output_dir=output_dir,
            common_output_dir=output_dir,
        )

        if not extracted_cards_v2 or not isinstance(extracted_cards_v2, list):
            raise ValueError(
                """Extraction initiale échouée :
                aucune carte détectée par contour_detection_service."""
            )

        # Enregistrement des cartes extraites par contour_detection_service
        for i, card_img_np in enumerate(extracted_cards_v2):
            card_path = os.path.join(output_dir, f"contour_card_{i}.png")
            cv2.imwrite(card_path, card_img_np)

        # 3. Analyse et comparaison avec DB
        db_hashes = db.query(CardHash).all()
        analyzed_cards: List[AnalyzedCard] = []

        for i, card_image_np in enumerate(extracted_cards_v2):
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

            if (
                best_match_id is not None
                and min_distance <= SIMILARITY_HAMMING_THRESHOLD
            ):
                is_similar = True

            # Si carte similaire, tenter de télécharger
            # la carte correspondante à partir de l'URL
            if is_similar and best_match_id:
                try:
                    prefix, suffix = best_match_id.split("-")
                    image_url = (
                        f"https://images.pokemontcg.io/{prefix}/{suffix}_hires.png"
                    )
                    if debug:
                        print(
                            f"""Téléchargement de la carte correspondante
                            {best_match_id} : {image_url}"""
                        )

                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        matched_card_filepath = os.path.join(
                            matched_cards_dir,
                            f"matched_card_{i}_id_{best_match_id}.png",
                        )
                        with open(matched_card_filepath, "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                f_out.write(chunk)
                        if debug:
                            print(
                                f"""Carte correspondante sauvegardée sous :
                                {matched_card_filepath}"""
                            )
                    else:
                        if debug:
                            print(
                                f"""Erreur téléchargement image {image_url} :
                                statut {response.status_code}"""
                            )
                except Exception as e:
                    if debug:
                        print(
                            f"""Erreur téléchargement/sauvegarde carte
                            {best_match_id} : {e}"""
                        )

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
            message="""Analyse complétée avec succès (Roboflow +
            contours + correspondance base avec téléchargement cartes).""",
            cards=analyzed_cards,
        )

    except Exception as e:
        print(f"Erreur durant l'analyse : {e}")
        raise e

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
