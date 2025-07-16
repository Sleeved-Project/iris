import os
import requests
from typing import List, Optional
import cv2
from PIL import Image
from sqlalchemy.orm import Session

from app.services.image_hash_service import image_hash_service
from app.schemas.analysis_schemas import (
    AnalysisResponse,
    AnalyzedCard,
    MatchedCardDetail,
)
from app.db.models.card_hash import CardHash
from app.services.card_extraction_service_v3 import card_extraction_service

HASH_SIZE_DIMENSION = 16
BITS_PER_HASH = HASH_SIZE_DIMENSION * HASH_SIZE_DIMENSION
TOTAL_HASH_BITS = BITS_PER_HASH * 2
SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5


def analyze_image_logic(
    image_path: str,
    db: Session,
    debug: bool = False,
    output_dir: str = "final_output",
    output_dir_low: str = "final_output2",
) -> AnalysisResponse:
    extracted_cards = card_extraction_service.extract_cards_from_image(image_path)
    if not extracted_cards or not isinstance(extracted_cards, list):
        raise ValueError(
            "Extraction des cartes échouée : aucune carte extraite."
        )

    db_hashes = db.query(CardHash).all()
    analyzed_cards: List[AnalyzedCard] = []
    source_filename = os.path.basename(image_path)

    for i in range(0, len(extracted_cards), 2):
        card_pair = extracted_cards[i : i + 2]

        best_card_result = None
        best_similarity_percentage = -1.0
        best_card_i = i

        worst_card_result = None
        worst_similarity_percentage = 101.0
        worst_card_i = i

        for j, card_image_np in enumerate(card_pair):
            card_image_pil = Image.fromarray(
                cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB)
            )
            card_image_pil_grayscale = card_image_pil.convert("L")
            card_hash = image_hash_service.calculate_hashes(card_image_pil_grayscale)

            best_match_id: Optional[str] = None
            best_match_name: Optional[str] = None
            min_distance = float("inf")
            is_similar = False
            similarity_percentage = 0.0
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
                        db_card_hash_entry,
                        "name",
                        f"Card {db_card_hash_entry.id}",
                    )
                    similarity_percentage = current_percentage

                all_matches.append(
                    MatchedCardDetail(
                        card_id=db_card_hash_entry.id,
                        card_name=getattr(
                            db_card_hash_entry,
                            "name",
                            f"Card {db_card_hash_entry.id}",
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

            analyzed = AnalyzedCard(
                card_hash=card_hash,
                card_index=i // 2,
                is_similar=is_similar,
                similarity_percentage=round(similarity_percentage, 2),
                matched_card_id=best_match_id,
                matched_card_name=best_match_name,
                top_n_matches=top_n_results,
            )

            if similarity_percentage > best_similarity_percentage:
                best_similarity_percentage = similarity_percentage
                best_card_result = analyzed
                best_card_i = i + j

            if similarity_percentage < worst_similarity_percentage:
                worst_similarity_percentage = similarity_percentage
                worst_card_result = analyzed
                worst_card_i = i + j

        def download_and_save(card_result, card_index, output_dir):
            if not (card_result and card_result.matched_card_id):
                return
            try:
                prefix, suffix = card_result.matched_card_id.split("-")
                image_url = (
                    f"https://images.pokemontcg.io/{prefix}/{suffix}_hires.png"
                )
                if debug:
                    print(
                        f"Lien vers la carte {card_result.matched_card_id} : "
                        f"{image_url}"
                    )

                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    output_filename = f"card_{source_filename}_{card_index + 1}.png"
                    output_filepath = os.path.join(output_dir, output_filename)

                    with open(output_filepath, "wb") as f_out:
                        for chunk in response.iter_content(chunk_size=8192):
                            f_out.write(chunk)

                    if debug:
                        print(
                            f"Image téléchargée et sauvegardée sous : "
                            f"{output_filepath}"
                        )
                else:
                    print(
                        f"Erreur téléchargement image {image_url}: "
                        f"Status {response.status_code}"
                    )
            except Exception as e:
                print(
                    f"Erreur lors du téléchargement ou sauvegarde pour "
                    f"{card_result.matched_card_id}: {e}"
                )

        if debug:
            download_and_save(best_card_result, best_card_i, output_dir)
            download_and_save(worst_card_result, worst_card_i, output_dir_low)

        if best_card_result:
            analyzed_cards.append(best_card_result)
        # Tu peux aussi ajouter la carte la moins similaire ici si besoin

    return AnalysisResponse(
        message=(
            "Analyse complétée avec succès via le nouveau modèle "
            "d'extraction (cartes aux similarités les plus haute et basse "
            "par paire sauvegardées)."
        ),
        cards=analyzed_cards,
    )
