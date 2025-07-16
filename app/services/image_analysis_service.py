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
from app.services.card_extraction_service_v2 import card_extraction_service

SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5


class ImageAnalysisService:
    def __init__(self, db: Session, debug: bool = False):
        self.db = db
        self.debug = debug
        self.final_output_dir = "final_output"
        self.final_output_dir_low = "final_output2"
        if self.debug:
            os.makedirs(self.final_output_dir, exist_ok=True)
            os.makedirs(self.final_output_dir_low, exist_ok=True)

    async def analyze_image(self, temp_image_path: str) -> AnalysisResponse:
        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Image file not found for analysis.")

        extracted_cards = card_extraction_service.extract_cards_from_image(
            temp_image_path
        )
        if not extracted_cards or not isinstance(extracted_cards, list):
            raise ValueError(
                "Extraction des cartes échouée : aucune carte extraite."
            )

        db_hashes = self.db.query(CardHash).all()
        analyzed_cards: List[AnalyzedCard] = []
        source_filename = os.path.basename(temp_image_path)

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
                card_hash = image_hash_service.calculate_hashes(
                    card_image_pil_grayscale
                )

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
                            db_card_hash_entry, "name", f"Card {db_card_hash_entry.id}"
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

                if similarity_percentage > best_similarity_percentage:
                    best_similarity_percentage = similarity_percentage
                    best_card_result = AnalyzedCard(
                        card_hash=card_hash,
                        card_index=i // 2,
                        is_similar=is_similar,
                        similarity_percentage=round(similarity_percentage, 2),
                        matched_card_id=best_match_id,
                        matched_card_name=best_match_name,
                        top_n_matches=top_n_results,
                    )
                    best_card_i = i + j

                if similarity_percentage < worst_similarity_percentage:
                    worst_similarity_percentage = similarity_percentage
                    worst_card_result = AnalyzedCard(
                        card_hash=card_hash,
                        card_index=i // 2,
                        is_similar=is_similar,
                        similarity_percentage=round(similarity_percentage, 2),
                        matched_card_id=best_match_id,
                        matched_card_name=best_match_name,
                        top_n_matches=top_n_results,
                    )
                    worst_card_i = i + j

            if self.debug:
                self._download_and_save_card(
                    best_card_result,
                    source_filename,
                    best_card_i,
                    self.final_output_dir,
                )
                self._download_and_save_card(
                    worst_card_result,
                    source_filename,
                    worst_card_i,
                    self.final_output_dir_low,
                )

            if best_card_result:
                analyzed_cards.append(best_card_result)
            # Tu peux décommenter si tu veux aussi les cartes les moins similaires :
            # if worst_card_result and worst_card_result != best_card_result:
            #     analyzed_cards.append(worst_card_result)

        return AnalysisResponse(
            message=(
                """Analyse complétée avec succès via
                le nouveau modèle d'extraction """
                """(cartes aux similarités les plus
                haute et basse par paire sauvegardées)."""
            ),
            cards=analyzed_cards,
        )

    def _download_and_save_card(
        self,
        card_result: Optional[AnalyzedCard],
        source_filename: str,
        card_index: int,
        output_dir: str,
    ):
        if not (card_result and card_result.matched_card_id):
            return
        try:
            prefix, suffix = card_result.matched_card_id.split("-")
            image_url = f"https://images.pokemontcg.io/{prefix}/{suffix}_hires.png"
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
