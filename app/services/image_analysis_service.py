from typing import List, Optional
import cv2
from PIL import Image
from sqlalchemy.orm import Session

from app.services.contour_detection_service import detect_cards
from app.services.image_hash_service import image_hash_service
from app.schemas.analysis_schemas import (
    AnalyzedCard,
    MatchedCardDetail,
    AnalysisResponse,
)

from app.db.models.card_hash import CardHash

SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5


def analyze_image_logic(
    image_path: str,
    db: Session,
    debug: bool = False,
) -> AnalysisResponse:
    warped_images, _ = detect_cards(
        image_path,
        method="canny",
        min_area=5000,
        aspect_ratio_range=(0.6, 0.85),
        debug=debug,
        output_dir="/tmp/debug_output",
        common_output_dir="/tmp/common_debug_output",
    )

    db_hashes = db.query(CardHash).all()

    analyzed_cards: List[AnalyzedCard] = []

    for i, card_image_np in enumerate(warped_images):
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
                    card_hash,
                    db_hash_str,
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

        if best_match_id is not None and min_distance <= SIMILARITY_HAMMING_THRESHOLD:
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
        message="Analysis completed successfully", cards=analyzed_cards
    )
