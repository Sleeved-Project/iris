# app/controllers/analysis_controller.py
import io
import os
from typing import List, Dict, Optional
import cv2
from PIL import Image
from sqlalchemy.orm import Session
from fastapi import Depends

from app.services.contour_detection_service import detect_cards
from app.services.image_hash_service import image_hash_service
from app.dependencies.image_request_validators import ValidationResult
from app.schemas.analysis_schemas import AnalysisResponse, AnalyzedCard, MatchedCardDetail # Import MatchedCardDetail
from app.db.session import get_db
from app.db.models.card_hash import CardHash

# --- Configuration inspired by the provided codebase ---
HASH_SIZE_DIMENSION = 16
BITS_PER_HASH = HASH_SIZE_DIMENSION * HASH_SIZE_DIMENSION
TOTAL_HASH_BITS = BITS_PER_HASH * 2

# Minimum similarity threshold for Hamming distance. Lower is more exact.
SIMILARITY_HAMMING_THRESHOLD = 90 # Adjust as needed based on testing

# Number of top matches to return
TOP_N_RESULTS = 5

async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False
) -> AnalysisResponse:
    """
    Detects card-like objects in an image, extracts them, and calculates their perceptual hashes.
    Compares the calculated hashes with hashes stored in the database.
    """
    temp_image_path = None
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

        warped_images, _ = detect_cards(
            temp_image_path,
            method="canny",
            min_area=5000,
            aspect_ratio_range=(0.6, 0.85),
            debug=debug,
            output_dir="/tmp/debug_output",
            common_output_dir="/tmp/common_debug_output"
        )

        # Fetch all hashes from the database once
        db_hashes = db.query(CardHash).all()
        
        analyzed_cards: List[AnalyzedCard] = []
        for i, card_image_np in enumerate(warped_images):
            card_image_pil = Image.fromarray(cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB))
            
            # Convert to grayscale for improved robustness against exposure/color shifts
            card_image_pil_grayscale = card_image_pil.convert("L")
            
            # Use the image_hash_service to calculate hashes
            card_hash = image_hash_service.calculate_hashes(card_image_pil_grayscale)

            best_match_id: Optional[str] = None
            best_match_name: Optional[str] = None
            min_distance = float('inf') # Initialize with a very high distance

            is_similar = False
            similarity_percentage = 0.0

            # List to store all potential matches for sorting
            all_matches: List[MatchedCardDetail] = []

            # Compare against all hashes in the database
            for db_card_hash_entry in db_hashes:
                db_hash_str = db_card_hash_entry.hash
                
                current_distance, current_percentage = image_hash_service.calculate_similarity_metrics(card_hash, db_hash_str) #

                # Keep track of the overall best match
                if current_distance < min_distance:
                    min_distance = current_distance
                    best_match_id = db_card_hash_entry.id
                    # Assuming a 'name' field in CardHash or a related model for the best match
                    best_match_name = getattr(db_card_hash_entry, 'name', f"Card {db_card_hash_entry.id}") # Placeholder for name
                    similarity_percentage = current_percentage
                
                # Add current match to the list of all matches
                all_matches.append(MatchedCardDetail(
                    card_id=db_card_hash_entry.id,
                    card_name=getattr(db_card_hash_entry, 'name', f"Card {db_card_hash_entry.id}"), # Placeholder for name
                    similarity_percentage=round(current_percentage, 2),
                    hamming_distance=current_distance
                ))

            # Sort all matches by similarity percentage in descending order, then by Hamming distance ascending (for tie-breaking)
            all_matches.sort(key=lambda x: (x.similarity_percentage, -x.hamming_distance), reverse=True) # Sort desc by similarity, then asc by distance

            # Get the top N results
            top_n_results = all_matches[:TOP_N_RESULTS]

            # Determine if a best match is found and meets the similarity threshold
            if best_match_id is not None and min_distance <= SIMILARITY_HAMMING_THRESHOLD:
                is_similar = True
                # If you have a real name field in your CardHash model or a linked model,
                # ensure it's retrieved here, e.g., best_match_name = db_card_hash_entry.name
                # The placeholder f"Card {best_match_id}" is used for now.

            analyzed_cards.append(AnalyzedCard(
                card_hash=card_hash,
                card_index=i,
                is_similar=is_similar,
                similarity_percentage=round(similarity_percentage, 2),
                matched_card_id=best_match_id,
                matched_card_name=best_match_name,
                top_n_matches=top_n_results # Add the top N results
            ))

        return AnalysisResponse(
            message="Analysis completed successfully", cards=analyzed_cards
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise e
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)