import os
import cv2
import requests
import numpy as np
import shutil
from typing import List, Dict, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session

from app.services.image_hash_service import image_hash_service
from app.db.models.card_hash import CardHash
from app.services.card_extraction_service_v2 import card_extraction_service
from app.schemas.analysis_schemas import (
    AnalyzedCard,
    MatchedCardDetail,
    AnalysisResponse,
)

# Configuration constants
HASH_SIZE_DIMENSION = 16
BITS_PER_HASH = HASH_SIZE_DIMENSION * HASH_SIZE_DIMENSION
SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5
DEFAULT_DEBUG_DIR = "debug_output/v2"
MAX_WORKERS = 4


class DebugVisualizer:
    """Handles visual debugging for the image analysis process"""

    def __init__(self, debug: bool, base_dir: str = DEFAULT_DEBUG_DIR):
        self.debug = debug
        self.base_dir = base_dir
        self.dirs = {}

        if not debug:
            return

        print(f"Debug mode enabled: {self.debug}")

        # Create fresh output directory
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

        # Create directory structure
        self.dirs = {
            "root": base_dir,
            "input": os.path.join(base_dir, "1_input"),
            "detection": os.path.join(base_dir, "2_detection_obb"),
            "roboflow": os.path.join(base_dir, "3_roboflow_extraction"),
            "processed": os.path.join(base_dir, "4_processed_cards"),
            "matches": os.path.join(base_dir, "5_matched_cards"),
        }

        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def save_image(self, img, stage: str, name: str, suffix: str = "jpg") -> None:
        """Save an image to the specified stage directory"""
        if not self.debug or img is None or stage not in self.dirs:
            return

        filepath = os.path.join(self.dirs[stage], f"{name}.{suffix}")

        # Convert PIL to OpenCV if needed
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(filepath, img)


def analyze_card(
    card_data: Tuple[int, np.ndarray, List[CardHash], bool, Dict[str, str]],
) -> AnalyzedCard:
    """Process a single card image and find matches in database"""
    i, card_image_np, db_hashes, debug, debug_dirs = card_data

    # Convert to grayscale for perceptual hashing
    card_image_pil = Image.fromarray(cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB))
    card_image_gray = card_image_pil.convert("L")
    card_hash = image_hash_service.calculate_hashes(card_image_gray)

    # Debug output
    if debug and "roboflow" in debug_dirs and "processed" in debug_dirs:
        roboflow_img_path = os.path.join(debug_dirs["roboflow"], f"card_{i}.jpg")
        processed_img_path = os.path.join(debug_dirs["processed"], f"processed_{i}.jpg")
        if os.path.exists(roboflow_img_path):
            shutil.copy(roboflow_img_path, processed_img_path)

    # Find best match
    best_match_id = None
    best_match_name = None
    min_distance = float("inf")
    similarity_percentage = 0.0
    all_matches = []

    # Compare with database hashes
    for db_card in db_hashes:
        current_distance, current_percentage = (
            image_hash_service.calculate_similarity_metrics(card_hash, db_card.hash)
        )

        if current_distance < min_distance:
            min_distance = current_distance
            best_match_id = db_card.id
            best_match_name = getattr(db_card, "name", f"Card {db_card.id}")
            similarity_percentage = current_percentage

        all_matches.append(
            MatchedCardDetail(
                card_id=db_card.id,
                card_name=getattr(db_card, "name", f"Card {db_card.id}"),
                similarity_percentage=round(current_percentage, 2),
                hamming_distance=current_distance,
            )
        )

    # Sort matches and get top results
    all_matches.sort(
        key=lambda x: (x.similarity_percentage, -x.hamming_distance),
        reverse=True,
    )
    top_n_results = all_matches[:TOP_N_RESULTS]

    # Determine if match is good enough
    is_similar = (
        best_match_id is not None and min_distance <= SIMILARITY_HAMMING_THRESHOLD
    )

    # Download matched card image for debug visualization
    if is_similar and best_match_id and debug and "matches" in debug_dirs:
        try:
            prefix, suffix = best_match_id.split("-")

            # Create directory if it doesn't exist
            os.makedirs(debug_dirs["matches"], exist_ok=True)

            # Try both high and standard resolution
            for img_type in ["_hires.png", ".png"]:
                image_url = f"https://images.pokemontcg.io/{prefix}/{suffix}{img_type}"

                try:
                    response = requests.get(image_url, stream=True, timeout=5)
                    if response.status_code == 200:
                        # Define file paths
                        matched_card_filepath = os.path.join(
                            debug_dirs["matches"], f"matched_{i}_{best_match_id}.png"
                        )

                        # Save directly to final location
                        with open(matched_card_filepath, "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                f_out.write(chunk)

                        # Create comparison visualization
                        matched_img = cv2.imread(matched_card_filepath)
                        if matched_img is not None:
                            # Resize to same height for comparison
                            h, w = card_image_np.shape[:2]
                            m_h, m_w = matched_img.shape[:2]
                            new_w = int(m_w * (h / m_h))
                            matched_img = cv2.resize(matched_img, (new_w, h))

                            # Create side-by-side comparison
                            comparison = np.hstack([card_image_np, matched_img])

                            cv2.putText(
                                comparison,
                                "Detected",
                                (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                comparison,
                                f"{best_match_name} ({similarity_percentage:.1f}%)",
                                (w + 20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                            comparison_path = os.path.join(
                                debug_dirs["matches"], f"comparison_{i}.jpg"
                            )
                            cv2.imwrite(comparison_path, comparison)
                        break
                except Exception as e:
                    print(f"Error downloading image: {e}")
        except Exception as e:
            print(f"Error processing match visualization: {e}")

    return AnalyzedCard(
        card_hash=card_hash,
        card_index=i,
        is_similar=is_similar,
        similarity_percentage=round(similarity_percentage, 2),
        matched_card_id=best_match_id,
        matched_card_name=best_match_name,
        top_n_matches=top_n_results,
    )


def analyze_image_logic(
    image_path: str,
    db: Session,
    debug: bool = False,
    output_dir: str = DEFAULT_DEBUG_DIR,
) -> AnalysisResponse:
    """Analyzes an image to detect and identify cards using Roboflow OBB extraction"""
    # Initialize debug visualizer
    visualizer = DebugVisualizer(debug, output_dir)

    # 1. Read input image (needed for visualization and card extraction)
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    visualizer.save_image(input_image, "input", "original_input")

    # 2. Get Roboflow predictions
    try:
        result = card_extraction_service._infer_via_http(image_path)
        raw_predictions = result.get("predictions", [])

        # Visualize detections if in debug mode
        if debug and raw_predictions:
            obb_vis = input_image.copy()
            for pred in raw_predictions:
                if "points" in pred and pred["points"]:
                    points_list = []
                    for point in pred["points"]:
                        if isinstance(point, dict) and "x" in point and "y" in point:
                            points_list.append([int(point["x"]), int(point["y"])])

                    if len(points_list) >= 4:
                        points = np.array(points_list, dtype=np.int32)
                        cv2.polylines(obb_vis, [points], True, (0, 255, 0), 2)

                        confidence = pred.get("confidence", 0)
                        cv2.putText(
                            obb_vis,
                            f"{confidence:.2f}",
                            (points_list[0][0], points_list[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                elif all(k in pred for k in ["x", "y", "width", "height"]):
                    x = int(pred["x"] - pred["width"] / 2)
                    y = int(pred["y"] - pred["height"] / 2)
                    w = int(pred["width"])
                    h = int(pred["height"])
                    cv2.rectangle(obb_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

            visualizer.save_image(obb_vis, "detection", "cards_detection_obb")

        # Extract cards
        extracted_cards, _ = card_extraction_service.extract_cards_from_image(
            image_path
        )

        # Save extracted cards
        if debug:
            for i, card in enumerate(extracted_cards):
                visualizer.save_image(card, "roboflow", f"card_{i}")

        print(f"Found {len(extracted_cards)} cards")
    except Exception as e:
        print(f"Extraction failed: {e}")
        extracted_cards = []

    # 3. Verify cards were detected
    if not extracted_cards:
        return AnalysisResponse(
            message="No cards detected in image.",
            debug_path=visualizer.base_dir if debug else None,
            cards=[],
        )

    # 4. Process cards in parallel
    db_hashes = db.query(CardHash).all()
    card_data_list = [
        (i, card, db_hashes, debug, visualizer.dirs)
        for i, card in enumerate(extracted_cards)
    ]

    with ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(extracted_cards))
    ) as executor:
        analyzed_cards = list(executor.map(analyze_card, card_data_list))

    return AnalysisResponse(
        message=f"Analysis completed successfully. Found {len(analyzed_cards)} cards.",
        debug_path=visualizer.base_dir if debug else None,
        cards=analyzed_cards,
    )
