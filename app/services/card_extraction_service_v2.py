import os
import cv2
import numpy as np
import requests
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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

load_dotenv()

# Configuration constants
SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5
OUTPUT_WIDTH = 600
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * 1.4)  # Standard card ratio
MAX_WORKERS = min(4, os.cpu_count() or 2)

# API configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") or "dummy_key"
MODEL_ID = os.getenv("ROBOFLOW_MODEL_OBB_ID") or "dummy_model_id"


class CardExtractionService:
    def __init__(self, api_key: str, model_id: str):
        """Initialize the card extraction service with API credentials"""
        if not api_key or "VOTRE_CLE" in api_key:
            raise ValueError("Roboflow API key is not properly configured.")
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = f"https://serverless.roboflow.com/{self.model_id}"
        self.cache = {}  # Simple in-memory cache for recent predictions

    def _infer_via_http(self, image_path: str) -> dict:
        """Send image to Roboflow model and get predictions with caching"""
        # Quick cache check based on file size and modification time
        file_stat = os.stat(image_path)
        cache_key = f"{image_path}_{file_stat.st_size}_{file_stat.st_mtime}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Send request to Roboflow API
        params = {"api_key": self.api_key}
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(
                self.base_url, params=params, files=files, timeout=10
            )
            response.raise_for_status()
            result = response.json()

        # Cache the result (keep cache limited to 10 entries)
        if len(self.cache) > 10:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = result

        return result

    def extract_cards_from_image(
        self, image_path: str
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Extract cards using oriented bounding boxes from Roboflow predictions"""
        # Load image with OpenCV for better memory efficiency
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read image with OpenCV.")

        # Get predictions from Roboflow
        result = self._infer_via_http(image_path)
        cards = []
        raw_predictions = result.get("predictions", [])

        # Process all detections in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for detection in raw_predictions:
                if "points" in detection and detection["points"]:
                    futures.append(
                        executor.submit(self._process_obb_detection, image, detection)
                    )
                elif all(k in detection for k in ["x", "y", "width", "height"]):
                    futures.append(
                        executor.submit(self._process_bbox_detection, image, detection)
                    )

            # Collect results
            for future in futures:
                result = future.result()
                if result is not None:
                    cards.append(result)

        if not cards:
            raise ValueError("No cards detected in image by Roboflow.")

        return cards, raw_predictions

    def _process_obb_detection(
        self, image: np.ndarray, detection: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Process OBB (oriented bounding box) detection"""
        try:
            # Extract points from detection
            points_list = []
            for point in detection["points"]:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    points_list.append([int(point["x"]), int(point["y"])])

            if len(points_list) < 4:
                return None

            contour = np.array(points_list, dtype=np.int32)

            # Simplify contour to quadrilateral if needed
            if len(points_list) > 4:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) != 4:
                    rect = cv2.minAreaRect(contour)
                    corners = cv2.boxPoints(rect).astype(np.int32)
                else:
                    corners = approx.reshape(-1, 2)
            else:
                corners = np.array(points_list, dtype=np.int32)

            # Sort corners for perspective transform
            corners_sorted = self._sort_corners(corners)
            src_pts = corners_sorted.astype(np.float32)

            # Define destination points
            dst_pts = np.array(
                [
                    [0, 0],
                    [OUTPUT_WIDTH, 0],
                    [OUTPUT_WIDTH, OUTPUT_HEIGHT],
                    [0, OUTPUT_HEIGHT],
                ],
                dtype=np.float32,
            )

            # Apply perspective transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            return warped
        except Exception as e:
            print(f"Error processing OBB detection: {e}")
            return None

    def _process_bbox_detection(
        self, image: np.ndarray, detection: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Process standard bounding box detection"""
        try:
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
                return None

            # Resize to standard dimensions
            return cv2.resize(card_img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        except Exception as e:
            print(f"Error processing bbox detection: {e}")
            return None

    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in clockwise order starting from top-left (optimized)"""
        # Faster center calculation
        center = corners.mean(axis=0)

        # Use vectorized operations for angles
        diff = corners - center
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        sorted_idx = np.argsort(angles)
        sorted_corners = corners[sorted_idx]

        # Find top-left corner (minimum sum of x+y)
        sums = sorted_corners.sum(axis=1)
        min_sum_idx = np.argmin(sums)
        return np.roll(sorted_corners, -min_sum_idx, axis=0)


# Create singleton service instance
card_extraction_service = CardExtractionService(
    api_key=ROBOFLOW_API_KEY, model_id=MODEL_ID
)


# Use LRU cache for database hashes to avoid repeated queries
@lru_cache(maxsize=1)
def get_all_card_hashes(db_session):
    """Get all card hashes from database with caching"""
    return db_session.query(CardHash).all()


def process_card_batch(batch_data):
    """Process a batch of cards in parallel"""
    card_batch, db_hashes, start_idx = batch_data
    analyzed_cards = []

    for i, card_image_np in enumerate(card_batch):
        card_idx = start_idx + i

        # Convert to grayscale for hashing (optimized)
        card_image_pil = Image.fromarray(cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB))
        card_image_gray = card_image_pil.convert("L")
        card_hash = image_hash_service.calculate_hashes(card_image_gray)

        # Initialize variables
        best_match = {
            "id": None,
            "name": None,
            "distance": float("inf"),
            "percentage": 0.0,
        }
        all_matches = []

        # Compare with database hashes
        for db_card in db_hashes:
            distance, percentage = image_hash_service.calculate_similarity_metrics(
                card_hash, db_card.hash
            )

            if distance < best_match["distance"]:
                best_match.update(
                    {
                        "id": db_card.id,
                        "name": getattr(db_card, "name", f"Card {db_card.id}"),
                        "distance": distance,
                        "percentage": percentage,
                    }
                )

            all_matches.append(
                MatchedCardDetail(
                    card_id=db_card.id,
                    card_name=getattr(db_card, "name", f"Card {db_card.id}"),
                    similarity_percentage=round(percentage, 2),
                    hamming_distance=distance,
                )
            )

        # Sort matches and get top results
        all_matches.sort(
            key=lambda x: (x.similarity_percentage, -x.hamming_distance),
            reverse=True,
        )
        top_n_results = all_matches[:TOP_N_RESULTS]

        # Check if match is good enough
        is_similar = (
            best_match["id"] is not None
            and best_match["distance"] <= SIMILARITY_HAMMING_THRESHOLD
        )

        analyzed_cards.append(
            AnalyzedCard(
                card_hash=card_hash,
                card_index=card_idx,
                is_similar=is_similar,
                similarity_percentage=round(best_match["percentage"], 2),
                matched_card_id=best_match["id"],
                matched_card_name=best_match["name"],
                top_n_matches=top_n_results,
            )
        )

    return analyzed_cards


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    """Analyze an image to detect and match cards"""
    temp_image_path = None

    try:
        # Process input file
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{os.path.basename(validated_input.file.filename)}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("URL-based analysis not implemented.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Image file not found for analysis.")

        # Extract cards using OBB detection
        extracted_cards, _ = card_extraction_service.extract_cards_from_image(
            temp_image_path
        )

        if not extracted_cards:
            return AnalysisResponse(
                message="No cards detected in image.",
                cards=[],
            )

        # Get all card hashes from database (using cached function)
        db_hashes = get_all_card_hashes(db)

        # Process cards in batches for better parallelism
        batch_size = max(1, len(extracted_cards) // MAX_WORKERS)
        batches = []

        for i in range(0, len(extracted_cards), batch_size):
            batch = extracted_cards[i : i + batch_size]
            batches.append((batch, db_hashes, i))

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_card_batch, batches)

        # Combine results
        analyzed_cards = []
        for result in results:
            analyzed_cards.extend(result)

        return AnalysisResponse(
            message=(
                f"Analysis completed successfully. Found {len(analyzed_cards)} cards "
                "using Roboflow OBB extraction."
            ),
            cards=analyzed_cards,
        )

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise e

    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
