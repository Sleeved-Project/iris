# test_contour_detection.py

import os
import pytest
import cv2
import numpy as np

# Fixed import
from app.services.contour_detection_service import ContourDetectionService

TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)

TEST_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "test_results/contour"
)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class TestContourDetectionService:

    @pytest.fixture(scope="class")
    def service(self):
        return ContourDetectionService(debug=True, use_advanced_preprocessing=True)

    @pytest.mark.parametrize("image_name", [
        "card_normal_light.png",
        "ebay.png",
        "ebay_low.png",
        "ebay_high.png",
        "ebay2.png",
        "collection.png",
        "ebay5.png",
    ])
    def test_detect_card_contours(self, service, image_name):
        image_path = os.path.join(TEST_ASSETS_DIR, image_name)
        image = service.load_image(image_path)

        contours = service.find_card_like_contours(image)
        assert isinstance(contours, list), "The result should be a list"
        assert all(isinstance(c, (list, np.ndarray)) for c in contours)
        "Each contour must be an array"

        output_image = service.draw_detected_contours(image, contours)
        result_path = os.path.join(TEST_RESULTS_DIR, f"result_{image_name}")
        cv2.imwrite(result_path, output_image)

    def test_extract_card_hashes(self, service):
        image_path = os.path.join(TEST_ASSETS_DIR, "collection.png")
        hashes = service.extract_card_hashes(image_path)
        assert isinstance(hashes, list)
        assert all(isinstance(h, str) for h in hashes)
        assert len(hashes) > 0, "No cards detected to compute hashes"
