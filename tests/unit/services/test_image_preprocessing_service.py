# test_image_preprocessing.py

import os
import pytest
import cv2
from PIL import Image
import numpy as np

# Fixed import
from app.services.image_preprocessing_service import ImagePreprocessingService

TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)

TEST_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "test_results/image"
)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class TestImagePreprocessingService:

    @pytest.fixture(scope="class")
    def service(self):
        return ImagePreprocessingService(debug=True, use_advanced=True)

    @pytest.mark.parametrize("image_name", [
        "card_normal_light.png",
        "ebay.png",
        "ebay_low.png",
        "ebay_high.png",
        "ebay2.png",
        "collection.png",
        "ebay5.png",
    ])
    def test_preprocess_pil_pipeline(self, service, image_name):
        img_path = os.path.join(TEST_ASSETS_DIR, image_name)
        img = Image.open(img_path)
        processed = service.preprocess_pil_pipeline(img)

        assert isinstance(processed, Image.Image)
        save_path = os.path.join(TEST_RESULTS_DIR, f"pil_pipeline_{image_name}")
        processed.save(save_path)

    @pytest.mark.parametrize("image_name", [
        "card_normal_light.png",
        "ebay.png",
        "ebay_low.png",
        "ebay_high.png",
        "ebay2.png",
        "collection.png",
        "ebay5.png",
    ])
    def test_preprocess_cv_pipeline(self, service, image_name):
        img_path = os.path.join(TEST_ASSETS_DIR, image_name)
        img = cv2.imread(img_path)
        result = service.preprocess(img)

        assert isinstance(result, (np.ndarray,)), "Expected OpenCV result"
        assert result.ndim == 2, "Preprocessed image should be grayscale"

        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, f"cv_pipeline_{image_name}"), result)
