import os
import pytest
import numpy as np
import cv2

from app.services.image_preprocessing_service import preprocess_image

TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results", "preprocess")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class TestPreprocessImage:

    @pytest.mark.parametrize(
        "image_name",
        [
            "card_normal_light.png",
            "test_card_0_20250629_214455.jpg",
            "test_card_1_20250629_214456.jpg",
            "test_card_2_20250629_214456.jpg",
            "test_card_3_20250629_214456.jpg",
            "test_card_4_20250629_214456.jpg",
            "test_card_5_20250629_214456.jpg",
            "test_card_6_20250629_214456.jpg",
            "test_card_7_20250629_214456.jpg",
            "test_card_8_20250629_214456.jpg",
            "test_card_9_20250629_214456.jpg",
            "test_card_10_20250629_214456.jpg",
            "test_card_11_20250629_214456.jpg",
            "test_card_12_20250629_214456.jpg",
            "test_card_13_20250629_214456.jpg",
            "test_card_14_20250629_214456.jpg",
            "test_card_15_20250629_214456.jpg",
            "test_card_16_20250629_214456.jpg",
            "test_card_17_20250629_214456.jpg",
        ],
    )
    @pytest.mark.parametrize("method", ["canny", "light", "spec"])
    def test_preprocess_image_output(self, image_name, method):
        image_path = os.path.join(TEST_ASSETS_DIR, image_name)
        original, edged = preprocess_image(image_path, method=method)

        assert original is not None
        assert isinstance(original, np.ndarray)
        assert (
            original.ndim == 3
            and original.shape[2] == 3
        ), "Image originale doit être couleur"

        assert edged is not None
        assert isinstance(edged, np.ndarray)
        assert (
            edged.ndim == 2
        ), "Image prétraitée doit être 2D (niveaux de gris ou binaire)"

        image_base = os.path.splitext(image_name)[0]
        cv2.imwrite(
            os.path.join(
                TEST_RESULTS_DIR,
                f"{image_base}_{method}_original.png"
            ),
            original
        )
        cv2.imwrite(
            os.path.join(
                TEST_RESULTS_DIR,
                f"{image_base}_{method}_edged.png"
            ),
            edged
        )
