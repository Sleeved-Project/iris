import os
import pytest
import cv2
import numpy as np

# Import du script à tester
from app.services.card_detector import detect_cards

TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)

TEST_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "test_results", "card_detector"
)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Nouveau répertoire pour les sauvegardes communes
TEST_COMMON_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "test_results", "all_extracted_cards"
)
os.makedirs(TEST_COMMON_RESULTS_DIR, exist_ok=True)


class TestCardDetector:

    @pytest.mark.parametrize(
        "image_name",
        [
            "card_normal_light.png",
            "ebay.png",
            "ebay_low.png",
            "ebay_high.png",
            "ebay2.png",
            "collection.png",
            "ebay5.png",
            "wattapik.png",
            "wattapik2.png",
            "wattapik3.png",
        ],
    )
    def test_detect_cards(self, image_name):
        image_path = os.path.join(TEST_ASSETS_DIR, image_name)
        output_dir = os.path.join(TEST_RESULTS_DIR, os.path.splitext(image_name)[0])
        os.makedirs(output_dir, exist_ok=True)

        warped_images, contours = detect_cards(
            image_path,
            debug=True,
            method="canny",
            output_dir=output_dir,
            common_output_dir=TEST_COMMON_RESULTS_DIR,  # Ajout du dossier commun
        )

        # Vérifications
        assert isinstance(warped_images, list), "La sortie doit être une liste d'images"
        assert isinstance(
            contours, list
        ), "Les contours doivent être retournés sous forme de liste"

        if len(contours) == 0:
            print(f"[INFO] Aucun contour détecté dans {image_name}")
        else:
            for i, contour in enumerate(contours):
                assert isinstance(contour, np.ndarray)
                assert (
                    contour.shape[0] == 4
                ), f"Contour #{i} n'est pas un quadrilatère : {contour.shape}"
                assert cv2.isContourConvex(contour), f"Contour #{i} détecté non convexe"
                assert warped_images[i] is not None, f"Image découpée #{i} est None"
