import os
import pytest
import numpy as np
import cv2

from app.services.image_preprocessing_service import preprocess_image

# Répertoire des images de test
TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)

# Répertoire pour enregistrer les résultats du prétraitement
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results", "preprocess")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class TestPreprocessImage:

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
    def test_preprocess_image_output(self, image_name):
        image_path = os.path.join(TEST_ASSETS_DIR, image_name)
        original, edged = preprocess_image(image_path, method="canny")

        # Vérifie que l'image originale a bien été chargée
        assert original is not None, "L'image originale ne doit pas être None"
        assert isinstance(
            original, np.ndarray
        ), "L'image originale doit être un tableau numpy"
        assert (
            original.ndim == 3 and original.shape[2] == 3
        ), "L'image originale doit être en couleur (3 canaux)"

        # Vérifie que l'image prétraitée est bien générée
        assert edged is not None, "L'image prétraitée ne doit pas être None"
        assert isinstance(
            edged, np.ndarray
        ), "L'image prétraitée doit être un tableau numpy"
        assert (
            edged.ndim == 2
        ), "L'image prétraitée doit être une image binaire ou en niveaux de gris (2D)"

        # Sauvegarde les résultats pour inspection visuelle
        image_base = os.path.splitext(image_name)[0]
        cv2.imwrite(
            os.path.join(TEST_RESULTS_DIR, f"{image_base}_original.png"), original
        )
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, f"{image_base}_edged.png"), edged)
