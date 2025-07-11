import os
import pytest
import cv2
import numpy as np

# Import du script à tester
from app.services.contour_detection_service import detect_cards
from app.services.image_preprocessing_service import preprocess_image # Added for direct path testing if needed

# SOLUTION: Ensure the path is absolute and normalized
# This resolves '..' and converts the path to its absolute form,
# preventing potential issues with cv2.imread on different OS.
TEST_ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), # Current directory: .../iris/tests/unit/services
        "..",                      # Up to: .../iris/tests/unit
        "..",                      # Up to: .../iris/tests
        "assets",                  # Into: .../iris/tests/assets
        "test_images"              # Into: .../iris/tests/assets/test_images
    )
)

# Répertoires pour les résultats des tests
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
    def test_detect_cards(self, image_name):
        # Construction du chemin complet de l'image
        image_path = os.path.join(TEST_ASSETS_DIR, image_name)

        # Ajout d'un print pour le débogage sur le runner CI/CD
        # Cela affichera le chemin exact que cv2.imread va essayer d'ouvrir
        print(f"Attempting to load image from: {image_path}")

        # Création du répertoire de sortie spécifique à l'image
        output_dir = os.path.join(TEST_RESULTS_DIR, os.path.splitext(image_name)[0])
        os.makedirs(output_dir, exist_ok=True)

        # Appel de la fonction à tester
        warped_images, contours = detect_cards(
            image_path,
            debug=True,
            method="canny",  # Utilisation de la méthode "canny" pour le test
            output_dir=output_dir,
            common_output_dir=TEST_COMMON_RESULTS_DIR,  # Ajout du dossier commun
        )

        # Vérifications des résultats
        assert isinstance(warped_images, list), "La sortie doit être une liste d'images"
        assert isinstance(
            contours, list
        ), "Les contours doivent être retournés sous forme de liste"

        if len(contours) == 0:
            print(f"[INFO] Aucun contour détecté dans {image_name}")
        else:
            for i, contour in enumerate(contours):
                # Vérifie que chaque contour est un tableau numpy
                assert isinstance(contour, np.ndarray)
                # Vérifie que le contour a 4 points (pour un quadrilatère)
                assert (
                    contour.shape[0] == 4
                ), f"Contour #{i} n'est pas un quadrilatère : {contour.shape}"
                # Vérifie que le contour est convexe
                assert cv2.isContourConvex(contour), f"Contour #{i} détecté non convexe"
                # Vérifie que l'image découpée n'est pas None
                assert warped_images[i] is not None, f"Image découpée #{i} est None"
