# image_preprocessing_service.py
import cv2
import numpy as np  # Importez numpy si vous utilisez np.array ou d'autres fonctions numpy ici, même si ce n'est pas le cas directement pour les paramètres Canny/kernel.


def preprocess_image(image_path, method="canny"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Définition des paramètres pour chaque méthode dans un dictionnaire
    # Chaque clé est le nom de la méthode, et la valeur est un dictionnaire de ses paramètres
    # incluant (gaussian_blur_kernel, canny_threshold1, canny_threshold2, close_kernel_size)
    processing_configs = {
        "canny": {
            "gaussian_blur_kernel": (9, 9),
            "canny_threshold1": 70,
            "canny_threshold2": 200,
            "close_kernel_size": (15, 15),
        },
        "light": {
            "gaussian_blur_kernel": (5, 5),  # Moins de flou pour plus de détails
            "canny_threshold1": 40,
            "canny_threshold2": 100,
            "close_kernel_size": (7, 7),  # Kernel plus petit pour fermeture
        },
        "spec": {
            "gaussian_blur_kernel": (
                9,
                9,
            ),  # Peut être ajusté si "spec" a besoin de plus/moins de flou
            "canny_threshold1": 30,
            "canny_threshold2": 150,
            "close_kernel_size": (20, 20),  # Kernel encore plus grand
        },
    }

    # Récupérer la configuration pour la méthode demandée
    config = processing_configs.get(method)
    if config is None:
        raise ValueError(
            f"Méthode de prétraitement '{method}' non supportée. Méthodes disponibles : {list(processing_configs.keys())}"
        )

    # Appliquer les paramètres de la configuration choisie
    blurred = cv2.GaussianBlur(gray, config["gaussian_blur_kernel"], 0)
    edged = cv2.Canny(blurred, config["canny_threshold1"], config["canny_threshold2"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config["close_kernel_size"])
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return (
        image,
        edged,
    )
