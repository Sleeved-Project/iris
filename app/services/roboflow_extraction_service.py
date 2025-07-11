# app/services/roboflow_extraction_service.py

import os
import cv2
from inference_sdk import InferenceHTTPClient
from typing import List
import numpy as np


class RoboflowExtractionService:
    """
    Service pour extraire des objets (cartes) d'une image en utilisant un modèle
    de détection d'objets hébergé sur Roboflow.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        output_dir: str = "roboflow_output",
        upscale: bool = False,
        apply_sharpen: bool = True,
        max_image_dim: int = 1280,  # Réduction de la dimension maximale de l'image
    ):
        if not api_key or "VOTRE_CLE" in api_key:
            raise ValueError("La clé API de Roboflow n'est pas configurée.")

        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com", api_key=api_key
        )
        self.model_id = model_id
        self.output_dir = output_dir
        self.upscale = upscale
        self.apply_sharpen = apply_sharpen
        self.max_image_dim = max_image_dim  # Initialisation de la nouvelle option

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Les cartes extraites seront sauvegardées dans : {self.output_dir}")

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un filtre de netteté à l'image.
        """
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel_sharpen)

    def extract_cards_from_image(self, image_path: str) -> List[np.ndarray]:
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur: Image non trouvée ou illisible : {image_path}")
            return []  # Return empty list if image is not found or unreadable

        # Suppression totale du redimensionnement ici :
        # On envoie directement l'image originale à Roboflow
        print(f"Envoi de l'image à Roboflow (modèle : {self.model_id})...")
        extracted_cards: List[np.ndarray] = []
        try:
            result = self.client.infer(
                image_path, model_id=self.model_id
            )  # Use image_path directly
            print(
                f"{len(result.get('predictions', []))} détections reçues de Roboflow."
            )
        except Exception as e:
            print(f"Erreur durant l'analyse par Roboflow : {e}")
            return []  # Return empty list on Roboflow inference error

        for i, pred in enumerate(result.get("predictions", [])):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]

            # Petite marge autour de la carte (2%)
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(image.shape[1], int(x + w / 2))
            y2 = min(image.shape[0], int(y + h / 2))

            cropped_card = image[y1:y2, x1:x2]

            # Ensure cropped_card is not empty before processing
            if cropped_card.shape[0] == 0 or cropped_card.shape[1] == 0:
                print(
                    f"""Avertissement: Carte rognée vide
                    pour la prédiction {i+1}. Sautée."""
                )
                continue

            processed_card = cropped_card.copy()

            if self.upscale:
                processed_card = cv2.resize(
                    processed_card,
                    (0, 0),
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if self.apply_sharpen:
                processed_card = self._sharpen_image(processed_card)

            extracted_cards.append(processed_card)

            output_filename = os.path.join(
                self.output_dir, f"card_{image_name}_{i+1}.png"
            )
            try:
                cv2.imwrite(
                    output_filename, processed_card, [cv2.IMWRITE_PNG_COMPRESSION, 0]
                )
                print(f"Carte extraite sauvegardée : {output_filename}")
            except Exception as e:
                print(
                    f"Erreur lors de la sauvegarde de la carte {output_filename}: {e}"
                )

        print(f"Fin de l'extraction Roboflow. Retour de {len(extracted_cards)} cartes.")
        return extracted_cards


# --- Instance du Service (exemple) ---
# Chargez la clé API depuis l'environnement ou utilisez une valeur par défaut
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "WpEORCOAhsk8DPf1XxXd")
MODEL_ID = "sleeved/7"

# Création de l'instance avec paramètres qualité
roboflow_service = RoboflowExtractionService(
    api_key=ROBOFLOW_API_KEY,
    model_id=MODEL_ID,
    output_dir="roboflow_output",
    upscale=False,  # Désactivé pour éviter la perte de qualité
    apply_sharpen=True,  # Activé pour mieux détecter les contours plus tard
    max_image_dim=1280,  # Nouvelle limite de dimension pour l'envoi
)
