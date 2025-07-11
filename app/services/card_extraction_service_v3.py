import os
import supervision as sv
import requests
import cv2
import numpy as np
from typing import List, Optional


class CardExtractionService:
    """
    Service pour extraire, détourer et redresser les cartes
    depuis une image via un modèle Roboflow avec masque.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        output_dir: str = "card_extraction_output",
        save_outputs: bool = False,
        confidence_threshold: float = 0.9,  # seuil par défaut 90%
    ):
        if not api_key or "VOTRE_CLE" in api_key:
            raise ValueError("La clé API Roboflow n'est pas configurée.")

        self.api_key = api_key
        self.model_id = model_id
        self.output_dir = output_dir
        self.save_outputs = save_outputs
        self.confidence_threshold = confidence_threshold

        os.makedirs(self.output_dir, exist_ok=True)

    def _infer_via_http(self, image_path: str) -> dict:
        """
        Effectue une inference en POST multipart/form-data vers Roboflow Serverless.
        Utilise le paramètre api_key en query string, pas de header Authorization.
        """
        url = f"https://serverless.roboflow.com/{self.model_id}"
        params = {"api_key": self.api_key}

        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}

            response = requests.post(url, params=params, files=files)
            response.raise_for_status()
            return response.json()

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordonnage des 4 points d'un quadrilatère:
        top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def straighten_masked_object_perspective(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return None

        cnt = contours[0]

        # Approximation polygonale pour trouver les coins (on espère 4)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) != 4:
            # Si pas quadrilatère, retomber sur méthode classique (rotation simple)
            return self.straighten_masked_object(img, mask)

        pts = approx.reshape(4, 2)
        rect = self.order_points(pts)

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped

    def straighten_masked_object(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return None

        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        width, height = rect[1]

        if width > height:
            angle += 90
            width, height = height, width

        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = img[y : y + h, x : x + w]
        cropped_mask = mask_uint8[y : y + h, x : x + w]

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(cropped_img, M, (w, h), flags=cv2.INTER_CUBIC)
        rotated_mask = cv2.warpAffine(cropped_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        masked_rotated = cv2.bitwise_and(rotated_img, rotated_img, mask=rotated_mask)

        ys, xs = np.where(rotated_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            final_img = masked_rotated[y1 : y2 + 1, x1 : x2 + 1]
        else:
            final_img = masked_rotated

        return final_img

    def fix_upside_down(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
        )

        top = thresh[: h // 3, :]
        bottom = thresh[-h // 3 :, :]

        top_mass = cv2.countNonZero(top)
        bottom_mass = cv2.countNonZero(bottom)

        if bottom_mass > top_mass:
            return cv2.rotate(img, cv2.ROTATE_180)
        return img

    def extract_cards_from_image(self, image_path: str) -> List[np.ndarray]:
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        result_json = self._infer_via_http(image_path)
        detections = sv.Detections.from_inference(result_json)

        extracted_cards = []

        # On récupère la liste des scores dans le JSON
        # (assumant que c’est dans result_json["predictions"])
        scores = [
            pred.get("confidence", 0) for pred in result_json.get("predictions", [])
        ]

        for i, (mask, confidence) in enumerate(zip(detections.mask, scores)):
            if confidence < self.confidence_threshold:
                print(
                    f"""Détection {i} ignorée, confiance
                    {confidence:.2f} < {self.confidence_threshold}"""
                )
                continue

            card = self.straighten_masked_object_perspective(image, mask)
            if card is not None:
                card = self.fix_upside_down(card)

                # Version originale redressée
                extracted_cards.append(card)
                if self.save_outputs:
                    output_file = os.path.join(
                        self.output_dir, f"{image_name}_extracted_{i}_normal.png"
                    )
                    cv2.imwrite(output_file, card)
                    print(
                        f"""Carte extraite et redressée
                          sauvegardée : {output_file}"""
                    )

                # Version retournée à l'envers (180 degrés)
                card_flipped = cv2.rotate(card, cv2.ROTATE_180)
                extracted_cards.append(card_flipped)
                if self.save_outputs:
                    output_file_flipped = os.path.join(
                        self.output_dir, f"{image_name}_extracted_{i}_flipped.png"
                    )
                    cv2.imwrite(output_file_flipped, card_flipped)
                    print(
                        f"""Carte extraite et retournée sauvegardée :
                        {output_file_flipped}"""
                    )
                    # --- Ajout de l'image avec les boîtes
                    # et masques dessinés ---
                    # --- Ajout de l'image avec les boîtes,
                    # masques et scores dessinés ---
                if self.save_outputs:
                    image_with_overlays = image.copy()

                    for i, (box, mask, score) in enumerate(
                        zip(detections.xyxy, detections.mask, scores)
                    ):
                        if score < self.confidence_threshold:
                            continue

                        # Dessiner la boîte englobante (rectangle vert)
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(
                            image_with_overlays, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )

                        # Afficher le score au-dessus de la boîte
                        label = f"{score:.2f}"
                        cv2.putText(
                            image_with_overlays,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # Appliquer le masque en rouge semi-transparent
                        colored_mask = np.zeros_like(
                            image_with_overlays, dtype=np.uint8
                        )
                        colored_mask[mask] = [0, 0, 255]  # Rouge
                        image_with_overlays = cv2.addWeighted(
                            image_with_overlays, 1.0, colored_mask, 0.4, 0
                        )

                    output_overlay_path = os.path.join(
                        self.output_dir, f"{image_name}_with_boxes_masks_scores.png"
                    )
                    cv2.imwrite(output_overlay_path, image_with_overlays)
                    print(
                        f"""Image avec boîtes, masques et scores sauvegardée :
                        {output_overlay_path}"""
                    )

            else:
                print(f"Aucun contour trouvé pour le masque {i}, carte ignorée.")

        return extracted_cards


ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "WpEORCOAhsk8DPf1XxXd")
MODEL_ID = "sleeved-obb/7"  # adapte selon ton modèle Roboflow

card_extraction_service = CardExtractionService(
    api_key=ROBOFLOW_API_KEY,
    model_id=MODEL_ID,
    output_dir="card_extraction_output",
    save_outputs=True,
    confidence_threshold=0.9,  # ici seuil 89%
)
