import os
import requests
from PIL import Image
from typing import Dict, Any, List

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY_GRADING") or "dummy_key"
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID_GRADING") or "sleeved-global-grade-7q5xt/7"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}"


def class_to_number(class_name: str) -> int:
    """Convertit 'PSA_X' en entier X"""
    try:
        return int(class_name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def weighted_psa_from_predictions(predictions: List[Dict[str, Any]]):
    """Calcule le PSA pondéré à partir des prédictions Roboflow"""
    top_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)[
        :3
    ]
    total_conf = sum(p.get("confidence", 0) for p in top_preds)
    if total_conf == 0:
        return None, None

    weighted_psa = (
        sum(class_to_number(p["class"]) * p["confidence"] for p in top_preds)
        / total_conf
    )
    final_psa = round(weighted_psa)
    return round(weighted_psa, 2), final_psa


def preprocess_image(image_path: str, max_width: int = 1024) -> str:
    """Redimensionne l'image si nécessaire pour Roboflow"""
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size)
        temp_path = "/tmp/temp_resized.jpg"
        img.save(temp_path)
        return temp_path
    return image_path


def grade_card(image_path: str) -> Dict[str, Any]:
    """Envoie l'image à Roboflow et retourne les résultats formatés"""
    processed_path = preprocess_image(image_path)
    with open(processed_path, "rb") as f:
        url = f"{API_URL}?api_key={ROBOFLOW_API_KEY}&format=json&confidence=0"
        response = requests.post(url, files={"file": f})

    response.raise_for_status()
    result = response.json()

    predictions = result.get("predictions", [])
    weighted, final_psa = weighted_psa_from_predictions(predictions)

    # 🔹 Formatage pour le nouveau schéma
    top_class_matchs = [
        {
            "card_class": p.get("class", "unknown"),
            "confidence": round(p.get("confidence", 0) * 100, 2),
        }
        for p in predictions
    ]

    return {
        "weighted": weighted,
        "average_card_class": f"PSA_{final_psa}" if final_psa else "unknown",
        "top_class_matchs": top_class_matchs,
    }
