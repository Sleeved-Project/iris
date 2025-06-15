import cv2
import numpy as np
import os
from app.services.image_preprocessing_service import preprocess_image


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _detect_cards_with_method(
    image_path,
    method,
    min_area,
    aspect_ratio_range,
    debug,
    output_dir,
    common_output_dir,
    orig_image,
    image_area,
):
    """
    Fonction utilitaire pour tenter la détection avec une méthode spécifique.
    Retourne (warped_images, card_contours) ou ([], []) si aucune carte n'est trouvée.
    """
    print(f"--- Tentative de détection avec la méthode '{method}' ---")
    _, edged = preprocess_image(image_path, method=method)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_card_contours = []
    temp_warped_images = []

    if not contours:
        print(f"  Aucun contour trouvé avec la méthode '{method}'.")
        return [], []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours):
        # Vous pouvez réactiver les logs détaillés ici si besoin pendant le débogage.
        # print(f"\n--- Analyse du contour n°{i+1} (Méthode: {method}) ---")

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # Maintenir la tolérance

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area < min_area or area / image_area > 0.95:
                # print(f"  REJETÉ (Méthode {method}) : Aire hors plage.")
                continue

            ((_, _), (width_rect, height_rect), _) = cv2.minAreaRect(approx)
            if width_rect == 0 or height_rect == 0:
                # print(f"  REJETÉ (Méthode {method}) : Largeur ou hauteur nulle.")
                continue

            aspect_ratio = min(width_rect, height_rect) / max(width_rect, height_rect)
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                # print(f"  REJETÉ (Méthode {method}) : Rapport d'aspect hors plage.")
                continue

            # print(f"  ACCEPTÉ (Méthode {method}) : Contour n°{i+1}
            # détecté comme carte !")
            temp_card_contours.append(approx)
            warped = four_point_transform(orig_image, approx)
            temp_warped_images.append(warped)

            # Enregistrement des images de débogage avec le nom de la méthode
            if debug and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(
                    os.path.join(output_dir, f"{image_name}_card_{method}_{i}.png"),
                    warped,
                )
            if debug and common_output_dir:
                os.makedirs(common_output_dir, exist_ok=True)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(
                    os.path.join(
                        common_output_dir, f"{image_name}_card_{method}_{i}.png"
                    ),
                    warped,
                )
        # else:
        # print(f"  REJETÉ (Méthode {method}) : Pas 4 coins (a {len(approx)}
        # coins) ou non convexe.")

    return temp_warped_images, temp_card_contours


def detect_cards(
    image_path,
    method="canny",
    min_area=5000,
    aspect_ratio_range=(0.6, 0.95),
    debug=False,
    output_dir=None,
    common_output_dir=None,
):

    image, _ = preprocess_image(image_path, method="canny")
    orig = image.copy()
    h, w = image.shape[:2]
    image_area = h * w

    methods_to_try = [method]
    if method != "canny":
        methods_to_try.append("canny")
    if "light" not in methods_to_try:
        methods_to_try.append("light")
    if "spec" not in methods_to_try:
        methods_to_try.append("spec")

    final_warped_images = []
    final_card_contours = []

    print(f"--- Détection de cartes pour l'image : {os.path.basename(image_path)} ---")
    print(f"Séquence de méthodes de détection : {methods_to_try}")

    for current_method in methods_to_try:
        if not final_card_contours:
            temp_warped_images, temp_card_contours = _detect_cards_with_method(
                image_path=image_path,
                method=current_method,
                min_area=min_area,
                aspect_ratio_range=aspect_ratio_range,
                debug=debug,
                output_dir=output_dir,
                common_output_dir=common_output_dir,
                orig_image=orig,
                image_area=image_area,
            )
            if temp_card_contours:
                final_warped_images = temp_warped_images
                final_card_contours = temp_card_contours
                print(
                    (
                        f"Méthode '{current_method}' a détecté "
                        f"{len(final_card_contours)} carte(s). "
                        "Arrêt de la recherche."
                    )
                )
                break
        else:
            break

    print(
        f"\n--- Fin de la détection. Total de cartes détectées : "
        f"{len(final_card_contours)} ---"
    )
    return final_warped_images, final_card_contours
