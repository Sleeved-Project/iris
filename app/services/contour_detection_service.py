import cv2
import numpy as np
import os
from app.services.image_preprocessing_service import (
    preprocess_image,
)


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


def detect_cards(
    image_path,
    method="canny",
    min_area=5000,
    aspect_ratio_range=(0.6, 0.85),
    debug=False,
    output_dir=None,
    common_output_dir=None,
):

    image, edged = preprocess_image(image_path, method=method)
    orig = image.copy()
    h, w = image.shape[:2]
    image_area = h * w

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contours = []
    warped_images = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area < min_area or area / image_area > 0.95:
                continue

            ((_, _), (width_rect, height_rect), _) = cv2.minAreaRect(approx)
            if width_rect == 0 or height_rect == 0:
                continue

            aspect_ratio = min(width_rect, height_rect) / max(width_rect, height_rect)
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                continue

            card_contours.append(approx)
            warped = four_point_transform(orig, approx)
            warped_images.append(warped)

            if debug and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(
                    os.path.join(output_dir, f"{image_name}_card_{i}.png"), warped
                )

            if debug and common_output_dir:
                os.makedirs(common_output_dir, exist_ok=True)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(
                    os.path.join(common_output_dir, f"{image_name}_card_{i}.png"),
                    warped,
                )

    return warped_images, card_contours
