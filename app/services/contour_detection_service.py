# contour_detection.py

import cv2
import numpy as np
from PIL import Image
import imagehash
from app.services.image_preprocessing_service import ImagePreprocessingService


class ContourDetectionService:
    def __init__(self, min_width=30, min_height=40,
                 aspect_ratio_range=(0.4, 1.5), debug=False,
                 preprocess_mode='canny', use_advanced_preprocessing=True):

        self.min_width = min_width
        self.min_height = min_height
        self.aspect_ratio_range = aspect_ratio_range
        self.debug = debug

        self.preprocessor = ImagePreprocessingService(
            debug=debug,
            mode=preprocess_mode,
            use_advanced=use_advanced_preprocessing
        )

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image

    def find_card_like_contours(self, image):
        edged = self.preprocessor.preprocess(image)
        height, width = edged.shape[:2]
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        card_contours = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if w > 0.9 * width and h > 0.9 * height:
                    if self.debug:
                        print(f"""[DEBUG] Contour rejected (too large):
                              x={x}, y={y}, w={w}, h={h}""")
                    continue

                if (self.aspect_ratio_range[0]
                    <= aspect_ratio
                    <= self.aspect_ratio_range[1]
                        and w > self.min_width and h > self.min_height):
                    if self.debug:
                        print(f"""[DEBUG] Contour accepted: w={w}, h={h},
                              ratio={aspect_ratio:.2f}""")
                    card_contours.append(approx)

        return card_contours

    def draw_detected_contours(self, image, contours, color=(0, 255, 0), thickness=3):
        image_copy = image.copy()
        for contour in contours:
            cv2.drawContours(image_copy, [contour], -1, color, thickness)
        return image_copy

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def extract_card_hashes(self, image_path, hash_func=imagehash.phash):
        image = self.load_image(image_path)
        contours = self.find_card_like_contours(image)
        hashes = []

        for i, contour in enumerate(contours):
            warped = self.four_point_transform(image, contour)
            pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            hash_val = str(hash_func(pil_img))
            hashes.append(hash_val)

            if self.debug:
                debug_name = f"card_{i}.png"
                cv2.imwrite(debug_name, warped)
                print(f"[DEBUG] Card {i} saved: {debug_name}")
                print(f"[DEBUG] Hash: {hash_val}")

        return hashes

    def detect_cards_in_image(self, image_path, display=True):
        image = self.load_image(image_path)
        card_contours = self.find_card_like_contours(image)

        if self.debug:
            print(f"[DEBUG] {len(card_contours)} card(s) detected")

        output_image = self.draw_detected_contours(image, card_contours)

        if display:
            cv2.imshow("Detected Cards", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return card_contours
