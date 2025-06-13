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


def detect_cards(
    image_path,
    debug=True,
    output_dir=None,
    common_output_dir=None,
    method="adaptive",
    min_area=3000,
    aspect_ratio_range=(0.5, 0.9),
):
    """
    Detects card-like objects in images.
    Works with various card borders on different backgrounds.

    Args:
        image_path: Path to the image file
        debug: Whether to save debug images
        output_dir: Directory for debug output
        common_output_dir: Directory for processing steps
        method: Preprocessing method ('adaptive', 'pokemon', 'gradient', 'canny')
        min_area: Minimum area for contour to be considered a card
        aspect_ratio_range: Allowed aspect ratio range for cards

    Returns:
        tuple: (list of warped card images, list of contours)
    """
    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]

    # Create debug directories
    debug_dirs = {}
    if debug:
        debug_dirs["output"] = output_dir or os.path.join(
            "assets/test_images/debug_output", image_base
        )
        debug_dirs["common"] = common_output_dir or os.path.join(
            "assets/test_images/processing_steps"
        )
        debug_dirs["contours"] = os.path.join(
            debug_dirs["common"], "contours", image_base
        )
        debug_dirs["masks"] = os.path.join(debug_dirs["common"], "masks", image_base)

        for dir_path in debug_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    # Preprocess image
    orig_image, processed = preprocess_image(image_path, method=method)

    # Save debug images
    if debug:
        cv2.imwrite(
            os.path.join(debug_dirs["common"], f"{image_base}_original.jpg"), orig_image
        )
        cv2.imwrite(
            os.path.join(debug_dirs["masks"], f"{image_base}_{method}_mask.jpg"),
            processed,
        )

    # Find contours
    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Nombre de contours bruts trouvés: {len(contours)}")

    # Filter and sort contours
    filtered_contours = []
    warped_images = []

    if contours:
        # Sort by descending area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Debug image with all contours
        if debug:
            all_contours_img = orig_image.copy()
            cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(
                os.path.join(debug_dirs["contours"], f"{image_base}_all_contours.jpg"),
                all_contours_img,
            )

            # Show only top 5 contours
            top_contours_img = orig_image.copy()
            for i, cnt in enumerate(contours[:5]):
                color = (0, 255 - i * 40, i * 40)
                cv2.drawContours(top_contours_img, [cnt], -1, color, 3)
                # Add contour number and area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area = cv2.contourArea(cnt)
                    cv2.putText(
                        top_contours_img,
                        f"#{i+1}: {area:.0f}",
                        (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
            cv2.imwrite(
                os.path.join(debug_dirs["contours"], f"{image_base}_top5_contours.jpg"),
                top_contours_img,
            )

        # Process contours
        for i, cnt in enumerate(contours):
            if i >= 10:  # Limit to first 10 contours to save time
                break

            # Approximate contour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Check if approximated contour has 4 points (rectangle)
            if len(approx) == 4:
                # Get rectangle dimensions
                area = cv2.contourArea(approx)
                (_, _), (width, height), angle = cv2.minAreaRect(approx)

                # Avoid division by zero
                if min(width, height) > 0:
                    aspect_ratio = min(width, height) / max(width, height)
                    min_ratio, max_ratio = aspect_ratio_range

                    # Log contour info for debugging
                    print(
                        f"Contour #{i}: points={len(approx)}, "
                        f"convexe={cv2.isContourConvex(approx)}, "
                        f"area={area:.2f}, ratio={aspect_ratio:.2f}, "
                        f"angle={angle:.1f}"
                    )

                    if debug:
                        # Create image to visualize this contour
                        contour_debug = orig_image.copy()
                        cv2.drawContours(contour_debug, [approx], -1, (0, 0, 255), 3)

                        # Add contour info to image
                        info_text = (
                            f"Area: {area:.0f}, Ratio: {aspect_ratio:.2f}\n"
                            f"Convex: {cv2.isContourConvex(approx)}, "
                            f"Points: {len(approx)}"
                        )
                        y0, dy = 50, 30
                        for j, line in enumerate(info_text.split("\n")):
                            y = y0 + j * dy
                            cv2.putText(
                                contour_debug,
                                line,
                                (50, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                            )

                        cv2.imwrite(
                            os.path.join(
                                debug_dirs["contours"], f"{image_base}_contour_{i}.jpg"
                            ),
                            contour_debug,
                        )

                    # Check card criteria
                    if (
                        area >= min_area
                        and min_ratio <= aspect_ratio <= max_ratio
                        and cv2.isContourConvex(approx)
                    ):

                        filtered_contours.append(approx)

                        # Transform perspective
                        warped = four_point_transform(orig_image, approx)
                        warped_images.append(warped)

                        # Save debug images
                        if debug:
                            # Accepted contour with green border
                            accepted_img = orig_image.copy()
                            cv2.drawContours(accepted_img, [approx], -1, (0, 255, 0), 3)
                            cv2.putText(
                                accepted_img,
                                "CARTE DÉTECTÉE!",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )
                            cv2.imwrite(
                                os.path.join(
                                    debug_dirs["output"], f"{image_base}_card_{i}.jpg"
                                ),
                                accepted_img,
                            )

                            # Extracted card
                            cv2.imwrite(
                                os.path.join(
                                    debug_dirs["output"], f"{image_base}_warped_{i}.jpg"
                                ),
                                warped,
                            )

    print(f"Cartes détectées: {len(filtered_contours)}")
    return warped_images, filtered_contours
