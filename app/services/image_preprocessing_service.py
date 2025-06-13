import cv2
import numpy as np

# Preprocessing configuration for multi-color detection
preprocessing_config = {
    "color_ranges": [
        # Format: name, color_space, lower_bound, upper_bound
        ("yellow", "hsv", [15, 20, 100], [45, 255, 255]),
        ("blue", "hsv", [90, 50, 50], [130, 255, 255]),
        ("black", "hsv", [0, 0, 0], [180, 255, 80]),
    ],
    "dilate_kernel": (5, 5),
    "dilate_iterations": 2,
    "erode_iterations": 1,
    "canny_threshold1": 50,
    "canny_threshold2": 150,
    "close_kernel_size": (15, 15),
}


def preprocess_image(image_path, method=None):
    """
    Preprocesses an image to detect card contours using multi-color approach.
    """
    if not image_path or not isinstance(image_path, str):
        raise ValueError("Image path must be a valid string")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return process_multi_color(image, preprocessing_config)


def process_multi_color(image, config):
    """
    Processes image to detect contours of multiple color ranges
    """
    if image is None or image.size == 0:
        return None, None

    combined_mask = None

    for _, color_space, lower, upper in config["color_ranges"]:
        if color_space == "hsv":
            color_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            color_image = image

        mask = cv2.inRange(color_image, np.array(lower), np.array(upper))

        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    if combined_mask is None:
        return image, np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    kernel = np.ones(config["dilate_kernel"], np.uint8)
    combined_mask = cv2.dilate(
        combined_mask, kernel, iterations=config["dilate_iterations"]
    )
    combined_mask = cv2.erode(
        combined_mask, kernel, iterations=config["erode_iterations"]
    )

    edges = cv2.Canny(
        combined_mask, config["canny_threshold1"], config["canny_threshold2"]
    )

    close_kernel = np.ones(config["close_kernel_size"], np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    return image, edges
