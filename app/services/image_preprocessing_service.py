import os
import cv2


def preprocess_image(image_path, method="canny", output_dir="preprocessing_output"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters definition for each method, avec ajout de "precise"
    processing_configs = {
        "canny": {
            "gaussian_blur_kernel": (9, 9),
            "canny_threshold1": 80,
            "canny_threshold2": 200,
            "close_kernel_size": (20, 20),
        },
        "light": {
            "gaussian_blur_kernel": (5, 5),
            "canny_threshold1": 40,
            "canny_threshold2": 100,
            "close_kernel_size": (7, 7),
        },
        "spec": {
            "gaussian_blur_kernel": (9, 9),
            "canny_threshold1": 30,
            "canny_threshold2": 150,
            "close_kernel_size": (20, 20),
        },
        "precise": {
            "gaussian_blur_kernel": (3, 3),
            "canny_threshold1": 10,
            "canny_threshold2": 70,
            "close_kernel_size": (3, 3),
        },
    }

    config = processing_configs.get(method)
    if config is None:
        raise ValueError(
            f"Preprocessing method '{method}' not supported. "
            f"Available methods: {list(processing_configs.keys())}"
        )

    blurred = cv2.GaussianBlur(gray, config["gaussian_blur_kernel"], 0)
    edged = cv2.Canny(blurred, config["canny_threshold1"], config["canny_threshold2"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config["close_kernel_size"])
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # --- Enregistrement du résultat ---
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_{method}.png")
    cv2.imwrite(output_path, edged)
    print(f"Image prétraitée sauvegardée dans : {output_path}")

    return image, edged
