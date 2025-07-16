import cv2


def preprocess_image(image_path, method="canny"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processing_configs = {
        "canny": {
            "gaussian_blur_kernel": (9, 9),
            "canny_threshold1": 70,
            "canny_threshold2": 200,
            "close_kernel_size": (15, 15),
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

    return image, edged
