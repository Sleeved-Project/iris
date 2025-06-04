# app/services/image_preprocessing_service.py

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessingService:
    """
    Image preprocessing service to optimize card edge detection.
    Converts raw images (PIL.Image.Image) into a format better suited for
    edge detection through operations like grayscale conversion, Gaussian blur,
    and brightness/contrast normalization.
    """

    @staticmethod
    def _pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
        """
        Converts a PIL (Pillow) image to an OpenCV-compatible NumPy array (BGR).
        """
        # Convert to RGB if not already for consistent processing
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # Convert PIL RGB to OpenCV BGR
        opencv_image = np.array(pil_image)
        return cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _opencv_to_pil(opencv_image: np.ndarray) -> Image.Image:
        """
        Converts an OpenCV-compatible NumPy array (BGR) to a PIL (Pillow) image.
        """
        # Convert OpenCV BGR to PIL RGB
        # Check if it's a 3-channel image
        if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(opencv_image)

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """
        Converts an image to grayscale.
        This step is crucial because most edge detection algorithms (like Canny)
        perform better on grayscale images.

        Args:
            image (Image.Image): The input image in PIL format.

        Returns:
            Image.Image: The grayscale converted image in PIL format.
        """
        cv_image = ImagePreprocessingService._pil_to_opencv(image)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        return ImagePreprocessingService._opencv_to_pil(gray_image)

    @staticmethod
    def apply_gaussian_blur(
        image: Image.Image, kernel_size: tuple = (5, 5)
    ) -> Image.Image:
        """
        Applies a Gaussian blur to the image to reduce noise.
        Gaussian blur is essential before edge detection to smooth fine details
        and noise that might be interpreted as edges. A (5,5) kernel is a good
        compromise for most images, sufficiently reducing noise without
        excessively blurring important edges.

        Args:
            image (Image.Image): The input image in PIL format.
            kernel_size (tuple): The size of the Gaussian blur kernel (must be
                                 odd and positive). E.g., (5, 5) for a 5x5 pixel
                                 kernel.

        Returns:
            Image.Image: The blurred image in PIL format.
        """
        if not all(k % 2 == 1 for k in kernel_size) or any(k <= 0 for k in kernel_size):
            raise ValueError(
                "kernel_size must be odd and positive for both dimensions."
            )

        cv_image = ImagePreprocessingService._pil_to_opencv(image)
        # Ensure the image is grayscale for Gaussian blur if it's not already.
        # However, cv2.GaussianBlur can work on color images, but for Canny,
        # grayscale is preferred. If the full pipeline first calls
        # convert_to_grayscale, then the input here will already be grayscale.
        # For unit function flexibility, we convert it here if necessary.
        if (
            len(cv_image.shape) == 3
        ):  # If it's a color image, convert to grayscale first
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(cv_image, kernel_size, 0)
        return ImagePreprocessingService._opencv_to_pil(blurred_image)

    @staticmethod
    def normalize_brightness_contrast(image: Image.Image) -> Image.Image:
        """
        Normalizes the image's brightness and contrast using adaptive histogram
        equalization (CLAHE).
        This helps improve edge visibility under varying lighting conditions
        (low light, overexposure, reflections) without over-amplifying noise
        as global equalization would.

        Optimal Parameters:
        - clipLimit: The contrast threshold for histogram re-sampling.
                     A value of 2.0 to 4.0 is often a good starting point.
                     A higher value increases contrast.
        - tileGridSize: The size of the grid (in pixels) for which equalization
                        is applied locally. A size of (8, 8) is common.
                        Smaller tiles capture more local details but can
                        amplify noise; larger tiles smooth more.

        Args:
            image (Image.Image): The input image in PIL format.

        Returns:
            Image.Image: The image with normalized brightness/contrast in PIL format.
        """
        cv_image = ImagePreprocessingService._pil_to_opencv(image)

        # CLAHE only works on grayscale images.
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Create the CLAHE object with optimal parameters
        # clipLimit=3.0: A contrast threshold of 3.0 is a good balance for
        # improving contrast without introducing too much noise.
        # tileGridSize=(8, 8): An 8x8 pixel grid is a standard tile size that
        # works well for most images, allowing local adaptation without being
        # too granular.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        normalized_image = clahe.apply(cv_image)

        return ImagePreprocessingService._opencv_to_pil(normalized_image)

    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """
        Executes the complete image preprocessing pipeline.
        1. Grayscale conversion.
        2. Brightness and contrast normalization (CLAHE).
        3. Gaussian blur application for noise reduction.

        The order of these operations is important:
        - Grayscale first to simplify calculations and adapt subsequent
          algorithms.
        - Normalization before blur to ensure contrast is well distributed
          across the image before blurring can potentially "mix" pixel values.
        - Gaussian blur last, just before edge detection, to ensure that noise
          is minimized on the final image passed to Canny.

        Args:
            image (Image.Image): The input image in PIL format.

        Returns:
            Image.Image: The preprocessed image in PIL format, ready for edge
                         detection.
        """
        # Step 1: Convert to grayscale
        # Necessary for CLAHE and Canny.
        grayscale_image = ImagePreprocessingService.convert_to_grayscale(image)

        # Step 2: Normalize brightness/contrast
        # Improves pixel intensity distribution, which is beneficial for edge
        # detection in varying lighting conditions.
        normalized_image = ImagePreprocessingService.normalize_brightness_contrast(
            grayscale_image
        )

        # Step 3: Apply Gaussian blur
        # Reduces image noise, making detected edges sharper and less fragmented.
        # A (5,5) kernel is a good balance for reducing noise while preserving
        # card edges.
        blurred_image = ImagePreprocessingService.apply_gaussian_blur(
            normalized_image, kernel_size=(5, 5)
        )

        return blurred_image
