# tests/unit/services/test_image_preprocessing_service.py

import pytest
import os
import numpy as np
from PIL import Image, ImageChops  # Import ImageChops here
import cv2

from app.services.image_preprocessing_service import ImagePreprocessingService

# Path to the test images directory
TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "test_images"
)

# Directory to save preprocessed images for visual inspection
# Create this directory manually or ensure it's created by the script
OUTPUT_IMAGES_DIR = os.path.join(TEST_ASSETS_DIR, "preprocessed_outputs")
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)


# Create test images if they don't exist for simple tests
# For more robust tests, use real images in TEST_ASSETS_DIR
@pytest.fixture(scope="module")
def sample_image():
    """Fixture for a simple in-memory PIL image."""
    # Create a 100x100 RGB image
    img = Image.new("RGB", (100, 100), color="red")
    # Add a blue square to create edges
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (0, 0, 255))
    return img


@pytest.fixture(scope="module")
def card_normal_light_image():
    """Loads a well-lit card image from assets."""
    img_path = os.path.join(TEST_ASSETS_DIR, "card_normal_light.png")
    if not os.path.exists(img_path):
        pytest.skip(
            f"Test image not found: {img_path}. Create test images "
            f"in {TEST_ASSETS_DIR}"
        )
    return Image.open(img_path).convert("RGB")


@pytest.fixture(scope="module")
def card_low_light_image():
    """Loads an underexposed card image from assets."""
    img_path = os.path.join(TEST_ASSETS_DIR, "card_low_light.png")
    if not os.path.exists(img_path):
        pytest.skip(
            f"Test image not found: {img_path}. Create test images "
            f"in {TEST_ASSETS_DIR}"
        )
    return Image.open(img_path).convert("RGB")


@pytest.fixture(scope="module")
def card_overexposed_image():
    """Loads an overexposed card image from assets."""
    img_path = os.path.join(TEST_ASSETS_DIR, "card_overexposed.png")
    if not os.path.exists(img_path):
        pytest.skip(
            f"Test image not found: {img_path}. Create test images "
            f"in {TEST_ASSETS_DIR}"
        )
    return Image.open(img_path).convert("RGB")


# Utility functions for tests


def _pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Helper to convert PIL to OpenCV (BGR)."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    opencv_image = np.array(pil_image)
    return cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)


def _get_image_diff_pixels(img1: Image.Image, img2: Image.Image) -> int:
    """
    Calculates the number of differing pixels between two PIL images.
    Images must have the same size and mode.
    """
    if img1.size != img2.size or img1.mode != img2.mode:
        raise ValueError("Images must have the same size and mode.")
    diff = ImageChops.difference(img1, img2)
    # Convert to grayscale to count non-black pixels
    diff_gray = diff.convert("L")
    # Count non-zero (differing) pixels
    diff_array = np.array(diff_gray)
    return np.count_nonzero(diff_array)


class TestImagePreprocessingService:

    def test_pil_to_opencv_and_back(self, sample_image):
        """Tests round-trip conversion between PIL and OpenCV."""
        cv_image = ImagePreprocessingService._pil_to_opencv(sample_image)
        assert isinstance(cv_image, np.ndarray)
        # BGR channels
        assert cv_image.shape == (sample_image.height, sample_image.width, 3)

        pil_back = ImagePreprocessingService._opencv_to_pil(cv_image)
        assert isinstance(pil_back, Image.Image)
        assert pil_back.size == sample_image.size
        # Tolerance for small variations
        assert _get_image_diff_pixels(sample_image, pil_back) < 5

    def test_convert_to_grayscale(self, sample_image):
        """Tests grayscale conversion."""
        grayscale_image = ImagePreprocessingService.convert_to_grayscale(sample_image)
        assert isinstance(grayscale_image, Image.Image)
        assert grayscale_image.mode == "L"  # 'L' for grayscale
        assert grayscale_image.size == sample_image.size

        # Verify that the image is indeed grayscale
        # (all channels are equal for a pixel if 3 channels)
        cv_gray = _pil_to_opencv(grayscale_image)
        # In grayscale, dimensions are (H, W) or (H, W, 1)
        # If it's (H, W, 3), all 3 channels must be equal.
        if len(cv_gray.shape) == 3:
            assert np.all(cv_gray[:, :, 0] == cv_gray[:, :, 1])
            assert np.all(cv_gray[:, :, 1] == cv_gray[:, :, 2])

    @pytest.mark.parametrize("kernel", [(3, 3), (5, 5), (7, 7)])
    def test_apply_gaussian_blur(self, sample_image, kernel):
        """Tests applying Gaussian blur with different kernel sizes."""
        blurred_image = ImagePreprocessingService.apply_gaussian_blur(
            sample_image, kernel_size=kernel
        )
        assert isinstance(blurred_image, Image.Image)
        assert blurred_image.size == sample_image.size
        # Verify that the image is different from the original
        # (i.e., blur has been applied)
        # Convert sample_image to grayscale for comparison because
        # blurred_image is converted to grayscale by apply_gaussian_blur
        assert _get_image_diff_pixels(sample_image.convert("L"), blurred_image) > 0

        # Tests that blurring a completely blurred image doesn't change it (edge case)
        already_blurred_img = Image.new("RGB", (100, 100), color="blue")
        blurred_again = ImagePreprocessingService.apply_gaussian_blur(
            already_blurred_img, kernel_size=kernel
        )
        # Convert for comparison
        assert (
            _get_image_diff_pixels(already_blurred_img.convert("L"), blurred_again) == 0
        )

    def test_apply_gaussian_blur_invalid_kernel(self, sample_image):
        """Tests kernel size validation for Gaussian blur."""
        with pytest.raises(ValueError, match="kernel_size must be odd and positive"):
            ImagePreprocessingService.apply_gaussian_blur(
                sample_image, kernel_size=(4, 4)
            )
        with pytest.raises(ValueError, match="kernel_size must be odd and positive"):
            ImagePreprocessingService.apply_gaussian_blur(
                sample_image, kernel_size=(5, 0)
            )

    def test_normalize_brightness_contrast(self, sample_image):
        """Tests brightness/contrast normalization (CLAHE)."""
        normalized_image = ImagePreprocessingService.normalize_brightness_contrast(
            sample_image
        )
        assert isinstance(normalized_image, Image.Image)
        assert normalized_image.mode == "L"  # CLAHE produces a grayscale image
        assert normalized_image.size == sample_image.size

        # For a more significant test, we can verify that the histogram has changed
        # (which is the purpose of normalization).
        # This is a heuristic test; a perfect comparison
        # is difficult without a reference.
        original_gray_cv = cv2.cvtColor(
            _pil_to_opencv(sample_image), cv2.COLOR_BGR2GRAY
        )
        hist_original = cv2.calcHist([original_gray_cv], [0], None, [256], [0, 256])
        hist_normalized = cv2.calcHist(
            [np.array(normalized_image)], [0], None, [256], [0, 256]
        )

        # Ensure histograms are not identical (indicates processing)
        assert not np.array_equal(hist_original, hist_normalized)

    def test_preprocess_image_pipeline(self, sample_image):
        """Tests the complete image preprocessing pipeline."""
        # Save original image for visual inspection
        sample_image.save(os.path.join(OUTPUT_IMAGES_DIR, "original_sample_image.png"))

        preprocessed_image = ImagePreprocessingService.preprocess_image(sample_image)
        assert isinstance(preprocessed_image, Image.Image)
        assert preprocessed_image.mode == "L"  # Final result is grayscale
        assert preprocessed_image.size == sample_image.size

        # Verify that the preprocessed image is different from the original
        assert _get_image_diff_pixels(sample_image.convert("L"), preprocessed_image) > 0

        # Save preprocessed image for visual inspection
        preprocessed_image.save(
            os.path.join(OUTPUT_IMAGES_DIR, "preprocessed_sample_image.png")
        )

    @pytest.mark.parametrize(
        "image_fixture",
        [
            "card_normal_light_image",
            "card_low_light_image",
            "card_overexposed_image",
        ],
    )
    def test_preprocess_image_with_varying_lighting(self, request, image_fixture):
        """
        Tests the complete pipeline on images with
        varying lighting conditions.
        This test verifies execution without errors and the nature of the
        resulting images, indicating potential improvement for edge detection.
        """
        original_image = request.getfixturevalue(image_fixture)

        # Save original image for visual inspection
        original_image.save(
            os.path.join(OUTPUT_IMAGES_DIR, f"original_{image_fixture}.png")
        )

        preprocessed_image = ImagePreprocessingService.preprocess_image(original_image)

        assert isinstance(preprocessed_image, Image.Image)
        assert preprocessed_image.mode == "L"  # Always grayscale
        assert preprocessed_image.size == original_image.size

        # Heuristic check: Preprocessing should modify the image.
        # We cannot verify "edge quality" here,
        # but execution and expected output.
        original_gray = original_image.convert("L")
        assert _get_image_diff_pixels(original_gray, preprocessed_image) > 0, (
            f"The preprocessed image for '{image_fixture}' "
            f"does not differ from the original."
        )

        # Save preprocessed images for visual inspection
        file_name = f"preprocessed_{image_fixture}.png"
        preprocessed_image.save(os.path.join(OUTPUT_IMAGES_DIR, file_name))
