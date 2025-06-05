# image_preprocessing.py

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessingService:
    def __init__(self, debug=False, mode='canny', use_advanced=True):
        self.debug = debug
        self.mode = mode
        self.use_advanced = use_advanced

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        return image.convert('L')

    @staticmethod
    def apply_gaussian_blur(image: Image.Image, kernel_size=(5, 5)) -> Image.Image:
        img_np = np.array(image)
        blurred = cv2.GaussianBlur(img_np, kernel_size, 0)
        return Image.fromarray(blurred)

    @staticmethod
    def normalize_brightness_contrast(image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            norm = cv2.equalizeHist(img_np)
        else:
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            norm = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(norm)

    def preprocess_pil_pipeline(self, pil_img: Image.Image) -> Image.Image:
        gray = self.convert_to_grayscale(pil_img)
        blurred = self.apply_gaussian_blur(gray)
        normalized = self.normalize_brightness_contrast(blurred)
        return normalized

    def preprocess(self, image):
        """
        Global OpenCV preprocessing: conversion, blur, and edges
        """
        if self.use_advanced:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_img = self.preprocess_pil_pipeline(pil_img)
            image = np.array(pil_img)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        if self.mode == 'canny':
            edged = cv2.Canny(blurred, 75, 200)
        else:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.debug:
            cv2.imwrite("debug_preprocess.png", edged)
            print("[DEBUG] Preprocessed image saved as debug_preprocess.png")

        return edged
