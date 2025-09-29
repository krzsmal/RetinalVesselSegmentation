"""Image preprocessing utilities for vessel segmentation."""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from skimage import filters, morphology

from src.utils import get_logger
logger = get_logger(__name__)

class ImageProcessor:
    """Image preprocessing utilities."""
        
    @staticmethod
    def morphological_opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply morphological opening operation (remove small spurs/noise)."""
        
        # kernel = morphology.square(kernel_size)
        kernel = morphology.footprint_rectangle((kernel_size, kernel_size))
        return morphology.opening(image, kernel)

    @staticmethod
    def extract_green_channel(image: np.ndarray) -> np.ndarray:
        """Extract green channel from RGB image."""

        if len(image.shape) != 3:
            logger.warning("Image is not RGB, returning as is")
            return image
        
        return image[:, :, 1]
    
    @staticmethod
    def denoise(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """Apply median blur for noise removal."""

        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def normalize(image: np.ndarray, min_val: int = 0, max_val: int = 255) -> np.ndarray:
        """Normalize image to specified range."""

        return cv2.normalize(image, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image)

    @staticmethod
    def frangi_filter(image: np.ndarray, scales: Optional[List[float]] = None,
                      alpha: float = 0.5, beta: float = 0.5, 
                      gamma: float = 15, black_ridges: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Frangi filter for vessel enhancement."""

        if scales is None:
            scales = range(1, 6)
        
        filtered = filters.frangi(image, sigmas=scales, alpha=alpha, beta=beta, gamma=gamma, black_ridges=black_ridges)
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        binary = normalized.copy()
        binary[binary > 0] = 255
        
        return normalized, binary
    
    @staticmethod
    def remove_small_objects(binary_image: np.ndarray, min_size: int = 250) -> np.ndarray:
        """Remove small objects from binary image."""

        bool_image = binary_image > 0
        cleaned = morphology.remove_small_objects(bool_image, min_size=min_size)
        
        return (cleaned * 255).astype(np.uint8)
    
    @staticmethod
    def create_noise_mask(binary_image: np.ndarray, min_size: int = 250) -> np.ndarray:
        """Create a noise mask for the binary image."""

        cleaned = ImageProcessor.remove_small_objects(binary_image, min_size)
        noise_mask = binary_image.copy()
        noise_mask[cleaned > 0] = 0
        noise_mask = 255 - noise_mask
        
        return noise_mask
    
    @staticmethod
    def morphological_closing(image: np.ndarray, kernel_size: int = 8) -> np.ndarray:
        """Apply morphological closing operation."""

        # kernel = morphology.square(kernel_size)
        kernel = morphology.footprint_rectangle((kernel_size, kernel_size))

        return morphology.closing(image, kernel)
    
    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image using bitwise AND."""

        return cv2.bitwise_and(image, mask)
    
    @staticmethod
    def resize_image(image: np.ndarray, scale: float = None, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image by scale factor or to target size."""

        if scale is not None:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            return cv2.resize(image, (width, height))
        elif target_size is not None:
            return cv2.resize(image, target_size)
        else:
            return image
    
    @staticmethod
    def letterbox(image: np.ndarray, desired_size: int = 512, is_mask: bool = False) -> np.ndarray:
        """Resize and pad image to square size while maintaining aspect ratio."""

        old_size = image.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Use INTER_NEAREST for masks to preserve binary values
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_size[1], new_size[0]), interpolation=interpolation)
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        
        color = [0, 0, 0] if len(image.shape) == 3 else 0
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return padded

    @staticmethod
    def remove_letterbox(image: np.ndarray, original_shape: Tuple[int, int], desired_size: int = 512, is_mask: bool = False) -> np.ndarray:
        old_h, old_w = original_shape
        ratio = float(desired_size) / max(original_shape)
        new_h, new_w = int(old_h * ratio), int(old_w * ratio)

        delta_w = desired_size - new_w
        delta_h = desired_size - new_h
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2

        # 1) zdejmij padding (crop do new_h×new_w)
        cropped = image[top:desired_size - bottom, left:desired_size - right]

        # 2) przeskaluj do oryginalnego rozmiaru
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        restored = cv2.resize(cropped, (old_w, old_h), interpolation=interp)
        return restored
