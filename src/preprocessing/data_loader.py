"""Data loading and management utilities."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
import re

from src.utils import get_logger
logger = get_logger(__name__)

class DataLoader:
    """Data loader for retinal vessel images."""
    
    def __init__(self, config: Dict):
        """Initialize data loader."""

        self.config = config
        self.image_dir = config['data']['image_dir']
        self.train_dir = config['data']['train_dir']
        self.mask_dir = config['data']['mask_dir']
        self.ground_truth_dir = config['data']['ground_truth_dir']
    
    @staticmethod
    def _normalize_stem(stem: str) -> str:
        """Remove _L/_R suffix from filename stem."""
        
        return re.sub(r'_(?:l|r)$', '', stem, flags=re.IGNORECASE)

    def get_image_files(self, directory: str) -> List[str]:
        """Get list of image files in directory."""

        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
        
        logger.info(f"Found {len(files)} images in {directory}")
        return sorted(files)
    
    def load_image(self, filepath: str, grayscale: bool = False) -> Optional[np.ndarray]:
        """Load image from file."""

        if not os.path.exists(filepath):
            logger.error(f"Image file not found: {filepath}")
            return None
        
        if grayscale:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(filepath)

            # Convert BGR to RGB
            if image is not None and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            logger.error(f"Failed to load image: {filepath}")
            
        return image

    def load_image_pair(self, image_name: str, image_dir_override: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and corresponding ground truth mask."""

        img_dir = image_dir_override if image_dir_override is not None else self.image_dir
        image_path = os.path.join(img_dir, image_name)
        image = self.load_image(image_path)

        # Get base name and normalized base
        base_name = os.path.splitext(image_name)[0]
        norm_base = self._normalize_stem(base_name)

        # Search for mask with common extensions
        mask_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        mask = None
        for ext in mask_extensions:
            mask_name = f"{norm_base}{ext}"
            mask_path = os.path.join(self.ground_truth_dir, mask_name)
            if os.path.exists(mask_path):
                mask = self.load_image(mask_path, grayscale=True)
                break

        if mask is None:
            logger.warning(f"Ground truth mask not found for {image_name} (searched by base '{norm_base}')")

        return image, mask

    def load_roi_mask(self, image_name: str) -> Optional[np.ndarray]:
        """Load ROI mask for image."""

        # Get base name and normalized base
        base_name = os.path.splitext(image_name)[0]
        norm_base = self._normalize_stem(base_name)
        mask_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']

        for ext in mask_extensions:
            mask_name = f"{norm_base}_mask{ext}"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                return self.load_image(mask_path, grayscale=True)

        logger.warning(f"ROI mask not found for {image_name} (searched by base '{norm_base}')")
        return None

    def load_dataset(self, directory: str = None,
                    load_masks: bool = True,
                    limit: Optional[int] = None,
                    canonicalize_left: bool = False
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Load dataset from directory."""

        if directory is None:
            directory = self.image_dir

        # Get list of image files
        image_files = self.get_image_files(directory)
        if limit:
            image_files = image_files[:limit]

        images, masks, filenames = [], [], []

        # Helper to check if image is right eye
        def _is_right_eye(name: str) -> bool:
            stem = os.path.splitext(name)[0]
            return re.search(r'_(?:r)$', stem, flags=re.IGNORECASE) is not None

        # Load images and masks
        for filename in tqdm(image_files, desc="Loading images"):
            if load_masks:
                image, mask = self.load_image_pair(filename, image_dir_override=directory)
                if image is not None and mask is not None:
                    if canonicalize_left and _is_right_eye(filename):
                        image = np.fliplr(image)
                        mask  = np.fliplr(mask)
                    images.append(image)
                    masks.append(mask)
                    filenames.append(filename)
            else:
                image_path = os.path.join(directory, filename)
                image = self.load_image(image_path)
                if image is not None:
                    if canonicalize_left and _is_right_eye(filename):
                        image = np.fliplr(image)
                    images.append(image)
                    filenames.append(filename)

        if not images:
            logger.error("No training images found!")
        else:
            logger.info(f"Successfully loaded {len(images)} images")

        return images, masks if load_masks else [], filenames
