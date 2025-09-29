"""Traditional image processing model using Frangi filter."""

import numpy as np
import cv2
from typing import Dict, Any, Optional
import os

from src.utils import get_logger
logger = get_logger(__name__)

from src.models.base_model import BaseSegmentationModel
from src.preprocessing.image_processor import ImageProcessor


class TraditionalVesselSegmentation(BaseSegmentationModel):
    """Traditional vessel segmentation using Frangi filter."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize traditional segmentation model."""

        super().__init__(config, name="TraditionalFrangi")
        
        # Get traditional method specific config
        self.trad_config = config.get('traditional', {})
        self.resize_factor = self.trad_config.get('resize_factor', 2)
        self.denoise_kernel = self.trad_config.get('denoise_kernel', 7)
        self.use_clahe = self.trad_config.get('use_clahe', False)
        self.frangi_scales = self.trad_config.get('frangi_scales', [1, 2, 3, 4, 5])
        self.frangi_alpha = self.trad_config.get('frangi_alpha', 0.5)
        self.frangi_beta = self.trad_config.get('frangi_beta', 0.5) 
        self.frangi_gamma = self.trad_config.get('frangi_gamma', 15)
        self.frangi_black_ridges = self.trad_config.get('frangi_black_ridges', True)
        self.min_object_size = self.trad_config.get('min_object_size', 250)
        self.closing_kernel_size = self.trad_config.get('closing_kernel_size', 8)
        self.opening_kernel_size = self.trad_config.get('opening_kernel_size', 3)
        self.use_green_channel = self.trad_config.get('use_green_channel', True)
        self.clahe_tile_grid_size = self.trad_config.get('clahe_tile_grid_size', 8)
        self.clahe_clip_limit = self.trad_config.get('clahe_clip_limit', 2.0)
    
        self.processor = ImageProcessor()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image."""

        if self.use_green_channel and len(image.shape) == 3:
            processed = self.processor.extract_green_channel(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()

        processed = self.processor.denoise(processed, self.denoise_kernel)
        processed = self.processor.normalize(processed)

        if self.use_clahe:
            try:
                processed = self.processor.apply_clahe(processed, clip_limit=self.clahe_clip_limit, tile_grid_size=self.clahe_tile_grid_size)
            except Exception as e:
                logger.warning(f"CLAHE failed, continuing without it: {e}")
        return processed


    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Traditional method doesn't require training."""

        logger.info(f"{self.name} doesn't require training")
        return {}
    
    def predict_with_intermediate_results(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Process image and return intermediate results for visualization."""

        results = {}
        
        # Original image
        results['original'] = image.copy()
        
        # Preprocessed image
        processed = self.preprocess(image)
        results['preprocessed'] = processed
        
        # Frangi filter results
        frangi_filtered, frangi_binary = self.processor.frangi_filter(
            processed,
            scales=self.frangi_scales,
            alpha=self.frangi_alpha,
            beta=self.frangi_beta,
            gamma=self.frangi_gamma,
            black_ridges=self.frangi_black_ridges,
        )
        results['frangi_filtered'] = frangi_filtered
        results['frangi_binary'] = frangi_binary
        
        # Apply ROI mask
        if roi_mask is not None:
            frangi_binary_masked = self.processor.apply_mask(frangi_binary, roi_mask)
            results['roi_masked'] = frangi_binary_masked
        else:
            frangi_binary_masked = frangi_binary
        
        # Noise mask
        noise_mask = self.processor.create_noise_mask(frangi_binary_masked, self.min_object_size)
        results['noise_mask'] = noise_mask
        
        # Final vessels
        vessels = self.processor.apply_mask(frangi_binary_masked, noise_mask)
        vessels = self.processor.morphological_opening(vessels, self.opening_kernel_size)
        vessels = self.processor.morphological_closing(vessels, self.closing_kernel_size)
        results['final_vessels'] = vessels
        
        return results

    def predict(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict vessel mask using improved traditional pipeline."""

        return self.predict_with_intermediate_results(image, roi_mask)['final_vessels']
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """Traditional method doesn't require postprocessing."""
        return prediction

    def save_model(self, filepath: str) -> None:
        """Traditional method doesn't require saving model."""
        
        logger.info(f"{self.name} doesn't require saving model")
    
    def load_model(self, filepath: str) -> None:
        """Traditional method doesn't require loading model."""
        
        logger.info(f"{self.name} doesn't require loading model.")
