"""Base model class for vessel segmentation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import os
import pickle
import json
from datetime import datetime

from src.utils import get_logger
logger = get_logger(__name__)

from src.evaluation.metrics import VesselSegmentationMetrics


class BaseSegmentationModel(ABC):
    """Abstract base class for all vessel segmentation models."""
    
    def __init__(self, config: Dict[str, Any], name: str = "BaseModel"):
        """Initialize the base model."""

        self.config = config
        self.name = name
        self.model = None
        logger.info(f"Initialized {name} model")
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        pass
    
    @abstractmethod
    def train(self, 
              X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training masks
            X_val: Validation images (optional)
            y_val: Validation masks (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict vessel mask for an image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Predicted binary mask (H, W)
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the model."""

        return f"{self.name}"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""

        return f"{self.name}(config={self.config})"