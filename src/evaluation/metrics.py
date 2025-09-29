"""Evaluation metrics for vessel segmentation."""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score)
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from src.utils import get_logger
logger = get_logger(__name__)

class VesselSegmentationMetrics:
    """Class for calculating vessel segmentation metrics."""
    
    @staticmethod
    def _flatten_and_binarize(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
        """Flatten and binarize mask."""

        flat = mask.flatten()
        return (flat > threshold).astype(np.uint8)
    

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        
        y_true_flat = self._flatten_and_binarize(y_true)
        y_pred_flat = self._flatten_and_binarize(y_pred)
        
        return confusion_matrix(y_true_flat, y_pred_flat)
    

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics."""

        y_true_flat = self._flatten_and_binarize(y_true)
        y_pred_flat = self._flatten_and_binarize(y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            logger.warning("Confusion matrix has unexpected size")
            tn = fp = fn = tp = 0
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        metrics['f1_score'] = f1_score(y_true_flat, y_pred_flat)
        
        # Sensitivity (Recall, True Positive Rate)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # IoU (Intersection over Union)
        metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # Dice coefficient
        metrics['dice'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Arithmetic mean of sensitivity and specificity
        metrics['arithmetic_mean'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Geometric mean of sensitivity and specificity
        metrics['geometric_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
        
        # Store confusion matrix values
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def create_comparison_image(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create color-coded comparison image."""

        # Binarize masks
        true_bin = (y_true > 0).astype(np.uint8)
        pred_bin = (y_pred > 0).astype(np.uint8)
        
        height, width = true_bin.shape
        comparison = np.zeros((height, width, 3), dtype=np.uint8)
        
        comparison[(pred_bin == 1) & (true_bin == 1)] = [0, 255, 0] # True Positives - Green
        comparison[(pred_bin == 1) & (true_bin == 0)] = [255, 0, 0] # False Positives - Red
        comparison[(pred_bin == 0) & (true_bin == 1)] = [0, 0, 255] # False Negatives - Blue

        return comparison
    
    def add_legend_to_comparison(self, comparison_image: np.ndarray) -> np.ndarray:
        """Add legend to comparison image."""
        
        legend_image = comparison_image.copy()
        
        height = comparison_image.shape[0]
        scale_factor = height / 1000.0
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5 * scale_factor
        thickness = int(5 * scale_factor)
        box_size = int(60 * scale_factor)
        padding = int(20 * scale_factor)
        text_offset = int(40 * scale_factor)
        
        labels = [
            ("TP", (0, 255, 0)),
            ("FP", (255, 0, 0)),
            ("FN", (0, 0, 255))
        ]
        
        for i, (label, color) in enumerate(labels):
            y_top = padding + i * (box_size + padding)
            y_bottom = y_top + box_size
            
            cv2.rectangle(legend_image, (padding, y_top), (padding + box_size, y_bottom), color, -1)
            cv2.putText(legend_image, label, (padding + box_size + padding, y_top + text_offset), font, font_scale, color, thickness)

        return legend_image
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix with values."""
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        labels = np.array([
            [f"TN\n{tn:,}", f"FP\n{fp:,}"],
            [f"FN\n{fn:,}", f"TP\n{tp:,}"]
        ])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=False,
                   xticklabels=["Predicted 0", "Predicted 1"],
                   yticklabels=["True 0", "True 1"],
                   ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate classification report."""

        y_true_flat = (y_true.flatten() > 0).astype(np.uint8)
        y_pred_flat = (y_pred.flatten() > 0).astype(np.uint8)
        
        return classification_report(y_true_flat, y_pred_flat, target_names=['Background', 'Vessel'])
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics for display."""

        output = []
        output.append("=" * 50)
        output.append("EVALUATION METRICS")
        output.append("=" * 50)
        output.append(f"Accuracy:          {metrics.get('accuracy', 0):.4f}")
        output.append(f"Sensitivity:       {metrics.get('sensitivity', 0):.4f}")
        output.append(f"Specificity:       {metrics.get('specificity', 0):.4f}")
        output.append(f"Precision:         {metrics.get('precision', 0):.4f}")
        output.append(f"F1 Score:          {metrics.get('f1_score', 0):.4f}")
        output.append(f"IoU:               {metrics.get('iou', 0):.4f}")
        output.append(f"Dice Coefficient:  {metrics.get('dice', 0):.4f}")
        output.append("-" * 50)
        output.append(f"Arithmetic Mean:   {metrics.get('arithmetic_mean', 0):.4f}")
        output.append(f"Geometric Mean:    {metrics.get('geometric_mean', 0):.4f}")
        output.append("=" * 50)
        
        return "\n".join(output)