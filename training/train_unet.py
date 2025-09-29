"""Training script for U-Net model."""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_logger
logger = get_logger(__name__)

from src.utils.config import config
from src.preprocessing.data_loader import DataLoader
from src.models.unet_model import UNetVesselSegmentation
from src.evaluation.metrics import VesselSegmentationMetrics


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='Train U-Net model for vessel segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()


def main():
    """Main training function."""

    args = parse_args()
    logger.info("Starting U-Net model training")
    cfg = config.load_config(args.config)
    
    # Load training data
    data_loader = DataLoader(cfg)
    logger.info("Loading training data...")

    train_images, train_masks, train_files = data_loader.load_dataset(
        directory=cfg['data']['train_dir'],
        load_masks=True,
        limit=cfg['unet'].get('max_images', None),
        canonicalize_left=True
    )
    
    if len(train_images) == 0:
        logger.error("No training images found!")
        return
    
    logger.info(f"Loaded {len(train_images)} training images")
    
    # Split data into train/val/test
    test_size = float(cfg['unet'].get('test_size', 0.15))
    val_size = float(cfg['unet'].get('val_size', 0.15))
    logger.info(f"Initial split: test_size={test_size}, val_size={val_size}")

    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        train_images, train_masks, train_files,
        test_size=(test_size + val_size),
        random_state=cfg['unet']['random_state'],
        shuffle=True
    )

    # Split temp into val and test
    relative_val = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.5
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp,
        test_size=(1.0 - relative_val),
        random_state=cfg['unet']['random_state'],
        shuffle=True
    )
    
    logger.info(f"Final split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"Train files: {[Path(f).name for f in f_train]}")
    logger.info(f"Val files:   {[Path(f).name for f in f_val]}")
    logger.info(f"Test files:  {[Path(f).name for f in f_test]}")
    
    # Initialize model
    logger.info("Initializing U-Net model...")
    model = UNetVesselSegmentation(cfg)
    
    # Train model
    logger.info("Starting training...")
    model.train(np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val))
    
    # Save model
    model_filename = f"{cfg['unet']['name']}.pt"
    model_path = os.path.join(cfg['data']['model_dir'], model_filename)
    logger.info(f"Saving model to {model_path}...")
    model.save_model(model_path)
    
    # Evaluate on TEST set
    logger.info("Evaluating on test set...")
    metrics_calculator = VesselSegmentationMetrics()

    all_metrics = []
    for i, (image, mask) in enumerate(tqdm(zip(X_test, y_test), total=len(X_test), desc="Evaluating")):
        prediction = model.predict(image)

        mask_resized = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Ensure binary mask (0/255)
        mask_resized = (mask_resized > 127).astype(np.uint8) * 255

        metrics = metrics_calculator.calculate_metrics(mask_resized, prediction)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    if all_metrics:
        keys = [k for k in all_metrics[0].keys() if k not in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']]
        for key in keys:
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

        logger.info(metrics_calculator.format_metrics(avg_metrics))
    else:
        logger.warning("No metrics computed on test set (empty?)")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()