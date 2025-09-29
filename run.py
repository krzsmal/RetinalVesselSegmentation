#!/usr/bin/env python
"""Main script to run the Retinal Vessel Segmentation application."""

import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.utils.logger import setup_logger, get_logger
setup_logger(
    log_level="INFO",
    enable_console=True
)
logger = get_logger(__name__)

def check_requirements():
    """Check if all required packages are installed."""

    try:
        import streamlit
        import sklearn
        import cv2
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import yaml
        import tqdm
        import xgboost
        import optuna
        import torch
        logger.info("All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False


def run_streamlit():
    """Run the Streamlit application."""

    app_path = Path(__file__).parent / "src" / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        logger.error(f"Streamlit app not found at {app_path}")
        return
    
    logger.info("Starting Streamlit application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def run_training(model_type: str, config_path: str):
    """Run training script for specified model."""

    if model_type == "classifier":
        script_path = Path(__file__).parent / "training" / "train_classifier.py"
    elif model_type == "unet":
        script_path = Path(__file__).parent / "training" / "train_unet.py"
    else:
        logger.error(f"Unknown model type: {model_type}")
        return
    
    if not script_path.exists():
        logger.error(f"Training script not found at {script_path}")
        return
    
    logger.info(f"Starting {model_type} training with config {config_path}...")
    subprocess.run([sys.executable, str(script_path), "--config", config_path])


def run_classifier_tuning(config_path: str):
    """Run hyperparameter tuning for the classifier model."""

    script_path = Path(__file__).parent / "training" / "tune_classifier.py"
    
    if not script_path.exists():
        logger.error(f"Tuning script not found at {script_path}")
        return
    
    logger.info(f"Starting classifier hyperparameter tuning with config {config_path}...")
    subprocess.run([sys.executable, str(script_path), "--config", config_path])


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Retinal Vessel Segmentation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--train",
        choices=["classifier", "unet"],
        help="Train a specific model instead of running the app"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all requirements are installed"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--tune_classifier",
        action="store_true",
        help="Run hyperparameter tuning for the classifier model"
    )
    
    args = parser.parse_args()
    
    if args.check:
        if check_requirements():
            print("All requirements are satisfied")
        else:
            print("Some requirements are missing")
        return
    
    if args.train:
        run_training(args.train, args.config)
    elif args.tune_classifier:
        run_classifier_tuning(args.config)
    else:
        run_streamlit()


if __name__ == "__main__":
    main()