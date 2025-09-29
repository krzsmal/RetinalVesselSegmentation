"""Bayesian hyperparameter optimization for the XGBoost classifier with Optuna."""

import os
import json
from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import optuna
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_logger
logger = get_logger(__name__)

from src.utils.config import config
from src.preprocessing.data_loader import DataLoader
from src.models.classifier_model import XGBoostVesselSegmentation
from src.evaluation.metrics import VesselSegmentationMetrics


def parse_args():
    """Parse command line arguments."""

    p = argparse.ArgumentParser(description="Optuna tuning for XGBoost vessel classifier")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    return p.parse_args()


def load_train_images(cfg, canonicalize_left=True):
    """Load images and masks."""

    dl = DataLoader(cfg)
    logger.info("Loading training data...")
    images, masks, files = dl.load_dataset(
        directory=cfg["data"]["train_dir"],
        load_masks=True,
        limit=cfg["classifier"].get("max_images", None),
        canonicalize_left=canonicalize_left,
    )
    if len(images) == 0:
        raise RuntimeError("No training images found.")
    logger.info(f"Loaded {len(images)} images for tuning.")
    return images, masks, files


def make_train_val_split(images, masks, files, val_size=0.25, seed=42):
    """Split by images."""

    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(
        images, masks, files,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}")
    logger.info(f"Train files: {[Path(f).name for f in f_train]}")
    logger.info(f"Val   files: {[Path(f).name for f in f_val]}")
    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)


def mean_f1_over_images(model, val_images, val_masks) -> float:
    """ Compute mean F1 score over a set of images."""

    metrics_calc = VesselSegmentationMetrics()
    f1_scores = []

    for img, gt in tqdm(zip(val_images, val_masks), total=len(val_images), desc="Validating"):
        pred = model.predict(img)

        if getattr(model, "resize_factor", 1) > 1:
            gt_aligned = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_aligned = (gt_aligned > 127).astype(np.uint8) * 255
        else:
            gt_aligned = gt

        m = metrics_calc.calculate_metrics(gt_aligned, pred)
        f1_scores.append(m.get("f1_score", 0.0))

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def suggest_params(trial, base_xgb: dict):
    """ Suggest XGBoost hyperparameters for the trial."""

    xgb = deepcopy(base_xgb)

    # Set defaults if not present
    xgb.setdefault("eval_metric", "aucpr")
    xgb.setdefault("tree_method", "hist")
    xgb.setdefault("device", "cuda")
    xgb.setdefault("n_jobs", -1)
    xgb.setdefault("early_stopping_rounds", 100)

    # Suggest key params
    xgb["n_estimators"] = 2000 
    xgb["learning_rate"] = trial.suggest_float("lr", 0.03, 0.15)
    xgb["max_depth"] = trial.suggest_int("max_depth", 5, 9)
    xgb["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 20)
    xgb["subsample"] = trial.suggest_float("subsample", 0.6, 0.9)
    xgb["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)
    xgb["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True)
    xgb["reg_lambda"] = trial.suggest_float("reg_lambda", 0.5, 5.0, log=True)
    xgb["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)

    return xgb


def build_objective(cfg, X_train, y_train, X_val, y_val):
    """Build the Optuna objective function."""
    
    base_cfg = deepcopy(cfg)
    base_xgb = deepcopy(base_cfg["classifier"].get("xgb_params", {}))

    # Ensure feature caching is on
    base_cfg["classifier"].setdefault("features", {})
    base_cfg["classifier"]["features"]["use_cache"] = True
    base_cfg["classifier"]["features"].setdefault("cache_dir", ".cache/features")
    Path(base_cfg["classifier"]["features"]["cache_dir"]).mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        xgb_params = suggest_params(trial, base_xgb)

        trial_cfg = deepcopy(base_cfg)
        trial_cfg["classifier"]["xgb_params"] = xgb_params
        trial_cfg["classifier"]["threshold"] = 0.99

        model = XGBoostVesselSegmentation(trial_cfg)
        model.train(np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val))

        f1 = mean_f1_over_images(model, X_val, y_val)
        return f1

    return objective


def main():
    args = parse_args()
    logger.info("Loading config...")
    cfg = config.load_config(args.config)

    # Read tuning settings
    classifier_cfg = cfg.get("classifier", {})
    tune_cfg = classifier_cfg.get("hyperparameter_tuning", {})
    n_trials = int(tune_cfg.get("n_trials", 30))
    val_size = float(tune_cfg.get("val_size", 0.25))
    seed = int(tune_cfg.get("seed", 42))
    train_best = bool(tune_cfg.get("train_best", True))
    logger.info(f"Tuning settings -> n_trials={n_trials}, val_size={val_size}, seed={seed}, train_best={train_best}")

    # Load and split data
    images, masks, files = load_train_images(cfg)
    X_train, X_val, y_train, y_val = make_train_val_split(
        images, masks, files, val_size=val_size, seed=seed
    )

    # Build and run study
    study = optuna.create_study(direction="maximize")
    study.optimize(build_objective(cfg, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best F1 on validation: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best classifier config only
    classifier_cfg = deepcopy(cfg["classifier"])
    xgb_from_cfg = deepcopy(classifier_cfg.get("xgb_params", {}))
    xgb_best = suggest_params(optuna.trial.FixedTrial(study.best_params), xgb_from_cfg)
    classifier_cfg["xgb_params"] = xgb_best
    classifier_cfg["threshold"] = 0.99
    classifier_cfg.setdefault("features", {})
    classifier_cfg["features"]["use_cache"] = True
    classifier_cfg["features"].setdefault("cache_dir", ".cache/features")

    Path("optuna").mkdir(parents=True, exist_ok=True)
    with open("optuna/best_classifier_config.json", "w") as f:
        json.dump(classifier_cfg, f, indent=2)
    logger.info("Saved best classifier config to optuna/best_classifier_config.json")

    # Retrain best model on TRAIN+VAL
    if train_best:
        logger.info("Retraining final model on TRAIN+VAL with best params...")
        all_imgs = np.concatenate([X_train, X_val], axis=0)
        all_msks = np.concatenate([y_train, y_val], axis=0)
        best_cfg = deepcopy(cfg)
        best_cfg["classifier"] = classifier_cfg
        final_model = XGBoostVesselSegmentation(best_cfg)
        final_model.train(all_imgs, all_msks, None, None)

        model_filename = f"{best_cfg['classifier']['name']}_optuna.pkl"
        model_path = os.path.join(cfg["data"]["model_dir"], model_filename)
        final_model.save_model(model_path)


if __name__ == "__main__":
    main()
