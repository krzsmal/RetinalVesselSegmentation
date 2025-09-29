"""XGBoost classifier model for vessel segmentation."""

import numpy as np
import cv2
import pickle
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, DMatrix
from tqdm import tqdm
import os
from pathlib import Path

from src.utils import get_logger
logger = get_logger(__name__)

from src.models.base_model import BaseSegmentationModel
from src.preprocessing.image_processor import ImageProcessor
from src.models.feature_extractor import FeatureExtractor


class XGBoostVesselSegmentation(BaseSegmentationModel):
    """XGBoost classifier for vessel segmentation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost segmentation model."""

        super().__init__(config, name="XGBoost")

        # Classifier-specific config
        self.clf_config = config.get('classifier', {})
        self.window_size = int(self.clf_config.get('features', {}).get('window_size', 5))
        self.xgb_params = self.clf_config.get('xgb_params', {})
        self.random_state = int(self.clf_config.get('random_state', 42))
        self.neg_sample_rate = float(self.clf_config.get('neg_sample_rate', 0.10))
        self.pos_sample_rate = float(self.clf_config.get('pos_sample_rate', 1.0))
        self.max_samples_per_image = int(self.clf_config.get('max_samples_per_image', 1_000_000))
        self.threshold = float(self.clf_config.get('threshold', 0.99))
        self.resize_factor = int(self.clf_config.get('resize_factor', 2))
        self.denoise_kernel = int(self.clf_config.get('denoise_kernel', 5))

        # Post-processing config
        self.postproc_config = config.get('postprocessing', {})
        self.min_object_size = int(self.postproc_config.get('min_object_size', 120))
        self.closing_kernel_size = int(self.postproc_config.get('closing_kernel_size', 5))
        self.opening_kernel_size = int(self.postproc_config.get('opening_kernel_size', 3))

        self.processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor(config)
        self.classifier: Optional[XGBClassifier] = None
        self.half_window = self.window_size // 2

    def prepare_training_data(self, images, masks):
        """ Prepare training data by sampling pixels from images and masks."""

        X_list, y_list = [], []
        total_pos_pixels, total_neg_pixels = 0, 0

        logger.info(
            f"In-flight sampling: pos_rate={self.pos_sample_rate}, "
            f"neg_rate={self.neg_sample_rate}, max_samples_per_image={self.max_samples_per_image}"
        )

        for img_idx, (image, mask) in enumerate(tqdm(zip(images, masks), total=len(images), desc="Processing images")):
            
            # resize mask
            if self.resize_factor > 1:
                mask_resized = self.processor.resize_image(mask, scale=1 / self.resize_factor)
            else:
                mask_resized = mask.copy()
            _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

            # count full class distribution in ROI
            h, w = mask_binary.shape
            half = self.half_window
            roi = mask_binary[half:h-half, half:w-half]
            pos_pixels = int(np.count_nonzero(roi))
            neg_pixels = int(roi.size - pos_pixels)
            total_pos_pixels += pos_pixels
            total_neg_pixels += neg_pixels

            # try to load from cache
            cache_path = self.feature_extractor._train_cache_path(image, mask)
            if self.feature_extractor.cache_enabled and cache_path.exists():
                try:
                    data = np.load(cache_path, allow_pickle=False)
                    Xc = data["X"].astype(np.float32); yc = data["y"].astype(np.uint8)
                    X_list.append(Xc); y_list.append(yc)
                    logger.info(f"[img {img_idx:02d}] loaded from cache: {cache_path.name} (X={len(Xc):,})")
                    continue
                except Exception as e:
                    logger.warning(f"[img {img_idx:02d}] cache read failed ({e}), recomputing...")

            # Convolutional feature extraction
            fmap = self.feature_extractor._precompute_feature_maps(image)
            X_full, _coords = self.feature_extractor._build_feature_stack(fmap)

            # sampling pixels based on ROI mask
            y_full = (roi.reshape(-1) > 0).astype(np.uint8)

            rng = np.random.default_rng(self.random_state + img_idx)
            idx_pos = np.flatnonzero(y_full == 1)
            idx_neg = np.flatnonzero(y_full == 0)

            take_pos = idx_pos if self.pos_sample_rate >= 1.0 else idx_pos[rng.random(len(idx_pos)) < self.pos_sample_rate]
            take_neg = idx_neg if self.neg_sample_rate >= 1.0 else idx_neg[rng.random(len(idx_neg)) < self.neg_sample_rate]

            idx_sel = np.concatenate([take_pos, take_neg])
            if len(idx_sel) > self.max_samples_per_image:
                idx_sel = rng.choice(idx_sel, size=self.max_samples_per_image, replace=False)

            Xc = X_full[idx_sel]
            yc = y_full[idx_sel]
            X_list.append(Xc); y_list.append(yc)

            logger.info(f"[img {img_idx:02d}] computed: X={len(Xc):,} (pos_total={pos_pixels:,}, neg_total={neg_pixels:,})")

            # save to cache
            if self.feature_extractor.cache_enabled:
                try:
                    np.savez_compressed(cache_path, X=Xc.astype(np.float32), y=yc.astype(np.uint8))
                    logger.info(f"[img {img_idx:02d}] cached {cache_path.name}")
                except Exception as e:
                    logger.warning(f"[img {img_idx:02d}] cache write failed ({e})")

        # concatenate all samples
        X = np.concatenate(X_list, axis=0) if len(X_list) > 1 else X_list[0]
        y = np.concatenate(y_list, axis=0) if len(y_list) > 1 else y_list[0]

        logger.info(
            f"TOTAL sampled: X={len(X):,}, pos={int(np.sum(y==1)):,}, neg={int(np.sum(y==0)):,}; "
            f"FULL class counts (no sampling): pos={total_pos_pixels:,}, neg={total_neg_pixels:,}"
        )
        self._full_class_counts = (total_pos_pixels, total_neg_pixels)
        return X, y

    def train(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train the XGBoost classifier"""

        X_features, y_labels = self.prepare_training_data(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val_feat, y_val_lbl = self.prepare_training_data(X_val, y_val)
        else:
            X_features, X_val_feat, y_labels, y_val_lbl = train_test_split(
                X_features, y_labels, test_size=0.15,
                random_state=self.random_state, shuffle=True
            )
            logger.info(f"Internal validation split created: train={len(X_features)}, val={len(X_val_feat)}")

        if hasattr(self, "_full_class_counts"):
            full_pos, full_neg = self._full_class_counts
        else:
            full_pos = int(np.sum(y_labels == 1))
            full_neg = int(np.sum(y_labels == 0))

        if full_pos == 0:
            logger.warning("No positive pixels in full counts, scale_pos_weight set to 1.0")
            spw = 1.0
        else:
            spw = max(1.0, full_neg / full_pos)
        logger.info(f"scale_pos_weight (FULL counts) ~ {spw:.2f} (neg={full_neg}, pos={full_pos})")

        # XGBoost parameters
        params = {
            'n_estimators':      self.xgb_params.get('n_estimators', 800),
            'max_depth':         self.xgb_params.get('max_depth', 6),
            'learning_rate':     self.xgb_params.get('learning_rate', 0.1),
            'subsample':         self.xgb_params.get('subsample', 0.8),
            'colsample_bytree':  self.xgb_params.get('colsample_bytree', 0.8),
            'min_child_weight':  self.xgb_params.get('min_child_weight', 1),
            'reg_alpha':         self.xgb_params.get('reg_alpha', 0),
            'reg_lambda':        self.xgb_params.get('reg_lambda', 1),
            'random_state':      self.xgb_params.get('random_state', self.random_state),
            'n_jobs':            self.xgb_params.get('n_jobs', -1),
            'objective':         self.xgb_params.get('objective', 'binary:logistic'),
            'eval_metric':       self.xgb_params.get('eval_metric', 'aucpr'),
            'tree_method':       self.xgb_params.get('tree_method', 'hist'),
            'device':            self.xgb_params.get('device', 'cuda'),
            'scale_pos_weight':  self.xgb_params.get('scale_pos_weight', spw),
            'early_stopping_rounds': self.xgb_params.get('early_stopping_rounds', 50),
        }

        logger.info(f"Training XGBoost with params: {params}")
        self.classifier = XGBClassifier(**params)

        # Train with early stopping on validation set
        self.classifier.fit(
            X_features, y_labels,
            eval_set=[(X_val_feat, y_val_lbl)],
            verbose=10
        )

        self.model = self.classifier

        if hasattr(self.classifier, "best_iteration"):
            logger.info(f"Best iteration (early stopping): {self.classifier.best_iteration}")
        if hasattr(self.classifier, "best_score"):
            logger.info(f"Best eval score: {self.classifier.best_score}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict vessel mask for an image."""

        if self.classifier is None:
            raise ValueError("Model is not trained yet")

        # extract features
        features, coordinates = self.feature_extractor.extract_features_from_image(image)
        X = np.asarray(features, dtype=np.float32)

        # Force CPU prediction if model was trained on GPU
        try:
            self.classifier.set_params(device="cpu")
            logger.info("Forcing CPU prediction mode")
        except Exception:
            pass

        # predict probabilities
        if hasattr(self.classifier, "predict_proba"):
            logger.info("Using predict_proba for probability estimation")
            proba = self.classifier.predict_proba(X)
            proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
        else:
            try:
                self.classifier.get_booster().set_param({'device': 'cpu'})
                logger.info("Using booster.predict for probability estimation")
            except Exception:
                pass
            dmat = DMatrix(X)
            proba = self.classifier.get_booster().predict(dmat)

        # thresholding
        thr = float(getattr(self, "threshold", 0.99))
        labels = (proba >= thr).astype(np.uint8)

        # Create empty mask and fill in positive predictions
        _, green_processed = self.feature_extractor.preprocess(image)
        h, w = green_processed.shape
        prediction_mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y), lab in zip(coordinates, labels):
            if lab:
                prediction_mask[y, x] = 255

        # Resize back to original size
        if self.resize_factor > 1:
            H, W = image.shape[:2]
            prediction_mask = self.processor.resize_image(prediction_mask, target_size=(W, H))

        return prediction_mask

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input image for feature extraction."""
        
        return self.feature_extractor.preprocess(image)

    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """Apply post-processing to prediction."""

        mask = (prediction > 0).astype(np.uint8) * 255
        mask = self.processor.remove_small_objects(mask, min_size=self.min_object_size)
        mask = self.processor.morphological_opening(mask, kernel_size=self.opening_kernel_size)
        mask = self.processor.morphological_closing(mask, kernel_size=self.closing_kernel_size)

        return (mask > 0).astype(np.uint8) * 255


    def save_model(self, filepath: str) -> None:
        """Save the trained model."""

        if self.classifier is None:
            logger.warning("Model is not trained yet")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""

        if not os.path.exists(filepath):
            error_msg = f"Model file not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(filepath, 'rb') as f:
                self.classifier = pickle.load(f)
            self.model = self.classifier
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            error_msg = f"Failed to load model from {filepath}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)