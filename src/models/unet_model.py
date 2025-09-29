"""U-Net deep learning model for vessel segmentation (PyTorch)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.utils import get_logger
logger = get_logger(__name__)

from src.models.base_model import BaseSegmentationModel
from src.preprocessing.image_processor import ImageProcessor


# Dice coefficient and loss functions
def dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice coefficient for binary segmentation."""

    y_true = y_true.float()
    y_pred = y_pred.float()

    intersection = (y_true * y_pred).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3)) + smooth)
    return dice.mean()


def iou_metric(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """IoU metric for binary segmentation."""
    
    y_pred_binary = (y_pred > threshold).float()

    intersection = (y_true * y_pred_binary).sum(dim=(2, 3))
    union = y_true.sum(dim=(2, 3)) + y_pred_binary.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        """Initialize combined loss."""

        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for combined loss."""

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dice_loss = 1 - dice_coefficient(targets, probs)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# U-Net architecture components
class ConvBlock(nn.Module):
    """Convolutional block with two convolutions, batch norm, and activation."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu', dropout: float = 0.0):
        """Initialize ConvBlock."""

        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1, inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    """U-Net architecture for image segmentation."""

    def __init__(self, encoder_filters: list, decoder_filters: list, activation: str = 'relu', dropout: float = 0.5):
        """Initialize U-Net model."""

        super().__init__()

        # Encoder (contracting path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        in_channels = 3
        for i, out_channels in enumerate(encoder_filters):
            block_dropout = dropout if i >= 2 else 0.0
            self.encoder_blocks.append(ConvBlock(in_channels, out_channels, activation, block_dropout))

            # Add pooling layer except for the last encoder block
            if i < len(encoder_filters) - 1:
                self.encoder_pools.append(nn.MaxPool2d(2))

            in_channels = out_channels

        # Decoder (expanding path)
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i, out_channels in enumerate(decoder_filters):
            # Transpose convolution for upsampling
            self.decoder_upsamples.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

            # Skip connection adds encoder features
            skip_channels = encoder_filters[-(i + 2)]
            block_dropout = dropout if i < 2 else 0.0
            self.decoder_blocks.append(ConvBlock(out_channels + skip_channels, out_channels, activation, block_dropout))

            in_channels = out_channels

        # Final output layer
        self.output_conv = nn.Conv2d(decoder_filters[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Encoder path
        encoder_features = []
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.encoder_pools + [None])):
            x = block(x)
            encoder_features.append(x)

            if pool is not None:
                x = pool(x)

        # Decoder path
        for i, (upsample, decoder_block) in enumerate(zip(self.decoder_upsamples, self.decoder_blocks)):
            x = upsample(x)

            # Get corresponding encoder feature for skip connection
            skip_feature = encoder_features[-(i + 2)]

            # Concatenate skip connection
            x = torch.cat([x, skip_feature], dim=1)
            x = decoder_block(x)

        # Output layer (logits)
        x = self.output_conv(x)
        return x


class UNetVesselSegmentation(BaseSegmentationModel):
    """PyTorch U-Net model for vessel segmentation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize U-Net segmentation model."""

        super().__init__(config, name="UNetPyTorch")

        # Get U-Net specific config
        self.unet_config = config.get('unet', {})
        self.input_size = int(self.unet_config.get('input_size', 256))

        # Architecture config
        self.arch_config = self.unet_config.get('architecture', {})
        self.encoder_filters = list(self.arch_config.get('encoder_filters', [32, 64, 128, 256, 512]))
        self.decoder_filters = list(self.arch_config.get('decoder_filters', [256, 128, 64, 32]))
        self.dropout_rate = float(self.arch_config.get('dropout_rate', 0.5))
        self.activation = str(self.arch_config.get('activation', 'relu'))

        # Preprocessing config
        self.use_green_channel = bool(self.unet_config.get('use_green_channel', True))
        self.denoise_kernel = int(self.unet_config.get('denoise_kernel', 0))
        self.threshold = float(self.unet_config.get('threshold', 0.7))

        # Training config
        self.train_config = self.unet_config.get('training', {})
        self.batch_size = int(self.train_config.get('batch_size', 2))
        self.epochs = int(self.train_config.get('epochs', 2))
        self.learning_rate = float(self.train_config.get('learning_rate', 0.0001))

        # Callback config
        self.early_stopping_patience = int(self.train_config.get('early_stopping_patience', 25))
        self.reduce_lr_patience = int(self.train_config.get('reduce_lr_patience', 10))
        self.lr_reduction_factor = float(self.train_config.get('lr_reduction_factor', 0.5))
        self.min_lr = float(self.train_config.get('min_lr', 1e-7))

        # Post-processing config
        self.postproc_config = config.get('postprocessing', {})
        self.min_object_size = int(self.postproc_config.get('min_object_size', 120))
        self.closing_kernel_size = int(self.postproc_config.get('closing_kernel_size', 5))
        self.opening_kernel_size = int(self.postproc_config.get('opening_kernel_size', 3))

        # Initialize components
        self.processor = ImageProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for training")

        # Build model
        self._build_model()

    def _build_model(self):
        """Build U-Net model with current configuration."""

        self.model = UNet(
            encoder_filters=self.encoder_filters,
            decoder_filters=self.decoder_filters,
            activation=self.activation,
            dropout=self.dropout_rate
        ).to(self.device)

        # Initialize optimizer and scheduler only in constructor
        if not hasattr(self, 'criterion'):
            self.criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.lr_reduction_factor,
                patience=self.reduce_lr_patience, min_lr=self.min_lr, verbose=True
            )

            # Mixed precision training
            self.use_amp = self.device.type == 'cuda'
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for U-Net."""

        if self.use_green_channel:
            if len(image.shape) == 3:
                # Extract green channel
                green_channel = self.processor.extract_green_channel(image)

                # Apply denoising
                if self.denoise_kernel > 0:
                    green_channel = self.processor.denoise(green_channel, self.denoise_kernel)

                # Convert to 3-channel by repeating green: G -> (G,G,G)
                processed = np.stack([green_channel, green_channel, green_channel], axis=-1)
            else:
                processed = np.stack([image, image, image], axis=-1)
        else:
            # Use original RGB image
            processed = image.copy()

            # Apply denoising
            if self.denoise_kernel > 0:
                processed = self.processor.denoise(processed, self.denoise_kernel)

        # Apply letterbox
        processed = self.processor.letterbox(processed, self.input_size)

        # Normalize
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))

        return processed

    def prepare_data(self, images: np.ndarray, masks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""

        X_processed = []
        y_processed = []

        for image, mask in zip(images, masks):
            # Preprocess image
            image_processed = self.preprocess(image)
            X_processed.append(image_processed)

            # Process mask with nearest neighbor interpolation
            mask_processed = self.processor.letterbox(mask, self.input_size, is_mask=True)
            mask_processed = (mask_processed > 0).astype(np.float32)

            # Convert to CHW format
            if len(mask_processed.shape) == 2:
                mask_processed = mask_processed[np.newaxis, ...]
            else:
                mask_processed = np.transpose(mask_processed, (2, 0, 1))

            y_processed.append(mask_processed)

        # Convert to tensors
        X_tensor = torch.from_numpy(np.stack(X_processed)).to(self.device)
        y_tensor = torch.from_numpy(np.stack(y_processed)).to(self.device)

        return X_tensor, y_tensor

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the U-Net model."""

        logger.info("Preparing training data (PyTorch)...")
        X_train_tensor, y_train_tensor = self.prepare_data(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self.prepare_data(X_val, y_val)
            has_validation = True
        else:
            X_val_tensor = y_val_tensor = None
            has_validation = False

        logger.info(f"Training data shape: {X_train_tensor.shape}")
        if has_validation:
            logger.info(f"Validation data shape: {X_val_tensor.shape}")

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        self.model.train()

        epoch_pbar = tqdm(range(self.epochs), desc="Training U-Net", unit="epoch")

        for epoch in epoch_pbar:
            epoch_losses = []
            epoch_dice_scores = []

            # Shuffle training data
            n_samples = X_train_tensor.shape[0]
            indices = torch.randperm(n_samples)

            batch_pbar = tqdm(
                range(0, n_samples, self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=False
            )

            for i in batch_pbar:
                batch_indices = indices[i:i + self.batch_size]
                batch_x = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Calculate metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    dice_score = dice_coefficient(batch_y, probs)
                    epoch_dice_scores.append(dice_score.item())

                epoch_losses.append(loss.item())

                if len(epoch_losses) > 0 and len(epoch_dice_scores) > 0:
                    batch_pbar.set_postfix({'loss': f"{np.mean(epoch_losses[-10:]):.4f}", 'dice': f"{np.mean(epoch_dice_scores[-10:]):.4f}"})

            # Validation phase
            val_loss = None
            val_dice = None
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_tensor)
                    val_loss = self.criterion(val_logits, y_val_tensor).item()
                    val_probs = torch.sigmoid(val_logits)
                    val_dice = dice_coefficient(y_val_tensor, val_probs).item()
                self.model.train()

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            avg_loss = np.mean(epoch_losses)
            avg_dice = np.mean(epoch_dice_scores)

            epoch_stats = {'loss': f"{avg_loss:.4f}", 'dice': f"{avg_dice:.4f}"}
            
            if has_validation:
                epoch_stats.update({'val_loss': f"{val_loss:.4f}", 'val_dice': f"{val_dice:.4f}"})

            epoch_pbar.set_postfix(epoch_stats)

            # Also log to logger for file output
            log_msg = f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.4f} - dice: {avg_dice:.4f}"
            if has_validation:
                log_msg += f" - val_loss: {val_loss:.4f} - val_dice: {val_dice:.4f}"

            logger.info(log_msg)

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")

        # Return training summary for BaseModel compatibility
        return {
            "final_epoch": epoch + 1,
            "best_val_loss": best_val_loss if has_validation else None,
            "epochs_trained": epoch + 1
        }

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict vessel mask for an image."""

        if self.model is None:
            raise ValueError("Model is not trained yet")

        original_shape = image.shape[:2]

        # Preprocess image
        processed = self.preprocess(image)

        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(processed[np.newaxis, ...]).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
            prob_map = probs[0, 0].cpu().numpy()

        # Remove letterbox padding and threshold
        prob_resized = self.processor.remove_letterbox(prob_map, original_shape, self.input_size, is_mask=False)
        prediction_binary = (prob_resized > self.threshold).astype(np.uint8) * 255

        return prediction_binary

    def auto_threshold(self, image: np.ndarray, ground_truth: np.ndarray,
                      threshold_range: Tuple[float, float] = (0.1, 0.9),
                      num_thresholds: int = 20) -> float:
        """Automatically find the best threshold for the given image."""

        if self.model is None:
            raise ValueError("Model is not trained yet")

        original_shape = image.shape[:2]

        # Get probability map
        processed = self.preprocess(image)
        input_tensor = torch.from_numpy(processed[np.newaxis, ...]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
            prob_map = probs[0, 0].cpu().numpy()

        # Remove letterbox padding
        prob_resized = self.processor.remove_letterbox(prob_map, original_shape, self.input_size, is_mask=False)

        # Test different thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

        # Optimize using F1 score with ground truth
        best_f1 = 0
        best_threshold = 0.5
        gt_flat = (ground_truth.flatten() > 127).astype(int)

        for threshold in thresholds:
            pred_binary = (prob_resized > threshold).astype(int)
            pred_flat = pred_binary.flatten()

            # Skip if all predictions are same class
            if len(np.unique(pred_flat)) == 1:
                continue

            f1 = f1_score(gt_flat, pred_flat, average='binary', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logger.info(f"Auto-threshold with ground truth: {best_threshold:.3f} (F1: {best_f1:.3f})")

        return best_threshold

    def set_threshold(self, threshold: float) -> None:
        """Set the prediction threshold."""
        self.threshold = float(threshold)
        logger.info(f"Threshold set to: {self.threshold:.3f}")

    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """Apply post-processing to prediction."""

        mask = (prediction > 0).astype(np.uint8) * 255
        mask = self.processor.remove_small_objects(mask, min_size=self.min_object_size)
        mask = self.processor.morphological_opening(mask, kernel_size=self.opening_kernel_size)
        mask = self.processor.morphological_closing(mask, kernel_size=self.closing_kernel_size)

        return (mask > 0).astype(np.uint8) * 255

    def save_model(self, filepath: str) -> None:
        """Save the trained model with configuration."""

        if self.model is None:
            logger.warning("Model is not trained yet")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create checkpoint with model state and architecture config only
        checkpoint = {
            "model_state": self.model.state_dict(),
            "config": {
                "encoder_filters": self.encoder_filters,
                "decoder_filters": self.decoder_filters,
                "activation": self.activation,
                "dropout": self.dropout_rate,
                "input_size": self.input_size,
                "use_green_channel": self.use_green_channel,
                "denoise_kernel": self.denoise_kernel
            }
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Model and config saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model with configuration."""

        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return

        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Handle both new and old formats
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                # New format - load architecture config from checkpoint
                config = checkpoint["config"]
                self.encoder_filters = config.get("encoder_filters", self.encoder_filters)
                self.decoder_filters = config.get("decoder_filters", self.decoder_filters)
                self.activation = config.get("activation", self.activation)
                self.dropout_rate = config.get("dropout", self.dropout_rate)
                self.input_size = config.get("input_size", self.input_size)
                self.use_green_channel = config.get("use_green_channel", self.use_green_channel)
                self.denoise_kernel = config.get("denoise_kernel", self.denoise_kernel)

                self._build_model()

                # Load model state
                self.model.load_state_dict(checkpoint["model_state"])
                logger.info(f"Model architecture loaded from {filepath}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded from {filepath}")

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise