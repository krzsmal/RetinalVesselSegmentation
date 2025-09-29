"""Streamlit application for retinal vessel segmentation."""

import streamlit as st
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_logger
logger = get_logger(__name__)

from src.utils.config import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.image_processor import ImageProcessor
from src.evaluation.metrics import VesselSegmentationMetrics
from src.models.traditional_model import TraditionalVesselSegmentation
from src.models.classifier_model import XGBoostVesselSegmentation
from src.models.unet_model import UNetVesselSegmentation


@st.cache_resource(show_spinner=False)
def load_unet_cached(model_path: str, mtime: float, cfg):
    """Load U-Net model into cache. mtime in key ensures refresh."""

    m = UNetVesselSegmentation(cfg)
    m.load_model(model_path)
    return m


@st.cache_resource(show_spinner=False)
def load_classifier_cached(model_path: str, mtime: float, cfg):
    """Load classifier model into cache. mtime in key ensures refresh."""

    m = XGBoostVesselSegmentation(cfg)
    m.load_model(model_path)
    return m


class VesselSegmentationApp:
    """Main Streamlit application."""

    def __init__(self):
        """Initialize the application."""

        st.set_page_config(
            page_title="Retinal Vessel Segmentation System",
            page_icon="👁️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        self.config = config.load_config()
        self.data_loader = DataLoader(self.config)
        self.processor = ImageProcessor()
        self.metrics_calculator = VesselSegmentationMetrics()

        self.traditional_model = None
        self.classifier_model = None
        self.unet_model = None

    @staticmethod
    def _is_right_eye(name: str) -> bool:
        """Check if image name indicates a right eye"""

        stem = os.path.splitext(name)[0]
        return re.search(r'_(?:r)$', stem, flags=re.IGNORECASE) is not None

    def get_model_files(self, extension):
        """Get list of model files with specific extension."""

        model_dir = self.config['data']['model_dir']
        if not os.path.exists(model_dir):
            return []

        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith(extension):
                model_files.append(file)

        return sorted(model_files)

    def _auto_load_model(self, model_type, model_files, load_function):
        """Helper function to auto-load models."""

        should_autoload = not st.session_state.get(f'{model_type}_loaded')
        if not should_autoload:
            return

        config_name = self.config[model_type]['name']
        extension = '.pkl' if model_type == 'classifier' else '.pt'
        default_file = config_name + extension

        if default_file in model_files:
            selected_file = default_file
        else:
            selected_file = model_files[0]
            logger.info(f"Default {default_file} not found, using first available: {selected_file}")

        model_path = os.path.join(self.config['data']['model_dir'], selected_file)
        try:
            with st.spinner(f"Loading {selected_file}..."):
                mtime = os.path.getmtime(model_path)
                model = load_function(model_path, mtime, self.config)

            st.session_state[f'{model_type}_model'] = model
            setattr(self, f'{model_type}_model', model)
            st.session_state[f'{model_type}_loaded'] = True
            st.session_state[f'{model_type}_file'] = selected_file

            if model_type == 'classifier':
                st.session_state['selected_classifier_file'] = selected_file

            logger.info(f"Successfully auto-loaded {model_type}: {selected_file}")
        except Exception as e:
            st.session_state[f'{model_type}_loaded'] = False
            st.session_state[f'{model_type}_file'] = None
            st.sidebar.error(f"Auto-load failed: {str(e)}")
            logger.error(f"Failed to auto-load {model_type}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _load_selected_model(self, model_type, selected_file, load_function):
        """Helper function to load a selected model."""

        model_path = os.path.join(self.config['data']['model_dir'], selected_file)
        with st.spinner(f"Loading {selected_file}..."):
            try:
                mtime = os.path.getmtime(model_path)
                model = load_function(model_path, mtime, self.config)
                st.session_state[f'{model_type}_model'] = model
                setattr(self, f'{model_type}_model', model)
                st.session_state[f'{model_type}_loaded'] = True
                st.session_state[f'{model_type}_file'] = selected_file
            except Exception as e:
                st.sidebar.error(f"Failed to load: {str(e)}")
                st.session_state[f'{model_type}_loaded'] = False
                st.session_state[f'{model_type}_file'] = None

    def load_models(self):
        """Load models into session state if not already loaded."""

        # Traditional
        if "traditional_model" not in st.session_state:
            st.session_state.traditional_model = TraditionalVesselSegmentation(self.config)
        self.traditional_model = st.session_state.get('traditional_model', None)

        # Classifier
        st.session_state.setdefault('classifier_loaded', False)
        st.session_state.setdefault('classifier_file', None)
        self.classifier_model = st.session_state.get('classifier_model', None)

        # U-Net
        st.session_state.setdefault('unet_loaded', False)
        st.session_state.setdefault('unet_file', None)
        self.unet_model = st.session_state.get('unet_model', None)

    def run(self):
        """Run the main application."""

        st.title("👁️ Retinal Vessel Segmentation System")

        # Load models
        with st.spinner("Initializing components..."):
            self.load_models()

        self.setup_sidebar()

        # Main content
        if st.session_state.get('selected_image') and st.session_state['selected_image'] != "Select an image...":
            self.process_image()
        else:
            self.show_introduction()

    def setup_sidebar(self):
        """Setup sidebar with controls."""

        st.sidebar.header("Configuration")

        # Image selection
        image_files = self.data_loader.get_image_files(
            self.config['data']['image_dir']
        )

        if not image_files:
            st.sidebar.warning("No images found in the configured directory")
            return

        selected_image = st.sidebar.selectbox(
            "Select Image",
            ["Select an image..."] + image_files,
            key='selected_image'
        )

        # Processing method selection
        st.sidebar.subheader("Processing Method")
        methods = ["Traditional Processing (Frangi Filter)"]

        # Check for available model files
        classifier_files = self.get_model_files('.pkl')
        unet_files = self.get_model_files('.pt')

        if classifier_files:
            methods.append("Machine Learning (XGBoost)")
        if unet_files:
            methods.append("Deep Learning (U-Net)")

        selected_method = st.sidebar.radio(
            "Select Method",
            methods,
            key='selected_method'
        )

        # Auto-load U-Net
        if "U-Net" in selected_method and unet_files:
            self._auto_load_model('unet', unet_files, load_unet_cached)

        # Auto-load Classifier
        if "XGBoost" in selected_method and classifier_files:
            self._auto_load_model('classifier', classifier_files, load_classifier_cached)

        # Model selection
        if "XGBoost" in selected_method:
            if classifier_files:
                selected_classifier = st.sidebar.selectbox(
                    "Select Classifier Model",
                    classifier_files,
                    index=classifier_files.index(st.session_state.get('classifier_file', classifier_files[0]))
                          if st.session_state.get('classifier_file') in classifier_files else 0,
                    key='selected_classifier_file',
                    help="Choose a trained XGBoost model file"
                )

                # Load from cache when selection changes
                if selected_classifier and st.session_state.get('classifier_file') != selected_classifier:
                    self._load_selected_model('classifier', selected_classifier, load_classifier_cached)

                # Show current loaded model status
                if st.session_state.get('classifier_model'):
                    st.sidebar.success(f"Model: {st.session_state.get('classifier_file', 'None')}")
                else:
                    st.sidebar.warning("No classifier model loaded")
            else:
                st.sidebar.warning("No classifier models found in models directory")
                st.sidebar.info("Train a model first using: python training/train_classifier.py")

        if "U-Net" in selected_method:
            if unet_files:
                selected_unet = st.sidebar.selectbox(
                    "Select U-Net Model",
                    unet_files,
                    index=unet_files.index(st.session_state.get('unet_file', unet_files[0]))
                          if st.session_state.get('unet_file') in unet_files else 0,
                    key='selected_unet_file',
                    help="Choose a trained U-Net model file"
                )

                # Load from cache when selection changes
                if selected_unet and st.session_state.get('unet_file') != selected_unet:
                    self._load_selected_model('unet', selected_unet, load_unet_cached)

                # Show current loaded model status
                if st.session_state.get('unet_model'):
                    st.sidebar.success(f"Model: {st.session_state.get('unet_file', 'None')}")
                else:
                    st.sidebar.warning("No U-Net model loaded")
            else:
                st.sidebar.warning("No U-Net models found in models directory")
                st.sidebar.info("Train a model first using: python training/train_unet.py")

        # Advanced options
        with st.sidebar.expander("Advanced Options"):

            # General settings
            st.markdown("#### General Settings")
            st.checkbox("Show comparison with ground truth", key='show_comparison', value=True)
            st.checkbox("Display evaluation metrics", key='show_metrics', value=True)

            if "Traditional" in selected_method:
                st.checkbox("Show intermediate results", key='show_intermediate')

                st.markdown("### Processing Parameters")
                
                # Preprocessing options
                st.checkbox("Use green channel", key='use_green_channel', value=True, help="Extract green channel for better vessel contrast")

                # Denoising
                st.slider("Denoise kernel size", 3, 15, 7, step=2, key='denoise_kernel', help="Kernel size for denoising (must be odd)")

                # Frangi filter parameters
                st.markdown("### Frangi Filter Settings")
                st.slider("Frangi scales (max)", 1, 10, 5, key='frangi_max_scale', help="Maximum scale for multiscale Frangi filter")
                st.slider("Frangi alpha", 0.1, 1.0, 0.5, step=0.1, key='frangi_alpha', help="Frangi vesselness measure sensitivity")
                st.slider("Frangi beta", 0.1, 1.0, 0.5, step=0.1, key='frangi_beta', help="Frangi background suppression")
                st.slider("Frangi gamma", 5, 50, 15, key='frangi_gamma', help="Frangi structure sensitivity")
                st.checkbox("Black ridges", key='frangi_black_ridges', value=True, help="Detect dark structures on bright background")

                # Morphological operations
                st.markdown("### Morphological Operations")
                st.slider("Min object size", 50, 1000, 250, step=50, key='min_object_size', help="Minimum connected component size in pixels")
                st.slider("Opening kernel", 1, 7, 3, step=2, key='opening_kernel_size', help="Kernel size for morphological opening")
                st.slider("Closing kernel", 2, 16, 8, step=2, key='closing_kernel_size', help="Kernel size for morphological closing")

                # CLAHE
                st.markdown("### CLAHE Settings")
                st.checkbox("Apply CLAHE", key='use_clahe', value=False, help="Apply Contrast Limited Adaptive Histogram Equalization")
                st.slider("Clip limit", 1.0, 5.0, 2.0, step=0.1, key='clahe_clip_limit',
                        help="Controls the degree of contrast enhancement - higher values increase contrast")
                st.slider("Tile grid size", 1, 16, 8, step=1, key='clahe_tile_grid_size',
                        help="Defines the number of regions for adaptive histogram equalization")

            elif "XGBoost" in selected_method:
                st.checkbox("Enable post-processing", key='enable_postprocess')

                st.slider("Prediction threshold", 0.1, 0.99, 
                    float(getattr(st.session_state.get('classifier_model', None), 'threshold', 0.99) or 0.99),
                    key='classifier_threshold', help="Threshold for binary segmentation (XGBoost)")

            elif "U-Net" in selected_method:
                st.checkbox("Enable post-processing", key='enable_postprocess')
                st.markdown("#### U-Net Settings")

                default_threshold = st.session_state.get('optimized_unet_threshold', 0.70)
                st.slider("Prediction threshold", 0.1, 0.99, default_threshold, key='unet_threshold',
                        help="Threshold for binary segmentation")

                # Auto-threshold button
                if st.button("Auto-optimize threshold", key='auto_threshold_btn',
                           help="Automatically find the best threshold for the current image",
                           use_container_width=True):
                    if st.session_state.get('selected_image') and st.session_state['selected_image'] != "Select an image...":
                        st.session_state['trigger_auto_threshold'] = True
                        st.rerun()
                    else:
                        st.warning("Please select an image first")

    def show_introduction(self):
        """Show introduction and project information."""

        st.markdown("### Comparative Analysis of Traditional, Machine Learning, and Deep Learning Approaches")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Project Overview", "🔧 Methods", "🚀 Getting Started", "📊 Features"])

        with tab1:
            st.markdown("""
            ## About This Project

            This application is a comprehensive medical image analysis system designed for **retinal blood vessel segmentation**.
            It implements and compares three different computational paradigms to automatically detect vessels in fundus images:

            - **Traditional Frangi filter-based processing**
            - **Machine Learning (XGBoost classifier with handcrafted features)**
            - **Deep Learning (U-Net convolutional neural networks in PyTorch)**

            ### 🎯 Research Objectives
            - Compare traditional image processing vs. modern AI approaches
            - Evaluate performance across multiple segmentation pipelines
            - Provide an interactive platform for medical image analysis
            - Enable reproducible research in retinal vessel detection

            ### 📈 Clinical Significance
            Retinal vessel analysis is crucial for:
            - **Diabetic retinopathy** diagnosis and monitoring
            - **Hypertensive retinopathy** assessment
            - **Cardiovascular disease** risk evaluation
            - **Glaucoma** and other retinal pathology detection
            """)

        with tab2:
            st.markdown("""
            ## 🔧 Segmentation Methods

            ### 1. Traditional Image Processing
            **Frangi Filter-Based Enhancement**
            - Green channel extraction for optimal vessel contrast
            - Median denoising and optional CLAHE
            - Multiscale Frangi vesselness filtering with black ridge detection
            - ROI mask application and noise suppression
            - Morphological opening/closing for refinement
            - Step-by-step **intermediate visualization**

            ### 2. Machine Learning Approach
            **XGBoost Classifier with Convolutional Feature Extraction**
            - Multi-scale vesselness filters (Frangi, Sato)
            - Multi-orientation Gabor features for texture
            - Hessian eigenvalue analysis and orientation features
            - Black-hat morphology (multiple radii)
            - Local statistics: mean, std, max in sliding windows
            - MAD (Mean Absolute Deviation) and gradient features
            - Feature caching for faster training & inference
            - Class balancing and random subsampling
            - Early stopping during training for better generalization
            - Model persistence: saved to .pkl for later inference
            - Post-processing: small object removal, morphological opening/closing
            - Hyperparameter optimization with **Optuna** Bayesian search

            ### 3. Deep Learning Method
            **U-Net Convolutional Neural Network (PyTorch)**
            - Custom encoder–decoder with skip connections
            - Configurable filters, activations, dropout
            - BCE + Dice combined loss
            - Mixed precision training with gradient scaling
            - Early stopping and learning rate scheduling
            - Automatic **per-image threshold optimization** (F1-score based)
            - Post-processing: small object removal, morphological opening/closing
            - Architecture & weights saved in .pt checkpoints
            """)

        with tab3:
            st.markdown("""
            ## 🚀 How to Get Started

            ### Step 1: Select Your Data
            📁 Choose a retinal fundus image from the sidebar dropdown

            ### Step 2: Choose Processing Method
            🎛️ Select one of the three segmentation approaches:
            - **Traditional Processing** (always available)
            - **Machine Learning** (requires trained XGBoost .pkl)
            - **Deep Learning** (requires trained U-Net .pt)

            ### Step 3: Load Models (if needed)
            🔄 For ML/DL methods:
            - Select model file from the sidebar
            - Models auto-load if defaults are found

            ### Step 4: Configure Options
            ⚙️ Adjust advanced parameters:
            - Thresholds (manual or auto-optimize for U-Net)
            - Post-processing (morphological refinement)
            - Visualization (intermediate steps, metrics, comparison)

            ### Step 5: Analyze Results
            📊 Review outputs:
            - Segmented vessel maps
            - Performance metrics
            - Ground truth comparisons
            - Confusion matrices
            """)

        with tab4:
            st.markdown("""
            ## 📊 System Features

            ### 🔍 Analysis Capabilities
            - **Real-time Processing**: instant segmentation results
            - **Multi-method Comparison**: evaluate three paradigms
            - **Intermediate Visualization**: available for traditional pipeline
            - **Ground Truth Validation**: compare with expert annotations
            - **ROI Mask Support** and automatic eye orientation correction

            ### 📈 Evaluation Metrics
            - **Accuracy**: Overall correctness of classification. Shows the proportion of pixels correctly labeled as vessel or background.
            - **Sensitivity (Recall)**: Ability to correctly identify vessel pixels. Important for detecting all vessels in the image.
            - **Specificity**: Ability to correctly identify background pixels. Helps to avoid false vessel detections.
            - **Precision**: Proportion of predicted vessel pixels that are correct. Measures reliability of the vessel predictions.
            - **F1-Score**: Harmonic mean of precision and recall. Balances detection completeness and reliability.
            - **IoU (Intersection over Union)**: Measures overlap between predicted and true vessel masks. Widely used in segmentation evaluation.
            - **Dice Coefficient**: Another overlap metric similar to F1. Emphasizes agreement between prediction and ground truth.
            - **Arithmetic Mean & Geometric Mean**: Combined measures of sensitivity and specificity. Provide a robust assessment of model performance.
            - **Confusion Matrix** visualization: Shows true positives, false positives, false negatives, and true negatives. Offers detailed insight into classification performance.
                        
            ### 🎨 Visualization Tools
            - **Color-coded overlays**: TP=🟢, FP=🔴, FN=🔵
            - **Confusion Matrices**: detailed classification breakdown
            - **Interactive Controls**: adjustable thresholds and parameters
            - **Training feedback**: epoch loss & Dice evolution (U-Net)

            ### 💾 Model Management & Training
            - **Automatic Loading**: default model detection from config
            - **Training Scripts**:
              - `train_classifier.py` – trains XGBoost 
              - `train_unet.py` – trains U-Net
            - **Hyperparameter Tuning**:
              - `tune_classifier.py` – Optuna-based Bayesian optimization for XGBoost
            - **Artifacts**:
              - Best configs saved to `optuna/best_classifier_config.json`
              - Models saved as .pkl (XGBoost) and .pt (U-Net)
            - **Reproducibility**: YAML config-driven pipelines with automatic directory setup
            - **Caching**: feature caching for efficient ML experiments
            - **Entry Point (`run.py`)** - single unified script to:
                - Launch the Streamlit application
                - Train models (`--train classifier` / `--train unet`)
                - Run requirement checks (`--check`)
                - Perform hyperparameter tuning (`--tune_classifier`)
            """)

        st.markdown("---")

        # System status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available Methods", "3", help="Traditional, ML, and DL approaches")
        with col2:
            model_count = len(self.get_model_files('.pkl')) + len(self.get_model_files('.pt'))
            st.metric("Trained Models", str(model_count), help="Available ML/DL models")
        with col3:
            image_count = len(self.data_loader.get_image_files(self.config['data']['image_dir']))
            st.metric("Test Images", str(image_count), help="Images ready for analysis")

    def process_image(self):
        """Process the selected image."""

        image_name = st.session_state.get('selected_image', None)
        method = st.session_state.get('selected_method', None)

        # Load image
        image_path = os.path.join(self.config['data']['image_dir'], image_name)
        image = self.data_loader.load_image(image_path)

        if image is None:
            st.error(f"Failed to load image: {image_name}")
            return

        # Load ground truth and ROI
        _, ground_truth = self.data_loader.load_image_pair(image_name)
        roi_mask = self.data_loader.load_roi_mask(image_name)

        # Process based on selected method
        if "Traditional" in method:
            self.process_traditional(image, ground_truth, roi_mask, image_name)
        elif "XGBoost" in method:
            self.process_classifier(image, ground_truth, image_name)
        elif "U-Net" in method:
            self.process_unet(image, ground_truth, image_name)

    def process_traditional(self, image, ground_truth, roi_mask, image_name):
        """Process using traditional method."""

        st.header("Traditional Processing (Frangi Filter)")

        # Update parameters from advanced options
        if st.session_state.get('use_green_channel') is not None:
            self.traditional_model.use_green_channel = st.session_state['use_green_channel']
        
        if st.session_state.get('denoise_kernel'):
            self.traditional_model.denoise_kernel = st.session_state['denoise_kernel']
        
        # CLAHE parameters
        if st.session_state.get('use_clahe') is not None:
            self.traditional_model.use_clahe = st.session_state['use_clahe']

        if st.session_state.get('clahe_clip_limit'):
            self.traditional_model.clahe_clip_limit = st.session_state['clahe_clip_limit']

        if st.session_state.get('clahe_tile_grid_size'):
            self.traditional_model.clahe_tile_grid_size = st.session_state['clahe_tile_grid_size']

        # Frangi parameters
        if st.session_state.get('frangi_max_scale'):
            self.traditional_model.frangi_scales = list(range(1, st.session_state['frangi_max_scale'] + 1))
        
        if st.session_state.get('frangi_alpha'):
            self.traditional_model.frangi_alpha = st.session_state['frangi_alpha']
        
        if st.session_state.get('frangi_beta'):
            self.traditional_model.frangi_beta = st.session_state['frangi_beta']
        
        if st.session_state.get('frangi_gamma'):
            self.traditional_model.frangi_gamma = st.session_state['frangi_gamma']
        
        if st.session_state.get('frangi_black_ridges') is not None:
            self.traditional_model.frangi_black_ridges = st.session_state['frangi_black_ridges']
        
        # Morphological parameters
        if st.session_state.get('min_object_size'):
            self.traditional_model.min_object_size = st.session_state['min_object_size']
        
        if st.session_state.get('opening_kernel_size'):
            self.traditional_model.opening_kernel_size = st.session_state['opening_kernel_size']
        
        if st.session_state.get('closing_kernel_size'):
            self.traditional_model.closing_kernel_size = st.session_state['closing_kernel_size']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, width='stretch')

        was_right = self._is_right_eye(image_name)
        image_for_infer = np.fliplr(image) if was_right else image

        with st.spinner("Processing..."):
            if st.session_state.get('show_intermediate'):
                results = self.traditional_model.predict_with_intermediate_results(
                    image_for_infer, roi_mask
                )
                prediction = results['final_vessels']
            else:
                prediction = self.traditional_model.predict(image_for_infer, roi_mask)

        if was_right:
            prediction = np.fliplr(prediction)

        with col2:
            st.subheader("Segmented Vessels")
            st.image(prediction, width='stretch', channels="GRAY")

        # Show intermediate results
        if st.session_state.get('show_intermediate') and 'results' in locals():
            self.show_intermediate_results(results)

        # Show comparison and metrics
        if ground_truth is not None:
            if st.session_state.get('show_comparison'):
                self.show_comparison(prediction, ground_truth)

            if st.session_state.get('show_metrics'):
                self.show_metrics(prediction, ground_truth)

    def process_classifier(self, image, ground_truth, image_name):
        """Process using XGBoost classifier."""

        st.header("Machine Learning (XGBoost)")

        if not st.session_state.get('classifier_model'):
            st.error("Please load a classifier model from the sidebar first!")
            return

        self.classifier_model = st.session_state['classifier_model']

        thr = st.session_state.get('classifier_threshold', None)
        if thr is not None:
            self.classifier_model.threshold = float(thr)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, width='stretch')

        was_right = self._is_right_eye(image_name)
        image_for_infer = np.fliplr(image) if was_right else image

        with st.spinner("Extracting features and classifying..."):
            prediction = self.classifier_model.predict(image_for_infer)

        if was_right:
            prediction = np.fliplr(prediction)

        # Apply post-processing
        if st.session_state.get('enable_postprocess'):
            with st.spinner("Applying post-processing..."):
                prediction = self.classifier_model.postprocess(prediction)

        with col2:
            st.subheader("Segmented Vessels")
            st.image(prediction, width='stretch', channels="GRAY")

        # Show comparison and metrics
        if ground_truth is not None:
            # Resize ground truth to match prediction
            if getattr(self.classifier_model, "resize_factor", 1) > 1:
                ground_truth_resized = cv2.resize(ground_truth, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
                ground_truth_resized = (ground_truth_resized > 127).astype(np.uint8) * 255
            else:
                ground_truth_resized = ground_truth

            if st.session_state.get('show_comparison'):
                self.show_comparison(prediction, ground_truth_resized)

            if st.session_state.get('show_metrics'):
                self.show_metrics(prediction, ground_truth_resized)

    def process_unet(self, image, ground_truth, image_name):
        """Process using U-Net model."""

        st.header("Deep Learning (U-Net)")

        if not st.session_state.get('unet_model'):
            st.error("Please load a U-Net model from the sidebar first!")
            return

        self.unet_model = st.session_state['unet_model']

        # Handle auto-threshold trigger
        if st.session_state.get('trigger_auto_threshold'):
            st.session_state['trigger_auto_threshold'] = False

            was_right = self._is_right_eye(image_name)
            image_for_auto = np.fliplr(image) if was_right else image

            ground_truth_for_auto = ground_truth
            if was_right and ground_truth is not None:
                ground_truth_for_auto = np.fliplr(ground_truth)

            with st.spinner("Auto-optimizing threshold..."):
                try:
                    optimal_threshold = self.unet_model.auto_threshold(
                        image_for_auto,
                        ground_truth=ground_truth_for_auto,
                        threshold_range=(0.1, 0.9),
                        num_thresholds=20
                    )

                    st.session_state['optimized_unet_threshold'] = optimal_threshold

                    if 'unet_threshold' in st.session_state:
                        del st.session_state['unet_threshold']

                except Exception as e:
                    st.error(f"Auto-threshold failed: {str(e)}")
                    logger.error(f"Auto-threshold error: {e}")

                st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, width='stretch')

        was_right = self._is_right_eye(image_name)
        image_for_infer = np.fliplr(image) if was_right else image

        with st.spinner("Running U-Net inference..."):
            self.unet_model.threshold = st.session_state.get('unet_threshold', 0.70)
            prediction = self.unet_model.predict(image_for_infer)

        if was_right:
            prediction = np.fliplr(prediction)

        # Apply post-processing
        if st.session_state.get('enable_postprocess'):
            prediction = self.unet_model.postprocess(prediction)

        with col2:
            st.subheader("Segmented Vessels")
            st.image(prediction, width='stretch', channels="GRAY")

        # Show comparison and metrics
        if ground_truth is not None:
            ground_truth_resized = cv2.resize(
                ground_truth,
                (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            if st.session_state.get('show_comparison'):
                self.show_comparison(prediction, ground_truth_resized)

            if st.session_state.get('show_metrics'):
                self.show_metrics(prediction, ground_truth_resized)

    def show_intermediate_results(self, results):
        """Show intermediate processing results."""

        st.subheader("Intermediate Processing Results")

        cols = st.columns(3)

        # Select which intermediate results to show
        intermediate_keys = [
            ('preprocessed', 'Preprocessed (Green Channel)'),
            ('frangi_filtered', 'Frangi Filter'),
            ('frangi_binary', 'Binary Frangi'),
            ('noise_mask', 'Noise Mask'),
            ('roi_masked', 'ROI Applied')
        ]

        for idx, (key, title) in enumerate(intermediate_keys):
            if key in results:
                col = cols[idx % 3]
                with col:
                    st.markdown(f"**{title}**")
                    st.image(results[key], width='stretch', channels="GRAY")

    def show_comparison(self, prediction, ground_truth):
        """Show comparison between prediction and ground truth."""

        st.subheader("Comparison with Ground Truth")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Ground Truth**")
            st.image(ground_truth, width='stretch', channels="GRAY")

        with col2:
            st.markdown("**Prediction**")
            st.image(prediction, width='stretch', channels="GRAY")

        with col3:
            st.markdown("**Comparison**")
            comparison = self.metrics_calculator.create_comparison_image(
                ground_truth, prediction
            )
            comparison_with_legend = self.metrics_calculator.add_legend_to_comparison(
                comparison
            )
            st.image(comparison_with_legend, width='stretch')

            st.markdown(
                "**Legend:**  \n"
                "🟢 Green: True Positives (TP)  \n"
                "🔴 Red: False Positives (FP)  \n"
                "🔵 Blue: False Negatives (FN)  \n"
            )

    def show_metrics(self, prediction, ground_truth):
        """Show evaluation metrics."""
        st.subheader("Evaluation Metrics")

        with st.spinner("Computing evaluation metrics..."):
            metrics = self.metrics_calculator.calculate_metrics(ground_truth, prediction)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
        with col2:
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            st.metric("IoU", f"{metrics['iou']:.4f}")
        with col4:
            st.metric("Dice Coefficient", f"{metrics['dice']:.4f}")
            st.metric("Geometric Mean", f"{metrics['geometric_mean']:.4f}")

        with st.expander("Confusion Matrix"):
            ph = st.empty()
            with ph.container():
                with st.spinner("Building confusion matrix..."):

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        cm = self.metrics_calculator.calculate_confusion_matrix(
                            ground_truth, prediction
                        )
                        fig = self.metrics_calculator.plot_confusion_matrix(cm)
                        fig.set_size_inches(6, 4)
                        st.pyplot(fig, width='content')


def main():
    """Main entry point."""
    app = VesselSegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
