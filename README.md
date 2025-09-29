# Retinal Vessel Segmentation System

A comprehensive **medical image analysis system** for automatic **retinal blood vessel segmentation** in fundus images. The project implements and compares three computational approaches: **Traditional Frangi filter**, **Machine Learning (XGBoost classifier)**, and **Deep Learning (U-Net CNN)**. It includes an interactive **Streamlit application**, training scripts, hyperparameter tuning, evaluation metrics, and comparison with expert annotations.

The project utilizes the publicly available **Fundus Image Dataset** provided by FAU: [https://www5.cs.fau.de/research/data/fundus-images/](https://www5.cs.fau.de/research/data/fundus-images/)


## Technologies

<img src="https://img.shields.io/badge/PyTorch-222222?logo=pytorch" height="30"> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv" height="30"> <img src="https://img.shields.io/badge/scikit learn-3499cd?logo=scikit-learn" height="30"> <img src="https://img.shields.io/badge/Optuna-002C76?logo=optuna" height="30"> <img src="https://img.shields.io/badge/Streamlit-222629?logo=streamlit" height="30"> <img src="https://img.shields.io/badge/SciPy-14181e?logo=scipy" height="30"> <img src="https://img.shields.io/badge/Numpy-013243?logo=numpy" height="30"> <img src="https://img.shields.io/badge/tqdm-4051b5?logo=tqdm" height="30"> 

## Features

### Segmentation Approaches

1. **Traditional Image Processing (Frangi Filter)**

   * Green channel extraction
   * Median denoising and optional CLAHE
   * Multiscale Frangi vesselness filtering
   * Noise removal and morphological refinement
   * Step-by-step intermediate visualization

2. **Machine Learning (XGBoost Classifier)**

   * Multi-scale vesselness features (Frangi, Sato)
   * Gabor filters for texture
   * Hessian eigenvalue and orientation features
   * Black-hat morphology with multiple radii
   * Local statistics: mean, std, max
   * MAD and gradient features
   * Feature caching for efficient training
   * Class balancing and random subsampling
   * Early stopping during training for better generalization
   * Model saved as `.pkl` for later inference
   * Post-processing: small object removal, opening, closing
   * Hyperparameter tuning via **Optuna**

3. **Deep Learning (U-Net CNN)**

   * Custom encoder-decoder architecture with skip connections
   * BCE + Dice combined loss
   * Mixed precision training
   * Early stopping and learning rate scheduling
   * Automatic per-image threshold optimization (F1-score)
   * Post-processing: morphological refinement
   * Model and architecture saved as `.pt` checkpoints

### Evaluation Metrics

* **Accuracy**: Overall correctness of classification. Shows the proportion of pixels correctly labeled as vessel or background.
* **Sensitivity (Recall)**: Ability to correctly identify vessel pixels. Important for detecting all vessels in the image.
* **Specificity**: Ability to correctly identify background pixels. Helps to avoid false vessel detections.
* **Precision**: Proportion of predicted vessel pixels that are correct. Measures reliability of the vessel predictions.
* **F1-Score**: Harmonic mean of precision and recall. Balances detection completeness and reliability.
* **IoU (Intersection over Union)**: Measures overlap between predicted and true vessel masks. Widely used in segmentation evaluation.
* **Dice Coefficient**: Another overlap metric similar to F1. Emphasizes agreement between prediction and ground truth.
* **Arithmetic Mean & Geometric Mean**: Combined measures of sensitivity and specificity. Provide a robust assessment of model performance.
* **Confusion Matrix** visualization: Shows true positives, false positives, false negatives, and true negatives. Offers detailed insight into classification performance.

## Screenshots

<table>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/f248d20b-1d9e-4ffb-9e8d-443b8d7b40d7" alt="Frangi Filter">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/ac60200d-5d21-466b-accd-9b99763c8c4f" alt="XGBoost">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/34d095ef-22f8-4af7-a390-6922648ef6cc" alt="U-Net">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/c878e52a-f30c-4c9d-9a59-ebced99d8713" alt="Overview">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/f229eb4a-0b4d-4726-b18b-6a5904fe4f95" alt="Methods">
    </td>
  </tr>
    <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1dddcf93-b09e-4f0c-8dc3-972060409f43" alt="Getting Started">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/61c04c23-6a0c-45c2-9bc5-962decf8caa7" alt="Features">
    </td>
</table>

## Project Structure

```
.
├── run.py                        # Main application entry point
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── src/                          # Source code modules
│   ├── app/                      # Streamlit web application
│   │   └── streamlit_app.py      # Streamlit app code
│   ├── models/                   # Model implementations
│   │   ├── base_model.py         # Abstract base class
│   │   ├── traditional_model.py  # Frangi filter approach
│   │   ├── classifier_model.py   # XGBoost implementation
│   │   ├── unet_model.py         # U-Net deep learning
│   │   └── feature_extractor.py  # Feature extraction utilities for ML
│   ├── preprocessing/            # Data loading and image processing
│   │   ├── data_loader.py        # Data loading utilities
│   │   └── image_processor.py    # Image processing functions
│   ├── evaluation/               # Metrics and performance evaluation
│   │   └── metrics.py            # Comprehensive evaluation metrics
│   └── utils/                    # Configuration and utilities
│       ├── config.py             # YAML configuration management
│       └── logger.py             # Logging system
├── training/                     # Standalone training scripts
│   ├── train_classifier.py       # XGBoost training
│   ├── train_unet.py             # U-Net training
│   └── tune_classifier.py        # Hyperparameter optimization for XGBoost
├── data/                         # Dataset directories
│   ├── images/                   # Input retinal images
│   ├── train_images/             # Training images
│   ├── masks/                    # Segmentation masks
│   └── ground_truth/             # Ground truth annotations
├── models/                       # Saved trained models
└── optuna/                       # Hyperparameter tuning results
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/krzsmal/RetinalVesselSegmentation
cd RetinalVesselSegmentation
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python run.py
```

Options:

* `--train classifier` – train XGBoost model
* `--train unet` – train U-Net model
* `--check` – verify requirements
* `--tune_classifier` – run hyperparameter tuning for XGBoost

### 4. Open Streamlit UI

Use the sidebar to:

* Select a retinal fundus image
* Choose a segmentation method
* Load trained models
* Configure parameters
* Visualize segmentation results and metrics

### Configuration
Use `config.yaml` to set paths, model names, training parameters, and preprocessing options.