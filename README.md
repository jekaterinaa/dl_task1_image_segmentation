# Multi-Class Semantic Segmentation with U-Net

This project implements a U-Net model for multi-class semantic segmentation, targeting three classes: **Cat**, **Dog**, and **Car**, along with the background.

## Project Overview

- **Model**: U-Net architecture implemented in PyTorch.
- **Dataset**: Open Images V7 segmentation subset, filtered for the target classes.
- **Pre-trained Model**: A pre-trained model is provided and can be loaded from `unet_trained.pth`.
- **Notebook**: The main workflow is in `task1.ipynb`.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure you have Python 3.11.12 installed.

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd Task1_dl
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Running the Project

The main workflow is contained in the Jupyter Notebook `task1.ipynb`. Open the notebook and execute the cells sequentially to:

1. **Download and Filter Data**: Use FiftyOne to download the Open Images V7 dataset and filter it for the target classes.
2. **Combine Masks**: Generate combined segmentation masks for the target classes.
3. **Data Augmentation**: Apply augmentations to the training and testing datasets.
4. **Model Training**: Train the U-Net model using a hybrid loss function (Focal Loss + Dice Loss).
5. **Model Evaluation**: Evaluate the model on the test set and visualize predictions.
6. **Pre-trained Model**: Load the pre-trained model weights from `unet_trained.pth` to skip training and directly test the model.

## Pre-trained Model

The pre-trained model weights are saved in `unet_trained.pth`. You can load it from file to skip training it from scratch.

## Key Files

- **`task1.ipynb`**: Main notebook for the entire workflow.
- **`unet_model.py`**: U-Net model implementation.
- **`loss_function.py`**: Custom loss functions (Dice Loss, Focal Loss, Final Loss).
- **`augmentations.py`**: Data augmentation pipelines.
- **`model_training_functions.py`**: Utilities for model training.
- **`visualisation_functions.py`**: Functions for visualizing prediction masks.

## Notes

- Ensure the dataset is downloaded before training or testing.
- The pre-trained model can be used to skip the training phase and directly evaluate or test the model.