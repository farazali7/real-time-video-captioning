# Source Directory

This directory contains the core Python scripts for training and inference in the machine learning project. It includes model definitions, utility functions, and main scripts to kick off processes like training models and running inference with trained models.


## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Inference](#inference)
3. [Training](#training)


## Repository Structure

```
src
├── models
│     └── ... # Model definitions and architecture files.
├── utils
│     └── ... # Utility functions for data loading and processing.
├── init.py # Initializes the Python module structure.
├── inference.py # Script for running inference using trained models.
├── metrics.py # Metrics and evaluation functions.
└── train.py # Script for training models.
```


## Inference

The `inference.py` script is responsible for loading trained models and performing inference on test data. It handles the following:
- Reading configuration from a config file for model and data loading.
- Loading the student and teacher models from the checkpoints based on the given `run_name`.
- Preparing the test data loader with the appropriate dataset.
- Running the inference on the test data to generate predictions.
- Printing both the ground truth and predicted captions for comparison.

This script is typically used after a model has been trained and a checkpoint has been saved. It requires the correct `run_name` to locate the checkpoint file.


## Training

The `train.py` script is designed for training models using a knowledge distillation approach. Key functionalities include:
- Setting up the WANDB environment for logging and tracking experiments.
- Defining and loading datasets for training, validation, and testing phases.
- Initializing the student and teacher models with specified configurations.
- Configuring callbacks for model checkpointing and monitoring.
- Orchestrating the training process using the PyTorch Lightning Trainer.
- Plotting the loss function to visualize the training progress.
- Conducting the training loop, including forward and backward passes, and updating model weights.

It uses the PyTorch Lightning framework to streamline the training loop and relies on WANDB for experiment tracking. The script also includes provisions for model evaluation post-training.