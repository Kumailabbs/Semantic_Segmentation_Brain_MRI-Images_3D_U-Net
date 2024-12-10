3D U-Net for Brain Tumor Segmentation on BraTS 2020 Dataset
This repository provides a complete pipeline for training a 3D U-Net model for semantic segmentation of brain tumors using the BraTS 2020 dataset. It includes scripts for data preparation, model training, evaluation, and visualization.

Overview
1. Dataset Preparation
Supports multimodal MRI scans (T1, T1c, T2, FLAIR).
Processes NIfTI files using nibabel.
Preprocessing includes:
Normalization of intensity values to [0, 1].
Mapping tumor labels (0, 1, 2, 4) to 0, 1, 2, 3.
Cropping and patch extraction for optimized memory usage.
Saves preprocessed data in .npy format for training and validation.
2. 3D U-Net Model
Implements a custom 3D U-Net architecture:
Adjustable input shape (e.g., 128x128x128).
Multiple convolutional layers with ReLU activation.
Dropout for regularization and concatenation in skip connections.
Output: Multi-class segmentation with a softmax activation layer.
3. Training Pipeline
Loss Function:
Weighted Dice Loss combined with Categorical Focal Loss to handle class imbalance.
Metrics:
Accuracy and Intersection over Union (IoU).
Data Generators:
Custom data loaders for efficient batch-wise training on 3D data.
Hyperparameters:
Configurable learning rate, batch size, and class weights.
Saves models after training for inference or fine-tuning.
4. Evaluation and Visualization
Quantitative metrics: Mean IoU on validation data.
Visualizes MRI slices, ground truth masks, and predictions side-by-side.
How to Use
Prepare Data:
Place the BraTS 2020 dataset in the appropriate folder structure.
Run the data preprocessing scripts to generate .npy files.
Train Model:
Use the training script to train the 3D U-Net model.
Monitor performance via metrics and loss plots.
Evaluate and Predict:
Evaluate using IoU or make predictions on unseen data.
Requirements
Python 3.8+
TensorFlow/Keras
nibabel
numpy
matplotlib
segmentation-models-3D
Acknowledgments
This project is inspired by the BraTS 2020 challenge and leverages the 3D U-Net architecture for medical image segmentation.
