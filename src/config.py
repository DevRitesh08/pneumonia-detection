"""Configuration constants for the Pneumonia Detection System."""

# Reproducibility
RANDOM_STATE = 42

# Data
DATA_DIR = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 4 on Linux

# Model
MODEL_NAME = "resnet18"  # Options: resnet18, resnet50
NUM_CLASSES = 2
FREEZE_BACKBONE = True
DROPOUT = 0.3

# Training
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Artifact paths
MODEL_PATH = "pneumonia_model.pth"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
