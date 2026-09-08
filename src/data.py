"""Data loading and preprocessing for chest X-ray images."""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, List

from src.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_STATE
)


def get_train_transforms() -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Validation/Test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Load chest X-ray data and return train/val/test DataLoaders.

    Expects directory structure:
        data/train/NORMAL/  data/train/PNEUMONIA/
        data/test/NORMAL/   data/test/PNEUMONIA/

    Handles class imbalance using WeightedRandomSampler.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(DATA_DIR)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training data not found at {train_dir}")

    # Load datasets
    full_train = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms())
    class_names = full_train.classes

    # 80/20 train-val split
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    # Print class distribution
    targets = [full_train.targets[i] for i in train_dataset.indices]
    class_counts = [targets.count(i) for i in range(len(class_names))]
    print(f"\nTraining class distribution:")
    for name, count in zip(class_names, class_counts):
        print(f"  {name}: {count}")

    # WeightedRandomSampler to handle class imbalance
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = torch.tensor([class_weights[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, class_names
