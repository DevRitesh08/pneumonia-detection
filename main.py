"""
Pneumonia Detection from Chest X-Rays using Transfer Learning
Modular entry point using src/ directory components.
"""

import numpy as np
import torch

from src.config import EPOCHS, LEARNING_RATE, MODEL_NAME, CLASS_NAMES, MODEL_PATH, CONFUSION_MATRIX_PATH
from src.data import get_dataloaders
from src.model import get_model, train_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load Data
    print("Loading chest X-ray data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = get_model(MODEL_NAME)

    # 3. Training
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )

    # 4. Plot training curves
    plot_training_curves(history, save_path="training_curves.png")

    # 5. Final Evaluation on Test Set
    print("\n" + "=" * 50)
    print("FINAL EVALUATION (TEST SET)")
    print("=" * 50)

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    results = evaluate_model(model, test_loader, CLASS_NAMES, device)

    # 6. Save Confusion Matrix
    plot_confusion_matrix(
        results["labels"], results["predictions"], CLASS_NAMES,
        title="Confusion Matrix (Test Set)", save_path=CONFUSION_MATRIX_PATH,
    )
    plot_confusion_matrix(
        results["labels"], results["predictions"], CLASS_NAMES,
        title="Normalized Confusion Matrix", save_path="confusion_matrix_normalized.png",
        normalize=True,
    )

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
