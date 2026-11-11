"""
Calibration module for CNN models with temperature scaling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import sys
import os

from network.cnn_transformer_hybrid import (
    RuneHybridNet,
    train_transforms,
    load_inference_model,
)
from rune_config import config, CALIBRATION_DATA_DIR

from calibrated_model import CalibratedModel


def get_optimal_temperature(logits, labels):
    # 1. Initialize T as a learnable parameter starting at 1.0
    temperature = nn.Parameter(torch.ones(1) * 1.5)

    # 2. Set up the optimizer (L-BFGS is standard for this)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll_loss = nn.CrossEntropyLoss()

    # 3. Define the optimization closure
    def eval_loss():
        optimizer.zero_grad()
        loss = nll_loss(logits / temperature, labels)
        loss.backward()
        return loss

    # 4. Run the optimization
    optimizer.step(eval_loss)

    return temperature.item()


def calibrate_model(model, calibration_data_dir, device, batch_size=128):
    print(f"Calibrating model using data from: {calibration_data_dir}")

    # Load calibration dataset
    image_size = config["image_size"][::-1]
    transform = train_transforms(image_size)

    calibration_dataset = datasets.ImageFolder(
        root=calibration_data_dir, transform=transform
    )
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=batch_size, shuffle=False
    )

    # Collect all logits and labels
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in calibration_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Get optimal temperature
    optimal_temperature = get_optimal_temperature(all_logits, all_labels)
    print(f"Optimal temperature found: {optimal_temperature:.4f}")

    return optimal_temperature


def save_calibrated_model(base_model, classes, optimal_temperature, model_file):
    # Create calibrated model file name
    calibrated_file = model_file.replace(".pth", "_calibrated.pth")

    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "classes": classes,
            "temperature": optimal_temperature,
            "model_type": "calibrated",
        },
        calibrated_file,
    )

    print(
        f"Saved calibrated model with temperature {optimal_temperature:.4f} to {calibrated_file}"
    )
    return calibrated_file


def load_calibrated_model(model_file, device="cuda"):
    checkpoint = torch.load(model_file, map_location=device)

    # Check if this is a calibrated model
    if "temperature" not in checkpoint:
        raise ValueError(
            f"Model file {model_file} is not a calibrated model (missing temperature)"
        )

    temperature = checkpoint["temperature"]
    classes = checkpoint["classes"]

    # Note: The actual model loading depends on the base model architecture
    # This function assumes the base model is already defined elsewhere
    print(f"Loaded calibrated model with temperature: {temperature}")
    print(f"Classes: {classes}")

    return checkpoint, classes, temperature


def main():
    model_size = sys.argv[1]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model file paths
    model_file = f"../models/{model_size}_hybrid_rune_classifier.pth"
    calibrated_file = model_file.replace(".pth", "_calibrated.pth")

    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Error: Model file not found: {model_file}")
        print("Please train the model first using train_cnn.py")
        sys.exit(1)

    # Load the trained model
    print(f"Loading trained model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)

    # Extract classes from checkpoint
    classes = checkpoint.get("classes", [])
    if not classes:
        print("Warning: No classes found in model checkpoint")
        # You might need to define classes based on your dataset
        classes = []

    # Create model architecture
    num_classes = len(classes) if classes else 35  # Default to 35 if classes not found
    model = RuneHybridNet(num_classes=num_classes, size=model_size).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully. Number of classes: {num_classes}")

    # Perform calibration
    print("\n--- Starting Model Calibration ---")
    optimal_temperature = calibrate_model(model, CALIBRATION_DATA_DIR, device)

    # Save calibrated model
    calibrated_file = save_calibrated_model(
        model, classes, optimal_temperature, model_file
    )

    print(f"\nCalibration complete!")
    print(f"Optimal temperature: {optimal_temperature:.4f}")


if __name__ == "__main__":
    main()
