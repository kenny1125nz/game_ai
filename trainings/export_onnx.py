#!/usr/bin/env python3
"""
ONNX export for hybrid rune classifier models.
Supports both base models and calibrated models.
"""
import torch
import torch.nn as nn
import torch.onnx
import onnx
import sys
import os

# Add path to import the model
sys.path.append("trainings/rune")
from rune.network.cnn_transformer_hybrid import load_inference_model
from rune.calibrated_model import CalibratedModel


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <model_size> [model_type]")
        print("  model_size: small, medium, or large")
        print("  model_type: base (default) or calibrated")
        sys.exit(1)

    size = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "base"

    # Validate model type
    if model_type not in ["base", "calibrated"]:
        print(
            f"Error: Invalid model type '{model_type}'. Must be 'base' or 'calibrated'."
        )
        sys.exit(1)

    # Determine model file path based on model type
    if model_type == "calibrated":
        model_path = f"models/{size}_hybrid_rune_classifier_calibrated.pth"
        onnx_path = f"../game_bot/models/{size}_hybrid_rune_classifier_calibrated.onnx"
    else:
        model_path = f"models/{size}_hybrid_rune_classifier.pth"
        onnx_path = f"../game_bot/models/{size}_hybrid_rune_classifier.onnx"

    # Create output directory
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Load model on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model information
    classes = checkpoint.get("classes", [])
    temperature = checkpoint.get("temperature", 1.0)

    # Load the base model using the existing function

    base_model = load_inference_model(model_path, device=device, size=size)
    # Determine which model to export
    if model_type == "calibrated":
        # Create calibrated model wrapper
        export_model = CalibratedModel(base_model, temperature)
    else:
        export_model = base_model

    export_model.to(device)
    export_model.eval()

    # Create dummy input (batch=1, channels=3, height=32, width=72)
    dummy_input = torch.randn(1, 3, 32, 72, device=device)

    # Export to ONNX
    torch.onnx.export(
        export_model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Exported {model_type} model to: {onnx_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
