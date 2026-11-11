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
from crnn import CRNN
from ocr_config import CANVAS_H, CANVAS_W, ALPHABET


def main():

    # Determine model file path based on model type
    model_path = f"./models/d2r_ocr_model.pth"
    onnx_path = f"../../game_bot/models/d2r_ocr.onnx"

    # Create output directory
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Load model on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model information
    classes = checkpoint.get("classes", [])
    temperature = checkpoint.get("temperature", 1.0)

    # Load the base model using the existing function

    export_model = CRNN(num_chars=len(ALPHABET)).to(device)
    checkpoint = torch.load(model_path)
    export_model.load_state_dict(checkpoint)

    export_model.to(device)
    export_model.eval()

    # Create dummy input (batch=1, channels=3, height=32, width=72)
    dummy_input = torch.randn(2, 3, CANVAS_H, CANVAS_W, device=device)

    # Export to ONNX
    torch.onnx.export(
        export_model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {1: "batch_size"}},
    )

    print(f"Exported model to: {onnx_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
