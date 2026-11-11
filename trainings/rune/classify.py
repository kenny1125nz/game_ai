import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import sys
from network.cnn_transformer_hybrid import load_inference_model, inference_transforms
from rune_config import config, TRAINING_DATA_DIR

import matplotlib.pyplot as plt

image_height, image_width = (32, 72)

# --- CONFIG (Must match your training script exactly) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    "Amn",
    "Background",
    "Ber",
    "Cham",
    "Dol",
    "El",
    "Eld",
    "Eth",
    "Fal",
    "Gul",
    "Hel",
    "Io",
    "Ist",
    "Ith",
    "Jah",
    "Ko",
    "Lem",
    "Lo",
    "Lum",
    "Mal",
    "Nef",
    "Ohm",
    "Ort",
    "Pul",
    "Ral",
    "Rune",
    "Shael",
    "Sol",
    "Sur",
    "Tal",
    "Thul",
    "Tir",
    "Um",
    "Vex",
    "Zod",
]
RARE_ORDER = [
    "El",
    "Eld",
    "Tir",
    "Nef",
    "Eth",
    "Ith",
    "Tal",
    "Ral",
    "Ort",
    "Thul",
    "Amn",
    "Sol",
    "Shael",
    "Dol",
    "Hel",
    "Io",
    "Lum",
    "Ko",
    "Fal",
    "Lem",
    "Pul",
    "Um",
    "Mal",
    "Ist",
    "Gul",
    "Vex",
    "Ohm",
    "Lo",
    "Sur",
    "Ber",
    "Jah",
    "Cham",
    "Zod",
]
model = load_inference_model(
    "../../../models/small_hybrid_rune_classifier.pth", size="small"
)


def show_single_tensor(tensor):
    # 1. Move to CPU and remove the batch dimension if it exists
    img = tensor.cpu().detach()

    # 2. Undo the Normalize((0.5,), (0.5,)) to get back to [0, 1] range
    img = img * 0.5 + 0.5
    img = img.squeeze().numpy()

    plt.figure(figsize=(4, 2))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def classify(chip):
    # 4. CLASSIFICATION (The Neural Validation)
    input_tensor = (
        inference_transforms(size=(image_height, image_width))(chip)
        .unsqueeze(0)
        .to(DEVICE)
    )
    show_single_tensor(input_tensor)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, class_idx = torch.max(probs, 0)
        rune_name = CLASSES[class_idx]

        return (rune_name, conf.item(), probs)


if __name__ == "__main__":
    rune = sys.argv[1]
    index = int(sys.argv[2])
    image_path = f"{TRAINING_DATA_DIR}/{rune}/{rune}_{index}.png"
    img = cv2.imread(image_path)
    result = classify(img)
    print(result)
