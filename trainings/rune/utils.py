"""
Utility functions for CNN training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor (default: 1)
        gamma: Focusing parameter (default: 2)
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return loss.mean()


def diagnose_errors(model, val_loader, classes, device='cuda'):
    """
    Diagnose model errors by analyzing confusion matrix.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        classes: List of class names
        device: Device to use (cuda/cpu)
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print the "Troublemakers"
    print("\n--- TOP CONFUSIONS ---")
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            if i != j and val > 0:
                print(f"Model confused {classes[i]} for {classes[j]} ({val} times)")


def show_tensor_batch(images, labels, max_images=8):
    """
    Visualize a batch of tensor images.
    
    Args:
        images: Batch of images (tensor)
        labels: Batch of labels (tensor)
        max_images: Maximum number of images to display
    """
    # Move from GPU to CPU if necessary
    images = images.cpu()
    
    # Create a grid
    num_to_show = min(len(images), max_images)
    fig, axes = plt.subplots(1, num_to_show, figsize=(20, 5))
    
    # Handle the case where only 1 image is in the batch (axes isn't a list)
    if num_to_show == 1:
        axes = [axes]

    for i in range(num_to_show):
        # 1. Undo the Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        # img = (tensor * std) + mean
        img = images[i] * 0.5 + 0.5 
        
        # 2. Permute dimensions: (C, H, W) -> (H, W, C)
        # This is the critical step for RGB images
        img = img.permute(1, 2, 0).numpy()
        
        # 3. Clip values to [0, 1] to prevent floating point warnings
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()