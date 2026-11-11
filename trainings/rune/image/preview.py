#!/usr/bin/env python3
"""
Preview script for rune image processing pipeline.
Displays original image and each processing stage for visual inspection.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
from processor import process_image
from config import TRAINING_DATA_DIR

# Constants


def load_image(size:str,rune: str, index: int):
    """
    Simplified image loading: directly construct file path and load image.
    
    Args:
        rune: Rune name (e.g., "Amn", "Ber")
        index: Image index (e.g., 0, 1, 2)
        
    Returns:
        numpy array of the loaded image
    """
    # Construct the file path
    file_path = Path(TRAINING_DATA_DIR)/size / rune / f"{rune}_{index}.png"
    
    # Check if file exists
    if not file_path.exists():
        # Try capitalized version
        rune_capitalized = rune.capitalize()
        file_path = Path(TRAINING_DATA_DIR) / rune_capitalized / f"{rune_capitalized}_{index}.png"
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        rune = rune_capitalized
    
    # Load image using OpenCV
    image = cv2.imread(str(file_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    
    return image


def display_images(original_image, processing_stages):

    # Prepare all images for display
    images_to_display = []
    
    # Add original image
    if len(original_image.shape) == 2:  # Grayscale
        original_display = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_display = original_image.copy()
    images_to_display.append(("Original", original_display))
    
    # Add processing stages
    for stage_name, stage_image in processing_stages:
        # Convert to appropriate format for display if needed
        if len(stage_image.shape) == 2:  # Grayscale
            display_image = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = stage_image.copy()
        images_to_display.append((stage_name, display_image))
    
    # Resize all images to the same height for stacking
    target_height = 200  # Fixed height for all images
    resized_images = []
    
    for stage_name, img in images_to_display:
        height, width = img.shape[:2]
        # Calculate new width while maintaining aspect ratio
        new_width = int(width * (target_height / height))
        resized_img = cv2.resize(img, (new_width, target_height))
        resized_images.append((stage_name, resized_img))
    
    # Stack images horizontally using numpy
    stacked_images = []
    for stage_name, img in resized_images:
        # Add label to the image
        labeled_img = img.copy()
        cv2.putText(labeled_img, stage_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        stacked_images.append(labeled_img)
    
    # Use numpy.hstack to combine all images horizontally
    composite_image = np.hstack(stacked_images)
    
    # Display the composite image
    cv2.imshow("Image Processing Pipeline", composite_image)
    
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    size = sys.argv[1]
    rune = sys.argv[2]
    index = int(sys.argv[3])
    print(f"Loading {rune} rune at index {index}...")

    # Simplified image loading
    original_image = load_image(size,rune, index)
    processing_stages = process_image(original_image, debug=True)
    
    display_images(original_image, processing_stages)


if __name__ == "__main__":
    main()