import os
import cv2
import numpy as np
from image.processor import process_image
from rune_config import TRAINING_DATA_DIR
from tqdm import tqdm
import sys


def process_all_images(data_dir, processed_dir="./d2r/screen/runes/model/processed/"):
    """Process all images in training_data directory and save processed results to processed directory."""

    # Define paths

    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Get all rune directories
    rune_dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    print(f"Found {len(rune_dirs)} rune directories to process")

    total_images = 0
    processed_images = 0

    # Count total images first for progress bar
    for rune_dir in rune_dirs:
        rune_path = os.path.join(data_dir, rune_dir)
        image_files = [
            f
            for f in os.listdir(rune_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        total_images += len(image_files)

    print(f"Total images to process: {total_images}")

    # Process each rune directory
    for rune_dir in rune_dirs:
        rune_path = os.path.join(data_dir, rune_dir)
        output_rune_dir = os.path.join(processed_dir, rune_dir)

        # Create output directory for this rune
        os.makedirs(output_rune_dir, exist_ok=True)

        # Get all image files in the rune directory
        image_files = [
            f
            for f in os.listdir(rune_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"\nProcessing {len(image_files)} images in '{rune_dir}'...")

        # Process each image
        for image_file in tqdm(image_files, desc=f"{rune_dir}"):
            try:
                # Load image
                image_path = os.path.join(rune_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue

                # Process image using processor.py
                processed = process_image(image, debug=False)

                # Check if processed is a numpy array (not debug output)
                if isinstance(processed, list):
                    # In debug mode, we get a list of tuples, take the last processed image
                    # The last element should be the thinned image
                    for stage_name, stage_img in processed:
                        if stage_name == "thinned":
                            processed = stage_img
                            break
                    else:
                        # If thinned not found, use the last image
                        processed = processed[-1][1]

                # Save processed image
                output_path = os.path.join(output_rune_dir, image_file)
                cv2.imwrite(output_path, processed)

                processed_images += 1

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_images} out of {total_images} images")
    print(f"Processed images saved to: {processed_dir}")


if __name__ == "__main__":
    data_dir = f"{TRAINING_DATA_DIR}"

    process_all_images(data_dir)
