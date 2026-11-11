from pathlib import Path
import cv2
import easyocr
import shutil

# Initialize easyOCR reader
reader = easyocr.Reader(["en"])

# Target directory
dir_path = Path("./20260330_055130_452178/crops")

# Get all PNG files in the crops directory
png_files = list(dir_path.glob("*.png"))

for f in png_files:
    image_path = str(f)
    # Extract text using easyOCR
    try:
        # Get text results (detail=0 returns text only)
        result = reader.readtext(image_path, detail=0)
        extracted_text = " ".join(result).strip()
    except Exception as e:
        print(f"  OCR error: {e}")
        extracted_text = ""

    # Determine destination directory
    if extracted_text:
        # Clean text for directory name
        clean_text = "".join(
            c for c in extracted_text if c.isalnum() or c.isspace()
        ).strip()
        if not clean_text:
            clean_text = "unknown"

        # Use first word or first 20 chars as directory name
        dir_name = clean_text.split()[0] if clean_text.split() else clean_text
        dir_name = dir_name[:20]
    else:
        # Empty text - put in noise directory
        dir_name = "noise"

    # Create destination directory if it doesn't exist
    dest_dir = dir_path / dir_name
    dest_dir.mkdir(exist_ok=True)

    # Move the image to the destination directory
    dest_path = dest_dir / f.name
    shutil.move(str(f), str(dest_path))

    # print(f"  Moved to: {dir_name}/")
