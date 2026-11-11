import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNES = ["El", "Eld", "Tir", "Nef", "Eth", "Ith", "Tal", "Ral", "Ort", "Thul", "Amn",
         "Sol", "Shael", "Dol", "Hel", "Io", "Lum", "Ko", "Fal", "Lem", "Pul", "Um",
         "Mal", "Ist", "Gul", "Vex", "Ohm", "Lo", "Sur", "Ber", "Jah", "Cham", "Zod"]

FONT_DIR="./src//d2r/screen/fonts/"
FONT_FILE="exocetblizzardot-medium.otf"

def create_rune_templates(font_path):
    # D2 text is usually small, around 14-20pt
    font = ImageFont.truetype(font_path, 25)
    
    for rune in RUNES:
        # Draw white text on black background
        img = Image.new('L', (200, 50), color=0)
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"{rune} Rune", font=font, fill=255, spacing=10)
        
        # Crop to the exact bounds of the white pixels
        numpy_img = np.array(img)
        coords = cv2.findNonZero(numpy_img)
        x, y, w, h = cv2.boundingRect(coords)
        binary_template = numpy_img[y:y+h, x:x+w]
        
        # Crop to content and save
        cv2.imwrite(f"{FONT_DIR}/../runes/templates/{rune.lower()}.png",binary_template)

create_rune_templates(FONT_DIR+FONT_FILE)