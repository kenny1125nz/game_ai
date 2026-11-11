import os
import shutil

import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from rune_config import config, FONT_PATH, SCREENSHOT_DIR, TRAINING_DATA_DIR

# from config import image_width,image_height

# --- CONFIGURATION ---
RUNES = [
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
    "Rune",
]


# Target Orange/Amber Color (with some jitter)
RUNE_COLOR = (0, 164, 255)


def generate_d2r_style_sample2(bg, rune_text=None):
    """
    Creates a D2R label with JITTER to prevent model overfitting.
    """
    image_width, image_height = config["image_size"]

    scale = random.uniform(0.8, 1.2)
    w = int(image_width * scale)
    h = int(image_height * scale * scale)

    h_bg, w_bg = bg.shape[:2]
    sx = random.randint(0, w_bg - w)
    sy = random.randint(0, h_bg - h)
    image = bg[sy : sy + h, sx : sx + w]

    f_min, fmax = config["font"]
    f_size = random.randint(f_min, fmax)
    font = ImageFont.truetype(FONT_PATH, int(f_size * scale))

    # 2. Add SPATIAL JITTER (Move the text position slightly)
    # Instead of perfectly centered, we shift it +/- 3 pixels
    # This prevents the NN from memorizing specific pixel coordinates.
    if rune_text is not None:
        left, top, right, bottom = font.getbbox(rune_text)
        tw, th = right - left, int((bottom - top) * 1.4)

        # Position the text randomly within the available "slack" space
        tx = (w - tw) // 2
        ty = (h - th) // 2

        # 4. Create Canvas and Box
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(image)
        canvas = canvas.convert("RGBA")

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        # Jitter box opacity and padding
        box_opacity = random.randint(100, 180)

        # Ensure box doesn't clip out of canvas either
        box_rect = [
            max(0, tx - 2),
            max(0, ty - 2),
            min(w, tx + tw + 2),
            min(h, ty + th + 2),
        ]

        draw_ov.rectangle(box_rect, fill=(0, 0, 0, box_opacity))
        canvas = Image.alpha_composite(canvas, overlay)
        # 5. Draw Text
        draw = ImageDraw.Draw(canvas)
        draw.text((tx, ty), rune_text, font=font, fill=(255, 164, 0, 255))  # RUNE_COLOR

        # 6. Final Resizing (Crucial for Resolution Invariance)
        # This brings the 'randomly scaled' render back to your fixed NN input size
        image = canvas.resize((image_width, image_height), resample=Image.LANCZOS)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def create_factory(target, count=20):
    if not os.path.exists(target):
        os.makedirs(target)

    for rune in RUNES:
        os.makedirs(os.path.join(target, rune), exist_ok=True)

    # Load all screenshots into memory
    bg_images = [
        cv2.imread(os.path.join(SCREENSHOT_DIR, f)) for f in os.listdir(SCREENSHOT_DIR)
    ]

    print(f"Generating {count * len(RUNES)} images...")

    for rune in RUNES:
        for i in range(count):
            # 1. Get a random background patch
            bg = random.choice(bg_images)
            final_img = generate_d2r_style_sample2(bg, rune)
            file_path = os.path.join(target, rune, f"{rune}_{i}.png")
            cv2.imwrite(file_path, final_img)

    print("Data Factory Complete!")


def generate_negatives(target, count=20):
    """
    Generates background patches and 'hard negatives'
    (things that might fool a color filter).
    """
    neg_dir = os.path.join(target, "Background")

    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)

    for rune in RUNES:
        os.makedirs(os.path.join(neg_dir, rune), exist_ok=True)

    # Load screenshots
    bg_images = [
        cv2.imread(os.path.join(SCREENSHOT_DIR, f)) for f in os.listdir(SCREENSHOT_DIR)
    ]

    print(f"Generating {count} negative samples...")

    for i in range(count):
        bg = random.choice(bg_images)
        if random.random() > 0.7:
            fake_text = random.choice(
                ["Gold", "Potion", "Key", "Super", "Healing", "134", "Stash"]
            )
            final_img = generate_d2r_style_sample2(bg, fake_text)
        else:
            final_img = generate_d2r_style_sample2(bg)

        # Save to 'Background' folder
        cv2.imwrite(os.path.join(neg_dir, f"neg_{i}.png"), final_img)

    print("Negative Data Generation Complete!")


if __name__ == "__main__":
    count = int(sys.argv[1])

    output = f"{TRAINING_DATA_DIR}/"

    if os.path.exists(output):
        shutil.rmtree(output)

    generate_negatives(output, count * 4)
    create_factory(output, count)
