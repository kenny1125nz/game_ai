import os

import random
import cv2

# from config import image_width,image_height


class D2RTraininngContext:
    def __init__(self, dir):
        print(f"loading background from:{dir}")

        screenshot_files = [
            f
            for f in os.listdir(dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]

        self.images = list(map(lambda f: cv2.imread(f"{dir}/{f}"), screenshot_files))

        if not self.images:
            raise FileNotFoundError(f"No screenshot files found in {dir}")
        print(f"loaded background :{len(self.images)}")

    def get_random_background(self, size):
        w, h = size
        bg = random.choice(self.images)

        if bg is not None:
            h_bg, w_bg = bg.shape[:2]
            scale = (w_bg / 1920, h_bg / 1080)
            w, h = int(size[0] * scale[0]), int(size[1] * scale[1])
            sx = random.randint(100, int(w_bg - w))
            sy = random.randint(100, int(h_bg - h) - 400)  # only use centre part
            image = bg[sy : sy + h, sx : sx + w]

            image = cv2.resize(image, size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image
