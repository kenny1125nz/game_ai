import cv2
import numpy as np
import sys
from ocr_config import CANVAS_W, CANVAS_H


class RoiDetector:

    def find_contours(self, hsv_image, lower, upper):
        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 2))
        connected = cv2.dilate(mask, kernel, iterations=2)

        # 4. Find the clusters
        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def find_all_text_contours(self, screenshot):
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        contours = []

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([179, 40, 255])

        # 2. Blue (Magic) - Specific Blue Hue
        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # 3. Yellow/Gold (Rare/Unique) - Yellow to Orange Hue
        lower_gold = np.array([15, 80, 100])
        upper_gold = np.array([35, 255, 255])

        # 4. Green (Set) - Green Hue
        lower_green = np.array([40, 80, 100])
        upper_green = np.array([80, 255, 255])

        # 5. Orange (Runes/Crafted) - Bright Orange
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([15, 255, 255])

        contours.extend(self.find_contours(hsv, lower_white, upper_white))
        contours.extend(self.find_contours(hsv, lower_blue, upper_blue))
        contours.extend(self.find_contours(hsv, lower_gold, upper_gold))
        contours.extend(self.find_contours(hsv, lower_green, upper_green))
        contours.extend(self.find_contours(hsv, lower_orange, upper_orange))
        return contours

    def find_all_text_contours2(self, screenshot):
        hsv_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # 1. White (Common/Superior) - Low Saturation, High Value
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([179, 40, 255])

        # 2. Blue (Magic) - Specific Blue Hue
        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # 3. Yellow/Gold (Rare/Unique) - Yellow to Orange Hue
        lower_gold = np.array([15, 80, 100])
        upper_gold = np.array([35, 255, 255])

        # 4. Green (Set) - Green Hue
        lower_green = np.array([40, 80, 100])
        upper_green = np.array([80, 255, 255])

        # 5. Orange (Runes/Crafted) - Bright Orange
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([15, 255, 255])

        # Create masks
        m1 = cv2.inRange(hsv_image, lower_white, upper_white)
        m2 = cv2.inRange(hsv_image, lower_blue, upper_blue)
        m3 = cv2.inRange(hsv_image, lower_gold, upper_gold)
        m4 = cv2.inRange(hsv_image, lower_green, upper_green)
        m5 = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Merge all into one "Interest Mask"
        final_mask = cv2.bitwise_or(
            m1, cv2.bitwise_or(m2, cv2.bitwise_or(m3, cv2.bitwise_or(m4, m5)))
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 2))
        connected = cv2.dilate(final_mask, kernel, iterations=2)

        # 4. Find the clusters
        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def preprocess_standardized(self, roi):
        # 2. Resize ROI once based on screen scale (keep aspect ratio)
        tw, th = CANVAS_W, CANVAS_H
        h, w = roi.shape[:2]

        # 1. Create the empty black canvas
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)

        # 2. Determine how much of the ROI actually fits
        # We clip the ROI to the canvas limits
        clip_h = min(h, th)
        clip_w = min(w, tw)

        # 3. Paste the clipped ROI
        # We center it vertically for better alignment with the model's 'sight'
        y_offset = (th - clip_h) // 2
        canvas[y_offset : y_offset + clip_h, :clip_w] = roi[:clip_h, :clip_w]

        return canvas

    def get_item_rois_v2(self, screenshot):

        crops = []
        coords = []

        h_screen, w_screen = screenshot.shape[:2]
        scale = (w_screen / 1920, h_screen / 1080)
        contours = self.find_all_text_contours(screenshot)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 5. Filter by "Label Shape"
            # Typical D2R labels are at least 30px wide and 10-40px tall
            if (
                30 * scale[0] < w < CANVAS_W * scale[0]
                and 12 * scale[1] < h < CANVAS_H * scale[1] * 0.65
            ):
                # 6. PADDING: The box is bigger than the text!
                # We expand the crop by ~5-8 pixels to ensure we get the full black box
                pad_h = 8
                pad_w = 10

                y1 = max(0, y - pad_h)
                y2 = min(h_screen, y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(w_screen, x + w + pad_w)

                roi = screenshot[y1:y2, x1:x2]
                roi = cv2.resize(
                    roi,
                    None,
                    fx=1 / scale[0],
                    fy=1 / scale[1],
                    interpolation=cv2.INTER_AREA,
                )

                roi = self.preprocess_standardized(roi)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                crops.append(roi)
                coords.append((x1, y1, x2 - x1, y2 - y1))
                # cv2.imwrite(f"{x1}_{y1}.png", roi)

        return crops, coords


if __name__ == "__main__":
    file = f"../data/screenshots/Diablo 2 Resurrected Screenshot 2026.02.27 - 01.03.55.36.png"
    screenshot = cv2.imread(file)
    detector = RoiDetector()
    rois = detector.get_item_rois_v2(screenshot)
