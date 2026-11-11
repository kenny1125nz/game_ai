import json
import os
import sys
import cv2
from pathlib import Path


class DataViewer:
    def __init__(self):
        self.window_name = "Data Viewer"
        self.window_x = 100
        self.window_y = 100

    def load_data(self, image_path: str, json_path: str | None = None):
        if json_path is None:
            json_path = image_path.replace(".png", ".json").replace(".jpg", ".json")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        return image, metadata

    def visualize_detections(self, image, metadata):
        image_with_boxes = image.copy()

        if "detections" not in metadata:
            return image_with_boxes

        detections = metadata["detections"]

        for detection in detections:
            if "roi" not in detection:
                continue

            roi = detection["roi"]
            if len(roi) != 4:
                continue

            x, y, w, h = roi

            color = (0, 255, 0)
            thickness = 1

            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, thickness)

            model_class = detection.get("model_class", "Unknown")
            model_conf = detection.get("model_conf", 0.0)

            label = f"{model_class}: {model_conf:.3f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            text_x = x
            text_y = y - 5 if y - 5 > text_height else y + text_height + 5

            cv2.putText(
                image_with_boxes,
                label,
                (text_x + 2, text_y - 2),
                font,
                font_scale,
                color,
                thickness,
            )

        image_with_boxes = cv2.resize(image_with_boxes, (1920, 1080))
        return image_with_boxes

    def display_navigation(self, image_files, json_files, start_index=0):
        current_index = start_index
        total_files = len(image_files)

        cv2.namedWindow(self.window_name)

        while 0 <= current_index < total_files:
            image_path = str(image_files[current_index])
            json_path = str(json_files[current_index])

            try:
                image, metadata = self.load_data(image_path, json_path)
                image_with_boxes = self.visualize_detections(image, metadata)

                cv2.imshow(self.window_name, image_with_boxes)
                cv2.moveWindow(self.window_name, self.window_x, self.window_y)

                key = cv2.waitKey(0)

                if key == ord("q") or key == 27:  # 'q' or ESC
                    current_index = total_files
                    print("stopped")
                    break
                elif key == ord("s"):
                    # save_path = image_path.replace('.png', '_annotated.png').replace('.jpg', '_annotated.jpg')
                    # cv2.imwrite(save_path, image_with_boxes)
                    print(f"Saved: {image_path}")
                elif key == 2424832 or key == ord("a"):
                    current_index = ((current_index - 1) + total_files) % total_files
                elif key == 2555904 or key == ord("d"):
                    current_index = ((current_index + 1) + total_files) % total_files
                else:
                    print(f"Key pressed: {key} (No action assigned)")

            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                current_index = min(total_files - 1, current_index + 1)

        cv2.destroyAllWindows()

    def batch_process(self, directory: str):
        dir_path = Path(directory)

        png_files = list(dir_path.glob("*.png"))
        json_files = []

        valid_png_files = []

        for png_file in sorted(png_files):
            json_file = png_file.with_suffix(".json")
            if json_file.exists():
                valid_png_files.append(png_file)
                json_files.append(json_file)
            else:
                print(f"No JSON for: {png_file.name}")

        if not valid_png_files:
            print(f"No valid PNG/JSON pairs found in {directory}")
            return

        print(f"Found {len(valid_png_files)} image/JSON pairs")
        print("Starting navigation viewer...")

        self.display_navigation(valid_png_files, json_files)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = f"./data/{sys.argv[1]}"
    else:
        directory = "./data/D2R/20260328_164053_346253"

    viewer = DataViewer()
    viewer.batch_process(directory)
