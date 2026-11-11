from typing import Tuple, Any
import json
import os
from datetime import datetime
import cv2


class DataCapture:
    def __init__(self, game_id: str, target_location: str = "./data/"):
        self.target_location = target_location
        self.game_id = game_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def save(self, image, meata: dict, crops=[]) -> Tuple[str, str] | None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = os.path.join(self.target_location, self.game_id, self.session_id)
            os.makedirs(save_dir, exist_ok=True)

            image_filename = f"{timestamp}.png"
            image_path = os.path.join(save_dir, image_filename)

            if not isinstance(image, bytes):
                success = cv2.imwrite(image_path, image)
                if not success:
                    raise ValueError(f"Failed to save image to {image_path}")
            else:
                with open(image_path, "wb") as f:
                    f.write(image)

            json_filename = f"{timestamp}.json"
            json_path = os.path.join(save_dir, json_filename)

            with open(json_path, "w") as f:
                json.dump(meata, f, indent=2)

            if len(crops) > 0 and "detections" in meata:
                self.save_crops(crops, save_dir, timestamp, meata["detections"])

            return image_path, json_path

        except Exception as e:
            return None

    def save_crops(self, crops, dir, timestamp, detections: list):
        save_dir = f"{dir}/crops/"
        os.makedirs(save_dir, exist_ok=True)

        for detected in detections:
            idx = detected["index"]
            label = detected["model_class"]
            initial = label[0]
            crop = crops[idx]

            target_dir = f"{save_dir}/{initial}/{label}"
            os.makedirs(target_dir, exist_ok=True)
            file_name = f"{target_dir}/crop_{timestamp}_{idx}.png"
            cv2.imwrite(file_name, crop)
