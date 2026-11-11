#!/usr/bin/env python3
"""
Generic ONNX Classifier with temperature scaling calibration support.
"""
import cv2
import numpy as np
import onnxruntime as ort
import onnx
import os
import time
import statistics
from pathlib import Path
import random
import sys


class ONNXClassifier:
    """Generic ONNX classifier with temperature scaling calibration support."""

    def __init__(self, model_path, preprocess_fn=None, classes=None):
        """
        Initialize ONNX classifier.

        Args:
            model_path: Path to ONNX model file
            preprocess_fn: Function to preprocess input data
            classes: List of class names
        """
        self.model_path = model_path
        self.preprocess_fn = preprocess_fn
        self.classes = classes

        # Performance tracking
        self._predict_count = 0
        self._predict_times = []
        self._last_print_count = 0

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Setup execution providers (use GPU if available)
        self.providers = self._get_providers()
        self.session = ort.InferenceSession(model_path, providers=self.providers)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _read_temperature_from_metadata(self, user_temperature):
        """Read temperature from model metadata if available."""
        # If user provided temperature, use it
        if user_temperature is not None:
            return user_temperature

        # Try to read from ONNX model metadata
        try:
            onnx_model = onnx.load(self.model_path)
            if onnx_model.metadata_props:
                for prop in onnx_model.metadata_props:
                    if prop.key == "temperature":
                        return float(prop.value)
        except Exception:
            pass

        return None

    def _get_providers(self):
        available = ort.get_available_providers()
        # if "CUDAExecutionProvider" in available:
        #     return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _print_performance_stats(self):
        """Print performance statistics for the last 50 predictions."""
        if len(self._predict_times) == 0:
            return

        # Get the last 50 times (or all if less than 50)
        recent_times = self._predict_times[-50:]

        avg_time = statistics.mean(recent_times)
        min_time = min(recent_times)
        max_time = max(recent_times)
        median_time = statistics.median(recent_times)

        print(f"\n{'='*60}")
        print(f"Performance Statistics (last {len(recent_times)} predictions):")
        print(f"  Total calls: {self._predict_count}")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Minimum time: {min_time:.2f} ms")
        print(f"  Maximum time: {max_time:.2f} ms")
        print(f"  Median time:  {median_time:.2f} ms")
        print(f"  FPS (approx): {1000/avg_time:.1f}")
        print(f"{'='*60}\n")

    def _preprocess(self, input_data):
        if self.preprocess_fn:
            return self.preprocess_fn(input_data)
        # Default: assume input is already preprocessed
        return np.array(input_data, dtype=np.float32)

    def predict(self, input_data, top_k=5):
        # Start timing
        start_time = time.perf_counter()

        # Preprocess
        processed_data = self._preprocess(input_data)

        # Ensure correct shape
        if len(processed_data.shape) == len(self.input_shape) - 1:
            processed_data = np.expand_dims(processed_data, axis=0)

        # Run inference
        output = self.session.run(
            [self.output_name], {self.input_name: processed_data}
        )[0]

        # Get predictions
        predictions = self._process_output(output, top_k)

        # End timing
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Update performance statistics
        self._predict_count += 1
        self._predict_times.append(inference_time)

        # Print statistics every 50 calls
        # if self._predict_count % 50 == 0:
        #     self._print_performance_stats()

        return predictions, output

    def _process_output(self, output, top_k=5):
        # Apply softmax
        exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
        probs = exp_output / np.sum(exp_output, axis=-1, keepdims=True)

        # Get top predictions
        batch_probs = probs[0] if len(probs.shape) > 1 else probs
        top_indices = np.argsort(batch_probs)[-top_k:][::-1]

        if self.classes:
            return [
                (self.classes[idx], float(batch_probs[idx]), idx) for idx in top_indices
            ]
        else:
            return [
                (f"Class_{idx}", float(batch_probs[idx]), idx) for idx in top_indices
            ]

    def get_model_info(self):
        return self.model_path


def rune_preprocess(img):
    """Preprocess function for rune images."""
    # 1. Load image (OpenCV loads in BGR)
    # 2. BGR -> RGB (Crucial: transforms.ToTensor() expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. transforms.ToTensor() logic:
    # Scale to [0, 1] and convert to float32
    img = img.astype(np.float32) / 255.0

    # 5. transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Math: (pixel - mean) / std
    img = (img - 0.5) / 0.5

    # 6. HWC -> CHW (Height, Width, Channel -> Channel, Height, Width)
    img = np.transpose(img, (2, 0, 1))

    # 7. Add Batch Dimension (N, C, H, W)
    img = np.expand_dims(img, axis=0)

    return img


class RealData:
    def __init__(self, root_dir):
        self.dataset = self.__load__(root_dir)

    def __load__(self, root_dir):
        print("Loading captured real data")
        data = {}
        directory_path = Path(root_dir)
        for image_path in directory_path.rglob("*.png"):
            dir_name = image_path.parent.name
            if dir_name in data:
                samples = data[dir_name]
                samples.append(image_path)
            else:
                samples = [image_path]
                data[dir_name] = samples

        label = random.choice(list(data.keys()))
        samples = data[label]
        return data

    def get_sample(self, clz, idx):
        rune_samples = self.dataset[clz]
        if rune_samples is not None:
            cnt = len(rune_samples)
            return rune_samples[(idx + cnt) % cnt]
        random_label = random.choice(list(self.dataset.keys()))
        return random.choice(self.dataset[random_label])

    def get_samples(self, clz):
        return self.dataset[clz]


CLASSES = [
    "Amn",
    "Background",
    "Ber",
    "Cham",
    "Dol",
    "El",
    "Eld",
    "Eth",
    "Fal",
    "Gul",
    "Hel",
    "Io",
    "Ist",
    "Ith",
    "Jah",
    "Ko",
    "Lem",
    "Lo",
    "Lum",
    "Mal",
    "Nef",
    "Ohm",
    "Ort",
    "Pul",
    "Ral",
    "Rune",
    "Shael",
    "Sol",
    "Sur",
    "Tal",
    "Thul",
    "Tir",
    "Um",
    "Vex",
    "Zod",
]


if __name__ == "__main__":
    rune = sys.argv[1]
    # idx = int(sys.argv[2])

    data = RealData("./data/")
    classifier = ONNXClassifier(
        model_path="tiny_hybrid_rune_classifier.onnx",
        preprocess_fn=rune_preprocess,
        classes=CLASSES,
    )
    samples = data.get_samples(rune)
    for sample_path in samples:
        img = cv2.imread(sample_path)
        result = classifier.predict(img, top_k=1)[0][0]

    classifier._print_performance_stats()
