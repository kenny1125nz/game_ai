import torch
from ocr.roi_detector import RoiDetector, CANVAS_H, CANVAS_W
import onnxruntime as ort
import numpy as np
import cv2
from datetime import datetime


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-%() '"

char_to_int = {char: i + 1 for i, char in enumerate(ALPHABET)}
int_to_char = {i + 1: char for i, char in enumerate(ALPHABET)}
int_to_char[0] = ""  # CTC Blank
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_np(x, axis=-1):
    """
    Computes softmax along a specific axis.
    Subtracting the max (np.max) prevents overflow.
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class D2rOCR:
    def __init__(self, model_path, capture):
        self.roi_detector = RoiDetector()
        self.providers = self._get_providers()
        self.session = ort.InferenceSession(model_path, providers=self.providers)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.capture = capture

    def _get_providers(self):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _greedy_decode_onnx(self, item_logits):
        """
        item_logits: NumPy array [63, NumClasses]
        """
        # 1. Get Probabilities
        probs = softmax_np(item_logits, axis=1)

        # 2. Get the index and confidence for each of the 63 steps
        indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        char_list = []
        conf_list = []

        last_char = None
        for i in range(len(indices)):
            idx = indices[i]
            conf = confidences[i]

            # CTC Logic: Ignore Blank (0) and repeats
            if idx != 0 and idx != last_char:
                char_list.append(int_to_char[idx])
                conf_list.append(conf)

            last_char = idx

        word_text = "".join(char_list)
        word_confidence = np.mean(conf_list) if conf_list else 0.0

        return word_text, word_confidence

    def _preprocess(self, bgr_images):
        batch_data = []
        for idx, roi in enumerate(bgr_images):
            img_rgb = roi[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            img_normalized = (img_rgb - 0.5) / 0.5
            batch_data.append(img_normalized)
        return batch_data

    def _batch_inference(self, bgr_images):
        if not bgr_images:
            return []

        # Preprocess
        batch_data = self._preprocess(bgr_images)

        # Create the Batch [N, 3, 32, 256]
        input_tensor = np.stack(batch_data).astype(np.float32)

        # 2. Run Inference
        # input_names and output_names must match your torch.onnx.export call
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        logits = self.session.run(None, ort_inputs)[0]

        # ONNX outputs are typically [SeqLen, Batch, Classes] or [Batch, SeqLen, Classes]
        # If you didn't permute inside the ONNX export, it's likely [SeqLen, Batch, Classes]
        # Let's assume standard CRNN output: [63, N, NumClasses]

        results = []
        num_items = logits.shape[1]  # N

        for i in range(num_items):
            # Extract the sequence for item 'i'
            item_logits = logits[:, i, :]  # [63, NumClasses]

            # Now pass to your greedy decode
            text, conf = self._greedy_decode_onnx(item_logits)
            results.append((text, conf))

        return results

    def read_screen(self, screenshot):
        bgr_crops, coords = self.roi_detector.get_item_rois_v2(screenshot)
        result = self._batch_inference(bgr_crops)
        words = list(map(lambda i: i[0], result))
        confs = list(map(lambda i: float(i[1]), result))

        detected = []
        detections = []
        for idx, word in enumerate(words):
            if not word == "":
                detected.append((word, confs[idx], coords[idx]))
            detections.append(
                {
                    "index": idx,
                    "roi": coords[idx],
                    "model_class": word,
                    "model_conf": confs[idx],
                }
            )
        if len(detections) > 0:
            self.capture.save(
                screenshot, {"model": "OCR", "detections": detections}, bgr_crops
            )

        return detected
