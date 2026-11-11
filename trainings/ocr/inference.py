import torch
import time
from PIL import Image
import torchvision.transforms as T
from crnn import CRNN
from ocr_config import MODEL_DIR, ALPHABET
import sys
from pathlib import Path
from roi_detector import RoiDetector
import cv2
import torch.nn.functional as F

char_to_int = {char: i + 1 for i, char in enumerate(ALPHABET)}
int_to_char = {i + 1: char for i, char in enumerate(ALPHABET)}
int_to_char[0] = ""  # CTC Blank


def greedy_decode_with_confidence(logits):
    # 1. Apply Softmax to turn raw scores into probabilities [63, NumClasses]
    probs = F.softmax(logits, dim=1)

    # 2. Get the highest probability and the index for each of the 63 time-steps
    confidences, indices = torch.max(probs, dim=1)

    # 3. Convert to CPU/Numpy for processing
    indices = indices.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()

    char_list = []
    conf_list = []

    last_char = None
    for i in range(len(indices)):
        idx = indices[i]
        conf = confidences[i]

        # CTC Logic: Ignore 'Blank' (usually index 0) and repeated characters
        if idx != 0 and idx != last_char:
            char_list.append(int_to_char[idx])
            conf_list.append(conf)

        last_char = idx

    # 4. Calculate Final Word Confidence
    # We take the average probability of all detected characters
    word_text = "".join(char_list)
    word_confidence = sum(conf_list) / len(conf_list) if conf_list else 0.0

    return word_text, word_confidence


def greedy_decode(logits):
    # logits shape: [TimeSteps, Classes]
    indices = torch.argmax(logits, dim=1).tolist()
    tokens = []
    for i in range(len(indices)):
        if indices[i] != 0 and (i == 0 or indices[i] != indices[i - 1]):
            tokens.append(int_to_char[indices[i]])
    return "".join(tokens)


def read_screen_text(model, transform, image_path):
    # Load image (or grab from your OpenCV screen capture)
    img = Image.open(image_path).convert("RGB")

    # If it's a full screenshot, you'd crop the item label here first
    # For now, let's assume 'img' is a cropped text strip
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        logits = model(img_tensor)  # [63, 1, 72]
        logits = logits.squeeze(1)  # [63, 72]

    prediction = greedy_decode(logits)
    return prediction


def batch_inference(model, transform, images):
    if not images:
        return []

    # 1. Preprocess all at once
    tensors = [transform(img) for img in images]
    input_batch = torch.stack(tensors).to(DEVICE)  # [N, 1, 32, 256]

    # 2. Single GPU Pass
    with torch.no_grad():
        logits = model(input_batch)  # [63, N, Classes]
        logits = logits.permute(1, 0, 2)  # [N, 63, Classes]

    # 3. Decode results
    results = []
    for i in range(logits.size(0)):
        results.append(greedy_decode_with_confidence(logits[i]))
    return results


def visualize_detections(image, metadata):
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


# --- TEST IT ---
if __name__ == "__main__":
    # 2. Load the "Brain"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_chars=len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}d2r_ocr_model.pth"))
    model.eval()
    print("model laoded")

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    # path = sys.argv[1]

    # for item in Path(path).iterdir():
    #     if item.is_file():
    #         result = read_screen_text(model, transform, f"{path}/{item.name}")
    #         print(f"{item.name}:{result}")

    # file = f"../data/screenshots/Diablo 2 Resurrected Screenshot 2026.02.27 - 01.03.55.36.png"
    file = "C:\\apps\\projects\\game_ai\\game_bot\\data\\D2R\\20260328_191101_535724\\20260328_191809_828937.png"
    screenshot = cv2.imread(file)
    detector = RoiDetector()

    start = time.perf_counter()
    crops, coords = detector.get_item_rois_v2(screenshot)
    end = time.perf_counter()
    print(f"detecting time:{(end-start)*1000} ms")
    start = end

    result = batch_inference(model, transform, crops)
    end = time.perf_counter()
    print(f"inference time:{(end-start)*1000} ms")

    detections = list(
        map(
            lambda item: {
                "roi": item[1],
                "model_class": result[item[0]][0],
                "model_conf": result[item[0]][1],
            },
            enumerate(coords),
        )
    )

    img = visualize_detections(screenshot, {"detections": detections})
    cv2.imshow("detected", img)
    cv2.waitKey(200000)
