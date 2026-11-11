import cv2
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from classifier.onnx_classifier import ONNXClassifier
from display.overlay import Overlay
from typing import Optional

# --- CONFIG (Must match your training script exactly) ---
CLASSES =['Amn', 'Background', 'Ber', 'Cham', 'Dol', 'El', 'Eld', 'Eth', 'Fal', 'Gul', 'Hel', 'Io', 'Ist', 'Ith', 'Jah', 'Ko', 'Lem', 'Lo', 'Lum', 'Mal', 'Nef', 'Ohm', 'Ort', 'Pul', 'Ral', 'Rune', 'Shael', 'Sol', 'Sur', 'Tal', 'Thul', 'Tir', 'Um', 'Vex', 'Zod']
RARE_ORDER=['El','Eld','Tir','Nef','Eth','Ith','Tal','Ral','Ort','Thul','Amn','Sol','Shael','Dol','Hel','Io','Lum','Ko','Fal','Lem','Pul','Um','Mal','Ist','Gul','Vex','Ohm','Lo','Sur','Ber','Jah','Cham','Zod']


CROPS_DIR="./data/crops/"


class CNNRuneDetector:
    """CNN-based rune detector that loads and reuses classifier instance."""
    
    def __init__(self, model_size='small', threshold=0.65, calibrated=False , data_capture=None,overlay:Optional[Overlay]=None):
        """
        Initialize the rune detector.
        
        Args:
            model_size: Model size ('small', 'medium', 'large')
            threshold: Confidence threshold for detection
        """
        self.model_size = model_size
        self.threshold = threshold
        self.image_width, self.image_height = (72, 32)
        self.data_capture = data_capture
        self.last_findings=[]
        self.overlay=overlay
        
        model_file =f"./models/{model_size}_hybrid_rune_classifier.onnx"
        if calibrated:
            model_file =f"./models/{model_size}_hybrid_rune_classifier_calibrated.onnx"

        
        # Initialize ONNX classifier
        self.classifier = ONNXClassifier(
            model_path=model_file,
            preprocess_fn=self.rune_preprocess,
            classes=CLASSES
        )
        
        print(f"Initialized CNNRuneDetector with {model_size} model, threshold={threshold}")
    
    @staticmethod
    def get_orange_blobs(screenshot_bgr):
        """Method 1: Quickly find all potential orange 'islands'."""
        lower = np.array([0, 140, 240]) 
        upper = np.array([10, 180, 255])
        mask = cv2.inRange(screenshot_bgr, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 2. PROPOSAL (Now finds whole words, not just letters)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def get_orange_chips(screenshot_bgr, image_width, image_height, scale:tuple[float,float]):
        """Method 1: Quickly find all potential orange 'islands'."""    
        # screenshot_bgr =cv2.resize(screenshot_bgr,(1920,1080))
        sh, sw, sc = screenshot_bgr.shape

        lower = np.array([0, 140, 240]) 
        upper = np.array([10, 180, 255])
        mask = cv2.inRange(screenshot_bgr, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #todo different screen size
        chips = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if  (30*scale[0] < w < image_width*1.2*scale[0] and 13*scale[1] < h <  image_height*scale[1]):
                # print(f"found roi-> {w}:{h}")
                cx, cy = x + w // 2, y + h // 2
                y1, y2 = y,y+h
                x1, x2 = x,x+w

                if w<image_width*scale[0]:
                    gap = (image_width*scale[0]-w)//2
                    x1=x1-gap
                    x2=x2+gap

                if h<image_height*scale[1]:
                    gap = (image_height*scale[1]-h)//2
                    y1=y1-gap
                    y2=y2+gap

                def norm (n,mx):
                    return min(max(0,int(n)),mx)

                y1,y2,x1,x2 = norm(y1,sh),norm(y2,sh),norm(x1,sw),norm(x2,sw),     
                chip = screenshot_bgr[y1:y2,x1:x2]
                chip = cv2.resize(chip, (image_width,image_height))
                chips.append({
                    "chip":chip,
                    "center":(cx,cy),
                    "roi":(x,y,w,h)
                })
        return chips
    
    @staticmethod
    def save_rune_chip(rune_name, chip):
        timestamp_str =  datetime.now().strftime("%Y%m%d%H%M%S")
        file_dir=f"{CROPS_DIR}{rune_name}"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path  =f"{file_dir}/{rune_name}_{timestamp_str}.png"
        # print(f"saving {file_dir}")
        cv2.imwrite(file_path, chip)
    
    @staticmethod
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
    
    def display_detections(self, detections):        
        if self.overlay is not None :
            display_items = list(map(lambda d:{
                "text": f"{d["model_class"]} / {d["model_conf"]:.3f}",
                "roi": d['roi'],
                "active": True
            },detections))
            self.overlay.update(display_items)


    def find_runes(self, screenshot_bgr, scale:tuple[float,float], highrune_min_idx=20) -> list[tuple]:
        """
        Find runes in a screenshot.
        
        Args:
            screenshot_bgr: Screenshot image in BGR format
            scale: Scaling factor tuple (x_scale, y_scale)
            highrune_min_idx: Minimum index in RARE_ORDER to consider as high rune
            
        Returns:
            List of tuples (rune_name, x, y, confidence)
        """
        found_runes = []

        chips = self.get_orange_chips(screenshot_bgr, self.image_width, self.image_height, scale)
        
        detections =[]
        for chip in chips:
            chip_img = chip['chip']
            cx, cy= chip['center']
            roi = chip['roi']
            predictions, _ = self.classifier.predict(chip_img, top_k=1)
            
            if not predictions:
                continue
                
            rune_name, conf, class_idx = predictions[0]

            # if conf < self.threshold and not rune_name == 'Rune' and not rune_name == "Background":
            #     print(f"low confidence: {rune_name}/{conf:.4f} at {cx},{cy}")
            #     self.save_rune_chip(rune_name, chip)
            #     continue

            # 5. FILTERING (Rarity & Background Check)
            if rune_name == "Background" or rune_name == 'Rune':
                continue

            detections.append({
                "roi":roi,
                "model_class":rune_name,
                "model_conf":conf
            })

            self.save_rune_chip(rune_name, chip_img)

            if conf > self.threshold :
                print(f"detected {rune_name}/{conf:.4f} at {cx},{cy}")

            try:
                rarity_idx = RARE_ORDER.index(rune_name, 0)                                
                if rarity_idx >= highrune_min_idx:
                    found_runes.append((rune_name, cx, cy, conf))
            except ValueError:
                # Rune not in RARE_ORDER list, skip it
                continue


        detected_runes = list(map(lambda detected:f"{detected['model_class']}/{detected['roi']}",detections))
        if not self.last_findings == detected_runes:
            self.display_detections(detections)

            if self.data_capture is not None and len(detections) >0 :
                self.data_capture(screenshot_bgr,{
                    "model": self.classifier.get_model_info(),
                    "detections":detections
                })

        self.last_findings = detected_runes
        return found_runes

