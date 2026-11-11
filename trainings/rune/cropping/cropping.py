import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import sys
from config import config

import matplotlib.pyplot as plt

# --- CONFIG (Must match your training script exactly) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES =['Amn', 'Background', 'Ber', 'Cham', 'Dol', 'El', 'Eld', 'Eth', 'Fal', 'Gul', 'Hel', 'Io', 'Ist', 'Ith', 'Jah', 'Ko', 'Lem', 'Lo', 'Lum', 'Mal', 'Nef', 'Ohm', 'Ort', 'Pul', 'Ral', 'Rune', 'Shael', 'Sol', 'Sur', 'Tal', 'Thul', 'Tir', 'Um', 'Vex', 'Zod']
RARE_ORDER=['El','Eld','Tir','Nef','Eth','Ith','Tal','Ral','Ort','Thul','Amn','Sol','Shael','Dol','Hel','Io','Lum','Ko','Fal','Lem','Pul','Um','Mal','Ist','Gul','Vex','Ohm','Lo','Sur','Ber','Jah','Cham','Zod']
SIZE='small'


def show_single_tensor(tensor):
    # 1. Move to CPU and remove the batch dimension if it exists
    img = tensor.cpu().detach()
    
    # 2. Undo the Normalize((0.5,), (0.5,)) to get back to [0, 1] range
    img = img * 0.5 + 0.5
    img = img.squeeze().numpy()    
    
    plt.figure(figsize=(4, 2))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def get_orange_blobs(screenshot_bgr):
    """Method 1: Quickly find all potential orange 'islands'."""
    lower = np.array([0, 140, 240]) 
    upper = np.array([10, 180, 255])
    mask = cv2.inRange(screenshot_bgr, lower, upper)
    
    # Simple morphology to connect any fragmented letters
    # kernel = np.ones((3,3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # 2. PROPOSAL (Now finds whole words, not just letters)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_orange_chips(screenshot_bgr,  image_width,image_height):
    """Method 1: Quickly find all potential orange 'islands'."""    
    # screenshot_bgr =cv2.resize(screenshot_bgr,(1920,1080))

    lower = np.array([0, 140, 240]) 
    upper = np.array([10, 180, 255])
    mask = cv2.inRange(screenshot_bgr, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i_w= int(image_width)
    i_h= int(image_height)
    #todo different screen size
    chips = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (30 < w < 200 and 13 < h < 30):
            continue
        
        w_offset = i_w//2
        h_offset=i_h//2
        if w<i_w :
            w_offset = w//2

        cx, cy = x + w // 2, y + h // 2
        y1, y2 = max(0, cy - h_offset), min(screenshot_bgr.shape[0], cy +h_offset) #fixed height
        x1, x2 = max(0, cx -  w_offset), min(screenshot_bgr.shape[1], cx +  w_offset) 
        chip = screenshot_bgr[y1:y2, x1:x2]

        if w<image_width:
            gap = (image_width-w)//2
            chip = cv2.copyMakeBorder(chip, 0, 0,gap,gap,   cv2.BORDER_CONSTANT, value=0)

        chip = cv2.resize(chip, (image_width,image_height))
        chips.append(chip)
    return chips

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    chips =  get_orange_chips(img,72,32)
    for i, chip in enumerate(chips):
        cv2.imwrite(f'./data/img{i}.png', chip)
