
import time
import numpy as np
from typing import Any, Optional, Tuple
import cv2

from common.game import Game, GameInstruction
from common.easy_ocr import extract_region, extract_text
from common.input_controller import InputController
from d2r.state import D2RGameState, GameStage, Act, PlayerClass
from d2r.cnn_rune_detector import find_runes
import cv2
import os
import numpy as np

def find_image_patch(image_path:str, screen_data: Any, scale:tuple[float,float]) -> bool:
    # Construct full path to the template image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, image_path)
    
    # Load template image
    template = cv2.imread(full_image_path)
    if template is None:
        print(f"Warning: Could not load template image from {full_image_path}")
        return False
    
    # Scale template according to screen resolution
    # Template images are designed for 1920x1080 resolution
    # scale[0] = screen_width / 1920, scale[1] = screen_height / 1080
    if scale[0] != 1.0 or scale[1] != 1.0:
        new_width = int(template.shape[1] * scale[0])
        new_height = int(template.shape[0] * scale[1])
        # Ensure minimum dimensions
        if new_width > 0 and new_height > 0:
            template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            print(f"Warning: Template would be resized to invalid dimensions: {new_width}x{new_height}")
            return False
    
    # Check if template is larger than screen (shouldn't happen with valid scaling)
    if template.shape[0] > screen_data.shape[0] or template.shape[1] > screen_data.shape[1]:
        print(f"Warning: Template {template.shape} is larger than screen {screen_data.shape}")
        return False
    
    # cv2.imshow('searching',template)
    # cv2.waitKey(1)

    # Perform template matching
    result = cv2.matchTemplate(screen_data, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Use a threshold of 0.8 (same as ScreenReader.find_template)
    threshold = 0.8
    return max_val >= threshold


def detect_game_stage( screen_data: Any,scale:tuple[float,float]) -> GameStage:
    if find_image_patch('play.png',screen_data,scale) :
        return GameStage.PLAYERS
    if find_image_patch('loading.png',screen_data,scale) or   find_image_patch('creating.png',screen_data,scale) :
        return GameStage.CREATING
    if find_image_patch('health.png',screen_data,scale) :
        return GameStage.PLAYING
    return GameStage.NONE
