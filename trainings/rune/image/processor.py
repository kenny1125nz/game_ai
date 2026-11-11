import cv2
import numpy as np
from typing import Union, List, Tuple
from skimage.morphology import skeletonize

def _convert_to_binary( image: np.ndarray) -> np.ndarray:
    lower = np.array([0, 90, 144]) 
    upper = np.array([10, 210, 255])
    return cv2.inRange(image, lower, upper)

def process_image(image: np.ndarray, debug: bool = False) -> Union[np.ndarray, List[Tuple[str, np.ndarray]]]:
    stages = []
    # Stage 1: Convert to binary
    binary_image = _convert_to_binary(image)
    if debug:
        stages.append(("binary", binary_image))

    # Stage 2: Thin strokes
    thinned = skeletonize(binary_image.astype(bool)).astype(np.uint8) * 255
    if debug:            
        stages.append(("thinned", thinned))        
        return stages
    
    # cv2.imshow('processed',thinned)
    # cv2.waitKey(0)
    
    return thinned

