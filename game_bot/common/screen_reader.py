"""Screen capture and image processing using OpenCV with Windows API window detection."""
import cv2
import numpy as np
from typing import Tuple, Optional
import win32gui
import mss

class ScreenReader:
    """
    Captures and processes game screen.
    
    Supports capturing a specific region of the screen by finding
    a window by its title using Windows API (win32gui).
    Window dimensions are automatically detected from the window.
    """
    
    def __init__(self, window_title: str):
        """
        Initialize the screen reader with a window title.
        
        Args:
            window_title: The title of the window to capture (partial match supported).
        
        Raises:
            RuntimeError: If required libraries are not available or window not found.
        """

        
        self._window_title = window_title.lower()
        self._width = 0
        self._height = 0
        self._x = 0
        self._y = 0
        self._hwnd = None
        self._mss = mss.mss()
        
        # Find the window on initialization
        self._find_window()
    
    @property
    def window_title(self) -> str:
        """Get the target window title."""
        return self._window_title
    
    @property
    def position(self) -> Tuple[int, int]:
        """Get the current capture position (x, y)."""
        return (self._x, self._y)
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the current capture resolution (width, height)."""
        return (self._width, self._height)
    
    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Get the full capture region (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)
    
    @property
    def is_window_found(self) -> bool:
        """Check if the target window has been found."""
        return self._hwnd is not None
    
    def _find_window(self) -> None:
        """
        Find the window by title using Windows API.
        
        Uses win32gui to enumerate windows and find one matching the title.
        
        Raises:
            RuntimeError: If window is not found.
        """
        found_hwnd = None
        
        def enum_callback(hwnd, _):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self._window_title in title.lower():
                    found_hwnd = hwnd
                    return False  # Stop enumeration
            return True  # Continue enumeration
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception:
            pass  # EnumWindows raises when callback returns False
        
        if not found_hwnd:
            raise RuntimeError(f"Window '{self._window_title}' not found")
        
        self._hwnd = found_hwnd
        self._update_window_rect()
        title = win32gui.GetWindowText(found_hwnd)
        print(f"Window '{title}' found at ({self._x}, {self._y}) size ({self._width}x{self._height})")
    
    def _update_window_rect(self) -> None:
        """
        Update the window position and size from the current hwnd.
        
        Raises:
            RuntimeError: If window handle is invalid.
        """
        if self._hwnd is None:
            raise RuntimeError("Window handle is invalid")
        
        try:
            rect = win32gui.GetWindowRect(self._hwnd)
            
            title_bar = 60
            self._width = rect[2] - rect[0]
            if self._width>3800:
                self._height = rect[3] - rect[1]
            else :
                self._height = rect[3] - rect[1]-title_bar
                self._width =  self._width- 26

            self._x = rect[0]            
            if(self._height>=2100):
                self._y = rect[1]
            else :
                self._y = rect[1]+title_bar 
        except Exception:
            raise RuntimeError("Failed to get window rectangle")
    
    def capture(self) -> np.ndarray:
        """
        Capture the screen region and return it as an image.
        
        Returns:
            np.ndarray: The captured screen image as a BGR numpy array
                        (OpenCV format).
        
        Raises:
            RuntimeError: If window not found or capture fails.
        """
        if not self.is_window_found:
            raise RuntimeError("Window not found")
        
        # Check if we have valid dimensions
        if self._width <= 0 or self._height <= 0:
            raise RuntimeError("Invalid window dimensions")
        
        monitor = {
            "left": self._x,
            "top": self._y,
            "width": self._width,
            "height": self._height
        }
        
        try:
            screenshot = self._mss.grab(monitor)
            # Convert to numpy array and BGR format for OpenCV
            img = np.array(screenshot)
            # mss returns BGRA, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            raise RuntimeError(f"Screen capture failed: {e}")
    
    def find_template(
        self,
        template_path: str,
        threshold: float = 0.8
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a template image on screen.
        
        Args:
            template_path: Path to the template image file.
            threshold: Matching threshold (0.0 to 1.0).
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box (x, y, width, height)
                of the match if found, None otherwise.
        """
        screen = self.capture()
        template = cv2.imread(template_path)
        
        if template is None:
            print(f"Warning: Could not load template image from {template_path}")
            return None
        
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            h, w = template.shape[:2]
            return (max_loc[0], max_loc[1], w, h)
        
        return None
