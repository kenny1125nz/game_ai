"""Mouse and keyboard input simulation."""

import pyautogui


class InputController:
    """Sends simulated inputs to the game."""
    
    def __init__(self):
        pyautogui.FAILSAFE = True
    
    def click(self, x, y, button='left'):
        """Click at position."""
        pyautogui.click(x, y, button=button)
    
    def press_key(self, key):
        """Press a keyboard key."""
        pyautogui.press(key)
    
    def type_text(self, text):
        """Type text."""
        pyautogui.typewrite(text)
