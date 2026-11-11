"""D2R-specific instruction execution controller."""

import pyautogui
from common.game import GameInstruction
import time




def Execute(instruction: GameInstruction, offset_x: int = 0, offset_y: int = 0):
    """
    Execute a D2R game instruction.
    
    Args:
        instruction: The game instruction to execute.
        offset_x: X offset to apply to all coordinates.
        offset_y: Y offset to apply to all coordinates.
        
    Returns:
        bool: True if the instruction was executed successfully, False otherwise.
    """
    try:
        action_type = instruction.action_type
        params = instruction.parameters
        
        if action_type == "teleport":
            position = params["position"]
            x = position[0]+offset_x
            y= position[1]+offset_y
            
            # Press F1 key (teleport hotkey)
            pyautogui.moveTo(x,y)
            pyautogui.press("f1")
            
            return True
            
        elif action_type == "click":
            position = params["position"]
            x = position[0]+offset_x
            y= position[1]+offset_y
            button = params.get("button", "left")
            
            # Move mouse to target position and click
            pyautogui.moveTo(x,y)
            pyautogui.click(x,y, button=button)
            
            return True
        elif action_type == "wait":
            seconds = int(params["value"])
            time.sleep(seconds)
            
            return True            
        elif action_type == "key":
            # Key press action
            if "key" not in params:
                print(f"Key instruction missing key parameter: {params}")
                return False
            
            key = params["key"]
            pyautogui.press(key)
            
            print(f"Pressed key: {key}")
            return True
            
        elif action_type == "type":
            # Type text action
            if "text" not in params:
                print(f"Type instruction missing text parameter: {params}")
                return False
            
            text = params["text"]
            pyautogui.typewrite(text)
            
            print(f"Typed text: {text}")
            return True
            
        else:
            print(f"Unknown action type: {action_type}")
            return False
            
    except Exception as e:
        print(f"Error executing instruction {instruction}: {e}")
        return False
    
def execute_multi(instructions: list[GameInstruction], offset_x: int = 0, offset_y: int = 0):
    for command in instructions:
        Execute (command,offset_x,offset_y)    