"""Diablo II: Resurrected Game implementation."""

import time
import numpy as np
from typing import Any, Optional, Tuple
import cv2

from common.game import Game, GameInstruction
from common.input_controller import InputController
from d2r.state import D2RGameState, GameStage, Act, PlayerClass
from d2r.d2r_config import buttons
from d2r.data.data_capture import DataCapture

# from screen.stage.game_stage import detect_game_stage

class D2RGame(Game):
    """
    Diablo II: Resurrected game implementation.
    
    Implements the Game interface for D2R, providing screen parsing,
    decision making, and instruction execution specific to the game.
    """
    
    # Health and mana region definitions for 1920x1080 resolution
    HEALTH_REGION = (443, 855, 150, 25)   # (x, y, width, height)
    MANA_REGION = (1443, 855, 150, 25)    # (x, y, width, height)
    
    def __init__(self,rune_detector):
        """
        Initialize D2R game instance.
        
        Args:
            use_paddle_ocr: Whether to use PaddleOCR (faster) instead of Tesseract.
        """
        self._state: D2RGameState = D2RGameState()
        self._input_controller = InputController()
        self._rule_detecor = rune_detector
        self._data_capture = DataCapture("Diablo2R","./data/")
        
    @property
    def name(self) -> str:
        """Get the game identifier."""
        return "d2r"
    
    @property
    def title(self) -> str:
        """Get the display title."""
        return "Diablo II: Resurrected"
    
    @property
    def current_state(self) -> D2RGameState:
        """Get the current game state."""
        return self._state
    
    def parse_screen(self, screen_data: Any, scale:tuple[float,float]) -> D2RGameState:
        """
        Parse screen data to extract D2R game state.
        
        Args:
            screen_data: Raw screen capture data (numpy array).
            scale: Scaling factor applied to the image (e.g., 0.5 for half-size).
            
        Returns:
            D2RGameState: Parsed game state from the screen.
        """
        state = D2RGameState(
            raw_data=screen_data,
            timestamp=time.time()
        )
        
        if not state.game_stage==GameStage.PLAYING:
            # Parse game stage
            state.game_stage = self._detect_game_stage(screen_data,scale)
        
        if state.game_stage == GameStage.PLAYING:
            # Parse in-game state
            state.act = self._detect_act(screen_data)
            state.scene = self._detect_scene(screen_data)
            state.player_class = self._detect_player_class(screen_data)
            state.items = self._detect_items(screen_data,scale)
            health = self._detect_health(screen_data)
            if health is not None:
                state.health = health[0]
                state.health_max = health[1]
                state.health_percent = health[0]/ health[1]
            mana = self._detect_mana(screen_data)
            if mana is not None:
                state.mana = mana[0]
                state.mana_max = mana[1]
                state.mana_percent = mana[0]/ mana[1]

        # Update internal state
        self._state = state
        return state
    

    def _is_centered(self,x,y)-> bool:
        print(x,y)
        if x>600 and  x<1500 and y>350 and y<700:
            return True
        return False

    def make_decision(self, state: D2RGameState) -> list[GameInstruction]:
        """
        Make a decision based on the current D2R game state.
        
        Args:
            state: The current D2R game state.
            
        Returns:
            Optional[GameInstruction]: The instruction to execute, or None.
        """
        # Handle different game stages
        if state.game_stage == GameStage.NONE:
            return []
        
        # 5. Print health and mana information
        if state.game_stage == GameStage.PLAYING:
            return self._make_playing_decision(state)
        elif state.game_stage==GameStage.PLAYERS:
            play=  GameInstruction(
                action_type = 'click',
                parameters = {"position":buttons['play'] } 
            )
            wait = GameInstruction(
                action_type = 'wait',
                parameters = {"value": 1 } 
            )

            hell=  GameInstruction(
                action_type = 'click',
                parameters = {"position":buttons['hell'] } 
            )

            return [play,wait,hell]
        elif state.game_stage.value=='loading':
            time.sleep(3)
            

        return []
    
    def send_instruction(self, instruction: GameInstruction) -> bool:
        """
        Send an instruction to D2R.
        
        Args:
            instruction: The game instruction to execute.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        action_type = instruction.action_type
        params = instruction.parameters
        
        try:
            if action_type == "click":
                self._input_controller.click(
                    params.get("x", 0),
                    params.get("y", 0),
                    params.get("button", "left")
                )
            elif action_type == "key":
                self._input_controller.press_key(params.get("key", ""))
            elif action_type == "type":
                self._input_controller.type_text(params.get("text", ""))
            else:
                return False
            
            return True
        except Exception:
            return False
    
    def reset_state(self) -> None:
        """Reset the game state to initial conditions."""
        self._state = D2RGameState()
    
    # Private helper methods for screen parsing
    
    def _detect_game_stage(self, screen_data: Any, scale:tuple[float,float]) -> GameStage:
        """Detect the current game stage from screen."""
        # return detect_game_stage(screen_data, scale)
        return GameStage.PLAYING
    
    def _detect_act(self, screen_data: Any) -> Optional[Act]:
        """Detect the current act from screen."""
        # TODO: Implement act detection logic
        return None
    
    def _detect_scene(self, screen_data: Any) -> str:
        """Detect the current scene/area from screen."""
        # TODO: Implement scene detection logic
        return ""
    
    def _detect_player_class(self, screen_data: Any) -> Optional[PlayerClass]:
        """Detect the player class from screen."""
        # TODO: Implement player class detection logic
        return None
    
    def _detect_items(self, screen_data: Any,scale:tuple[float,float]) -> list[Tuple]:
        """Detect the player class from screen."""
    
        return self._rule_detecor(screen_data,scale,highrune_min_idx=18)

    
    def _scale_coordinates(self, x: int, y: int, w: int, h: int, scale: float) -> Tuple[int, int, int, int]:
        """
        Scale coordinates based on image scaling factor.
        
        Args:
            x, y: Top-left coordinates.
            w, h: Width and height.
            scale: Scaling factor (e.g., 0.5 for half-size).
            
        Returns:
            Tuple[int, int, int, int]: Scaled (x, y, w, h)
        """
        scaled_x = int(x * scale)
        scaled_y = int(y * scale)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        return scaled_x, scaled_y, scaled_w, scaled_h
    
    def _parse_health_mana_text(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Parse health/mana text in "current/max" format (e.g., "1150/1155").
        
        Args:
            text: Text string in "current/max" format.
            
        Returns:
            Optional[Tuple[int, int]]: Tuple of (current, max) values, or None if parsing fails.
        """
        if not text or '/' not in text:
            return None
        
        try:
            current_str, max_str = text.split('/', 1)
            current = int(current_str.strip())
            max_val = int(max_str.strip())
            
            if max_val > 0:
                return (current, max_val)
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _calculate_percentage(self, current: int, max_val: int) -> float:
        """
        Calculate percentage from current and max values.
        
        Args:
            current: Current value.
            max_val: Maximum value.
            
        Returns:
            float: Percentage as float between 0.0 and 1.0.
        """
        if max_val <= 0:
            return 1.0
        
        percentage = current / max_val
        return max(0.0, min(1.0, percentage))  # Clamp to [0.0, 1.0]
    
    def _detect_resource(self, screen_data: Any, region: Tuple[int, int, int, int], scale: float = 1.0, type='unknown') -> Optional[tuple[int,int]]:
        """
        Detect resource percentage (health/mana) from a specific region.
        
        Args:
            screen_data: Screen image data.
            region: Tuple of (x, y, width, height) for 1920x1080 resolution.
            scale: Scaling factor applied to the image.
            
        Returns:
            float: Resource percentage (0.0 to 1.0).
        """
        
        try:
            # Unpack region coordinates
            base_x, base_y, base_w, base_h = region
            
            # Scale coordinates
            x, y, w, h = self._scale_coordinates(base_x, base_y, base_w, base_h, scale)
            
            # Extract text using configured OCR backend
            ocr_options = {
                "allowlist":"0123456789/",
                "width_ths": 2
            }
            text = '1000/1000'
            return self._parse_health_mana_text(text)
                
        except Exception as e:
            print(f"Resource detection error: {e}")
            return None
    
    def _detect_health(self, screen_data: Any) -> Optional[tuple[int,int]]:
        """
        Detect health percentage from screen.
        
        Args:
            screen_data: Screen image data.
            scale: Scaling factor applied to the image.
            
        Returns:
            float: Health percentage (0.0 to 1.0).
        """
        return self._detect_resource(screen_data, self.HEALTH_REGION, type='health') 
    
    def _detect_mana(self, screen_data: Any) -> Optional[tuple[int,int]]:
        """
        Detect mana percentage from screen.
        
        Args:
            screen_data: Screen image data.
            scale: Scaling factor applied to the image.
            
        Returns:
            float: Mana percentage (0.0 to 1.0).
        """
        return self._detect_resource(screen_data, self.MANA_REGION, type='mana')
    
    def _make_playing_decision(self, state: D2RGameState) -> list[GameInstruction]:

        for (item, x,y,conf) in state.items:
            if self._is_centered(x,y):
                return [ GameInstruction(
                    action_type = 'click',
                    parameters = {"position": (x,y)} 
                )]

            return  [GameInstruction(
                action_type = 'teleport',
                parameters = {"position": (x,y)} 
            )]


        return []