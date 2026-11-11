"""Abstract Game interface for game behavior abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """Base class for game state representation."""
    raw_data: Any = None
    timestamp: float = 0.0


@dataclass 
class GameInstruction:
    """Base class for game instructions/actions."""
    action_type: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class Game(ABC):
    """
    Abstract interface for game behavior.
    
    This interface provides a unified abstraction for interacting with different games,
    supporting state management, screen parsing, decision making, and instruction execution.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name/identifier of the game.
        
        Returns:
            str: The game name identifier (e.g., "d2r", "poe").
        """
        pass
    
    @property
    @abstractmethod
    def title(self) -> str:
        """
        Get the display title of the game.
        
        Returns:
            str: The human-readable game title (e.g., "Diablo II: Resurrected").
        """
        pass
    
    @property
    @abstractmethod
    def current_state(self) -> GameState:
        """
        Get the current game state.
        
        Returns:
            GameState: The current state of the game.
        """
        pass
    
    @abstractmethod
    def parse_screen(self, screen_data: Any, scale: float = 1.0) -> GameState:
        """
        Parse screen data to extract game state information.
        
        Args:
            screen_data: Raw screen capture data (e.g., image array from ScreenReader).
            scale: Scaling factor applied to the image (e.g., 0.5 for half-size).
            
        Returns:
            GameState: Parsed game state extracted from the screen.
        """
        pass
    
    @abstractmethod
    def make_decision(self, state: GameState) -> Optional[GameInstruction]:
        """
        Make a decision based on the current game state.
        
        Args:
            state: The current game state to evaluate.
            
        Returns:
            Optional[GameInstruction]: The instruction to execute, or None if no action needed.
        """
        pass
    
    @abstractmethod
    def send_instruction(self, instruction: GameInstruction) -> bool:
        """
        Send an instruction to the game.
        
        Args:
            instruction: The game instruction to execute.
            
        Returns:
            bool: True if the instruction was sent successfully, False otherwise.
        """
        pass
    
    def update(self, screen_data: Any, scale: float = 1.0) -> Optional[GameInstruction]:
        """
        Convenience method that combines parse_screen, make_decision, and send_instruction.
        
        This is the main loop logic that can be called each frame/tick.
        
        Args:
            screen_data: Raw screen capture data.
            scale: Scaling factor applied to the image (e.g., 0.5 for half-size).
            
        Returns:
            Optional[GameInstruction]: The instruction that was executed, or None.
        """
        state = self.parse_screen(screen_data, scale)
        instruction = self.make_decision(state)
        
        if instruction is not None:
            self.send_instruction(instruction)
            
        return instruction
    
    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the game state to initial conditions.
        """
        pass
