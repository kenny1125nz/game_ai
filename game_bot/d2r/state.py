"""Diablo II: Resurrected game state definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from common.game import GameState


class GameStage(Enum):
    """Game stage/phase enumeration."""
    NONE = "none"
    PLAYERS = "players"
    CREATING = "creating"
    JOINING = "joining"
    PLAYING = "playing"


class Act(Enum):
    """Game act enumeration."""
    ACT_1 = 1
    ACT_2 = 2
    ACT_3 = 3
    ACT_4 = 4
    ACT_5 = 5


class PlayerClass(Enum):
    """Player character class enumeration."""
    AMAZON = "amazon"
    NECROMANCER = "necromancer"
    BARBARIAN = "barbarian"
    SORCERESS = "sorceress"
    PALADIN = "paladin"
    DRUID = "druid"
    ASSASSIN = "assassin"


@dataclass
class D2RGameState(GameState):
    """
    Diablo II: Resurrected specific game state.
    
    Extends the base GameState with D2R-specific attributes.
    """
    # Game stage
    game_stage: GameStage = GameStage.NONE
    
    # Current act (only valid when playing)
    act: Optional[Act] = None
    
    # Current scene/area name
    scene: str = ""
    
    # Player information
    player_class: Optional[PlayerClass] = None
    
    # Health and mana as percentages (0.0 to 1.0)
    health: int =0
    mana: int=0
    health_max: int=100
    mana_max: int = 100
    health_percent: float = 1.0
    mana_percent: float = 1.0

    items: list =  field(default_factory=list)
    
    def is_playing(self) -> bool:
        """Check if currently in gameplay."""
        return self.game_stage == GameStage.PLAYING
    
    def is_low_health(self, threshold: float = 0.3) -> bool:
        """Check if health is below threshold."""
        return self.health_percent < threshold
    
    def is_low_mana(self, threshold: float = 0.2) -> bool:
        """Check if mana is below threshold."""
        return self.mana_percent < threshold
