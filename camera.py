"""Camera system for viewing the world."""

from typing import Tuple
from constants import WORLD_WIDTH, WORLD_HEIGHT


class Camera:
    """Camera system for viewing the world."""
    
    def __init__(self):
        self.x = WORLD_WIDTH / 2
        self.y = WORLD_HEIGHT / 2
        self.zoom = 0.3  # Start zoomed out more
    
    def world_to_screen(self, wx: float, wy: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        sx = int((wx - self.x) * self.zoom + screen_width / 2)
        sy = int((wy - self.y) * self.zoom + screen_height / 2)
        return sx, sy
    
    def screen_to_world(self, sx: int, sy: int, screen_width: int, screen_height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        wx = (sx - screen_width / 2) / self.zoom + self.x
        wy = (sy - screen_height / 2) / self.zoom + self.y
        return wx, wy
    
    def update(self, target_x: float, target_y: float):
        """Follow a target (e.g., average organism position)."""
        # Smooth camera follow
        self.x += (target_x - self.x) * 0.05
        self.y += (target_y - self.y) * 0.05
    
    def move_to(self, target_x: float, target_y: float, speed: float = 0.1):
        """Move camera to a specific location (for event alerts)."""
        # Faster movement for event alerts
        self.x += (target_x - self.x) * speed
        self.y += (target_y - self.y) * speed
    
    def move(self, dx: float, dy: float):
        """Move camera by world space delta."""
        self.x += dx
        self.y += dy

