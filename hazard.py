"""Hazard zones that damage organisms."""


class Hazard:
    """A dangerous environmental zone spawned by the AI."""
    def __init__(self, x: float, y: float, radius: float = 100.0, duration: float = 10.0, damage_rate: float = 10.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.duration = duration
        self.damage_rate = damage_rate
        self.age = 0.0

    def update(self, dt: float) -> bool:
        """Update hazard state. Returns False if hazard should be removed."""
        self.age += dt
        return self.age < self.duration

