"""Food particles in the environment."""

import random
import math
from typing import List, Optional


class Food:
    """Food particles in the environment."""
    
    def __init__(self, x: float, y: float, parent: Optional['Food'] = None):
        self.x = x
        self.y = y
        self.size = random.uniform(2, 5)
        self.energy = random.uniform(5, 15)
        self.food_type = random.choice(['plant', 'meat'])  # For food preference system
        
        # Food reproduction and spreading
        self.age = 0.0
        self.reproduction_timer = 0.0
        self.reproduction_interval = random.uniform(30.0, 60.0)  # Reproduce every 30-60 seconds (slower)
        self.parent = parent  # Track parent for spreading
        
        # Toxicity - some food is toxic to some organisms
        # Toxicity is a value 0-1, where 1 is highly toxic
        # Organisms with toxicity_resistance > toxicity can eat it safely
        self.toxicity = random.uniform(0, 1) if random.random() < 0.3 else 0.0  # 30% chance of being toxic
        if parent is not None:
            # Inherit toxicity from parent (with small chance of mutation)
            self.toxicity = parent.toxicity
            if random.random() < 0.1:  # 10% chance to mutate toxicity
                self.toxicity = random.uniform(0, 1)
    
    def update(self, dt: float, foods: List['Food'], world_width: float, world_height: float, growth_multiplier: float = 1.0) -> List['Food']:
        """Update food state and handle reproduction. Returns list of new food."""
        self.age += dt
        # Apply growth multiplier to reproduction timer (higher multiplier = faster reproduction)
        self.reproduction_timer += dt * growth_multiplier
        
        new_foods = []
        
        # Reproduce if enough time has passed and not too crowded
        if self.reproduction_timer >= self.reproduction_interval:
            # Check if there's space nearby (not too many food particles)
            nearby_count = 0
            for food in foods:
                if food is self:
                    continue
                dx = food.x - self.x
                dy = food.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < 100:  # Within 100 units
                    nearby_count += 1
            
            # Reproduce if not too crowded (max 5 nearby food)
            if nearby_count < 5:
                # Create 1-2 new food particles nearby but spread out
                num_offspring = random.randint(1, 2)
                for _ in range(num_offspring):
                    # Spread out from parent (30-80 units away)
                    spread_distance = random.uniform(30, 80)
                    spread_angle = random.uniform(0, 2 * math.pi)
                    new_x = self.x + math.cos(spread_angle) * spread_distance
                    new_y = self.y + math.sin(spread_angle) * spread_distance
                    
                    # Wrap around world edges
                    new_x = new_x % world_width
                    new_y = new_y % world_height
                    
                    # Create new food with this as parent
                    new_food = Food(new_x, new_y, parent=self)
                    new_foods.append(new_food)
                
                # Reset reproduction timer
                self.reproduction_timer = 0.0
                # Slightly randomize next reproduction interval
                self.reproduction_interval = random.uniform(30.0, 60.0)
        
        return new_foods

