"""
Virtual Life Simulation
A fluid environment where organisms evolve through DNA-based reproduction and mutation.
"""

import pygame
import numpy as np
import random
import math
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WORLD_WIDTH = 5000
WORLD_HEIGHT = 5000
FPS = 60

# Colors
BACKGROUND_COLOR = (20, 30, 50)
FOOD_COLOR = (100, 200, 100)

class DNA:
    """Digital DNA that describes an organism's characteristics."""
    
    def __init__(self, parent_dna: Optional['DNA'] = None, parent2_dna: Optional['DNA'] = None):
        if parent_dna is None:
            # Create random DNA for first generation
            self.food_preference = random.uniform(0, 1)  # 0 = herbivore, 1 = carnivore
            self.color_r = random.randint(50, 255)
            self.color_g = random.randint(50, 255)
            self.color_b = random.randint(50, 255)
            self.size = random.uniform(5, 20)
            self.shape_points = random.randint(3, 8)  # Number of points for polygon shape
            self.propulsion_strength = random.uniform(0.5, 3.0)
            self.propulsion_efficiency = random.uniform(0.3, 0.9)
            self.max_speed = random.uniform(1.0, 5.0)
            self.energy_efficiency = random.uniform(0.5, 1.5)
            self.metabolism = random.uniform(0.01, 0.05)
            self.vision_range = random.uniform(50, 200)
            self.reproduction_threshold = random.uniform(50, 100)
            self.aggression = random.uniform(0, 1)
        else:
            # Combine parent DNA with mutation
            if parent2_dna is None:
                # Asexual reproduction (clone with mutation)
                self._inherit_from_parent(parent_dna)
            else:
                # Sexual reproduction (combine two parents)
                self._combine_parents(parent_dna, parent2_dna)
            
            # Apply mutations
            self._mutate()
    
    def _inherit_from_parent(self, parent: 'DNA'):
        """Inherit all traits from a single parent."""
        self.food_preference = parent.food_preference
        self.color_r = parent.color_r
        self.color_g = parent.color_g
        self.color_b = parent.color_b
        self.size = parent.size
        self.shape_points = parent.shape_points
        self.propulsion_strength = parent.propulsion_strength
        self.propulsion_efficiency = parent.propulsion_efficiency
        self.max_speed = parent.max_speed
        self.energy_efficiency = parent.energy_efficiency
        self.metabolism = parent.metabolism
        self.vision_range = parent.vision_range
        self.reproduction_threshold = parent.reproduction_threshold
        self.aggression = parent.aggression
    
    def _combine_parents(self, parent1: 'DNA', parent2: 'DNA'):
        """Combine traits from two parents (randomly choose or average)."""
        # Some traits are inherited from one parent, others are averaged
        self.food_preference = random.choice([parent1.food_preference, parent2.food_preference])
        self.color_r = int((parent1.color_r + parent2.color_r) / 2)
        self.color_g = int((parent1.color_g + parent2.color_g) / 2)
        self.color_b = int((parent1.color_b + parent2.color_b) / 2)
        self.size = (parent1.size + parent2.size) / 2
        self.shape_points = random.choice([parent1.shape_points, parent2.shape_points])
        self.propulsion_strength = (parent1.propulsion_strength + parent2.propulsion_strength) / 2
        self.propulsion_efficiency = (parent1.propulsion_efficiency + parent2.propulsion_efficiency) / 2
        self.max_speed = (parent1.max_speed + parent2.max_speed) / 2
        self.energy_efficiency = (parent1.energy_efficiency + parent2.energy_efficiency) / 2
        self.metabolism = (parent1.metabolism + parent2.metabolism) / 2
        self.vision_range = (parent1.vision_range + parent2.vision_range) / 2
        self.reproduction_threshold = (parent1.reproduction_threshold + parent2.reproduction_threshold) / 2
        self.aggression = (parent1.aggression + parent2.aggression) / 2
    
    def _mutate(self):
        """Apply random mutations to DNA."""
        mutation_rate = 0.1  # 10% chance of mutation per trait
        
        if random.random() < mutation_rate:
            self.food_preference = np.clip(self.food_preference + random.uniform(-0.2, 0.2), 0, 1)
        
        if random.random() < mutation_rate:
            self.color_r = int(np.clip(self.color_r + random.randint(-30, 30), 50, 255))
        
        if random.random() < mutation_rate:
            self.color_g = int(np.clip(self.color_g + random.randint(-30, 30), 50, 255))
        
        if random.random() < mutation_rate:
            self.color_b = int(np.clip(self.color_b + random.randint(-30, 30), 50, 255))
        
        if random.random() < mutation_rate:
            self.size = np.clip(self.size + random.uniform(-2, 2), 5, 30)
        
        if random.random() < mutation_rate:
            self.shape_points = int(np.clip(self.shape_points + random.randint(-1, 1), 3, 10))
        
        if random.random() < mutation_rate:
            self.propulsion_strength = np.clip(self.propulsion_strength + random.uniform(-0.3, 0.3), 0.2, 4.0)
        
        if random.random() < mutation_rate:
            self.propulsion_efficiency = np.clip(self.propulsion_efficiency + random.uniform(-0.1, 0.1), 0.2, 1.0)
        
        if random.random() < mutation_rate:
            self.max_speed = np.clip(self.max_speed + random.uniform(-0.5, 0.5), 0.5, 6.0)
        
        if random.random() < mutation_rate:
            self.energy_efficiency = np.clip(self.energy_efficiency + random.uniform(-0.2, 0.2), 0.3, 2.0)
        
        if random.random() < mutation_rate:
            self.metabolism = np.clip(self.metabolism + random.uniform(-0.01, 0.01), 0.005, 0.08)
        
        if random.random() < mutation_rate:
            self.vision_range = np.clip(self.vision_range + random.uniform(-20, 20), 30, 250)
        
        if random.random() < mutation_rate:
            self.reproduction_threshold = np.clip(self.reproduction_threshold + random.uniform(-10, 10), 30, 150)
        
        if random.random() < mutation_rate:
            self.aggression = np.clip(self.aggression + random.uniform(-0.2, 0.2), 0, 1)

class Food:
    """Food particles in the environment."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.size = random.uniform(2, 5)
        self.energy = random.uniform(5, 15)
        self.food_type = random.choice(['plant', 'meat'])  # For food preference system

class Organism:
    """An organism with DNA-based characteristics."""
    
    def __init__(self, x: float, y: float, dna: Optional[DNA] = None):
        self.x = x
        self.y = y
        self.dna = dna if dna else DNA()
        
        # Physical properties from DNA
        self.size = self.dna.size
        self.color = (self.dna.color_r, self.dna.color_g, self.dna.color_b)
        self.shape_points = self.dna.shape_points
        
        # Movement properties
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(0, 2 * math.pi)
        self.rotation_speed = random.uniform(-0.1, 0.1)
        
        # Energy and life
        self.energy = random.uniform(30, 50)
        self.max_energy = 100
        self.age = 0
        
        # Behavior
        self.target = None  # Target food or organism
        self.last_reproduction = 0
        
    def update(self, organisms: List['Organism'], foods: List[Food], dt: float):
        """Update organism state."""
        self.age += dt
        
        # Consume energy (metabolism)
        self.energy -= self.dna.metabolism * dt
        
        # Die if no energy
        if self.energy <= 0:
            return False
        
        # Find target (food or prey)
        self._find_target(organisms, foods)
        
        # Move towards target or random movement
        self._move(dt)
        
        # Check for eating
        self._try_eat(organisms, foods)
        
        # Check for reproduction
        if self.energy > self.dna.reproduction_threshold and self.age - self.last_reproduction > 2.0:
            return self._try_reproduce(organisms)
        
        return True
    
    def _find_target(self, organisms: List['Organism'], foods: List[Food]):
        """Find nearest food or prey within vision range."""
        best_target = None
        best_distance = self.dna.vision_range
        
        # Look for food
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < best_distance:
                # Check food preference
                food_match = 1.0 - abs(self.dna.food_preference - (1.0 if food.food_type == 'meat' else 0.0))
                if food_match > 0.3:  # Will eat if preference matches reasonably
                    best_target = food
                    best_distance = distance
        
        # Look for prey (if carnivorous enough)
        if self.dna.food_preference > 0.5 and self.dna.aggression > 0.3:
            for org in organisms:
                if org is self or org.size >= self.size * 1.2:  # Don't attack larger organisms
                    continue
                
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < best_distance:
                    best_target = org
                    best_distance = distance
        
        self.target = best_target
    
    def _move(self, dt: float):
        """Move organism using propulsion system."""
        # Calculate desired direction
        if self.target:
            if isinstance(self.target, Food):
                dx = self.target.x - self.x
                dy = self.target.y - self.y
            else:  # Organism target
                dx = self.target.x - self.x
                dy = self.target.y - self.y
            
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > 0:
                desired_angle = math.atan2(dy, dx)
                # Smooth rotation
                angle_diff = desired_angle - self.angle
                # Normalize angle difference
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                self.angle += angle_diff * 0.1
        else:
            # Random movement
            self.angle += self.rotation_speed * dt
            if random.random() < 0.01:
                self.rotation_speed = random.uniform(-0.1, 0.1)
        
        # Apply propulsion
        propulsion_force = self.dna.propulsion_strength
        energy_cost = propulsion_force * (1.0 - self.dna.propulsion_efficiency) * 0.1
        
        if self.energy > energy_cost:
            self.vx += math.cos(self.angle) * propulsion_force * dt
            self.vy += math.sin(self.angle) * propulsion_force * dt
            self.energy -= energy_cost
        
        # Apply friction/drag (fluid environment)
        drag = 0.95
        self.vx *= drag
        self.vy *= drag
        
        # Limit speed
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > self.dna.max_speed:
            self.vx = (self.vx / speed) * self.dna.max_speed
            self.vy = (self.vy / speed) * self.dna.max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Wrap around world edges
        self.x = self.x % WORLD_WIDTH
        self.y = self.y % WORLD_HEIGHT
    
    def _try_eat(self, organisms: List['Organism'], foods: List[Food]):
        """Try to eat food or other organisms."""
        eat_radius = self.size * 1.5
        
        # Try eating food
        for food in foods[:]:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < eat_radius:
                # Check if food matches preference
                food_match = 1.0 - abs(self.dna.food_preference - (1.0 if food.food_type == 'meat' else 0.0))
                energy_gain = food.energy * food_match * self.dna.energy_efficiency
                self.energy = min(self.energy + energy_gain, self.max_energy)
                foods.remove(food)
                return
        
        # Try eating other organisms
        if self.dna.food_preference > 0.3:  # Carnivorous enough
            for org in organisms[:]:
                if org is self or org.size >= self.size * 1.1:
                    continue
                
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < eat_radius:
                    # Eat the organism
                    energy_gain = org.energy * 0.5 * self.dna.energy_efficiency
                    self.energy = min(self.energy + energy_gain, self.max_energy)
                    organisms.remove(org)
                    return
    
    def _try_reproduce(self, organisms: List['Organism']) -> bool:
        """Try to reproduce with nearby organism or asexually."""
        # Look for mate
        for org in organisms:
            if org is self:
                continue
            
            dx = org.x - self.x
            dy = org.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Mate if close enough and both have enough energy
            if distance < self.size * 3 and org.energy > org.dna.reproduction_threshold:
                # Sexual reproduction
                new_dna = DNA(parent_dna=self.dna, parent2_dna=org.dna)
                new_x = self.x + random.uniform(-20, 20)
                new_y = self.y + random.uniform(-20, 20)
                new_org = Organism(new_x, new_y, new_dna)
                organisms.append(new_org)
                
                # Both parents lose energy
                self.energy -= 20
                org.energy -= 20
                self.last_reproduction = self.age
                org.last_reproduction = org.age
                return True
        
        # Asexual reproduction if no mate found
        if random.random() < 0.01:  # Small chance
            new_dna = DNA(parent_dna=self.dna)
            new_x = self.x + random.uniform(-20, 20)
            new_y = self.y + random.uniform(-20, 20)
            new_org = Organism(new_x, new_y, new_dna)
            organisms.append(new_org)
            self.energy -= 15
            self.last_reproduction = self.age
            return True
        
        return False
    
    def get_shape(self) -> List[Tuple[int, int]]:
        """Get polygon points for rendering."""
        points = []
        for i in range(self.shape_points):
            angle = (2 * math.pi * i / self.shape_points) + self.angle
            px = self.x + math.cos(angle) * self.size
            py = self.y + math.sin(angle) * self.size
            points.append((int(px), int(py)))
        return points

class Camera:
    """Camera system for viewing the world."""
    
    def __init__(self):
        self.x = WORLD_WIDTH / 2
        self.y = WORLD_HEIGHT / 2
        self.zoom = 1.0
    
    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        sx = int((wx - self.x) * self.zoom + SCREEN_WIDTH / 2)
        sy = int((wy - self.y) * self.zoom + SCREEN_HEIGHT / 2)
        return sx, sy
    
    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        wx = (sx - SCREEN_WIDTH / 2) / self.zoom + self.x
        wy = (sy - SCREEN_HEIGHT / 2) / self.zoom + self.y
        return wx, wy
    
    def update(self, target_x: float, target_y: float):
        """Follow a target (e.g., average organism position)."""
        # Smooth camera follow
        self.x += (target_x - self.x) * 0.05
        self.y += (target_y - self.y) * 0.05

class Simulation:
    """Main simulation class."""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Virtual Life Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize world
        self.organisms: List[Organism] = []
        self.foods: List[Food] = []
        self.camera = Camera()
        
        # Create initial organisms
        for _ in range(20):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.organisms.append(Organism(x, y))
        
        # Create initial food
        self._spawn_food(100)
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.paused = False
    
    def _spawn_food(self, count: int):
        """Spawn food particles in the world."""
        for _ in range(count):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.foods.append(Food(x, y))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self._spawn_food(50)
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                self.camera.zoom = max(0.5, min(2.0, self.camera.zoom + event.y * 0.1))
    
    def update(self, dt: float):
        """Update simulation."""
        if self.paused:
            return
        
        # Update organisms
        for org in self.organisms[:]:
            if not org.update(self.organisms, self.foods, dt):
                self.organisms.remove(org)
        
        # Maintain food population
        if len(self.foods) < 50:
            self._spawn_food(10)
        
        # Update camera to follow center of mass
        if self.organisms:
            avg_x = sum(org.x for org in self.organisms) / len(self.organisms)
            avg_y = sum(org.y for org in self.organisms) / len(self.organisms)
            self.camera.update(avg_x, avg_y)
    
    def render(self):
        """Render the simulation."""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw food
        for food in self.foods:
            sx, sy = self.camera.world_to_screen(food.x, food.y)
            if -10 < sx < SCREEN_WIDTH + 10 and -10 < sy < SCREEN_HEIGHT + 10:
                pygame.draw.circle(self.screen, FOOD_COLOR, (sx, sy), int(food.size * self.camera.zoom))
        
        # Draw organisms
        for org in self.organisms:
            sx, sy = self.camera.world_to_screen(org.x, org.y)
            if -50 < sx < SCREEN_WIDTH + 50 and -50 < sy < SCREEN_HEIGHT + 50:
                # Draw organism shape
                points = org.get_shape()
                screen_points = [self.camera.world_to_screen(px, py) for px, py in points]
                pygame.draw.polygon(self.screen, org.color, screen_points)
                # Draw outline
                pygame.draw.polygon(self.screen, (255, 255, 255), screen_points, 1)
                
                # Draw energy bar
                bar_width = int(org.size * 2 * self.camera.zoom)
                bar_height = 3
                bar_x = sx - bar_width // 2
                bar_y = sy - int(org.size * self.camera.zoom) - 10
                energy_ratio = org.energy / org.max_energy
                pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        
        # Draw UI
        info_text = [
            f"Organisms: {len(self.organisms)}",
            f"Food: {len(self.foods)}",
            f"Zoom: {self.camera.zoom:.2f}",
            "SPACE: Pause | F: Add Food | Mouse Wheel: Zoom"
        ]
        y_offset = 10
        for text in info_text:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            self.screen.blit(pause_text, (SCREEN_WIDTH // 2 - 50, 10))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()

