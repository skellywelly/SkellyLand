"""Main simulation class."""

import pygame
import numpy as np
import random
import math
from typing import List, Tuple, Optional

from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, FPS,
    BACKGROUND_COLOR, FOOD_COLOR, INPUT_SIZE, MAX_SPEED_CAP
)
from neural_network import NeuralNetwork
from dna import DNA
from food import Food
from hazard import Hazard
from organism import Organism
from camera import Camera
from population_controller import PopulationControllerAI

class Simulation:
    """Main simulation class."""
    
    def __init__(self):
        # Get screen info and create maximized window
        screen_info = pygame.display.Info()
        # Create window at screen size (maximized)
        try:
            # Try to get screen dimensions
            import os
            if os.name == 'posix':  # Linux - try to get from environment or use screen info
                # Use screen dimensions for maximized window
                width = screen_info.current_w if hasattr(screen_info, 'current_w') else SCREEN_WIDTH
                height = screen_info.current_h if hasattr(screen_info, 'current_h') else SCREEN_HEIGHT
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            else:
                # For other systems, use screen info
                width = screen_info.current_w if hasattr(screen_info, 'current_w') else SCREEN_WIDTH
                height = screen_info.current_h if hasattr(screen_info, 'current_h') else SCREEN_HEIGHT
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        except:
            # Fallback to default size but resizable
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        
        # Get actual screen dimensions
        self.actual_width = self.screen.get_width()
        self.actual_height = self.screen.get_height()
        
        pygame.display.set_caption("Virtual Life Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize world
        self.organisms: List[Organism] = []
        self.foods: List[Food] = []
        self.camera = Camera()
        
        # Create initial organisms at random positions with random velocities to prevent clumping
        for _ in range(40):
            # Random position across the entire world
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            new_org = Organism(x, y)
            
            # Random velocity in random direction to prevent clustering
            ejection_speed = random.uniform(40.0, 100.0)
            random_angle = random.uniform(0, 2 * math.pi)
            new_org.vx = math.cos(random_angle) * ejection_speed
            new_org.vy = math.sin(random_angle) * ejection_speed
            
            self.organisms.append(new_org)
        
        # Create initial food
        self._spawn_food(50)  # Reduced initial food count
        
        # Initialize hazards list and Divine Energy
        self.hazards = []
        self.divine_energy = 100.0
        self.max_divine_energy = 100.0
        self.divine_regen_rate = 1.0

        # UI
        self.font = pygame.font.Font(None, 24)
        # Use bundled fonts for platform independence
        import os
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        dejavu_path = os.path.join(font_dir, "DejaVuSans.ttf")
        emoji_path = os.path.join(font_dir, "NotoColorEmoji.ttf")
        alert_font_size = 12  # Much smaller for subtle alerts
        
        # Load text font
        try:
            if os.path.exists(dejavu_path):
                self.alert_font = pygame.font.Font(dejavu_path, alert_font_size)
            else:
                self.alert_font = pygame.font.SysFont("sans", alert_font_size)
        except:
            self.alert_font = pygame.font.Font(None, alert_font_size)
        
        # Load emoji font for emoji rendering (same size as alert text)
        emoji_font_size = alert_font_size  # Match alert font size
        try:
            if os.path.exists(emoji_path):
                self.emoji_font = pygame.font.Font(emoji_path, emoji_font_size)
            else:
                self.emoji_font = None
        except:
            self.emoji_font = None
        
        # Alert system for major events
        self.current_alert = None  # (message, color, time_remaining, location_x, location_y)
        self.alert_duration = 3.0  # Show alert for 3 seconds
        self.alerts_enabled = True  # Toggle for event alerts
        self.camera_following_event = False  # Whether camera is following an event
        self.event_target_x = None
        self.event_target_y = None
        self.paused = False
        
        # Mouse dragging
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.camera_follow_enabled = False  # Auto-follow disabled by default
        
        # Monitoring - comprehensive statistics tracking
        self.monitor_timer = 0.0
        self.monitor_interval = 2.0  # Print stats every 2 seconds
        self.frame_count = 0
        self.total_time = 0.0
        self.show_terminal_output = False  # Toggle for terminal monitoring output (off by default)
        
        # Global parameter multipliers (adjustable via UI)
        self.metabolism_multiplier = 1.0
        self.flagella_impulse_multiplier = 1.0
        self.aggression_multiplier = 1.0
        self.reproduction_desire_multiplier = 1.0
        self.food_growth_multiplier = 1.0  # Controls food reproduction rate
        self.mutation_rate_multiplier = 1.0  # Controls DNA mutation rate

        # Population monitoring and control
        self.population_history = []  # Track population over time
        self.population_control_timer = 0.0
        self.population_control_interval = 10.0  # Adjust parameters every 10 seconds
        self.target_population_min = 60  # Minimum target population (centered around 70)
        self.target_population_max = 80  # Maximum target population (centered around 70)
        self.parameter_adjustment_rate = 0.05  # How much to adjust parameters per cycle
        self.last_parameter_changes = {}  # Track when parameters were last adjusted
        self.ai_save_timer = 0.0
        self.ai_save_interval = 300.0  # Save AI weights every 5 minutes

        # AI Population Controller Neural Network
        self.ai_controller = PopulationControllerAI()
        self.ai_controller.load_weights()  # Load previously learned weights if available

        # Graph data for bottom panel
        self.graph_history_size = 200  # Store last 200 data points
        self.graph_data = {
            'population': [],
            'metabolism': [],
            'flagella': [],
            'aggression': [],
            'reproduction': [],
            'food_growth': [],
            'mutation_rate': [],
            'food_count': [],
            'avg_energy': [],
        }
        self.graph_update_timer = 0.0
        self.graph_update_interval = 0.1  # Update graph every 0.1 seconds
        
        # Statistics counters (reset each interval)
        self.stats = {
            'deaths': 0,
            'deaths_combat': 0,
            'deaths_starvation': 0,
            'deaths_toxic': 0,
            'births': 0,
            'births_sexual': 0,
            'matings': 0,
            'fights': 0,
            'food_eaten': 0,
            'toxic_food_eaten': 0,
            'food_reproduced': 0,
        }
        
        # Cumulative statistics (never reset)
        self.cumulative_stats = {
            'total_births': 0,
            'total_deaths': 0,
            'total_matings': 0,
            'total_fights': 0,
            'total_food_eaten': 0,
        }
    
    def _spawn_food(self, count: int):
        """Spawn food particles in the world."""
        MAX_FOOD = 350  # Maximum food units allowed
        current_count = len(self.foods)
        if current_count >= MAX_FOOD:
            return  # Don't add more food if at or above cap
        
        # Only add up to the cap
        remaining_slots = MAX_FOOD - current_count
        actual_count = min(count, remaining_slots)
        
        for _ in range(actual_count):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.foods.append(Food(x, y))
    
    def _add_organisms(self, count: int):
        """Add new organisms to the simulation with ejection velocity to prevent clumping."""
        # Find center of existing organisms or use world center
        if len(self.organisms) > 0:
            center_x = sum(org.x for org in self.organisms) / len(self.organisms)
            center_y = sum(org.y for org in self.organisms) / len(self.organisms)
        else:
            center_x = WORLD_WIDTH / 2
            center_y = WORLD_HEIGHT / 2
        
        for i in range(count):
            # Spread in a circle around center
            angle = (2 * math.pi * i) / count
            distance = random.uniform(50, 150)
            x = center_x + math.cos(angle) * distance
            y = center_y + math.sin(angle) * distance
            # Wrap around world bounds
            x = x % WORLD_WIDTH
            y = y % WORLD_HEIGHT
            new_org = Organism(x, y)
            
            # Eject with velocity away from center
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x = math.cos(angle)
                dir_y = math.sin(angle)
            
            # Ejection velocity (40-100 pixels/second)
            ejection_speed = random.uniform(40.0, 100.0)
            new_org.vx = dir_x * ejection_speed
            new_org.vy = dir_y * ejection_speed
            
            self.organisms.append(new_org)
    
    def _reset_simulation(self):
        """Reset the simulation to initial state, including AI weights."""
        # Clear all organisms and food
        self.organisms.clear()
        self.foods.clear()

        # Reset camera
        self.camera.x = WORLD_WIDTH / 2
        self.camera.y = WORLD_HEIGHT / 2
        self.camera.zoom = 0.3  # Start zoomed out more

        # Reset AI controller to random weights (forget learned behavior)
        self.ai_controller = PopulationControllerAI()
        print("🔄 Reset AI controller to random weights")

        # Reset global parameters to defaults
        self.metabolism_multiplier = 1.0
        self.flagella_impulse_multiplier = 1.0
        self.aggression_multiplier = 1.0
        self.reproduction_desire_multiplier = 1.0
        self.food_growth_multiplier = 1.0
        self.mutation_rate_multiplier = 1.0

        # Reset statistics
        for key in self.stats:
            self.stats[key] = 0
        for key in self.cumulative_stats:
            self.cumulative_stats[key] = 0

        # Reset population monitoring
        self.population_history.clear()
        self.last_parameter_changes.clear()

        # Create initial organisms at random positions with random velocities to prevent clumping
        for _ in range(40):
            # Random position across the entire world
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            new_org = Organism(x, y)
            
            # Random velocity in random direction to prevent clustering
            ejection_speed = random.uniform(40.0, 100.0)
            random_angle = random.uniform(0, 2 * math.pi)
            new_org.vx = math.cos(random_angle) * ejection_speed
            new_org.vy = math.sin(random_angle) * ejection_speed
            
            self.organisms.append(new_org)

        # Create initial food
        self._spawn_food(100)

        # Initialize hazards list and Divine Energy
        self.hazards = []
        self.divine_energy = 100.0
        self.max_divine_energy = 100.0
        self.divine_regen_rate = 1.0  # Regain 1 energy per second

        print("🔄 Simulation completely reset (including AI learning)")
    
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
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                elif event.key == pygame.K_o:
                    self._add_organisms(10)
                elif event.key == pygame.K_m:
                    # Toggle terminal monitoring output
                    self.show_terminal_output = not self.show_terminal_output
                    print(f"Terminal output: {'ON' if self.show_terminal_output else 'OFF'}")
                elif event.key == pygame.K_a:
                    # Toggle event alerts
                    self.alerts_enabled = not self.alerts_enabled
                    if not self.alerts_enabled:
                        self.current_alert = None  # Clear current alert when disabling
                    print(f"Event alerts: {'ON' if self.alerts_enabled else 'OFF'}")
                elif event.key == pygame.K_s:
                    # Save AI weights manually
                    self.ai_controller.save_weights()
                    print("💾 Manually saved AI controller weights")
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in with + or =
                    self.camera.zoom = min(3.0, self.camera.zoom + 0.1)
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
                elif event.key == pygame.K_MINUS:
                    # Zoom out with -
                    self.camera.zoom = max(0.1, self.camera.zoom - 0.1)
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
                elif event.key == pygame.K_UP:
                    # Pan up with arrow key
                    self.camera.move(0, -100 / self.camera.zoom)  # Move 100 world units up
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
                elif event.key == pygame.K_DOWN:
                    # Pan down with arrow key
                    self.camera.move(0, 100 / self.camera.zoom)  # Move 100 world units down
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
                elif event.key == pygame.K_LEFT:
                    # Pan left with arrow key
                    self.camera.move(-100 / self.camera.zoom, 0)  # Move 100 world units left
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
                elif event.key == pygame.K_RIGHT:
                    # Pan right with arrow key
                    self.camera.move(100 / self.camera.zoom, 0)  # Move 100 world units right
                    self.camera_follow_enabled = False  # Disable auto-follow when manually controlling
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                    self.camera_follow_enabled = False  # Disable auto-follow when dragging
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    # Calculate mouse movement in screen space
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    
                    # Convert screen movement to world movement (inverse of zoom)
                    screen_width = self.screen.get_width()
                    world_dx = -dx / self.camera.zoom
                    world_dy = -dy / self.camera.zoom
                    
                    # Move camera
                    self.camera.move(world_dx, world_dy)
                    
                    # Update last mouse position
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                self.camera.zoom = max(0.1, min(3.0, self.camera.zoom + event.y * 0.1))
    
    def update(self, dt: float):
        """Update simulation."""
        if self.paused:
            return
        
        self.total_time += dt
        self.monitor_timer += dt
        
        # Update hazards
        self.hazards = [h for h in self.hazards if h.update(dt)]
        
        # Regenerate Divine Energy
        self.divine_energy = min(self.max_divine_energy, self.divine_energy + self.divine_regen_rate * dt)

        # Apply hazard effects to organisms
        for org in self.organisms:
            org.detected_hazard = None # Reset hazard detection
            for hazard in self.hazards:
                dx = org.x - hazard.x
                dy = org.y - hazard.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                # Check for detection
                if dist < org.dna.vision_range + hazard.radius:
                    if org.detected_hazard is None or dist < math.sqrt((org.x - org.detected_hazard.x)**2 + (org.y - org.detected_hazard.y)**2):
                        org.detected_hazard = hazard

                # Check for damage
                if dist < hazard.radius:
                    damage = hazard.damage_rate * dt
                    org.energy -= damage
                    if org.energy <= 0:
                        org._death_cause = 'hazard'

        # Track organism count before update
        org_count_before = len(self.organisms)
        food_count_before = len(self.foods)
        
        # Track matings by storing organism IDs before update
        org_ids_before = {id(org) for org in self.organisms}
        
        # Update organisms
        deaths = 0
        total_energy = 0.0
        total_speed = 0.0
        avg_age = 0.0
        
        # Track organisms in combat for fight counting
        organisms_in_combat = set()
        new_organisms_this_frame = []
        
        for org in self.organisms[:]:
            # Pass multipliers to organism for use during update
            org._temp_metabolism_mult = self.metabolism_multiplier
            org._temp_flagella_mult = self.flagella_impulse_multiplier
            org._temp_aggression_mult = self.aggression_multiplier
            org._temp_reproduction_desire_mult = self.reproduction_desire_multiplier
            org._temp_mutation_mult = self.mutation_rate_multiplier
            
            if not org.update(self.organisms, self.foods, dt):
                # Death event - create alert
                # Check if organism is still in list (might have been eaten/removed during update)
                if org not in self.organisms:
                    continue
                
                death_cause = 'starvation'
                if hasattr(org, '_death_cause'):
                    death_cause = org._death_cause
                
                self._create_alert(f"Death ({death_cause})", (255, 100, 100), org.x, org.y)
                
                self.organisms.remove(org)
                deaths += 1
                self.stats['deaths'] += 1
                self.cumulative_stats['total_deaths'] += 1
                # Track death cause
                if hasattr(org, '_death_cause'):
                    if org._death_cause == 'combat':
                        self.stats['deaths_combat'] += 1
                    elif org._death_cause == 'toxic':
                        self.stats['deaths_toxic'] += 1
                    elif org._death_cause == 'starvation':
                        self.stats['deaths_starvation'] += 1
                else:
                    self.stats['deaths_starvation'] += 1
            else:
                total_energy += org.energy
                speed = math.sqrt(org.vx * org.vx + org.vy * org.vy)
                total_speed += speed
                avg_age += org.age
                
                # Track fights (count unique combat pairs)
                if org.in_combat and org.combat_target:
                    pair = tuple(sorted([id(org), id(org.combat_target)]))
                    if pair not in organisms_in_combat:
                        organisms_in_combat.add(pair)
                        self.stats['fights'] += 1
                        self.cumulative_stats['total_fights'] += 1
                        # Fight event - create alert at midpoint between fighters
                        fight_x = (org.x + org.combat_target.x) / 2
                        fight_y = (org.y + org.combat_target.y) / 2
                        self._create_alert("Fight!", (255, 0, 0), fight_x, fight_y)
        
        # Find new organisms (born this frame)
        org_ids_after = {id(org) for org in self.organisms}
        new_org_ids = org_ids_after - org_ids_before
        new_organisms_this_frame = [org for org in self.organisms if id(org) in new_org_ids]
        
        # Debug: verify births are being detected
        if len(new_org_ids) > 0:
            if self.show_terminal_output:
                print(f"DEBUG: Detected {len(new_org_ids)} new organisms. Total organisms: {len(self.organisms)} (was {org_count_before})")
        
        # Track births and estimate matings
        births = len(new_organisms_this_frame)
        if births > 0:
            self.stats['births'] += births
            self.cumulative_stats['total_births'] += births
            
            # All births are from sexual reproduction (mating)
            # Track matings for all new organisms
            for new_org in new_organisms_this_frame:
                # All births come from mating, so count as sexual reproduction
                self.stats['births_sexual'] += 1
                self.stats['matings'] += 1
                self.cumulative_stats['total_matings'] += 1
                # Birth/Mating event - create alert at birth location
                self._create_alert("Birth!", (0, 255, 255), new_org.x, new_org.y)

        # Check for ecosystem collapse - population below 10 indicates collapse
        if len(self.organisms) < 10:
            print(f"💀 ECOSYSTEM COLLAPSE: Population dropped to {len(self.organisms)} organisms!")
            print("🌱 Respawning 40 new organisms to restart the ecosystem...")

            # Create alert for collapse
            self._create_alert("COLLAPSE! Respawning...", (255, 0, 0), WORLD_WIDTH/2, WORLD_HEIGHT/2)

            # Clear remaining organisms from collapsed population
            self.organisms.clear()

            # Spawn 40 new organisms at random positions with random velocities to prevent clumping
            for _ in range(40):
                # Random position across the entire world
                x = random.uniform(0, WORLD_WIDTH)
                y = random.uniform(0, WORLD_HEIGHT)
                new_org = Organism(x, y)
                
                # Random velocity in random direction to prevent clustering
                ejection_speed = random.uniform(40.0, 100.0)
                random_angle = random.uniform(0, 2 * math.pi)
                new_org.vx = math.cos(random_angle) * ejection_speed
                new_org.vy = math.sin(random_angle) * ejection_speed
                
                self.organisms.append(new_org)

            # Add some food to help the new population
            self._spawn_food(50)

            print(f"✅ Respawned 40 organisms. Population: {len(self.organisms)}")

        # Track food eaten
        food_count_after = len(self.foods)
        food_eaten_count = food_count_before - food_count_after
        if food_eaten_count > 0:
            self.stats['food_eaten'] += food_eaten_count
            self.cumulative_stats['total_food_eaten'] += food_eaten_count
            # Estimate toxic food eaten (rough estimate based on toxic food percentage)
            toxic_food_count_before = sum(1 for food in self.foods if food.toxicity > 0) if food_count_before > 0 else 0
            if toxic_food_count_before > 0 and food_count_before > 0:
                toxic_ratio = toxic_food_count_before / food_count_before
                self.stats['toxic_food_eaten'] += int(food_eaten_count * toxic_ratio)
        
        # Update food (reproduction and spreading)
        new_foods = []
        for food in self.foods:
            offspring = food.update(dt, self.foods, WORLD_WIDTH, WORLD_HEIGHT, self.food_growth_multiplier)
            new_foods.extend(offspring)
            if len(offspring) > 0:
                self.stats['food_reproduced'] += len(offspring)
        
        # Add new food from reproduction (respecting cap)
        MAX_FOOD = 350  # Maximum food units allowed
        current_count = len(self.foods)
        if current_count < MAX_FOOD:
            remaining_slots = MAX_FOOD - current_count
            # Only add as many as we have slots for
            foods_to_add = new_foods[:remaining_slots]
            self.foods.extend(foods_to_add)
        
        # Maintain minimum food population (but food can now reproduce naturally)
        if len(self.foods) < 20:  # Lower threshold, spawn less frequently
            self._spawn_food(5)  # Spawn fewer food at a time (will respect cap)
        
        # Update camera - follow center of mass (don't move camera for events, just point at them)
        if self.camera_follow_enabled and self.organisms:
            avg_x = sum(org.x for org in self.organisms) / len(self.organisms)
            avg_y = sum(org.y for org in self.organisms) / len(self.organisms)
            self.camera.update(avg_x, avg_y)
        
        # Update alert timer
        if self.current_alert is not None:
            message, color, time_remaining, x, y = self.current_alert
            time_remaining -= dt
            if time_remaining <= 0:
                self.current_alert = None
            else:
                self.current_alert = (message, color, time_remaining, x, y)
        
        # Print monitoring stats (if enabled)
        if self.monitor_timer >= self.monitor_interval:
            if self.show_terminal_output:
                self._print_comprehensive_monitoring_stats(deaths, total_energy, total_speed, avg_age)
            # Reset interval stats
            for key in self.stats:
                self.stats[key] = 0
            self.monitor_timer = 0.0

        # Population monitoring and control (every 10 seconds)
        self.population_control_timer += dt
        if self.population_control_timer >= self.population_control_interval:
            self._control_population()
            self.population_control_timer = 0.0

        # Periodic AI weight saving (every 5 minutes)
        self.ai_save_timer += dt
        if self.ai_save_timer >= self.ai_save_interval:
            self.ai_controller.save_weights()
            if self.show_terminal_output:
                print("💾 Saved AI controller weights")
            self.ai_save_timer = 0.0

        # Update graph data
        self.graph_update_timer += dt
        if self.graph_update_timer >= self.graph_update_interval:
            # Calculate current values
            current_pop = len(self.organisms)
            current_food = len(self.foods)
            avg_energy = sum(org.energy for org in self.organisms) / current_pop if current_pop > 0 else 0.0

            # Add data points
            self.graph_data['population'].append(current_pop)
            self.graph_data['metabolism'].append(self.metabolism_multiplier)
            self.graph_data['flagella'].append(self.flagella_impulse_multiplier)
            self.graph_data['aggression'].append(self.aggression_multiplier)
            self.graph_data['reproduction'].append(self.reproduction_desire_multiplier)
            self.graph_data['food_growth'].append(self.food_growth_multiplier)
            self.graph_data['mutation_rate'].append(self.mutation_rate_multiplier)
            self.graph_data['food_count'].append(current_food)
            self.graph_data['avg_energy'].append(avg_energy)

            # Limit history size
            for key in self.graph_data:
                if len(self.graph_data[key]) > self.graph_history_size:
                    self.graph_data[key].pop(0)

            self.graph_update_timer = 0.0
    
    def _update_communication_signals(self, dt: float):
        """Update signal propagation between organisms."""
        current_time = getattr(self, '_simulation_time', 0)

        # Collect all signals from all organisms
        all_signals = []
        for org in self.organisms:
            for signal_type, strength, x, y, time_emitted in org.signals_emitted:
                age = current_time - time_emitted
                # Signals decay with distance and time
                decayed_strength = strength * max(0, 1 - age * 0.05)  # Time decay
                if decayed_strength > 0.1:
                    all_signals.append((signal_type, decayed_strength, x, y, age))

        # Let each organism detect nearby signals
        for org in self.organisms:
            org.signals_detected = {}
            comm_range = getattr(org.dna, 'communication_range', 100)

            for signal_type, strength, x, y, age in all_signals:
                dx = org.x - x
                dy = org.y - y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < comm_range and distance > 0:  # Don't detect own signals
                    # Distance-based attenuation
                    distance_factor = max(0.1, 1 - distance / comm_range)
                    detected_strength = strength * distance_factor

                    if signal_type not in org.signals_detected:
                        org.signals_detected[signal_type] = []
                    org.signals_detected[signal_type].append((detected_strength, distance, age))

    def _create_alert(self, message: str, color: Tuple[int, int, int], x: float, y: float):
        """Create an alert for a major event (points at location but doesn't move camera)."""
        if not self.alerts_enabled:
            return  # Don't create alerts if disabled
        self.current_alert = (message, color, self.alert_duration, x, y)

    def _update_territory_markers(self, dt: float):
        """Update territory markers - they decay slower than regular signals."""
        # Territory markers are stored as special signals with slower decay
        # They influence organism behavior in marked areas
        pass  # Implementation would track persistent territory markers in simulation state
    
    def _spawn_hazard_zone(self):
        """AI God Power: Spawn a dangerous hazard zone."""
        # Spawn near center of organism mass or random if none
        if self.organisms:
            avg_x = sum(org.x for org in self.organisms) / len(self.organisms)
            avg_y = sum(org.y for org in self.organisms) / len(self.organisms)
            # Add some randomness
            x = avg_x + random.uniform(-200, 200)
            y = avg_y + random.uniform(-200, 200)
        else:
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            
        x = x % WORLD_WIDTH
        y = y % WORLD_HEIGHT
        
        hazard = Hazard(x, y, radius=150.0, duration=15.0, damage_rate=20.0)
        self.hazards.append(hazard)
        self._create_alert("⚠️ Hazard Zone Spawned!", (255, 50, 50), x, y)

    def _spawn_manna_drop(self):
        """AI God Power: Spawn a cluster of high-energy food."""
        # Spawn near center of organism mass
        if self.organisms:
            avg_x = sum(org.x for org in self.organisms) / len(self.organisms)
            avg_y = sum(org.y for org in self.organisms) / len(self.organisms)
            # Add some randomness
            center_x = avg_x + random.uniform(-100, 100)
            center_y = avg_y + random.uniform(-100, 100)
        else:
            center_x = random.uniform(0, WORLD_WIDTH)
            center_y = random.uniform(0, WORLD_HEIGHT)
            
        # Spawn cluster of golden food
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, 100)
            x = (center_x + math.cos(angle) * dist) % WORLD_WIDTH
            y = (center_y + math.sin(angle) * dist) % WORLD_HEIGHT
            
            food = Food(x, y)
            food.energy = 50.0  # High energy
            food.toxicity = 0.0  # Safe
            # Could add visual distinction later (e.g. color override in Food class or checking energy in render)
            self.foods.append(food)
            
        self._create_alert("✨ Manna Rain!", (255, 215, 0), center_x, center_y)

    def _draw_ui_panel(self, screen_width: int, screen_height: int):
        """Draw consolidated UI panel with all text information."""
        panel_width = 350
        panel_x = 10
        panel_y = 10
        panel_padding = 10
        line_height = 20
        
        # Calculate panel height based on content
        num_lines = 18  # Increased to accommodate new parameters
        panel_height = num_lines * line_height + panel_padding * 2
        
        # Create transparent surface for panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.set_alpha(220)  # Semi-transparent (220/255 opacity)
        
        # Draw panel background on transparent surface
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(panel_surface, (30, 40, 60), panel_rect)
        pygame.draw.rect(panel_surface, (100, 100, 100), panel_rect, 2)
        
        y_offset = panel_padding
        
        # Current alert (if any)
        if self.current_alert is not None:
            message, color, time_remaining, alert_x, alert_y = self.current_alert
            # Remove any emojis from message
            clean_message = message.replace("💀", "").replace("⚔️", "").replace("👶", "").replace("🧠", "").replace("📉", "").replace("📈", "").replace("⚖️", "").replace("🔄", "").strip()
            alert_text = f"Alert: {clean_message}"
            surface = self.font.render(alert_text, True, color)
            panel_surface.blit(surface, (panel_padding, y_offset))
            y_offset += line_height
        
        # Stats section
        stats_text = [
            f"Organisms: {len(self.organisms)}",
            f"Food: {len(self.foods)}",
            f"Total Births: {self.cumulative_stats['total_births']}",
            f"Total Deaths: {self.cumulative_stats['total_deaths']}",
        ]
        for text in stats_text:
            surface = self.font.render(text, True, (255, 255, 255))
            panel_surface.blit(surface, (panel_padding, y_offset))
            y_offset += line_height
        
        y_offset += 5  # Spacing
        
        # Parameters section
        current_pop = len(self.organisms)
        if current_pop < self.target_population_min:
            status_text = f"Status: Population Low ({current_pop})"
            status_color = (255, 100, 100)
        elif current_pop > self.target_population_max:
            status_text = f"Status: Population High ({current_pop})"
            status_color = (255, 150, 100)
        else:
            status_text = f"Status: Population Stable ({current_pop})"
            status_color = (100, 255, 100)
        
        status_surface = self.font.render(status_text, True, status_color)
        panel_surface.blit(status_surface, (panel_padding, y_offset))
        y_offset += line_height
        
        # Parameters
        param_text = [
            f"Metabolism: {self.metabolism_multiplier:.2f}",
            f"Flagella: {self.flagella_impulse_multiplier:.2f}",
            f"Aggression: {self.aggression_multiplier:.2f}",
            f"Reproduction: {self.reproduction_desire_multiplier:.2f}",
            f"Food Growth: {self.food_growth_multiplier:.2f}",
            f"Mutation Rate: {self.mutation_rate_multiplier:.2f}",
        ]
        for text in param_text:
            surface = self.font.render(text, True, (255, 255, 255))
            panel_surface.blit(surface, (panel_padding, y_offset))
            y_offset += line_height
        
        y_offset += 5  # Spacing
        
        # Settings
        settings_text = [
            f"Zoom: {self.camera.zoom:.2f}",
            f"Terminal: {'ON' if self.show_terminal_output else 'OFF'}",
            f"Alerts: {'ON' if self.alerts_enabled else 'OFF'}",
            f"Divine Energy: {int(self.divine_energy)}/{int(self.max_divine_energy)}"
        ]
        for text in settings_text:
            surface = self.font.render(text, True, (200, 200, 200))
            panel_surface.blit(surface, (panel_padding, y_offset))
            y_offset += line_height
        
        # Blit the transparent panel to the screen
        self.screen.blit(panel_surface, (panel_x, panel_y))
    
    
    def _render_text_with_emoji(self, text: str, color: Tuple[int, int, int]) -> pygame.Surface:
        """Render text with emoji support by combining text and emoji fonts."""
        if self.emoji_font is None:
            # No emoji font available, just render with text font
            return self.alert_font.render(text, True, color)
        
        # Simple emoji detection: check if character is outside ASCII range
        # This is a basic approach - emojis are typically in Unicode ranges
        parts = []
        current_text = ""
        current_emoji = ""
        
        for char in text:
            # Check if character is likely an emoji (outside ASCII, or in emoji ranges)
            # Emojis are typically in ranges: U+1F300-1F9FF, U+2600-26FF, U+2700-27BF, etc.
            code_point = ord(char)
            is_emoji = (code_point >= 0x1F300 and code_point <= 0x1F9FF) or \
                      (code_point >= 0x2600 and code_point <= 0x26FF) or \
                      (code_point >= 0x2700 and code_point <= 0x27BF) or \
                      (code_point >= 0x1F600 and code_point <= 0x1F64F) or \
                      (code_point >= 0x1F900 and code_point <= 0x1F9FF)
            
            if is_emoji:
                if current_text:
                    parts.append(('text', current_text))
                    current_text = ""
                current_emoji += char
            else:
                if current_emoji:
                    parts.append(('emoji', current_emoji))
                    current_emoji = ""
                current_text += char
        
        # Add remaining parts
        if current_text:
            parts.append(('text', current_text))
        if current_emoji:
            parts.append(('emoji', current_emoji))
        
        # Render each part and combine
        surfaces = []
        total_width = 0
        max_height = 0
        
        for part_type, part_text in parts:
            if part_type == 'text':
                surf = self.alert_font.render(part_text, True, color)
            else:  # emoji
                surf = self.emoji_font.render(part_text, True, color)
            surfaces.append(surf)
            total_width += surf.get_width()
            max_height = max(max_height, surf.get_height())
        
        # Combine surfaces
        if not surfaces:
            return self.alert_font.render("", True, color)
        
        combined = pygame.Surface((total_width, max_height), pygame.SRCALPHA)
        x_offset = 0
        for surf in surfaces:
            # Center vertically
            y_offset = (max_height - surf.get_height()) // 2
            combined.blit(surf, (x_offset, y_offset))
            x_offset += surf.get_width()
        
        return combined
    
    def _print_comprehensive_monitoring_stats(self, deaths: int, total_energy: float, total_speed: float, avg_age: float):
        """Print comprehensive monitoring statistics to console."""
        org_count = len(self.organisms)
        food_count = len(self.foods)
        toxic_food_count = sum(1 for food in self.foods if food.toxicity > 0)
        
        avg_energy = total_energy / org_count if org_count > 0 else 0.0
        avg_speed = total_speed / org_count if org_count > 0 else 0.0
        avg_age_val = avg_age / org_count if org_count > 0 else 0.0
        
        # Calculate FPS
        fps = self.frame_count / self.monitor_interval if self.monitor_interval > 0 else 0
        self.frame_count = 0
        
        # Calculate evolution metrics (trait averages and ranges)
        trait_stats = {}
        if org_count > 0:
            traits = ['size', 'aggression', 'reproduction_desire', 'toxicity_resistance', 
                     'food_preference', 'vision_range', 'metabolism', 'max_speed',
                     'flagella_count', 'flagella_length', 'energy_efficiency']
            
            for trait in traits:
                values = [getattr(org.dna, trait) for org in self.organisms]
                trait_stats[trait] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values)) if len(values) > 1 else 0.0
                }
        
        # Count organisms in combat
        in_combat = sum(1 for org in self.organisms if org.in_combat)
        
        # Age distribution
        ages = [org.age for org in self.organisms]
        max_age = max(ages) if ages else 0.0
        min_age = min(ages) if ages else 0.0
        
        # Energy distribution
        energies = [org.energy for org in self.organisms]
        min_energy = min(energies) if energies else 0.0
        max_energy = max(energies) if energies else 0.0
        
        # Neural network stats
        nn_stats = {}
        if org_count > 0:
            hidden_layers = [org.dna.nn_hidden_layers for org in self.organisms]
            neurons = [org.dna.nn_neurons_per_layer for org in self.organisms]
            learning_rates = [org.dna.nn_learning_rate for org in self.organisms]
            nn_stats = {
                'avg_hidden_layers': sum(hidden_layers) / len(hidden_layers),
                'avg_neurons': sum(neurons) / len(neurons),
                'avg_learning_rate': sum(learning_rates) / len(learning_rates),
            }
        
        # Print comprehensive stats
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE SIMULATION MONITORING - Time: {self.total_time:.1f}s | FPS: {fps:.1f}")
        print(f"{'='*100}")
        
        # Population Overview
        print(f"\n📊 POPULATION OVERVIEW")
        print(f"  Organisms: {org_count:4d} | Food: {food_count:4d} (Toxic: {toxic_food_count:4d}, {toxic_food_count/food_count*100:.1f}%)")
        print(f"  In Combat: {in_combat:4d} | Avg Energy: {avg_energy:6.1f} | Avg Speed: {avg_speed:5.2f} | Avg Age: {avg_age_val:5.1f}s")
        print(f"  Age Range: {min_age:.1f}s - {max_age:.1f}s | Energy Range: {min_energy:.1f} - {max_energy:.1f}")
        
        # Life Events (this interval)
        print(f"\n⚡ LIFE EVENTS (Last {self.monitor_interval:.1f}s)")
        print(f"  Births: {self.stats['births']:3d} (All from Mating: {self.stats['births_sexual']:3d})")
        print(f"  Deaths: {self.stats['deaths']:3d} (Combat: {self.stats['deaths_combat']:3d}, Starvation: {self.stats['deaths_starvation']:3d}, Toxic: {self.stats['deaths_toxic']:3d})")
        print(f"  Matings: {self.stats['matings']:3d} | Fights: {self.stats['fights']:3d}")
        print(f"  Food Eaten: {self.stats['food_eaten']:3d} (Toxic: {self.stats['toxic_food_eaten']:3d}) | Food Reproduced: {self.stats['food_reproduced']:3d}")
        
        # Cumulative Statistics
        print(f"\n📈 CUMULATIVE STATISTICS (Total)")
        print(f"  Total Births: {self.cumulative_stats['total_births']:6d} | Total Deaths: {self.cumulative_stats['total_deaths']:6d}")
        print(f"  Total Matings: {self.cumulative_stats['total_matings']:6d} | Total Fights: {self.cumulative_stats['total_fights']:6d}")
        print(f"  Total Food Eaten: {self.cumulative_stats['total_food_eaten']:6d}")
        
        # Evolution Metrics
        if org_count > 0:
            print(f"\n🧬 EVOLUTION METRICS (Trait Averages & Ranges)")
            print(f"  Size:           Avg={trait_stats['size']['avg']:5.1f} (Range: {trait_stats['size']['min']:.1f}-{trait_stats['size']['max']:.1f}, Std: {trait_stats['size']['std']:.2f})")
            print(f"  Aggression:     Avg={trait_stats['aggression']['avg']:5.3f} (Range: {trait_stats['aggression']['min']:.3f}-{trait_stats['aggression']['max']:.3f})")
            print(f"  Repro Desire:   Avg={trait_stats['reproduction_desire']['avg']:5.3f} (Range: {trait_stats['reproduction_desire']['min']:.3f}-{trait_stats['reproduction_desire']['max']:.3f})")
            print(f"  Tox Resistance: Avg={trait_stats['toxicity_resistance']['avg']:5.3f} (Range: {trait_stats['toxicity_resistance']['min']:.3f}-{trait_stats['toxicity_resistance']['max']:.3f})")
            print(f"  Food Pref:      Avg={trait_stats['food_preference']['avg']:5.3f} (0=herbivore, 1=carnivore)")
            print(f"  Vision Range:   Avg={trait_stats['vision_range']['avg']:5.1f} (Range: {trait_stats['vision_range']['min']:.1f}-{trait_stats['vision_range']['max']:.1f})")
            print(f"  Metabolism:     Avg={trait_stats['metabolism']['avg']:6.4f} (Range: {trait_stats['metabolism']['min']:.4f}-{trait_stats['metabolism']['max']:.4f})")
            print(f"  Max Speed:      Avg={trait_stats['max_speed']['avg']:5.2f} (Range: {trait_stats['max_speed']['min']:.2f}-{trait_stats['max_speed']['max']:.2f})")
            print(f"  Flagella Count: Avg={trait_stats['flagella_count']['avg']:5.1f} (Range: {int(trait_stats['flagella_count']['min'])}-{int(trait_stats['flagella_count']['max'])})")
            print(f"  Flagella Length: Avg={trait_stats['flagella_length']['avg']:5.1f} (Range: {trait_stats['flagella_length']['min']:.1f}-{trait_stats['flagella_length']['max']:.1f})")
            print(f"  Energy Efficiency: Avg={trait_stats['energy_efficiency']['avg']:5.3f} (Range: {trait_stats['energy_efficiency']['min']:.3f}-{trait_stats['energy_efficiency']['max']:.3f})")
        
        # Neural Network Evolution
        if org_count > 0:
            print(f"\n🧠 NEURAL NETWORK EVOLUTION")
            print(f"  Avg Hidden Layers: {nn_stats['avg_hidden_layers']:.2f} | Avg Neurons/Layer: {nn_stats['avg_neurons']:.2f}")
            print(f"  Avg Learning Rate: {nn_stats['avg_learning_rate']:.4f}")
        
        # Food Ecosystem
        if food_count > 0:
            toxic_food_avg_toxicity = sum(food.toxicity for food in self.foods if food.toxicity > 0) / toxic_food_count if toxic_food_count > 0 else 0.0
            avg_food_age = sum(food.age for food in self.foods) / food_count
            print(f"\n🌱 FOOD ECOSYSTEM")
            print(f"  Total Food: {food_count:4d} | Toxic Food: {toxic_food_count:4d} ({toxic_food_count/food_count*100:.1f}%)")
            if toxic_food_count > 0:
                print(f"  Avg Toxicity (toxic food): {toxic_food_avg_toxicity:.3f}")
            print(f"  Avg Food Age: {avg_food_age:.1f}s")
        
        # Performance
        print(f"\n⚙️  PERFORMANCE")
        print(f"  FPS: {fps:.1f} | Zoom: {self.camera.zoom:.2f} | Camera: ({self.camera.x:.0f}, {self.camera.y:.0f})")
        
        print(f"{'='*100}\n")

    def _control_population(self):
        """AI-driven population control using learned parameter adjustments."""
        current_population = len(self.organisms)

        # Track population history
        prev_population = self.population_history[-1] if self.population_history else current_population
        self.population_history.append(current_population)
        if len(self.population_history) > 6:
            self.population_history.pop(0)

        # Get current state
        current_state = self.ai_controller.get_state(self)

        # Learn from previous action if we have previous state
        if self.ai_controller.prev_state is not None:
            reward = self.ai_controller.calculate_reward(self, prev_population, current_population)
            self.ai_controller.learn(self.ai_controller.prev_state, self.ai_controller.prev_actions, reward, current_state)

        # Choose new actions
        actions = self.ai_controller.choose_actions(current_state)

        # Apply actions
        self.ai_controller.apply_actions(self, actions)

        # Store current state and actions for next learning cycle
        self.ai_controller.prev_state = current_state
        self.ai_controller.prev_actions = actions

        # Periodic experience replay training
        if len(self.ai_controller.memory) >= 50 and random.random() < 0.3:
            self.ai_controller.train_on_experience()

        # Terminal output
        if self.show_terminal_output:
            action_names = []
            if actions[0] != 0: action_names.append(f"Metabolism {'↑' if actions[0] == 1 else '↓'}")
            if actions[1] != 0: action_names.append(f"Flagella {'↑' if actions[1] == 1 else '↓'}")
            if actions[2] != 0: action_names.append(f"Aggression {'↑' if actions[2] == 1 else '↓'}")
            if actions[3] != 0: action_names.append(f"Reproduction {'↑' if actions[3] == 1 else '↓'}")

            if action_names:
                print(f"🤖 AI Control: {current_population} organisms - {', '.join(action_names)}")
            else:
                print(f"🤖 AI Control: {current_population} organisms - No changes")

    def _draw_graph_panel(self, screen_width: int, screen_height: int):
        """Draw a graph panel at the bottom of the screen showing various metrics over time."""
        panel_height = 150
        panel_y = screen_height - panel_height
        margin = 10
        graph_width = screen_width - 2 * margin
        graph_height = panel_height - 40  # Leave space for labels

        # Create transparent surface for panel
        panel_surface = pygame.Surface((screen_width, panel_height), pygame.SRCALPHA)
        panel_surface.set_alpha(220)  # Semi-transparent (220/255 opacity)

        # Draw panel background on transparent surface
        panel_rect = pygame.Rect(0, 0, screen_width, panel_height)
        pygame.draw.rect(panel_surface, (30, 40, 60), panel_rect)
        pygame.draw.rect(panel_surface, (100, 100, 100), panel_rect, 2)

        # Check if we have data
        if not self.graph_data['population']:
            # Still blit the empty panel
            self.screen.blit(panel_surface, (0, panel_y))
            return

        # Define graph lines with colors
        graph_lines = [
            ('population', (255, 100, 100), 0, 100, 'Population'),
            ('metabolism', (100, 255, 100), 0.5, 1.5, 'Metabolism'),
            ('flagella', (100, 150, 255), 0.5, 2.0, 'Flagella'),
            ('aggression', (255, 200, 100), 0.5, 2.0, 'Aggression'),
            ('reproduction', (255, 100, 255), 0.3, 2.0, 'Reproduction'),
            ('food_growth', (50, 255, 200), 0.1, 3.0, 'Food Growth'),
            ('mutation_rate', (255, 50, 200), 0.1, 3.0, 'Mutation'),
            ('food_count', (150, 255, 150), 0, 350, 'Food'),
            ('avg_energy', (200, 200, 255), 0, 150, 'Avg Energy'),
        ]

        # Draw each line on the panel surface
        for line_key, color, min_val, max_val, label in graph_lines:
            data = self.graph_data[line_key]
            if len(data) < 2:
                continue

            # Normalize data to graph height
            value_range = max_val - min_val
            if value_range == 0:
                continue

            points = []
            for i, value in enumerate(data):
                x = margin + (i / max(len(data) - 1, 1)) * graph_width
                normalized = (value - min_val) / value_range
                normalized = max(0, min(1, normalized))  # Clamp to 0-1
                y = graph_height - (normalized * graph_height)
                points.append((int(x), int(y)))

            # Draw line on panel surface
            if len(points) > 1:
                pygame.draw.lines(panel_surface, color, False, points, 2)

        # Draw labels on the right side
        label_y = 5
        label_x = screen_width - 120
        small_font = pygame.font.Font(None, 16)
        
        for line_key, color, min_val, max_val, label in graph_lines:
            if len(self.graph_data[line_key]) > 0:
                current_val = self.graph_data[line_key][-1]
                label_text = f"{label}: {current_val:.2f}"
                label_surface = small_font.render(label_text, True, color)
                panel_surface.blit(label_surface, (label_x, label_y))
                label_y += 18

        # Draw time axis label
        time_label = small_font.render("Time →", True, (200, 200, 200))
        panel_surface.blit(time_label, (margin, graph_height + 5))

        # Blit the transparent panel to the screen
        self.screen.blit(panel_surface, (0, panel_y))

    def render(self):
        """Render the simulation."""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Get current screen dimensions
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Draw food
        for food in self.foods:
            sx, sy = self.camera.world_to_screen(food.x, food.y, screen_width, screen_height)
            if -10 < sx < screen_width + 10 and -10 < sy < screen_height + 10:
                # Color food based on toxicity
                if food.toxicity > 0:
                    # Toxic food: red/purple tint (more toxic = more red)
                    toxicity_factor = food.toxicity
                    toxic_color = (
                        int(100 + toxicity_factor * 155),  # Red component
                        int(200 - toxicity_factor * 100),  # Green component (less green = more toxic)
                        int(100 - toxicity_factor * 50)    # Blue component
                    )
                    pygame.draw.circle(self.screen, toxic_color, (sx, sy), int(food.size * self.camera.zoom))
                    # Draw a warning ring around toxic food
                    pygame.draw.circle(self.screen, (255, 0, 0), (sx, sy), int(food.size * self.camera.zoom), 1)
                else:
                    # Normal food: green
                    pygame.draw.circle(self.screen, FOOD_COLOR, (sx, sy), int(food.size * self.camera.zoom))
        
        # Draw organisms
        champion_org = None
        if self.organisms:
            # Find the "Champion" (based on age, energy, children)
            # Simple heuristic: oldest organism with good energy is the champion
            champion_org = max(self.organisms, key=lambda o: o.age * 0.5 + o.energy * 0.1)

        for org in self.organisms:
            sx, sy = self.camera.world_to_screen(org.x, org.y, screen_width, screen_height)
            if -50 < sx < screen_width + 50 and -50 < sy < screen_height + 50:
                # Draw champion halo
                if org is champion_org:
                    # Gold pulsating halo
                    pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) * 0.5  # 0 to 1
                    halo_size = int((org.size * 1.5 + pulse * 5) * self.camera.zoom)
                    pygame.draw.circle(self.screen, (255, 215, 0, 100), (sx, sy), halo_size, 2)

                # Draw flagella first (behind organism)
                flagella_points_list = org.get_flagella_points()
                flagella_color = tuple(max(0, min(255, c - 30)) for c in org.color)  # Slightly darker than body
                
                for flagellum_points in flagella_points_list:
                    # Convert to screen coordinates
                    screen_flagellum = [self.camera.world_to_screen(px, py, screen_width, screen_height) for px, py in flagellum_points]
                    # Draw flagellum as a smooth line
                    if len(screen_flagellum) > 1:
                        thickness = int(org.dna.flagella_thickness * self.camera.zoom)
                        thickness = max(1, thickness)  # At least 1 pixel
                        pygame.draw.lines(self.screen, flagella_color, False, screen_flagellum, thickness)
                
                # Draw organism shape
                points = org.get_shape()
                screen_points = [self.camera.world_to_screen(px, py, screen_width, screen_height) for px, py in points]
                pygame.draw.polygon(self.screen, org.color, screen_points)
                # Draw outline
                pygame.draw.polygon(self.screen, (255, 255, 255), screen_points, 1)
                
                # Draw state indicator circle at center (priority: fighting > mating > feeding > avoiding > running > idle)
                state_color = (128, 128, 128)  # Default gray (idle)
                if org.in_combat or (hasattr(org, 'neural_fight') and org.neural_fight):
                    state_color = (255, 0, 0)  # Red - Fighting
                elif (hasattr(org, 'neural_mate') and org.neural_mate) or (org.dna.reproduction_desire > 0.4):
                    state_color = (255, 0, 255)  # Magenta - Seeking mate
                elif hasattr(org, 'neural_feed') and org.neural_feed:
                    state_color = (255, 255, 0)  # Yellow - Seeking food
                elif hasattr(org, 'neural_avoid_toxic') and org.neural_avoid_toxic:
                    state_color = (255, 165, 0)  # Orange - Avoiding toxic food
                elif hasattr(org, 'neural_run') and org.neural_run:
                    state_color = (0, 255, 255)  # Cyan - Running away
                
                # Draw state indicator circle
                indicator_radius = max(3, int(5 * self.camera.zoom))
                pygame.draw.circle(self.screen, state_color, (sx, sy), indicator_radius)
                # Draw a subtle outline for visibility
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), indicator_radius, 1)
                
                # Draw eyes (visual sensors for finding food)
                eye_positions = org.get_eye_positions()
                for eye_x, eye_y in eye_positions:
                    eye_sx, eye_sy = self.camera.world_to_screen(eye_x, eye_y, screen_width, screen_height)
                    eye_radius = max(2, int(3 * self.camera.zoom))
                    # Draw eye (white with black pupil)
                    pygame.draw.circle(self.screen, (255, 255, 255), (eye_sx, eye_sy), eye_radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), (eye_sx, eye_sy), max(1, int(eye_radius * 0.6)))
                
                # Draw energy bar
                bar_width = int(org.size * 2 * self.camera.zoom)
                bar_height = 3
                bar_x = sx - bar_width // 2
                bar_y = sy - int(org.size * self.camera.zoom) - 10
                energy_ratio = org.energy / org.max_energy
                pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        
        # Draw alert if active
        # Draw hazards
        for hazard in self.hazards:
            sx, sy = self.camera.world_to_screen(hazard.x, hazard.y, screen_width, screen_height)
            radius = int(hazard.radius * self.camera.zoom)
            
            # Draw semi-transparent red circle
            hazard_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(hazard_surface, (255, 50, 50, 100), (radius, radius), radius)
            # Pulsing effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
            inner_radius = int(radius * (0.8 + pulse * 0.2))
            pygame.draw.circle(hazard_surface, (255, 0, 0, 150), (radius, radius), inner_radius, 2)
            
            self.screen.blit(hazard_surface, (sx - radius, sy - radius))

        # Draw consolidated UI panel
        self._draw_ui_panel(screen_width, screen_height)
        
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            pause_x = screen_width // 2 - pause_text.get_width() // 2
            self.screen.blit(pause_text, (pause_x, 10))

        # Draw graph panel at the bottom
        self._draw_graph_panel(screen_width, screen_height)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*80)
        print("Virtual Life Simulation - Starting...")
        print("="*80)
        print("Controls:")
        print("  SPACE - Pause/Unpause")
        print("  F - Add Food")
        print("  O - Add Organisms")
        print("  R - Reset Simulation (includes AI)")
        print("  S - Save AI Weights Manually")
        print("  M - Toggle Terminal Output")
        print("  ESC - Exit")
        print("  Mouse Wheel - Zoom")
        print("  Left Click + Drag - Pan View")
        print("="*80 + "\n")
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            self.frame_count += 1
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        print("\n" + "="*80)
        print("Simulation Ended")
        print("="*80)

        # Save AI weights on exit
        print("💾 Saving AI controller weights...")
        self.ai_controller.save_weights()
        print("✅ AI weights saved successfully")

        pygame.quit()

