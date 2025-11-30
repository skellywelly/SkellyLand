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

# Neural Network Constants
INPUT_SIZE = 19  # Number of input neurons (sensors) - increased for toxicity, mate quality, and overcrowding
# OUTPUT_SIZE is now variable: flagella_count + 3 (turn_left, turn_right, reproduction)

class NeuralNetwork:
    """Neural network that controls organism behavior. Can learn and adapt."""
    
    def __init__(self, hidden_layers: int, neurons_per_layer: int, learning_rate: float, 
                 output_size: int, initial_weights: Optional[List[np.ndarray]] = None):
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.output_size = output_size  # Variable output size (flagella_count + 3)
        
        # Initialize weights if not provided
        if initial_weights is None:
            self.weights = self._initialize_weights()
        else:
            self.weights = [w.copy() for w in initial_weights]
        
        # Experience buffer for learning
        self.experience_buffer = []
        self.max_buffer_size = 50
    
    def _initialize_weights(self):
        """Initialize network weights randomly."""
        weights = []
        
        # Input to first hidden layer
        weights.append(np.random.randn(INPUT_SIZE, self.neurons_per_layer) * 0.5)
        
        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            weights.append(np.random.randn(self.neurons_per_layer, self.neurons_per_layer) * 0.5)
        
        # Last hidden to output (variable size)
        weights.append(np.random.randn(self.neurons_per_layer, self.output_size) * 0.5)
        
        return weights
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Normalize inputs to [-1, 1]
        inputs = np.clip(inputs, -1, 1)
        
        activation = inputs
        for i, weight in enumerate(self.weights):
            activation = np.dot(activation, weight)
            if i < len(self.weights) - 1:  # Not the last layer
                # ReLU activation for hidden layers
                activation = np.maximum(0, activation)
            else:
                # Tanh for output layer (bounded outputs)
                activation = np.tanh(activation)
        
        return activation
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Get network predictions (same as forward)."""
        return self.forward(inputs)
    
    def train_step(self, inputs: np.ndarray, target_outputs: np.ndarray):
        """Single training step using gradient descent."""
        # Forward pass
        activations = [inputs]
        for i, weight in enumerate(self.weights):
            z = np.dot(activations[-1], weight)
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = np.tanh(z)  # Tanh
            activations.append(activation)
        
        output = activations[-1]
        error = target_outputs - output
        
        # Backward pass (simplified gradient descent)
        gradients = []
        delta = error * (1 - output ** 2)  # Derivative of tanh
        
        for i in range(len(self.weights) - 1, -1, -1):
            grad = np.outer(activations[i], delta)
            gradients.insert(0, grad)
            
            if i > 0:
                # Backpropagate through ReLU
                relu_derivative = (activations[i] > 0).astype(float)
                delta = np.dot(delta, self.weights[i].T) * relu_derivative
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * gradients[i]
            # Clip weights to prevent explosion
            self.weights[i] = np.clip(self.weights[i], -5, 5)
    
    def learn_from_experience(self, reward: float, inputs: np.ndarray, outputs: np.ndarray):
        """Learn from experience using reward signal."""
        # Store experience
        self.experience_buffer.append({
            'inputs': inputs.copy(),
            'outputs': outputs.copy(),
            'reward': reward
        })
        
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Train on recent experiences with high rewards
        if len(self.experience_buffer) >= 5:
            # Get experiences with positive rewards
            positive_experiences = [e for e in self.experience_buffer if e['reward'] > 0]
            
            if positive_experiences:
                # Train on best experiences
                best_experience = max(positive_experiences, key=lambda x: x['reward'])
                # Create target: reinforce the outputs that led to positive reward
                target = best_experience['outputs'] + reward * 0.1
                target = np.clip(target, -1, 1)
                self.train_step(best_experience['inputs'], target)
    
    def get_weights_copy(self) -> List[np.ndarray]:
        """Get a copy of the weights (for reproduction)."""
        return [w.copy() for w in self.weights]
    
    def mutate_weights(self, mutation_rate: float = 0.1):
        """Randomly mutate network weights."""
        for weight in self.weights:
            mask = np.random.random(weight.shape) < mutation_rate
            weight[mask] += np.random.randn(*weight.shape)[mask] * 0.1

class DNA:
    """Digital DNA that describes an organism's characteristics."""
    
    def __init__(self, parent_dna: Optional['DNA'] = None, parent2_dna: Optional['DNA'] = None):
        if parent_dna is None:
            # Create random DNA for first generation
            self.food_preference = random.uniform(0, 1)  # 0 = herbivore, 1 = carnivore
            self.color_r = random.randint(50, 255)
            self.color_g = random.randint(50, 255)
            self.color_b = random.randint(50, 255)
            self.size = random.uniform(20, 50)  # Larger organisms
            # Complex irregular shape system
            self.shape_point_count = random.randint(4, 10)  # Number of points (4-10 for complexity)
            # Each point has a radius multiplier (0.5 to 1.5) and angle offset (-0.3 to 0.3)
            # This creates irregular, complex shapes
            self.shape_radii = [random.uniform(0.6, 1.4) for _ in range(self.shape_point_count)]
            self.shape_angle_offsets = [random.uniform(-0.2, 0.2) for _ in range(self.shape_point_count)]
            # Flagella properties
            self.flagella_count = random.randint(1, 4)  # Number of flagella
            self.flagella_length = random.uniform(15, 40)  # Length of flagella
            self.flagella_thickness = random.uniform(1.5, 3.5)  # Thickness of flagella
            self.flagella_beat_frequency = random.uniform(2.0, 8.0)  # Beats per second
            self.flagella_wave_amplitude = random.uniform(0.3, 0.8)  # Wave amplitude (0-1)
            self.flagella_wave_length = random.uniform(0.5, 2.0)  # Wave length factor
            # Propulsion (now based on flagella)
            self.propulsion_strength = random.uniform(0.8, 4.0)  # Overall propulsion strength (increased from 0.5-3.0)
            self.propulsion_efficiency = random.uniform(0.3, 0.9)
            self.max_speed = random.uniform(1.5, 6.0)  # Increased from 1.0-5.0
            self.energy_efficiency = random.uniform(0.5, 1.5)
            self.metabolism = random.uniform(0.002, 0.008)  # Much lower metabolism - live longer
            self.vision_range = random.uniform(50, 200)
            self.reproduction_threshold = random.uniform(50, 100)
            self.aggression = random.uniform(0, 1)
            self.reproduction_desire = random.uniform(0, 1)  # Desire to mate (0 = low, 1 = high)
            self.toxicity_resistance = random.uniform(0, 1)  # Resistance to toxic food (0 = none, 1 = full)
            self.min_mating_age = random.uniform(10, 30)  # Minimum age in seconds before can mate (10-30 seconds)
            self.overcrowding_threshold_base = random.uniform(3, 10)  # Base threshold for overcrowding (organism count)
            self.overcrowding_distance_base = random.uniform(50, 300)  # Base distance to check for overcrowding
            # Neural network architecture (DNA-controlled)
            self.nn_hidden_layers = random.randint(1, 3)  # Number of hidden layers
            self.nn_neurons_per_layer = random.randint(4, 12)  # Neurons per hidden layer
            self.nn_learning_rate = random.uniform(0.01, 0.1)  # Learning rate
        else:
            # Combine parent DNA with mutation
            if parent2_dna is None:
                # Single parent (shouldn't happen in normal reproduction, but handle gracefully)
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
        self.shape_point_count = parent.shape_point_count
        self.shape_radii = parent.shape_radii.copy()
        self.shape_angle_offsets = parent.shape_angle_offsets.copy()
        self.flagella_count = parent.flagella_count
        self.flagella_length = parent.flagella_length
        self.flagella_thickness = parent.flagella_thickness
        self.flagella_beat_frequency = parent.flagella_beat_frequency
        self.flagella_wave_amplitude = parent.flagella_wave_amplitude
        self.flagella_wave_length = parent.flagella_wave_length
        self.propulsion_strength = parent.propulsion_strength
        self.propulsion_efficiency = parent.propulsion_efficiency
        self.max_speed = parent.max_speed
        self.energy_efficiency = parent.energy_efficiency
        self.metabolism = parent.metabolism
        self.vision_range = parent.vision_range
        self.reproduction_threshold = parent.reproduction_threshold
        self.aggression = parent.aggression
        self.reproduction_desire = parent.reproduction_desire
        self.toxicity_resistance = parent.toxicity_resistance
        self.min_mating_age = parent.min_mating_age
        self.overcrowding_threshold_base = parent.overcrowding_threshold_base
        self.overcrowding_distance_base = parent.overcrowding_distance_base
        self.nn_hidden_layers = parent.nn_hidden_layers
        self.nn_neurons_per_layer = parent.nn_neurons_per_layer
        self.nn_learning_rate = parent.nn_learning_rate
    
    def _combine_parents(self, parent1: 'DNA', parent2: 'DNA'):
        """Combine traits from two parents (randomly choose or average)."""
        # Some traits are inherited from one parent, others are averaged
        self.food_preference = random.choice([parent1.food_preference, parent2.food_preference])
        self.color_r = int((parent1.color_r + parent2.color_r) / 2)
        self.color_g = int((parent1.color_g + parent2.color_g) / 2)
        self.color_b = int((parent1.color_b + parent2.color_b) / 2)
        self.size = (parent1.size + parent2.size) / 2
        # Combine shape parameters from both parents
        # Use the point count from one parent, then blend radii and offsets
        self.shape_point_count = random.choice([parent1.shape_point_count, parent2.shape_point_count])
        max_points = max(parent1.shape_point_count, parent2.shape_point_count)
        
        # Blend radii and offsets, handling different point counts
        self.shape_radii = []
        self.shape_angle_offsets = []
        for i in range(self.shape_point_count):
            # Interpolate between parents' values
            idx1 = i % parent1.shape_point_count
            idx2 = i % parent2.shape_point_count
            self.shape_radii.append((parent1.shape_radii[idx1] + parent2.shape_radii[idx2]) / 2)
            self.shape_angle_offsets.append((parent1.shape_angle_offsets[idx1] + parent2.shape_angle_offsets[idx2]) / 2)
        self.flagella_count = random.choice([parent1.flagella_count, parent2.flagella_count])
        self.flagella_length = (parent1.flagella_length + parent2.flagella_length) / 2
        self.flagella_thickness = (parent1.flagella_thickness + parent2.flagella_thickness) / 2
        self.flagella_beat_frequency = (parent1.flagella_beat_frequency + parent2.flagella_beat_frequency) / 2
        self.flagella_wave_amplitude = (parent1.flagella_wave_amplitude + parent2.flagella_wave_amplitude) / 2
        self.flagella_wave_length = (parent1.flagella_wave_length + parent2.flagella_wave_length) / 2
        self.propulsion_strength = (parent1.propulsion_strength + parent2.propulsion_strength) / 2
        self.propulsion_efficiency = (parent1.propulsion_efficiency + parent2.propulsion_efficiency) / 2
        self.max_speed = (parent1.max_speed + parent2.max_speed) / 2
        self.energy_efficiency = (parent1.energy_efficiency + parent2.energy_efficiency) / 2
        self.metabolism = (parent1.metabolism + parent2.metabolism) / 2
        self.vision_range = (parent1.vision_range + parent2.vision_range) / 2
        self.reproduction_threshold = (parent1.reproduction_threshold + parent2.reproduction_threshold) / 2
        self.aggression = (parent1.aggression + parent2.aggression) / 2
        self.reproduction_desire = (parent1.reproduction_desire + parent2.reproduction_desire) / 2
        self.toxicity_resistance = (parent1.toxicity_resistance + parent2.toxicity_resistance) / 2
        self.min_mating_age = (parent1.min_mating_age + parent2.min_mating_age) / 2
        self.overcrowding_threshold_base = (parent1.overcrowding_threshold_base + parent2.overcrowding_threshold_base) / 2
        self.overcrowding_distance_base = (parent1.overcrowding_distance_base + parent2.overcrowding_distance_base) / 2
        self.nn_hidden_layers = random.choice([parent1.nn_hidden_layers, parent2.nn_hidden_layers])
        self.nn_neurons_per_layer = random.choice([parent1.nn_neurons_per_layer, parent2.nn_neurons_per_layer])
        self.nn_learning_rate = (parent1.nn_learning_rate + parent2.nn_learning_rate) / 2
    
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
            self.size = np.clip(self.size + random.uniform(-3, 3), 15, 60)
        
        # Mutate shape complexity
        if random.random() < mutation_rate:
            # Change point count (add or remove points)
            change = random.randint(-1, 1)
            new_count = int(np.clip(self.shape_point_count + change, 4, 12))
            if new_count != self.shape_point_count:
                if new_count > self.shape_point_count:
                    # Add new points with random values
                    for _ in range(new_count - self.shape_point_count):
                        self.shape_radii.append(random.uniform(0.6, 1.4))
                        self.shape_angle_offsets.append(random.uniform(-0.2, 0.2))
                else:
                    # Remove points
                    self.shape_radii = self.shape_radii[:new_count]
                    self.shape_angle_offsets = self.shape_angle_offsets[:new_count]
                self.shape_point_count = new_count
        
        # Mutate individual point radii (creates irregularity)
        if random.random() < mutation_rate:
            # Mutate a random point's radius
            idx = random.randint(0, len(self.shape_radii) - 1)
            self.shape_radii[idx] = np.clip(self.shape_radii[idx] + random.uniform(-0.2, 0.2), 0.3, 1.7)
        
        # Mutate individual point angle offsets (creates more irregularity)
        if random.random() < mutation_rate:
            # Mutate a random point's angle offset
            idx = random.randint(0, len(self.shape_angle_offsets) - 1)
            self.shape_angle_offsets[idx] = np.clip(self.shape_angle_offsets[idx] + random.uniform(-0.1, 0.1), -0.4, 0.4)
        
        if random.random() < mutation_rate:
            self.flagella_count = int(np.clip(self.flagella_count + random.randint(-1, 1), 1, 6))
        
        if random.random() < mutation_rate:
            self.flagella_length = np.clip(self.flagella_length + random.uniform(-5, 5), 10, 50)
        
        if random.random() < mutation_rate:
            self.flagella_thickness = np.clip(self.flagella_thickness + random.uniform(-0.5, 0.5), 1.0, 5.0)
        
        if random.random() < mutation_rate:
            self.flagella_beat_frequency = np.clip(self.flagella_beat_frequency + random.uniform(-1.0, 1.0), 1.0, 12.0)
        
        if random.random() < mutation_rate:
            self.flagella_wave_amplitude = np.clip(self.flagella_wave_amplitude + random.uniform(-0.1, 0.1), 0.1, 1.0)
        
        if random.random() < mutation_rate:
            self.flagella_wave_length = np.clip(self.flagella_wave_length + random.uniform(-0.3, 0.3), 0.3, 3.0)
        
        if random.random() < mutation_rate:
            self.propulsion_strength = np.clip(self.propulsion_strength + random.uniform(-0.3, 0.3), 0.5, 5.0)  # Increased max from 4.0 to 5.0
        
        if random.random() < mutation_rate:
            self.propulsion_efficiency = np.clip(self.propulsion_efficiency + random.uniform(-0.1, 0.1), 0.2, 1.0)
        
        if random.random() < mutation_rate:
            self.max_speed = np.clip(self.max_speed + random.uniform(-0.5, 0.5), 0.5, 7.0)  # Increased max from 6.0 to 7.0
        
        if random.random() < mutation_rate:
            self.energy_efficiency = np.clip(self.energy_efficiency + random.uniform(-0.2, 0.2), 0.3, 2.0)
        
        if random.random() < mutation_rate:
            self.metabolism = np.clip(self.metabolism + random.uniform(-0.002, 0.002), 0.001, 0.015)
        
        if random.random() < mutation_rate:
            self.vision_range = np.clip(self.vision_range + random.uniform(-20, 20), 30, 250)
        
        if random.random() < mutation_rate:
            self.reproduction_threshold = np.clip(self.reproduction_threshold + random.uniform(-10, 10), 30, 150)
        
        if random.random() < mutation_rate:
            self.aggression = np.clip(self.aggression + random.uniform(-0.2, 0.2), 0, 1)
        
        if random.random() < mutation_rate:
            self.reproduction_desire = np.clip(self.reproduction_desire + random.uniform(-0.2, 0.2), 0, 1)
        
        if random.random() < mutation_rate:
            self.toxicity_resistance = np.clip(self.toxicity_resistance + random.uniform(-0.2, 0.2), 0, 1)
        
        if random.random() < mutation_rate:
            self.min_mating_age = np.clip(self.min_mating_age + random.uniform(-2, 2), 5, 40)  # Can mutate between 5-40 seconds
        
        if random.random() < mutation_rate:
            self.overcrowding_threshold_base = np.clip(self.overcrowding_threshold_base + random.uniform(-1, 1), 2, 15)
        
        if random.random() < mutation_rate:
            self.overcrowding_distance_base = np.clip(self.overcrowding_distance_base + random.uniform(-20, 20), 30, 400)
        
        if random.random() < mutation_rate:
            self.nn_hidden_layers = int(np.clip(self.nn_hidden_layers + random.randint(-1, 1), 1, 4))
        
        if random.random() < mutation_rate:
            self.nn_neurons_per_layer = int(np.clip(self.nn_neurons_per_layer + random.randint(-2, 2), 3, 16))
        
        if random.random() < mutation_rate:
            self.nn_learning_rate = np.clip(self.nn_learning_rate + random.uniform(-0.02, 0.02), 0.005, 0.2)

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
    
    def update(self, dt: float, foods: List['Food'], world_width: float, world_height: float) -> List['Food']:
        """Update food state and handle reproduction. Returns list of new food."""
        self.age += dt
        self.reproduction_timer += dt
        
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

class Organism:
    """An organism with DNA-based characteristics."""
    
    def __init__(self, x: float, y: float, dna: Optional[DNA] = None):
        self.x = x
        self.y = y
        self.dna = dna if dna else DNA()
        
        # Physical properties from DNA
        self.size = self.dna.size
        self.color = (self.dna.color_r, self.dna.color_g, self.dna.color_b)
        self.shape_point_count = self.dna.shape_point_count
        self.shape_radii = self.dna.shape_radii.copy()
        self.shape_angle_offsets = self.dna.shape_angle_offsets.copy()
        
        # Movement properties
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(0, 2 * math.pi)
        self.rotation_speed = random.uniform(-0.1, 0.1)
        
        # Flagella animation state - one phase per flagellum
        self.flagella_phases = [random.uniform(0, 2 * math.pi) for _ in range(self.dna.flagella_count)]
        
        # Energy and life
        self.energy = random.uniform(50, 80)  # Start with more energy
        self.max_energy = 150  # Higher max energy
        self.age = 0
        
        # Behavior
        self.target = None  # Target food or organism
        self.last_reproduction = 0
        self.in_combat = False  # Whether currently fighting
        self.combat_target = None  # Organism being fought
        self.last_interaction = 0  # Time since last interaction (prevent spam)
        self._death_cause = None  # Track cause of death for statistics
        
        # Neural network (DNA-controlled architecture)
        # Output size: turn_left, turn_right, flagella_count outputs, mate, fight, run, chase, feed, avoid_toxic, reproduction, overcrowding_threshold_mod, overcrowding_distance_mod
        # Total: 2 + flagella_count + 9 = flagella_count + 11
        output_size = self.dna.flagella_count + 11
        self.brain = NeuralNetwork(
            hidden_layers=self.dna.nn_hidden_layers,
            neurons_per_layer=self.dna.nn_neurons_per_layer,
            learning_rate=self.dna.nn_learning_rate,
            output_size=output_size
        )
        
        # Learning state
        self.last_inputs = None
        self.last_outputs = None
        self.last_reward = 0.0
        self.learning_timer = 0.0
        self.backprop_timer = 0.0
        self.backprop_interval = 3.0  # Backpropagate every 3 seconds
        self.experience_history = []  # Store recent experiences for batch learning
        self.energy_history = []  # Track energy over time
        self.food_found_count = 0  # Track food finding success
        self.last_food_eaten = None  # Track last food eaten for learning (Food object or None)
        self.last_food_was_toxic = False  # Track if last food was toxic
        self.last_food_was_harmful = False  # Track if last food was harmful
        self.last_mate_quality = 0.0  # Track quality of last mate (0-1) for learning
        self.last_mated = False  # Track if organism mated recently
        
    def update(self, organisms: List['Organism'], foods: List[Food], dt: float):
        """Update organism state."""
        self.age += dt
        self.learning_timer += dt
        self.backprop_timer += dt
        
        # Consume energy (metabolism) - apply global multiplier
        old_energy = self.energy
        metabolism_mult = getattr(self, '_temp_metabolism_mult', 1.0)
        self.energy -= self.dna.metabolism * metabolism_mult * dt
        
        # Track energy history
        self.energy_history.append(self.energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
        
        # Die if no energy
        if self.energy <= 0:
            if self._death_cause is None:
                self._death_cause = 'starvation'  # Default to starvation if not set
            return False
        
        # Get sensory inputs for neural network
        inputs = self._get_sensory_inputs(organisms, foods)
        
        # Get neural network decision
        outputs = self.brain.predict(inputs)
        self.last_inputs = inputs
        self.last_outputs = outputs
        
        # Store experience for batch learning
        reward = self._calculate_reward(old_energy, organisms, foods)
        self.experience_history.append({
            'inputs': inputs.copy(),
            'outputs': outputs.copy(),
            'reward': reward
        })
        if len(self.experience_history) > 20:
            self.experience_history.pop(0)
        
        # Periodic backpropagation - learn from success/failure every few seconds
        if self.backprop_timer >= self.backprop_interval:
            self._backpropagate_success_failure()
            self.backprop_timer = 0.0
        
        # Execute neural network decisions (sets behavioral flags)
        self._execute_neural_network_behavior(outputs, organisms, foods, dt)
        
        # Find target based on neural network decisions
        self._find_target(organisms, foods)
        
        # Check for interactions with other organisms (fight or mate) - neural network controlled
        interaction_result = self._interact_with_organisms(organisms, dt)
        if interaction_result == 'died':
            return False  # This organism died in combat
        
        # Check for eating - neural network controlled (feed decision)
        if not self.in_combat and hasattr(self, 'neural_feed') and self.neural_feed:
            food_eaten = self._try_eat(organisms, foods)
            if food_eaten:
                self.food_found_count += 1
                # Track food eaten (will be done in Simulation class)
        
        return True
    
    def _get_sensory_inputs(self, organisms: List['Organism'], foods: List[Food]) -> np.ndarray:
        """Get sensory inputs for neural network."""
        inputs = np.zeros(INPUT_SIZE)
        
        # Find nearest food
        nearest_food_dist = 1.0
        nearest_food_angle = 0.0
        nearest_food_type = 0.0
        nearest_food_toxicity = 0.0
        nearest_food_beneficial = 0.0  # 1.0 if toxic and we can benefit, -1.0 if toxic and harmful, 0.0 if not toxic
        
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            normalized_dist = min(distance / self.dna.vision_range, 1.0)
            
            if normalized_dist < nearest_food_dist:
                nearest_food_dist = normalized_dist
                angle = math.atan2(dy, dx) - self.angle
                # Normalize angle to [-1, 1]
                nearest_food_angle = angle / math.pi
                nearest_food_type = 1.0 if food.food_type == 'meat' else 0.0
                nearest_food_toxicity = food.toxicity
                
                # Calculate if this toxic food would be beneficial (resistance > toxicity + 0.2)
                if food.toxicity > 0:
                    resistance_excess = self.dna.toxicity_resistance - food.toxicity
                    if resistance_excess > 0.2:
                        nearest_food_beneficial = 1.0  # Beneficial toxic food
                    elif self.dna.toxicity_resistance >= food.toxicity:
                        nearest_food_beneficial = 0.0  # Safe but not beneficial
                    else:
                        nearest_food_beneficial = -1.0  # Harmful toxic food
                else:
                    nearest_food_beneficial = 0.0  # Not toxic
        
        # Find nearest organism and evaluate mate quality
        nearest_org_dist = 1.0
        nearest_org_angle = 0.0
        nearest_org_size_ratio = 0.0
        nearest_org_energy_ratio = 0.0
        nearest_org_speed = 0.0
        nearest_org_complexity = 0.0
        nearest_org = None
        
        for org in organisms:
            if org is self:
                continue
            dx = org.x - self.x
            dy = org.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            normalized_dist = min(distance / self.dna.vision_range, 1.0)
            
            if normalized_dist < nearest_org_dist:
                nearest_org_dist = normalized_dist
                angle = math.atan2(dy, dx) - self.angle
                nearest_org_angle = angle / math.pi
                nearest_org_size_ratio = org.size / self.size  # Relative size
                nearest_org_energy_ratio = org.energy / org.max_energy  # Mate's energy level (0-1)
                # Mate's speed (normalized)
                org_speed = math.sqrt(org.vx * org.vx + org.vy * org.vy)
                nearest_org_speed = min(org_speed / org.dna.max_speed / 20.0, 1.0)  # Normalize to 0-1
                # Mate's complexity (shape complexity - number of points and variation)
                # More points and more variation = more complex
                shape_variation = np.std(org.shape_radii) if len(org.shape_radii) > 0 else 0.0
                nearest_org_complexity = min((org.shape_point_count / 10.0 + shape_variation / 0.5) / 2.0, 1.0)  # Normalize to 0-1
                nearest_org = org
        
        # Input features
        inputs[0] = nearest_food_dist
        inputs[1] = nearest_food_angle
        inputs[2] = nearest_food_type
        inputs[3] = nearest_org_dist
        inputs[4] = nearest_org_angle
        inputs[5] = nearest_org_size_ratio
        inputs[6] = self.energy / self.max_energy  # Normalized energy (0-1)
        inputs[7] = math.sqrt(self.vx * self.vx + self.vy * self.vy) / self.dna.max_speed  # Normalized speed
        inputs[8] = (self.angle % (2 * math.pi)) / (2 * math.pi)  # Normalized angle
        inputs[9] = self.dna.food_preference
        aggression_mult = getattr(self, '_temp_aggression_mult', 1.0)
        inputs[10] = self.dna.aggression * aggression_mult  # Apply global aggression multiplier
        inputs[11] = min(self.age / 100.0, 1.0)  # Normalized age
        inputs[12] = nearest_food_toxicity  # Toxicity level of nearest food (0-1)
        inputs[13] = nearest_food_beneficial  # -1 = harmful, 0 = neutral, 1 = beneficial
        inputs[14] = nearest_org_energy_ratio  # Nearest organism's energy level (0-1) - mate quality indicator
        inputs[15] = nearest_org_speed  # Nearest organism's speed (0-1) - mate quality indicator
        inputs[16] = nearest_org_complexity  # Nearest organism's shape complexity (0-1) - mate quality indicator
        
        # Overcrowding detection - count nearby organisms
        nearby_count = 0
        if hasattr(self, 'overcrowding_distance'):
            check_distance = self.overcrowding_distance
        else:
            check_distance = self.dna.overcrowding_distance_base if hasattr(self.dna, 'overcrowding_distance_base') else 150.0
        
        for org in organisms:
            if org is self:
                continue
            dx = org.x - self.x
            dy = org.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= check_distance:
                nearby_count += 1
        
        # Normalize nearby count (assume max 20 organisms nearby is "very crowded")
        inputs[17] = min(nearby_count / 20.0, 1.0)  # Normalized nearby organism count (0-1)
        
        # Overcrowding status (1.0 if overcrowded, 0.0 if not)
        if hasattr(self, 'is_overcrowded'):
            inputs[18] = 1.0 if self.is_overcrowded else 0.0
        else:
            inputs[18] = 0.0
        
        # Note: Energy level (inputs[6]) already provides information about fullness
        # The neural network can learn to avoid food when energy is high (close to 1.0)
        # The neural network can learn to seek mates with high energy, speed, and complexity
        # The neural network can learn to fight more when overcrowded (inputs[17] and inputs[18])
        
        return inputs
    
    def _calculate_reward(self, old_energy: float, organisms: List['Organism'], foods: List[Food]) -> float:
        """Calculate reward signal for learning - focused on food finding, movement, and food safety."""
        reward = 0.0
        
        # Strong reward for improving energy (finding food)
        energy_change = self.energy - old_energy
        energy_ratio = self.energy / self.max_energy
        old_energy_ratio = old_energy / self.max_energy
        
        # Learning about food safety - reward/penalty based on what was eaten
        if self.last_food_eaten is not None:
            if self.last_food_was_harmful:
                if self.last_food_was_toxic:
                    # Strong negative reward for eating toxic food (when not resistant)
                    reward -= 3.0  # Strong penalty for eating harmful toxic food
                else:
                    # Negative reward for overeating (eating when full)
                    reward -= 1.5  # Penalty for overeating
            else:
                # Positive reward for eating safe food
                if self.last_food_was_toxic:
                    # Bonus for eating beneficial toxic food (highly resistant)
                    reward += 2.0  # Strong positive for eating beneficial toxic food
                else:
                    # Normal positive for eating safe, non-toxic food
                    reward += 1.0  # Positive reward for safe food
        
        if energy_change > 0:
            # Gaining energy - but penalize if already full (overeating)
            if old_energy_ratio >= 0.95:
                # Strong negative reward for eating when full (overeating is harmful)
                reward -= energy_change * 2.0  # Strong negative for overeating
            else:
                # Normal positive reward for eating when not full
                reward += energy_change * 0.5  # Strong positive for gaining energy
        else:
            reward += energy_change * 0.1  # Smaller negative for losing energy
        
        # Reward for being near food (movement toward food) - but consider food safety and energy level
        energy_ratio = self.energy / self.max_energy
        is_full = energy_ratio >= 0.95
        
        nearest_food_dist = float('inf')
        nearest_food = None
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < nearest_food_dist:
                nearest_food_dist = distance
                nearest_food = food
            
            # Reward/penalty for being close to food depends on energy level and food safety
            if distance < self.size * 2:
                if is_full:
                    # Negative reward for being near food when full (should avoid)
                    reward -= 0.5  # Penalty for approaching food when full
                else:
                    # Check if food is safe for this organism
                    is_toxic = food.toxicity > 0
                    can_resist = self.dna.toxicity_resistance >= food.toxicity
                    resistance_excess = self.dna.toxicity_resistance - food.toxicity if is_toxic else 0
                    
                    if is_toxic and not can_resist:
                        # Negative reward for being near harmful toxic food
                        reward -= 1.5  # Strong penalty for approaching harmful toxic food
                    elif is_toxic and resistance_excess > 0.2:
                        # Positive reward for being near beneficial toxic food
                        reward += 1.5  # Bonus for approaching beneficial toxic food
                    else:
                        # Normal positive reward for safe food
                        reward += 1.0  # Strong reward for reaching safe food
        
        # Reward for moving toward/away from food (if food is visible) - consider food safety
        if nearest_food_dist < self.dna.vision_range and nearest_food is not None:
            # Calculate if we're moving toward food
            food_dx = 0
            food_dy = 0
            for food in foods:
                dx = food.x - self.x
                dy = food.y - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.dna.vision_range and dist < nearest_food_dist:
                    food_dx = dx / dist if dist > 0 else 0
                    food_dy = dy / dist if dist > 0 else 0
            
            # Check food safety
            is_toxic = nearest_food.toxicity > 0
            can_resist = self.dna.toxicity_resistance >= nearest_food.toxicity
            resistance_excess = self.dna.toxicity_resistance - nearest_food.toxicity if is_toxic else 0
            
            # Dot product of velocity and direction to food
            if food_dx != 0 or food_dy != 0:
                speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
                if speed > 0:
                    vel_norm = math.sqrt(self.vx * self.vx + self.vy * self.vy)
                    vel_dir_x = self.vx / vel_norm
                    vel_dir_y = self.vy / vel_norm
                    alignment = vel_dir_x * food_dx + vel_dir_y * food_dy
                    
                    if is_full:
                        # Negative reward for moving toward food when full
                        reward -= alignment * 0.3  # Penalty for moving toward food when full
                    elif is_toxic and not can_resist:
                        # Negative reward for moving toward harmful toxic food
                        reward -= alignment * 0.5  # Strong penalty for approaching harmful toxic food
                    elif is_toxic and resistance_excess > 0.2:
                        # Positive reward for moving toward beneficial toxic food
                        reward += alignment * 0.4  # Bonus for approaching beneficial toxic food
                    else:
                        # Positive reward for moving toward safe food when not full
                        reward += alignment * 0.3  # Reward for moving toward safe food
        
        # Negative reward for low energy (encourages finding food)
        if self.energy < 30:
            reward -= 0.2
        
        # Small reward for movement (exploration)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        reward += speed * 0.005
        
        # Reward for seeking good mates (if mated recently)
        if hasattr(self, 'last_mated') and self.last_mated:
            # Reward based on mate quality - better mates = better reward
            # High quality mate (energy, speed, complexity) = strong positive reward
            if hasattr(self, 'last_mate_quality'):
                mate_quality_reward = self.last_mate_quality * 2.0  # Scale up mate quality to reward
                reward += mate_quality_reward
                
                # Reset mating tracking
                self.last_mated = False
                self.last_mate_quality = 0.0
        
        # Reward for approaching good potential mates (when not in combat and not full)
        if not is_full and not self.in_combat:
            # Find nearest organism for potential mating
            nearest_org_for_mating = None
            nearest_org_dist_for_mating = float('inf')
            for org in organisms:
                if org is self or org.in_combat:
                    continue
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < self.dna.vision_range and distance < nearest_org_dist_for_mating:
                    nearest_org_dist_for_mating = distance
                    nearest_org_for_mating = org
            
            if nearest_org_for_mating is not None and nearest_org_dist_for_mating < self.dna.vision_range * 0.8:
                # Calculate potential mate quality (reduced requirements - more lenient)
                potential_mate_energy = nearest_org_for_mating.energy / nearest_org_for_mating.max_energy
                potential_mate_speed = math.sqrt(nearest_org_for_mating.vx * nearest_org_for_mating.vx + 
                                                nearest_org_for_mating.vy * nearest_org_for_mating.vy)
                # More lenient speed normalization - easier to achieve good score
                potential_mate_speed_norm = min(potential_mate_speed / nearest_org_for_mating.dna.max_speed / 10.0, 1.0)  # Was /20.0
                potential_mate_shape_var = np.std(nearest_org_for_mating.shape_radii) if len(nearest_org_for_mating.shape_radii) > 0 else 0.0
                # More lenient complexity calculation - easier to achieve good score
                potential_mate_complexity = min((nearest_org_for_mating.shape_point_count / 5.0 + potential_mate_shape_var / 0.3) / 2.0, 1.0)  # Was /10.0 and /0.5
                potential_mate_quality = (potential_mate_energy + potential_mate_speed_norm + potential_mate_complexity) / 3.0
                
                # Calculate movement toward mate
                org_dx = nearest_org_for_mating.x - self.x
                org_dy = nearest_org_for_mating.y - self.y
                org_dist = math.sqrt(org_dx * org_dx + org_dy * org_dy)
                if org_dist > 0:
                    org_dir_x = org_dx / org_dist
                    org_dir_y = org_dy / org_dist
                    speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
                    if speed > 0:
                        vel_norm = math.sqrt(self.vx * self.vx + self.vy * self.vy)
                        vel_dir_x = self.vx / vel_norm
                        vel_dir_y = self.vy / vel_norm
                        alignment = vel_dir_x * org_dir_x + vel_dir_y * org_dir_y
                        # Reward for moving toward mates (increased rewards, lower quality threshold)
                        mate_seeking_reward = alignment * potential_mate_quality * 0.8  # Increased from 0.5
                        reward += mate_seeking_reward
                        
                        # Additional reward if neural network is trying to mate and approaching
                        if hasattr(self, 'neural_mate') and self.neural_mate and alignment > 0:
                            reward += 0.4  # Increased bonus (was 0.2)
                        
                        # Reward for being near potential mates (lower quality threshold - more lenient)
                        # This helps the neural network learn to seek mates
                        if potential_mate_quality > 0.3 and nearest_org_dist_for_mating < self.dna.vision_range * 0.7:  # Lower threshold (was 0.6), wider range (was 0.5)
                            reward += 0.5 * potential_mate_quality  # Increased reward (was 0.3)
                        
                        # General reward for being near any organism when seeking mates (encourages proximity)
                        should_seek_mate = (hasattr(self, 'neural_mate') and self.neural_mate) or (self.dna.reproduction_desire > 0.4)
                        if should_seek_mate and nearest_org_dist_for_mating < self.dna.vision_range * 0.5:
                            reward += 0.2  # Bonus for being close to potential mates
        
        # Reset food tracking after reward calculation (so it doesn't carry over)
        # This will be set again if food is eaten in the next frame
        if self.last_food_eaten is not None:
            self.last_food_eaten = None
            self.last_food_was_toxic = False
            self.last_food_was_harmful = False
        
        return reward
    
    def _backpropagate_success_failure(self):
        """Periodic backpropagation based on success/failure in movement and food finding."""
        if len(self.experience_history) < 5:
            return
        
        # Calculate overall success metric
        # Success = energy improvement + food found
        energy_trend = 0.0
        if len(self.energy_history) >= 10:
            recent_avg = sum(self.energy_history[-10:]) / 10
            older_avg = sum(self.energy_history[:10]) / 10 if len(self.energy_history) >= 20 else recent_avg
            energy_trend = recent_avg - older_avg
        
        # Success score: energy trend + food finding
        success_score = energy_trend * 0.1 + self.food_found_count * 2.0
        self.food_found_count = 0  # Reset counter
        
        # Normalize success score
        success_score = np.clip(success_score / 10.0, -1.0, 1.0)
        
        # Train on recent experiences with adjusted rewards based on success
        for exp in self.experience_history[-10:]:  # Last 10 experiences
            # Adjust reward based on overall success
            adjusted_reward = exp['reward'] + success_score * 0.5
            
            # Create target outputs that reinforce successful behaviors
            target_outputs = exp['outputs'].copy()
            
            # If successful, reinforce current outputs
            if success_score > 0:
                # Slightly increase outputs that led to success
                target_outputs = np.clip(target_outputs + success_score * 0.1, -1, 1)
            else:
                # If failing, try to adjust outputs
                target_outputs = np.clip(target_outputs - abs(success_score) * 0.05, -1, 1)
            
            # Train the network
            self.brain.train_step(exp['inputs'], target_outputs)
        
        # Clear old history
        if len(self.experience_history) > 15:
            self.experience_history = self.experience_history[-10:]
    
    def _execute_neural_network_behavior(self, outputs: np.ndarray, organisms: List['Organism'], 
                                         foods: List[Food], dt: float):
        """Execute actions based on neural network outputs."""
        # Outputs: [turn_left, turn_right, flagella_0, ..., flagella_N, mate, fight, run, chase, feed, avoid_toxic, reproduction]
        turn_left = outputs[0]
        turn_right = outputs[1]
        
        # Get per-flagella activities (one output per flagellum)
        flagella_activities = []
        for i in range(self.dna.flagella_count):
            flagella_output = outputs[2 + i]  # Skip turn_left (0) and turn_right (1)
            activity = (flagella_output + 1) / 2  # Convert from [-1,1] to [0,1]
            activity = max(0.1, activity)  # Minimum 10% activity per flagellum
            flagella_activities.append(activity)
        
        # Behavioral decision outputs (threshold at 0.0, values > 0 = active)
        # But also consider reproduction_desire and energy level to bias decisions
        base_idx = 2 + self.dna.flagella_count
        energy_ratio = self.energy / self.max_energy
        
        # Mate decision: combine neural output with reproduction_desire and energy
        # Higher reproduction_desire and energy = more likely to mate
        mate_output = outputs[base_idx + 0]
        # Increased bias - organisms are much more likely to want to mate
        reproduction_desire_mult = getattr(self, '_temp_reproduction_desire_mult', 1.0)
        mate_bias = (self.dna.reproduction_desire * reproduction_desire_mult * 0.7 + (energy_ratio - 0.2) * 0.5)  # Apply global reproduction_desire multiplier
        # Lower threshold - easier to trigger mate decision
        self.neural_mate = (mate_output + mate_bias) > -0.5  # Even lower threshold to make mating more common
        
        self.neural_fight = outputs[base_idx + 1] > 0.4  # Fight decision - higher threshold to make fights rarer
        self.neural_run = outputs[base_idx + 2] > 0.0  # Run away decision
        self.neural_chase = outputs[base_idx + 3] > 0.0  # Chase decision
        self.neural_feed = outputs[base_idx + 4] > 0.0  # Feed decision
        self.neural_avoid_toxic = outputs[base_idx + 5] > 0.0  # Avoid toxic food decision
        self.neural_reproduction = outputs[base_idx + 6] > 0.0  # Reproduction decision
        
        # Overcrowding detection parameters (neural network + DNA influence)
        # Neural outputs are in [-1, 1], convert to modifiers
        overcrowding_threshold_modifier = outputs[base_idx + 7]  # -1 to 1
        overcrowding_distance_modifier = outputs[base_idx + 8]  # -1 to 1
        
        # Combine neural network output with DNA base values
        # Modifier range: -0.5 to +0.5 (50% variation from base)
        threshold_mod = 1.0 + overcrowding_threshold_modifier * 0.5
        distance_mod = 1.0 + overcrowding_distance_modifier * 0.5
        
        # Calculate actual threshold and distance
        self.overcrowding_threshold = max(2, int(self.dna.overcrowding_threshold_base * threshold_mod))
        self.overcrowding_distance = max(30, self.dna.overcrowding_distance_base * distance_mod)
        
        # Average flagella activity for overall movement
        avg_flagella_activity = sum(flagella_activities) / len(flagella_activities) if flagella_activities else 0.3
        # Ensure minimum overall activity so organisms always move
        if avg_flagella_activity < 0.3:
            avg_flagella_activity = 0.3
        
        # Turn based on neural network and target (if chasing, running, or seeking)
        # If running, turn away from nearest threat
        if hasattr(self, 'neural_run') and self.neural_run:
            # Find nearest organism to run away from
            nearest_threat = None
            min_threat_dist = float('inf')
            for org in organisms:
                if org is self:
                    continue
                dx = org.x - self.x
                dy = org.y - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_threat_dist:
                    min_threat_dist = dist
                    nearest_threat = org
            
            if nearest_threat:
                # Turn away from threat
                threat_angle = math.atan2(nearest_threat.y - self.y, nearest_threat.x - self.x)
                desired_angle = threat_angle + math.pi  # Opposite direction
                angle_diff = desired_angle - self.angle
                # Normalize angle difference to [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                # Turn toward desired angle
                turn_angle = angle_diff * 0.2  # Turn rate
                self.angle += turn_angle
            else:
                # No threat, use normal turning
                turn_angle = (turn_right - turn_left) * 0.1
                self.angle += turn_angle
        elif self.target:
            # Turn toward target (food, mate, chase target, or fight target)
            target_angle = math.atan2(self.target.y - self.y, self.target.x - self.x)
            angle_diff = target_angle - self.angle
            # Normalize angle difference to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            # Combine neural network turning with target seeking
            neural_turn = (turn_right - turn_left) * 0.1
            target_turn = angle_diff * 0.15  # Turn toward target
            turn_angle = neural_turn + target_turn
            self.angle += turn_angle
        else:
            # No target, use neural network turning
            turn_angle = (turn_right - turn_left) * 0.1
            self.angle += turn_angle
        
        # Normalize angle
        self.angle = self.angle % (2 * math.pi)
        
        # Update flagella phases based on individual activity levels
        # Each flagellum has its own phase and can beat at different rates
        base_beat_frequency = self.dna.flagella_beat_frequency
        for i in range(len(self.flagella_phases)):
            # Each flagellum beats based on its own activity
            active_beat_frequency = base_beat_frequency * (0.1 + flagella_activities[i] * 0.9)  # 10-100% of base frequency
            self.flagella_phases[i] += active_beat_frequency * dt * 2 * math.pi
            if self.flagella_phases[i] > 2 * math.pi:
                self.flagella_phases[i] -= 2 * math.pi
        
        # Calculate thrust from flagella motion
        # The thrust comes from the flagella beating - wave motion creates forward propulsion
        # More active flagella = more thrust
        # The wave amplitude and frequency contribute to thrust
        # Use average activity for overall thrust calculation
        flagella_thrust_factor = avg_flagella_activity * (0.5 + self.dna.flagella_wave_amplitude * 0.5)
        
        # Base thrust from flagella properties
        # Longer flagella and more flagella = more thrust potential
        flagella_thrust_base = 1.0 + (self.dna.flagella_length / 40.0) * 0.5  # Boost from length
        flagella_thrust_base += (self.dna.flagella_count / 4.0) * 0.3  # Boost from count
        flagella_thrust_base = min(flagella_thrust_base, 2.0)  # Cap the multiplier
        
        # Simplified propulsion - ensure organisms always move
        # Base speed from flagella properties - much faster
        flagella_mult = getattr(self, '_temp_flagella_mult', 1.0)
        base_speed = 130.0  # Base pixels per second (increased from 100.0)
        speed_multiplier = avg_flagella_activity * (1.2 + self.dna.flagella_length / 35.0) * (1.2 + self.dna.flagella_count / 3.5)  # Increased multipliers
        target_speed = base_speed * speed_multiplier * self.dna.propulsion_strength * 0.4 * flagella_mult  # Add propulsion_strength and global multiplier
        
        # Don't let max_speed limit too much - scale it up
        effective_max_speed = self.dna.max_speed * 25.0  # Scale up max_speed (increased from 20.0)
        target_speed = min(target_speed, effective_max_speed)
        
        # Calculate desired velocity
        desired_vx = math.cos(self.angle) * target_speed
        desired_vy = math.sin(self.angle) * target_speed
        
        # Smoothly approach desired velocity (like acceleration)
        acceleration = 250.0  # pixels per second squared (increased from 200.0)
        vx_diff = desired_vx - self.vx
        vy_diff = desired_vy - self.vy
        self.vx += vx_diff * acceleration * dt
        self.vy += vy_diff * acceleration * dt
        
        # Apply friction/drag (fluid environment) - less drag
        drag = 0.97  # Less drag (was 0.95)
        self.vx *= drag
        self.vy *= drag
        
        # Energy cost based on movement
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        energy_cost = speed * 0.0005 * dt  # Reduced energy cost
        self.energy -= energy_cost
        
        # Limit speed (but with scaled up max)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > effective_max_speed:
            self.vx = (self.vx / speed) * effective_max_speed
            self.vy = (self.vy / speed) * effective_max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Wrap around world edges
        self.x = self.x % WORLD_WIDTH
        self.y = self.y % WORLD_HEIGHT
    
    def _find_target(self, organisms: List['Organism'], foods: List[Food]):
        """Find target based on neural network decisions (feed, chase, mate, fight)."""
        best_target = None
        best_distance = self.dna.vision_range
        
        # Neural network decides what to target based on behavioral outputs
        # Priority: feed > chase > mate > fight (but neural network can override)
        
        # Look for food if neural network wants to feed
        if hasattr(self, 'neural_feed') and self.neural_feed:
            for food in foods:
                dx = food.x - self.x
                dy = food.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < best_distance:
                    # Check if neural network wants to avoid toxic food
                    is_toxic = food.toxicity > 0
                    can_resist = self.dna.toxicity_resistance >= food.toxicity
                    if hasattr(self, 'neural_avoid_toxic') and is_toxic and self.neural_avoid_toxic and not can_resist:
                        # Skip toxic food if neural network decided to avoid it
                        continue
                    
                    # Check food preference
                    food_match = 1.0 - abs(self.dna.food_preference - (1.0 if food.food_type == 'meat' else 0.0))
                    if food_match > 0.1:  # Will seek any food
                        best_target = food
                        best_distance = distance
        
        # Look for organisms to chase if neural network wants to chase
        if hasattr(self, 'neural_chase') and self.neural_chase:
            for org in organisms:
                if org is self:
                    continue
                
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < best_distance:
                    best_target = org
                    best_distance = distance
        
        # Look for mates if neural network wants to mate OR if organism has high reproduction desire
        # This makes organisms more likely to seek mates
        should_seek_mate = (hasattr(self, 'neural_mate') and self.neural_mate) or (self.dna.reproduction_desire > 0.4)
        if should_seek_mate:
            for org in organisms:
                if org is self or org.in_combat:
                    continue
                
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < best_distance:
                    # Reduced requirements - accept mates with lower energy
                    org_energy_ratio = org.energy / org.max_energy
                    if org_energy_ratio > 0.2:  # Lower threshold (was 0.5) - accept more potential mates
                        best_target = org
                        best_distance = distance
        
        # Look for fight targets if neural network wants to fight
        if hasattr(self, 'neural_fight') and self.neural_fight:
            for org in organisms:
                if org is self or org.size >= self.size * 1.2:  # Don't attack larger organisms
                    continue
                
                dx = org.x - self.x
                dy = org.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < best_distance:
                    best_target = org
                    best_distance = distance
        
        # Look for prey (if carnivorous enough and wants to feed)
        if self.dna.food_preference > 0.3 and hasattr(self, 'neural_feed') and self.neural_feed:
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
    
    
    def _try_eat(self, organisms: List['Organism'], foods: List[Food]) -> bool:
        """Try to eat food or other organisms. Returns True if food was eaten."""
        eat_radius = self.size * 1.5
        
        # Try eating food
        for food in foods[:]:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < eat_radius:
                # Check if food is toxic to this organism
                is_toxic = food.toxicity > 0
                can_resist = self.dna.toxicity_resistance >= food.toxicity
                resistance_excess = self.dna.toxicity_resistance - food.toxicity  # How much resistance exceeds toxicity
                
                # Neural network decision: avoid toxic food if decision is to avoid
                if is_toxic and self.neural_avoid_toxic and not can_resist:
                    # Neural network decided to avoid this toxic food - don't eat it
                    continue
                
                if is_toxic and not can_resist:
                    # Toxic food damages the organism (not resistant enough)
                    damage = food.toxicity * food.energy * 2.0  # More toxic = more damage
                    self.energy -= damage
                    # Track for learning - this was toxic and harmful
                    self.last_food_eaten = food
                    self.last_food_was_toxic = True
                    self.last_food_was_harmful = True
                    foods.remove(food)
                    # Track toxic food death if it kills
                    if self.energy <= 0:
                        self._death_cause = 'toxic'
                    return True  # Food was consumed (but harmful)
                else:
                    # Safe to eat (either not toxic, or organism is resistant)
                    # Check if organism is full of energy - overeating is harmful
                    energy_ratio = self.energy / self.max_energy
                    is_full = energy_ratio >= 0.95  # Consider "full" at 95%+ energy
                    
                    if is_full:
                        # Overeating when full causes damage (detrimental effect)
                        # The more full, the more damage
                        overeating_damage = food.energy * (energy_ratio - 0.9) * 3.0  # Damage scales with how full
                        self.energy -= overeating_damage
                        # Track for learning - overeating is harmful
                        self.last_food_eaten = food
                        self.last_food_was_toxic = is_toxic
                        self.last_food_was_harmful = True  # Overeating is harmful
                        foods.remove(food)
                        # Track if overeating kills
                        if self.energy <= 0:
                            self._death_cause = 'starvation'  # Technically overeating, but categorize as starvation
                        return True  # Food was consumed (but harmful due to overeating)
                    
                    # Normal eating when not full
                    # Check if food matches preference
                    food_match = 1.0 - abs(self.dna.food_preference - (1.0 if food.food_type == 'meat' else 0.0))
                    energy_gain = food.energy * food_match * self.dna.energy_efficiency
                    
                    # Bonus for eating toxic food when highly resistant
                    # If resistance significantly exceeds toxicity, get bonus energy
                    if is_toxic and resistance_excess > 0.2:  # Resistance is 0.2+ higher than toxicity
                        # More excess resistance = more bonus (up to 2x energy)
                        bonus_multiplier = 1.0 + min(resistance_excess * 2.0, 1.0)  # 1.0 to 2.0 multiplier
                        energy_gain *= bonus_multiplier
                    
                    # Track for learning - this was safe food (or beneficial toxic food)
                    self.last_food_eaten = food
                    self.last_food_was_toxic = is_toxic
                    self.last_food_was_harmful = False  # Safe to eat
                    
                    self.energy = min(self.energy + energy_gain, self.max_energy)
                    foods.remove(food)
                    return True
        
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
                    return True
        
        return False
    
    def _check_overcrowding(self, organisms: List['Organism']):
        """Check if the area around this organism is overcrowded."""
        if not hasattr(self, 'overcrowding_threshold') or not hasattr(self, 'overcrowding_distance'):
            # Not initialized yet (neural network hasn't run)
            self.is_overcrowded = False
            return
        
        nearby_count = 0
        for org in organisms:
            if org is self:
                continue
            dx = org.x - self.x
            dy = org.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= self.overcrowding_distance:
                nearby_count += 1
        
        self.is_overcrowded = nearby_count >= self.overcrowding_threshold
    
    def _interact_with_organisms(self, organisms: List['Organism'], dt: float) -> Optional[str]:
        """Handle interactions with nearby organisms: fight or mate. Returns 'died' if this organism died."""
        interaction_range = self.size * 2.5  # Range for interactions
        
        # Update last interaction timer
        self.last_interaction += dt
        
        # If already in combat, continue fighting
        if self.in_combat and self.combat_target is not None:
            if self.combat_target not in organisms:
                # Target died, exit combat
                self.in_combat = False
                self.combat_target = None
                return None
            
            # Check if still close enough to fight
            dx = self.combat_target.x - self.x
            dy = self.combat_target.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > self.size * 3:  # Moved too far apart
                # Exit combat
                self.in_combat = False
                if self.combat_target.in_combat and self.combat_target.combat_target == self:
                    self.combat_target.in_combat = False
                    self.combat_target.combat_target = None
                self.combat_target = None
                return None
            
            # Continue fighting
            return self._fight_organism(self.combat_target, dt)
        
        # Look for nearby organisms to interact with
        for org in organisms:
            if org is self:
                continue
            
            # Skip if either organism is already in combat with someone else
            if (self.in_combat and self.combat_target != org) or (org.in_combat and org.combat_target != self):
                continue
            
            dx = org.x - self.x
            dy = org.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < interaction_range:
                # Check if both organisms want to interact (prevent spam)
                if self.last_interaction < 0.5 or org.last_interaction < 0.5:
                    continue
                
                # Neural network decides: fight, mate, run, or chase
                # Both organisms must be close enough
                if distance < self.size * 1.5:  # Very close = collision
                    # When organisms meet, they MUST interact - either fight or mate
                    # Fighting only happens if BOTH want to fight, otherwise default to mating
                    
                    # Check for overcrowding - increases fight probability
                    self._check_overcrowding(organisms)
                    org._check_overcrowding(organisms)
                    
                    # Overcrowding makes organisms more likely to fight
                    # If overcrowded, increase fight probability
                    self_fight_probability = 1.0 if (hasattr(self, 'neural_fight') and self.neural_fight) else 0.0
                    org_fight_probability = 1.0 if (hasattr(org, 'neural_fight') and org.neural_fight) else 0.0
                    
                    # Increase fight probability if overcrowded (reduced boost to make fights less common)
                    if hasattr(self, 'is_overcrowded') and self.is_overcrowded:
                        self_fight_probability = min(1.0, self_fight_probability + 0.3)  # Reduced boost
                    if hasattr(org, 'is_overcrowded') and org.is_overcrowded:
                        org_fight_probability = min(1.0, org_fight_probability + 0.3)  # Reduced boost
                    
                    # Check if both organisms want to fight (mutual aggression, including overcrowding boost)
                    # Higher threshold to make fights rarer - both must strongly want to fight
                    both_want_fight = (self_fight_probability > 0.7 and org_fight_probability > 0.7)
                    
                    if both_want_fight:
                        # Both want to fight - engage in combat
                        self.in_combat = True
                        self.combat_target = org
                        org.in_combat = True
                        org.combat_target = self
                        self.last_interaction = 0.0
                        org.last_interaction = 0.0
                        # Track fight (will be done in Simulation class)
                        return self._fight_organism(org, dt)
                    else:
                        # Default to mating (most of the time)
                        # Only require basic conditions: enough energy, age, and cooldown period
                        # Ensure parents have enough energy to survive mating (need at least 40 energy to mate safely)
                        # This accounts for energy cost (up to 18 for 3 offspring) + safety margin
                        min_energy_for_mating = 25.0  # Lowered minimum energy needed to survive mating (was 40.0)
                        # Use DNA-defined minimum mating age for each organism
                        if (self.energy > min_energy_for_mating and
                            org.energy > min_energy_for_mating and
                            self.age >= self.dna.min_mating_age and  # Must be old enough (DNA-defined)
                            org.age >= org.dna.min_mating_age and
                            self.age - self.last_reproduction > 1.5 and  # Shorter cooldown (1.5s instead of 2.0s)
                            org.age - org.last_reproduction > 1.5):
                            # Mate - this is the default behavior when organisms meet
                            self._mate_with_organism(org, organisms)
                            self.last_interaction = 0.0
                            org.last_interaction = 0.0
                            # Track mating (will be done in Simulation class)
                            return None
                        else:
                            # Can't mate yet (cooldown or low energy) - push apart to prevent spinning
                            # Push organisms apart slightly
                            if distance > 0:
                                push_force = 30.0
                                push_x = (dx / distance) * push_force * dt
                                push_y = (dy / distance) * push_force * dt
                                self.x -= push_x
                                self.y -= push_y
                                org.x += push_x
                                org.y += push_y
                            continue
        
        return None
    
    def _fight_organism(self, opponent: 'Organism', dt: float) -> Optional[str]:
        """Fight with another organism through collisions. Returns 'died' if this organism died."""
        # Calculate collision
        dx = opponent.x - self.x
        dy = opponent.y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < (self.size + opponent.size) * 0.9:  # Collision
            # Both organisms take damage based on size and speed
            # Larger and faster organisms deal more damage
            
            # Calculate relative speed
            rel_vx = self.vx - opponent.vx
            rel_vy = self.vy - opponent.vy
            relative_speed = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)
            
            # Damage is based on size ratio and relative speed
            self_damage = (opponent.size / self.size) * relative_speed * 0.5 * dt
            opponent_damage = (self.size / opponent.size) * relative_speed * 0.5 * dt
            
            self.energy -= self_damage
            opponent.energy -= opponent_damage
            
            # Push organisms apart slightly
            if distance > 0:
                push_force = 50.0
                push_x = (dx / distance) * push_force * dt
                push_y = (dy / distance) * push_force * dt
                self.x -= push_x
                self.y -= push_y
                opponent.x += push_x
                opponent.y += push_y
            
            # Check if either organism died
            if self.energy <= 0:
                self.in_combat = False
                if opponent.in_combat and opponent.combat_target == self:
                    opponent.in_combat = False
                    opponent.combat_target = None
                return 'died'
            
            if opponent.energy <= 0:
                # Winner gains energy from defeating opponent
                # Gain energy based on opponent's size and remaining energy (before death)
                # Larger opponents provide more energy
                energy_gain = opponent.size * 0.5 + min(opponent.energy + opponent_damage, 30)  # Gain from size + opponent's energy (capped)
                self.energy = min(self.energy + energy_gain, self.max_energy)  # Cap at max energy
                
                if opponent.in_combat and opponent.combat_target == self:
                    opponent.in_combat = False
                    opponent.combat_target = None
                self.in_combat = False
                self.combat_target = None
                return None
        
        return None
    
    def _mate_with_organism(self, partner: 'Organism', organisms: List['Organism']):
        """Mate with another organism to produce offspring."""
        # Calculate mate quality (for learning) - more lenient requirements
        partner_energy_ratio = partner.energy / partner.max_energy
        partner_speed = math.sqrt(partner.vx * partner.vx + partner.vy * partner.vy)
        # More lenient speed normalization
        partner_speed_normalized = min(partner_speed / partner.dna.max_speed / 10.0, 1.0)  # Was /20.0
        # Partner complexity - more lenient
        partner_shape_variation = np.std(partner.shape_radii) if len(partner.shape_radii) > 0 else 0.0
        partner_complexity = min((partner.shape_point_count / 5.0 + partner_shape_variation / 0.3) / 2.0, 1.0)  # Was /10.0 and /0.5
        
        # Mate quality score (0-1): average of energy, speed, and complexity
        # More lenient - give higher scores for lower requirements
        mate_quality = (partner_energy_ratio * 0.8 + partner_speed_normalized * 0.6 + partner_complexity * 0.6) / 2.0  # Weighted average favoring energy
        
        # Track mate quality for learning
        self.last_mate_quality = mate_quality
        self.last_mated = True
        # More lenient mate quality calculation for partner
        self_speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        self_speed_norm = min(self_speed / self.dna.max_speed / 10.0, 1.0)  # Was /20.0
        self_shape_var = np.std(self.shape_radii) if len(self.shape_radii) > 0 else 0.0
        self_complexity = min((self.shape_point_count / 5.0 + self_shape_var / 0.3) / 2.0, 1.0)  # Was /10.0 and /0.5
        partner.last_mate_quality = (self.energy / self.max_energy * 0.8 + self_speed_norm * 0.6 + self_complexity * 0.6) / 2.0
        partner.last_mated = True
        
        # Sexual reproduction - produce multiple offspring (3-5) with different DNA mixes
        # Number of offspring based on parents' energy and reproduction desire
        energy_factor = (self.energy / self.max_energy + partner.energy / partner.max_energy) / 2.0
        desire_factor = (self.dna.reproduction_desire + partner.dna.reproduction_desire) / 2.0
        # Higher energy and desire = more offspring (3-5)
        # Base is 3, can go up to 5 with high energy and desire
        base_offspring = 3
        bonus_offspring = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2 + energy_factor * 0.3 + desire_factor * 0.3])[0]
        num_offspring = base_offspring + bonus_offspring
        
        # Get parent weights once for reuse
        parent1_weights = self.brain.get_weights_copy()
        parent2_weights = partner.brain.get_weights_copy()
        architectures_compatible = (len(parent1_weights) == len(parent2_weights) and 
                                   all(w1.shape[0] == w2.shape[0] for w1, w2 in zip(parent1_weights, parent2_weights)))
        
        # Create multiple offspring, each with a different DNA mix
        for i in range(num_offspring):
            # Each offspring gets a different mix of parents' DNA
            # The DNA._combine_parents method already has randomness, but we can create multiple instances
            new_dna = DNA(parent_dna=self.dna, parent2_dna=partner.dna)
            
            # Spread offspring around the mating location
            angle = (2 * math.pi * i) / num_offspring  # Distribute evenly in a circle
            distance = random.uniform(20, 50)  # Distance from center
            new_x = (self.x + partner.x) / 2 + math.cos(angle) * distance
            new_y = (self.y + partner.y) / 2 + math.sin(angle) * distance
            # Wrap around world bounds
            new_x = new_x % WORLD_WIDTH
            new_y = new_y % WORLD_HEIGHT
            new_org = Organism(new_x, new_y, new_dna)
            
            # Give new organism a random initial velocity
            # Random direction and speed (0.5 to 2.0 times the parent's average max_speed)
            parent_avg_max_speed = (self.dna.max_speed + partner.dna.max_speed) / 2.0
            initial_speed = random.uniform(0.5, 2.0) * parent_avg_max_speed
            initial_angle = random.uniform(0, 2 * math.pi)
            new_org.vx = math.cos(initial_angle) * initial_speed
            new_org.vy = math.sin(initial_angle) * initial_speed
            
            # Inherit neural network weights with variation for each offspring
            if architectures_compatible:
                # Create varied weight combinations for each offspring
                # Mix ratio varies per offspring (0.3-0.7 instead of always 0.5)
                mix_ratio = random.uniform(0.3, 0.7)  # Different mix for each offspring
                averaged_weights = []
                for j, (w1, w2) in enumerate(zip(parent1_weights, parent2_weights)):
                    if j == len(parent1_weights) - 1:  # Last layer (output layer)
                        # Output layer may have different sizes - use child's output size
                        child_output_size = new_org.brain.output_size
                        # Take weighted average of matching dimensions
                        min_output = min(w1.shape[1], w2.shape[1], child_output_size)
                        avg_output = np.zeros((w1.shape[0], child_output_size))
                        # Weighted mix instead of 50/50
                        avg_output[:, :min_output] = w1[:, :min_output] * mix_ratio + w2[:, :min_output] * (1 - mix_ratio)
                        # If child needs more outputs, use from parent with more weight
                        if child_output_size > min_output:
                            if w1.shape[1] >= child_output_size:
                                avg_output[:, min_output:] = w1[:, min_output:child_output_size]
                            elif w2.shape[1] >= child_output_size:
                                avg_output[:, min_output:] = w2[:, min_output:child_output_size]
                        averaged_weights.append(avg_output)
                    else:
                        # Hidden layers - weighted mix
                        averaged_weights.append(w1 * mix_ratio + w2 * (1 - mix_ratio))
                new_org.brain.weights = averaged_weights
                # Vary mutation rate per offspring
                mutation_rate = random.uniform(0.03, 0.08)  # Different mutation for each
                new_org.brain.mutate_weights(mutation_rate)
            else:
                # If architectures don't match, just use child's randomly initialized weights
                pass
            
            # Ensure we're adding to the correct list (should be self.organisms from Simulation)
            # Add new organism to the list - this is the same reference as self.organisms
            organisms.append(new_org)
        
        # Both parents lose energy proportional to number of offspring (but survive)
        # Ensure parents don't die from mating - leave them with at least 20 energy (safety margin)
        energy_cost_per_offspring = 12  # Further reduced from 15
        total_energy_cost = energy_cost_per_offspring * num_offspring
        cost_per_parent = total_energy_cost / 2
        
        # Deduct energy but ensure parents survive (minimum 20 energy after mating for safety)
        # This accounts for metabolism that will be deducted in the same frame
        self.energy = max(20.0, self.energy - cost_per_parent)
        partner.energy = max(20.0, partner.energy - cost_per_parent)
        self.last_reproduction = self.age
        partner.last_reproduction = partner.age
    
    def get_shape(self) -> List[Tuple[int, int]]:
        """Get polygon points for rendering - complex irregular shape defined by DNA."""
        points = []
        
        # Generate irregular shape using DNA-defined radii and angle offsets
        for i in range(self.shape_point_count):
            # Base angle for this point (evenly distributed around circle)
            base_angle = (2 * math.pi * i / self.shape_point_count)
            
            # Apply angle offset for irregularity (mutations can create complex shapes)
            point_angle = base_angle + self.shape_angle_offsets[i] + self.angle
            
            # Use DNA-defined radius multiplier for this point
            # This creates bumps, indentations, and irregular protrusions
            radius_multiplier = self.shape_radii[i]
            effective_radius = self.size * radius_multiplier
            
            # Calculate point position
            px = self.x + math.cos(point_angle) * effective_radius
            py = self.y + math.sin(point_angle) * effective_radius
            points.append((int(px), int(py)))
        
        # Shape is always closed (last point connects to first)
        return points
    
    def get_flagella_points(self) -> List[List[Tuple[float, float]]]:
        """Get flagella points for rendering - animated based on DNA."""
        flagella_list = []
        
        # Calculate attachment points around the organism (opposite to movement direction)
        base_angle = self.angle + math.pi  # Opposite to movement
        
        for i in range(self.dna.flagella_count):
            # Distribute flagella around the back of the organism
            flagella_angle_offset = (2 * math.pi * i / self.dna.flagella_count) - (math.pi / 2)
            attachment_angle = base_angle + flagella_angle_offset
            
            # Attachment point on organism edge
            attach_x = self.x + math.cos(attachment_angle) * self.size * 0.8
            attach_y = self.y + math.sin(attachment_angle) * self.size * 0.8
            
            # Create flagellum as a curved line with wave motion
            segments = 8
            flagellum_points = []
            
            # Phase offset for each flagellum (creates wave propagation)
            # Use individual phase for each flagellum
            phase_offset = self.flagella_phases[i] if i < len(self.flagella_phases) else 0.0
            
            for seg in range(segments + 1):
                t = seg / segments  # 0 to 1 along flagellum
                
                # Base direction (away from organism, opposite to movement)
                flag_dir = attachment_angle
                
                # Wave motion perpendicular to flagellum direction
                # The wave creates a propulsive force - visualize this with the wave pattern
                wave_phase = phase_offset + t * self.dna.flagella_wave_length * math.pi * 2
                # Wave amplitude affects how much the flagellum curves (and thus thrust)
                wave_offset = math.sin(wave_phase) * self.dna.flagella_wave_amplitude * 0.6
                
                # Perpendicular direction for wave
                perp_angle = flag_dir + math.pi / 2
                
                # Calculate point along flagellum
                length_along = t * self.dna.flagella_length
                px = attach_x + math.cos(flag_dir) * length_along + math.cos(perp_angle) * wave_offset * self.dna.flagella_length
                py = attach_y + math.sin(flag_dir) * length_along + math.sin(perp_angle) * wave_offset * self.dna.flagella_length
                
                flagellum_points.append((px, py))
            
            flagella_list.append(flagellum_points)
        
        return flagella_list
    
    def get_eye_positions(self) -> List[Tuple[float, float]]:
        """Get positions of eyes (visual sensors) on the organism."""
        eyes = []
        # Eyes are positioned on the front of the organism (facing movement direction)
        # Number of eyes based on vision capability
        eye_count = max(1, int(self.dna.vision_range / 50))  # More vision = more eyes
        eye_count = min(eye_count, 4)  # Max 4 eyes
        
        for i in range(eye_count):
            # Distribute eyes around the front of the organism
            eye_angle_offset = (i - eye_count / 2 + 0.5) * 0.3  # Spread eyes
            eye_angle = self.angle + eye_angle_offset
            
            # Position eyes on the front edge of the organism
            eye_distance = self.size * 0.9
            eye_x = self.x + math.cos(eye_angle) * eye_distance
            eye_y = self.y + math.sin(eye_angle) * eye_distance
            
            eyes.append((eye_x, eye_y))
        
        return eyes

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
        
        # Create initial organisms
        for _ in range(40):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.organisms.append(Organism(x, y))
        
        # Create initial food
        self._spawn_food(50)  # Reduced initial food count
        
        # UI
        self.font = pygame.font.Font(None, 24)
        # Use bundled fonts for platform independence
        import os
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        dejavu_path = os.path.join(font_dir, "DejaVuSans.ttf")
        emoji_path = os.path.join(font_dir, "NotoColorEmoji.ttf")
        alert_font_size = 36
        
        # Load text font
        try:
            if os.path.exists(dejavu_path):
                self.alert_font = pygame.font.Font(dejavu_path, alert_font_size)
            else:
                self.alert_font = pygame.font.SysFont("sans", alert_font_size)
        except:
            self.alert_font = pygame.font.Font(None, alert_font_size)
        
        # Load emoji font for emoji rendering (smaller size to match text)
        emoji_font_size = 24  # Same size as normal text font
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
        """Add new organisms to the simulation."""
        for _ in range(count):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.organisms.append(Organism(x, y))
    
    def _reset_simulation(self):
        """Reset the simulation to initial state."""
        # Clear all organisms and food
        self.organisms.clear()
        self.foods.clear()
        
        # Reset camera
        self.camera.x = WORLD_WIDTH / 2
        self.camera.y = WORLD_HEIGHT / 2
        self.camera.zoom = 0.3  # Start zoomed out more
        
        # Create initial organisms
        for _ in range(40):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.organisms.append(Organism(x, y))
        
        # Create initial food
        self._spawn_food(100)
    
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
                    # Check if clicking on parameter control buttons
                    if self._handle_parameter_button_click(event.pos):
                        continue  # Button was clicked, don't start dragging
                    
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
            
            if not org.update(self.organisms, self.foods, dt):
                # Death event - create alert
                # Check if organism is still in list (might have been eaten/removed during update)
                if org not in self.organisms:
                    continue
                
                death_cause = 'starvation'
                if hasattr(org, '_death_cause'):
                    death_cause = org._death_cause
                
                self._create_alert(f"💀 Death ({death_cause})", (255, 100, 100), org.x, org.y)
                
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
                        self._create_alert("⚔️ Fight!", (255, 0, 0), fight_x, fight_y)
        
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
                self._create_alert("👶 Birth!", (0, 255, 255), new_org.x, new_org.y)
        
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
            offspring = food.update(dt, self.foods, WORLD_WIDTH, WORLD_HEIGHT)
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
    
    def _create_alert(self, message: str, color: Tuple[int, int, int], x: float, y: float):
        """Create an alert for a major event (points at location but doesn't move camera)."""
        if not self.alerts_enabled:
            return  # Don't create alerts if disabled
        self.current_alert = (message, color, self.alert_duration, x, y)
    
    def _draw_parameter_controls(self, screen_width: int, start_y: int):
        """Draw clickable parameter control buttons."""
        button_width = 80
        button_height = 25
        button_spacing = 10
        label_width = 150
        x_start = screen_width - label_width - button_width * 2 - button_spacing * 2 - 20
        
        y = start_y + 10
        controls = [
            ("Metabolism", self.metabolism_multiplier, 0.1, 3.0),
            ("Flagella", self.flagella_impulse_multiplier, 0.1, 3.0),
            ("Aggression", self.aggression_multiplier, 0.1, 3.0),
            ("Repro Desire", self.reproduction_desire_multiplier, 0.1, 3.0),
        ]
        
        for label, value, min_val, max_val in controls:
            # Draw label
            label_surface = self.font.render(f"{label}: {value:.2f}", True, (255, 255, 255))
            self.screen.blit(label_surface, (x_start, y))
            
            # Draw - button
            minus_rect = pygame.Rect(x_start + label_width, y, button_width, button_height)
            pygame.draw.rect(self.screen, (100, 100, 100), minus_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), minus_rect, 2)
            minus_text = self.font.render("-", True, (255, 255, 255))
            text_x = minus_rect.centerx - minus_text.get_width() // 2
            text_y = minus_rect.centery - minus_text.get_height() // 2
            self.screen.blit(minus_text, (text_x, text_y))
            
            # Draw + button
            plus_rect = pygame.Rect(x_start + label_width + button_width + button_spacing, y, button_width, button_height)
            pygame.draw.rect(self.screen, (100, 100, 100), plus_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), plus_rect, 2)
            plus_text = self.font.render("+", True, (255, 255, 255))
            text_x = plus_rect.centerx - plus_text.get_width() // 2
            text_y = plus_rect.centery - plus_text.get_height() // 2
            self.screen.blit(plus_text, (text_x, text_y))
            
            y += button_height + 5
    
    def _handle_parameter_button_click(self, pos: Tuple[int, int]) -> bool:
        """Handle clicks on parameter control buttons. Returns True if a button was clicked."""
        button_width = 80
        button_height = 25
        button_spacing = 10
        label_width = 150
        screen_width = self.screen.get_width()
        x_start = screen_width - label_width - button_width * 2 - button_spacing * 2 - 20
        
        # Calculate start_y (same as in _draw_parameter_controls)
        info_lines = 9  # Number of info text lines
        start_y = 10 + info_lines * 25 + 10
        
        controls = [
            ("metabolism", self.metabolism_multiplier, 0.1, 3.0),
            ("flagella", self.flagella_impulse_multiplier, 0.1, 3.0),
            ("aggression", self.aggression_multiplier, 0.1, 3.0),
            ("reproduction_desire", self.reproduction_desire_multiplier, 0.1, 3.0),
        ]
        
        y = start_y
        for control_name, value, min_val, max_val in controls:
            # Check - button
            minus_rect = pygame.Rect(x_start + label_width, y, button_width, button_height)
            if minus_rect.collidepoint(pos):
                step = 0.1
                if control_name == "metabolism":
                    self.metabolism_multiplier = max(min_val, self.metabolism_multiplier - step)
                elif control_name == "flagella":
                    self.flagella_impulse_multiplier = max(min_val, self.flagella_impulse_multiplier - step)
                elif control_name == "aggression":
                    self.aggression_multiplier = max(min_val, self.aggression_multiplier - step)
                elif control_name == "reproduction_desire":
                    self.reproduction_desire_multiplier = max(min_val, self.reproduction_desire_multiplier - step)
                return True
            
            # Check + button
            plus_rect = pygame.Rect(x_start + label_width + button_width + button_spacing, y, button_width, button_height)
            if plus_rect.collidepoint(pos):
                step = 0.1
                if control_name == "metabolism":
                    self.metabolism_multiplier = min(max_val, self.metabolism_multiplier + step)
                elif control_name == "flagella":
                    self.flagella_impulse_multiplier = min(max_val, self.flagella_impulse_multiplier + step)
                elif control_name == "aggression":
                    self.aggression_multiplier = min(max_val, self.aggression_multiplier + step)
                elif control_name == "reproduction_desire":
                    self.reproduction_desire_multiplier = min(max_val, self.reproduction_desire_multiplier + step)
                return True
            
            y += button_height + 5
        
        return False
    
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
        for org in self.organisms:
            sx, sy = self.camera.world_to_screen(org.x, org.y, screen_width, screen_height)
            if -50 < sx < screen_width + 50 and -50 < sy < screen_height + 50:
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
        if self.current_alert is not None:
            message, color, time_remaining, alert_x, alert_y = self.current_alert
            # Convert alert location to screen coordinates
            alert_sx, alert_sy = self.camera.world_to_screen(alert_x, alert_y, screen_width, screen_height)
            
            # Render alert text with emoji support
            alert_text = self._render_text_with_emoji(message, color)
            text_rect = alert_text.get_rect()
            
            # Position alert at top center of screen
            alert_x_pos = screen_width // 2 - text_rect.width // 2
            alert_y_pos = 50
            
            # Draw semi-transparent background
            bg_rect = pygame.Rect(alert_x_pos - 10, alert_y_pos - 5, text_rect.width + 20, text_rect.height + 10)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.set_alpha(200)
            bg_surface.fill((0, 0, 0))
            self.screen.blit(bg_surface, bg_rect)
            
            # Draw alert text
            self.screen.blit(alert_text, (alert_x_pos, alert_y_pos))
            
            # Draw indicator pointing to event location
            center_x = alert_x_pos + text_rect.width // 2
            center_y = alert_y_pos + text_rect.height
            
            if 0 <= alert_sx < screen_width and 0 <= alert_sy < screen_height:
                # Event is on screen - draw line and circle
                pygame.draw.line(self.screen, color, (center_x, center_y), (alert_sx, alert_sy), 2)
                # Draw a circle at event location
                pygame.draw.circle(self.screen, color, (alert_sx, alert_sy), 10, 2)
            else:
                # Event is off screen - draw arrow pointing in the direction
                # Calculate direction from screen center to event
                screen_center_x = screen_width // 2
                screen_center_y = screen_height // 2
                dx = alert_sx - screen_center_x
                dy = alert_sy - screen_center_y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance > 0:
                    # Normalize and scale to edge of screen
                    edge_distance = min(screen_width, screen_height) * 0.4
                    dir_x = dx / distance
                    dir_y = dy / distance
                    edge_x = screen_center_x + dir_x * edge_distance
                    edge_y = screen_center_y + dir_y * edge_distance
                    
                    # Draw line from alert to edge
                    pygame.draw.line(self.screen, color, (center_x, center_y), (int(edge_x), int(edge_y)), 2)
                    
                    # Draw arrowhead at edge
                    arrow_size = 15
                    perp_x = -dir_y
                    perp_y = dir_x
                    arrow1_x = int(edge_x - dir_x * arrow_size + perp_x * arrow_size * 0.5)
                    arrow1_y = int(edge_y - dir_y * arrow_size + perp_y * arrow_size * 0.5)
                    arrow2_x = int(edge_x - dir_x * arrow_size - perp_x * arrow_size * 0.5)
                    arrow2_y = int(edge_y - dir_y * arrow_size - perp_y * arrow_size * 0.5)
                    pygame.draw.polygon(self.screen, color, [
                        (int(edge_x), int(edge_y)),
                        (arrow1_x, arrow1_y),
                        (arrow2_x, arrow2_y)
                    ])
        
        # Draw UI
        info_text = [
            f"Organisms: {len(self.organisms)}",
            f"Food: {len(self.foods)}",
            f"Total Births: {self.cumulative_stats['total_births']}",
            f"Total Deaths: {self.cumulative_stats['total_deaths']}",
            f"Zoom: {self.camera.zoom:.2f}",
            f"Terminal Output: {'ON' if self.show_terminal_output else 'OFF'}",
            f"Event Alerts: {'ON' if self.alerts_enabled else 'OFF'}",
            "SPACE: Pause | F: Add Food | O: Add Organisms",
            "R: Reset | M: Toggle Terminal | A: Toggle Alerts | ESC: Exit",
            "+/-: Zoom | Arrow Keys: Pan | Mouse Wheel: Zoom | Drag: Pan"
        ]
        y_offset = 10
        for text in info_text:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        # Draw parameter control buttons
        self._draw_parameter_controls(screen_width, y_offset)
        
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            pause_x = screen_width // 2 - pause_text.get_width() // 2
            self.screen.blit(pause_text, (pause_x, 10))
        
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
        print("  R - Reset Simulation")
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
        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()

