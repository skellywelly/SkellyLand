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
INPUT_SIZE = 12  # Number of input neurons (sensors)
OUTPUT_SIZE = 4  # Number of output neurons (actions)

class NeuralNetwork:
    """Neural network that controls organism behavior. Can learn and adapt."""
    
    def __init__(self, hidden_layers: int, neurons_per_layer: int, learning_rate: float, 
                 initial_weights: Optional[List[np.ndarray]] = None):
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        
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
        
        # Last hidden to output
        weights.append(np.random.randn(self.neurons_per_layer, OUTPUT_SIZE) * 0.5)
        
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
            self.shape_points = random.randint(3, 8)  # Number of points for polygon shape
            self.shape_elongation = random.uniform(0.8, 1.2)  # Elongation factor for shape
            # Flagella properties
            self.flagella_count = random.randint(1, 4)  # Number of flagella
            self.flagella_length = random.uniform(15, 40)  # Length of flagella
            self.flagella_thickness = random.uniform(1.5, 3.5)  # Thickness of flagella
            self.flagella_beat_frequency = random.uniform(2.0, 8.0)  # Beats per second
            self.flagella_wave_amplitude = random.uniform(0.3, 0.8)  # Wave amplitude (0-1)
            self.flagella_wave_length = random.uniform(0.5, 2.0)  # Wave length factor
            # Propulsion (now based on flagella)
            self.propulsion_strength = random.uniform(0.5, 3.0)  # Overall propulsion strength
            self.propulsion_efficiency = random.uniform(0.3, 0.9)
            self.max_speed = random.uniform(1.0, 5.0)
            self.energy_efficiency = random.uniform(0.5, 1.5)
            self.metabolism = random.uniform(0.002, 0.008)  # Much lower metabolism - live longer
            self.vision_range = random.uniform(50, 200)
            self.reproduction_threshold = random.uniform(50, 100)
            self.aggression = random.uniform(0, 1)
            # Neural network architecture (DNA-controlled)
            self.nn_hidden_layers = random.randint(1, 3)  # Number of hidden layers
            self.nn_neurons_per_layer = random.randint(4, 12)  # Neurons per hidden layer
            self.nn_learning_rate = random.uniform(0.01, 0.1)  # Learning rate
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
        self.shape_elongation = parent.shape_elongation
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
        self.shape_points = random.choice([parent1.shape_points, parent2.shape_points])
        self.shape_elongation = (parent1.shape_elongation + parent2.shape_elongation) / 2
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
        
        if random.random() < mutation_rate:
            self.shape_points = int(np.clip(self.shape_points + random.randint(-1, 1), 3, 10))
        
        if random.random() < mutation_rate:
            self.shape_elongation = np.clip(self.shape_elongation + random.uniform(-0.1, 0.1), 0.6, 1.4)
        
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
            self.propulsion_strength = np.clip(self.propulsion_strength + random.uniform(-0.3, 0.3), 0.2, 4.0)
        
        if random.random() < mutation_rate:
            self.propulsion_efficiency = np.clip(self.propulsion_efficiency + random.uniform(-0.1, 0.1), 0.2, 1.0)
        
        if random.random() < mutation_rate:
            self.max_speed = np.clip(self.max_speed + random.uniform(-0.5, 0.5), 0.5, 6.0)
        
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
            self.nn_hidden_layers = int(np.clip(self.nn_hidden_layers + random.randint(-1, 1), 1, 4))
        
        if random.random() < mutation_rate:
            self.nn_neurons_per_layer = int(np.clip(self.nn_neurons_per_layer + random.randint(-2, 2), 3, 16))
        
        if random.random() < mutation_rate:
            self.nn_learning_rate = np.clip(self.nn_learning_rate + random.uniform(-0.02, 0.02), 0.005, 0.2)

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
        self.shape_elongation = self.dna.shape_elongation
        
        # Movement properties
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(0, 2 * math.pi)
        self.rotation_speed = random.uniform(-0.1, 0.1)
        
        # Flagella animation state
        self.flagella_phase = random.uniform(0, 2 * math.pi)  # Animation phase for each flagellum
        
        # Energy and life
        self.energy = random.uniform(50, 80)  # Start with more energy
        self.max_energy = 150  # Higher max energy
        self.age = 0
        
        # Behavior
        self.target = None  # Target food or organism
        self.last_reproduction = 0
        
        # Neural network (DNA-controlled architecture)
        self.brain = NeuralNetwork(
            hidden_layers=self.dna.nn_hidden_layers,
            neurons_per_layer=self.dna.nn_neurons_per_layer,
            learning_rate=self.dna.nn_learning_rate
        )
        
        # Learning state
        self.last_inputs = None
        self.last_outputs = None
        self.last_reward = 0.0
        self.learning_timer = 0.0
        
    def update(self, organisms: List['Organism'], foods: List[Food], dt: float):
        """Update organism state."""
        self.age += dt
        self.learning_timer += dt
        
        # Consume energy (metabolism)
        old_energy = self.energy
        self.energy -= self.dna.metabolism * dt
        
        # Die if no energy
        if self.energy <= 0:
            return False
        
        # Get sensory inputs for neural network
        inputs = self._get_sensory_inputs(organisms, foods)
        
        # Get neural network decision
        outputs = self.brain.predict(inputs)
        self.last_inputs = inputs
        self.last_outputs = outputs
        
        # Calculate reward from previous action
        reward = self._calculate_reward(old_energy, organisms, foods)
        if self.last_inputs is not None and self.last_outputs is not None:
            self.brain.learn_from_experience(reward, self.last_inputs, self.last_outputs)
        
        # Execute neural network decisions
        self._execute_neural_network_behavior(outputs, organisms, foods, dt)
        
        # Check for eating
        self._try_eat(organisms, foods)
        
        # Check for reproduction (neural network can also trigger this)
        if self.energy > self.dna.reproduction_threshold and self.age - self.last_reproduction > 2.0:
            if outputs[3] > 0.5:  # Reproduction decision from neural network
                return self._try_reproduce(organisms)
        
        return True
    
    def _get_sensory_inputs(self, organisms: List['Organism'], foods: List[Food]) -> np.ndarray:
        """Get sensory inputs for neural network."""
        inputs = np.zeros(INPUT_SIZE)
        
        # Find nearest food
        nearest_food_dist = 1.0
        nearest_food_angle = 0.0
        nearest_food_type = 0.0
        
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
        
        # Find nearest organism
        nearest_org_dist = 1.0
        nearest_org_angle = 0.0
        nearest_org_size_ratio = 0.0
        
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
        inputs[10] = self.dna.aggression
        inputs[11] = min(self.age / 100.0, 1.0)  # Normalized age
        
        return inputs
    
    def _calculate_reward(self, old_energy: float, organisms: List['Organism'], foods: List[Food]) -> float:
        """Calculate reward signal for learning."""
        reward = 0.0
        
        # Reward for maintaining/improving energy
        energy_change = self.energy - old_energy
        reward += energy_change * 0.1
        
        # Reward for being near food
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < self.size * 2:
                reward += 0.5
        
        # Small negative reward for low energy (encourages finding food)
        if self.energy < 30:
            reward -= 0.1
        
        # Reward for moving (exploration)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        reward += speed * 0.01
        
        return reward
    
    def _execute_neural_network_behavior(self, outputs: np.ndarray, organisms: List['Organism'], 
                                         foods: List[Food], dt: float):
        """Execute actions based on neural network outputs."""
        # Outputs: [turn_left, turn_right, flagella_activity, reproduction]
        turn_left = outputs[0]
        turn_right = outputs[1]
        flagella_activity = (outputs[2] + 1) / 2  # Convert from [-1,1] to [0,1] - controls flagella beating
        # Ensure minimum activity so organisms always move
        flagella_activity = max(0.3, flagella_activity)  # At least 30% activity
        
        # Turn based on neural network
        turn_angle = (turn_right - turn_left) * 0.1  # Turn rate
        self.angle += turn_angle
        
        # Normalize angle
        self.angle = self.angle % (2 * math.pi)
        
        # Update flagella phase based on activity level
        # Higher activity = faster beating = more thrust
        base_beat_frequency = self.dna.flagella_beat_frequency
        active_beat_frequency = base_beat_frequency * (0.3 + flagella_activity * 0.7)  # 30-100% of base frequency
        self.flagella_phase += active_beat_frequency * dt * 2 * math.pi
        if self.flagella_phase > 2 * math.pi:
            self.flagella_phase -= 2 * math.pi
        
        # Calculate thrust from flagella motion
        # The thrust comes from the flagella beating - wave motion creates forward propulsion
        # More active flagella = more thrust
        # The wave amplitude and frequency contribute to thrust
        flagella_thrust_factor = flagella_activity * (0.5 + self.dna.flagella_wave_amplitude * 0.5)
        
        # Base thrust from flagella properties
        # Longer flagella and more flagella = more thrust potential
        flagella_thrust_base = 1.0 + (self.dna.flagella_length / 40.0) * 0.5  # Boost from length
        flagella_thrust_base += (self.dna.flagella_count / 4.0) * 0.3  # Boost from count
        flagella_thrust_base = min(flagella_thrust_base, 2.0)  # Cap the multiplier
        
        # Simplified propulsion - ensure organisms always move
        # Base speed from flagella properties - much faster
        base_speed = 100.0  # Base pixels per second (increased from 30)
        speed_multiplier = flagella_activity * (1.0 + self.dna.flagella_length / 40.0) * (1.0 + self.dna.flagella_count / 4.0)
        target_speed = base_speed * speed_multiplier
        
        # Don't let max_speed limit too much - scale it up
        effective_max_speed = self.dna.max_speed * 20.0  # Scale up max_speed
        target_speed = min(target_speed, effective_max_speed)
        
        # Calculate desired velocity
        desired_vx = math.cos(self.angle) * target_speed
        desired_vy = math.sin(self.angle) * target_speed
        
        # Smoothly approach desired velocity (like acceleration)
        acceleration = 200.0  # pixels per second squared (increased)
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
        """Find nearest food or prey within vision range."""
        best_target = None
        best_distance = self.dna.vision_range
        
        # Look for food
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < best_distance:
                # Check food preference - organisms will eat any food, but prefer matching type
                food_match = 1.0 - abs(self.dna.food_preference - (1.0 if food.food_type == 'meat' else 0.0))
                # Lower threshold - organisms will seek any food, preference affects energy gain
                if food_match > 0.1:  # Will seek any food
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
                
                # Inherit neural network weights (average of parents)
                parent1_weights = self.brain.get_weights_copy()
                parent2_weights = org.brain.get_weights_copy()
                # Average weights if same architecture
                if (len(parent1_weights) == len(parent2_weights) and 
                    all(w1.shape == w2.shape for w1, w2 in zip(parent1_weights, parent2_weights))):
                    averaged_weights = [(w1 + w2) / 2 for w1, w2 in zip(parent1_weights, parent2_weights)]
                    new_org.brain.weights = averaged_weights
                    new_org.brain.mutate_weights(0.05)  # Small mutation
                
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
            
            # Inherit neural network weights (clone with mutation)
            new_org.brain.weights = self.brain.get_weights_copy()
            new_org.brain.mutate_weights(0.1)  # Mutation
            
            organisms.append(new_org)
            self.energy -= 15
            self.last_reproduction = self.age
            return True
        
        return False
    
    def get_shape(self) -> List[Tuple[int, int]]:
        """Get polygon points for rendering - shape defined by DNA."""
        points = []
        for i in range(self.shape_points):
            # Base angle for this point
            point_angle = (2 * math.pi * i / self.shape_points) + self.angle
            # Apply elongation: stretch along movement direction
            radius = self.size
            # Elongation creates oval/ellipse shapes
            elongation_factor = self.shape_elongation
            # Calculate radius based on angle relative to movement
            angle_from_movement = point_angle - self.angle
            # Elongate perpendicular to movement direction
            effective_radius = radius * (1.0 + (elongation_factor - 1.0) * abs(math.sin(angle_from_movement)))
            
            px = self.x + math.cos(point_angle) * effective_radius
            py = self.y + math.sin(point_angle) * effective_radius
            points.append((int(px), int(py)))
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
            phase_offset = self.flagella_phase + (i * math.pi / self.dna.flagella_count)
            
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

class Camera:
    """Camera system for viewing the world."""
    
    def __init__(self):
        self.x = WORLD_WIDTH / 2
        self.y = WORLD_HEIGHT / 2
        self.zoom = 0.5  # Start zoomed out
    
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
        for _ in range(20):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            self.organisms.append(Organism(x, y))
        
        # Create initial food
        self._spawn_food(100)
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.paused = False
        
        # Mouse dragging
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.camera_follow_enabled = True
        
        # Monitoring
        self.monitor_timer = 0.0
        self.monitor_interval = 2.0  # Print stats every 2 seconds
        self.frame_count = 0
        self.total_time = 0.0
    
    def _spawn_food(self, count: int):
        """Spawn food particles in the world."""
        for _ in range(count):
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
        self.camera.zoom = 0.5  # Start zoomed out
        
        # Create initial organisms
        for _ in range(20):
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
        
        # Update organisms
        deaths = 0
        total_energy = 0.0
        total_speed = 0.0
        avg_age = 0.0
        
        for org in self.organisms[:]:
            if not org.update(self.organisms, self.foods, dt):
                self.organisms.remove(org)
                deaths += 1
            else:
                total_energy += org.energy
                speed = math.sqrt(org.vx * org.vx + org.vy * org.vy)
                total_speed += speed
                avg_age += org.age
        
        # Maintain food population
        if len(self.foods) < 50:
            self._spawn_food(10)
        
        # Update camera to follow center of mass (only if not dragging)
        if self.camera_follow_enabled and self.organisms:
            avg_x = sum(org.x for org in self.organisms) / len(self.organisms)
            avg_y = sum(org.y for org in self.organisms) / len(self.organisms)
            self.camera.update(avg_x, avg_y)
        
        # Print monitoring stats
        if self.monitor_timer >= self.monitor_interval:
            self._print_monitoring_stats(deaths, total_energy, total_speed, avg_age)
            self.monitor_timer = 0.0
    
    def _print_monitoring_stats(self, deaths: int, total_energy: float, total_speed: float, avg_age: float):
        """Print monitoring statistics to console."""
        org_count = len(self.organisms)
        food_count = len(self.foods)
        
        avg_energy = total_energy / org_count if org_count > 0 else 0.0
        avg_speed = total_speed / org_count if org_count > 0 else 0.0
        avg_age_val = avg_age / org_count if org_count > 0 else 0.0
        
        # Calculate FPS
        fps = self.frame_count / self.monitor_interval if self.monitor_interval > 0 else 0
        self.frame_count = 0
        
        # Get sample organism stats
        sample_stats = ""
        if org_count > 0:
            sample = self.organisms[0]
            speed = math.sqrt(sample.vx * sample.vx + sample.vy * sample.vy)
            sample_stats = (
                f" | Sample: speed={speed:.1f}, energy={sample.energy:.1f}, "
                f"flagella={sample.dna.flagella_count}, size={sample.size:.1f}"
            )
        
        print(f"\n{'='*80}")
        print(f"Simulation Status (Time: {self.total_time:.1f}s)")
        print(f"{'='*80}")
        print(f"Organisms: {org_count:3d} | Food: {food_count:3d} | Deaths: {deaths:2d}")
        print(f"Avg Energy: {avg_energy:6.1f} | Avg Speed: {avg_speed:5.2f} | Avg Age: {avg_age_val:5.1f}")
        print(f"FPS: {fps:5.1f} | Zoom: {self.camera.zoom:.2f}{sample_stats}")
        print(f"{'='*80}")
    
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
            "SPACE: Pause | F: Add Food | O: Add Organisms",
            "R: Reset | Mouse Wheel: Zoom | Drag: Pan View"
        ]
        y_offset = 10
        for text in info_text:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
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

