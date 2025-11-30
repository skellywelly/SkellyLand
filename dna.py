import numpy as np
import random
from typing import Optional

class DNA:
    """Digital DNA that describes an organism's characteristics."""
    
    def __init__(self, parent_dna: Optional['DNA'] = None, parent2_dna: Optional['DNA'] = None, mutation_multiplier: float = 1.0):
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
            self.min_mating_age = random.uniform(20, 45)  # Minimum age in seconds before can mate (20-45 seconds)
            self.overcrowding_threshold_base = random.uniform(3, 10)  # Base threshold for overcrowding (organism count)
            self.overcrowding_distance_base = random.uniform(50, 300)  # Base distance to check for overcrowding
            # Neural network architecture (DNA-controlled)
            self.nn_hidden_layers = random.randint(1, 3)  # Number of hidden layers
            self.nn_neurons_per_layer = random.randint(4, 12)  # Neurons per hidden layer
            self.nn_learning_rate = random.uniform(0.01, 0.1)  # Learning rate
            # Cooperative behavior traits
            self.cooperative_hunting = random.uniform(0, 1)  # Tendency to hunt cooperatively (0 = solo, 1 = highly cooperative)
            self.cooperative_mating = random.uniform(0, 1)  # Tendency to mate cooperatively (0 = solo, 1 = highly cooperative)
        else:
            # Combine parent DNA with mutation
            if parent2_dna is None:
                # Single parent (shouldn't happen in normal reproduction, but handle gracefully)
                self._inherit_from_parent(parent_dna)
            else:
                # Sexual reproduction (combine two parents)
                self._combine_parents(parent_dna, parent2_dna)
            
            # Apply mutations
            self._mutate(mutation_multiplier)
    
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
        self.cooperative_hunting = parent.cooperative_hunting
        self.cooperative_mating = parent.cooperative_mating
    
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
        self.cooperative_hunting = (parent1.cooperative_hunting + parent2.cooperative_hunting) / 2
        self.cooperative_mating = (parent1.cooperative_mating + parent2.cooperative_mating) / 2
    
    def _mutate(self, multiplier: float = 1.0):
        """Apply random mutations to DNA."""
        mutation_rate = 0.1 * multiplier  # Base 10% chance adjusted by multiplier
        
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
            self.min_mating_age = np.clip(self.min_mating_age + random.uniform(-3, 3), 15, 60)  # Can mutate between 15-60 seconds
        
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

        if random.random() < mutation_rate:
            self.cooperative_hunting = np.clip(self.cooperative_hunting + random.uniform(-0.2, 0.2), 0, 1)

        if random.random() < mutation_rate:
            self.cooperative_mating = np.clip(self.cooperative_mating + random.uniform(-0.2, 0.2), 0, 1)

