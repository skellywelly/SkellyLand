"""Organism class with DNA-based characteristics and neural network control."""

import numpy as np
import random
import math
from typing import List, Optional, Tuple

from constants import INPUT_SIZE, MAX_SPEED_CAP, WORLD_WIDTH, WORLD_HEIGHT
from neural_network import NeuralNetwork
from dna import DNA
from food import Food

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

        # Pain and damage tracking (for neural network)
        self.pain_level = 0.0  # Current pain (0-1, decays over time)
        self.pain_decay_rate = 0.02  # How fast pain decays per second
        
        # Behavior
        self.target = None  # Target food or organism
        self.last_reproduction = 0
        self.in_combat = False  # Whether currently fighting
        self.combat_target = None  # Organism being fought
        self.last_interaction = 0  # Time since last interaction (prevent spam)
        self._death_cause = None  # Track cause of death for statistics
        
        # Neural network (DNA-controlled architecture)
        # Output size: turn_left, turn_right, flagella_count outputs, mate, fight, run, chase, feed, avoid_toxic, reproduction, overcrowding_threshold_mod, overcrowding_distance_mod, metabolic_rate_control, learning_rate_control, cooperative_hunt, cooperative_mate
        # Total: 2 + flagella_count + 11 = flagella_count + 13
        output_size = self.dna.flagella_count + 15
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
        
        # Consume energy (metabolism) - apply global multiplier and neural network control
        old_energy = self.energy
        metabolism_mult = getattr(self, '_temp_metabolism_mult', 1.0)
        neural_metabolism_mult = getattr(self, 'metabolic_rate_multiplier', 1.0)
        self.energy -= self.dna.metabolism * metabolism_mult * neural_metabolism_mult * dt

        # Update pain level (decays over time)
        self.pain_level = max(0.0, self.pain_level - self.pain_decay_rate * dt)
        
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

        # Perform cooperative behaviors
        self._perform_cooperative_behaviors(organisms, dt)

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

        # Use effective vision range for sensing
        vision_range = getattr(self, 'effective_vision_range', self.dna.vision_range)

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
        # Social interaction history (recent fights and matings)
        # Positive for recent mating, negative for recent fighting
        inputs[19] = self.last_mated * 0.5 - (1 if hasattr(self, "combat_target") and self.combat_target else 0) * 0.3
        
        # Environmental awareness - density gradients
        # Calculate food density in different directions
        food_front = 0
        food_back = 0
        org_front = 0
        org_back = 0
        
        # Sample points in front and back of the organism
        sample_distance = vision_range * 0.5
        front_x = self.x + math.cos(self.angle) * sample_distance
        front_y = self.y + math.sin(self.angle) * sample_distance
        back_x = self.x - math.cos(self.angle) * sample_distance
        back_y = self.y - math.sin(self.angle) * sample_distance
        
        # Count food in front and back regions
        for food in foods:
            dx_front = food.x - front_x
            dy_front = food.y - front_y
            dist_front = math.sqrt(dx_front * dx_front + dy_front * dy_front)
            
            dx_back = food.x - back_x
            dy_back = food.y - back_y
            dist_back = math.sqrt(dx_back * dx_back + dy_back * dy_back)
            
            if dist_front < sample_distance:
                food_front += 1
            if dist_back < sample_distance:
                food_back += 1
        
        # Count organisms in front and back regions
        for org in organisms:
            if org is self:
                continue
            
            dx_front = org.x - front_x
            dy_front = org.y - front_y
            dist_front = math.sqrt(dx_front * dx_front + dy_front * dy_front)
            
            dx_back = org.x - back_x
            dy_back = org.y - back_y
            dist_back = math.sqrt(dx_back * dx_back + dy_back * dy_back)
            
            if dist_front < sample_distance:
                org_front += 1
            if dist_back < sample_distance:
                org_back += 1
        
        # Normalize gradients (difference between front and back, scaled to 0-1)
        food_gradient = min(1.0, (food_front - food_back + 10) / 20.0)  # +10 to shift from -10..10 to 0..20, then /20
        org_gradient = min(1.0, (org_front - org_back + 10) / 20.0)    # Same normalization
        
        inputs[20] = food_gradient  # Food density gradient (0 = more food behind, 1 = more food in front)
        inputs[21] = org_gradient   # Organism density gradient (0 = more orgs behind, 1 = more orgs in front)
        
        # Additional environmental awareness
        # Total food and organism counts in vision range
        total_food_in_range = sum(1 for food in foods if math.sqrt((food.x - self.x)**2 + (food.y - self.y)**2) < vision_range)
        total_org_in_range = sum(1 for org in organisms if org is not self and math.sqrt((org.x - self.x)**2 + (org.y - self.y)**2) < vision_range)
        
        inputs[22] = min(total_food_in_range / 50.0, 1.0)  # Total food density (normalized)
        inputs[23] = min(total_org_in_range / 20.0, 1.0)  # Total organism density (normalized)
        
        # Advanced environmental awareness - quality and movement patterns
        # Food quality distribution (toxic vs safe food)
        safe_food_front = sum(1 for food in foods
                            if math.sqrt((food.x - front_x)**2 + (food.y - front_y)**2) < sample_distance
                            and food.toxicity == 0.0)
        safe_food_back = sum(1 for food in foods
                           if math.sqrt((food.x - back_x)**2 + (food.y - back_y)**2) < sample_distance
                           and food.toxicity == 0.0)
        toxic_food_front = sum(1 for food in foods
                             if math.sqrt((food.x - front_x)**2 + (food.y - front_y)**2) < sample_distance
                             and food.toxicity > 0.0)
        toxic_food_back = sum(1 for food in foods
                            if math.sqrt((food.x - back_x)**2 + (food.y - back_y)**2) < sample_distance
                            and food.toxicity > 0.0)
        
        # Normalize food quality gradients
        total_safe_front = safe_food_front + 1  # +1 to avoid division by zero
        total_safe_back = safe_food_back + 1
        safe_food_gradient = min(1.0, safe_food_front / total_safe_front - safe_food_back / total_safe_back + 1.0) / 2.0
        
        total_toxic_front = toxic_food_front + 1
        total_toxic_back = toxic_food_back + 1
        toxic_food_gradient = min(1.0, toxic_food_front / total_toxic_front - toxic_food_back / total_toxic_back + 1.0) / 2.0
        
        # Movement flow patterns (average velocity directions of nearby organisms)
        front_velocities = []
        back_velocities = []
        
        for org in organisms:
            if org is self:
                continue
            
            # Check if in front region
            if math.sqrt((org.x - front_x)**2 + (org.y - front_y)**2) < sample_distance:
                front_velocities.append((org.vx, org.vy))
            
            # Check if in back region
            if math.sqrt((org.x - back_x)**2 + (org.y - back_y)**2) < sample_distance:
                back_velocities.append((org.vx, org.vy))
        
        # Calculate average movement directions
        front_avg_angle = 0.0
        back_avg_angle = 0.0
        
        if front_velocities:
            avg_vx = sum(v[0] for v in front_velocities) / len(front_velocities)
            avg_vy = sum(v[1] for v in front_velocities) / len(front_velocities)
            front_avg_angle = math.atan2(avg_vy, avg_vx) / math.pi  # Normalized to [-1, 1]
        
        if back_velocities:
            avg_vx = sum(v[0] for v in back_velocities) / len(back_velocities)
            avg_vy = sum(v[1] for v in back_velocities) / len(back_velocities)
            back_avg_angle = math.atan2(avg_vy, avg_vx) / math.pi  # Normalized to [-1, 1]
        
        # Movement flow gradient (difference in movement directions)
        movement_flow_gradient = (front_avg_angle - back_avg_angle + 2.0) / 4.0  # Normalized to [0, 1]
        
        inputs[24] = safe_food_gradient  # Safe food concentration gradient
        inputs[25] = toxic_food_gradient  # Toxic food concentration gradient
        inputs[26] = front_avg_angle * 0.5 + 0.5  # Front region movement direction (normalized to [0, 1])
        inputs[27] = movement_flow_gradient  # Movement flow gradient between regions

        # Cooperative behavior inputs
        # Check for cooperative hunting opportunities
        cooperative_hunting_opportunity = 0.0
        if nearest_org and nearest_org.size > self.size * 1.5:  # Prey is much larger than self
            # Count nearby organisms that might help hunt
            helper_count = 0
            for org in organisms:
                if org is self or org is nearest_org:
                    continue
                dx = org.x - self.x
                dy = org.y - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.dna.vision_range * 0.8:  # Within cooperative range
                    # Check if this organism is also targeting the same prey or is aggressive
                    if (hasattr(org, 'target') and org.target == nearest_org) or org.dna.aggression > 0.6:
                        helper_count += 1
            cooperative_hunting_opportunity = min(helper_count / 3.0, 1.0)  # Normalize to 0-1

        # Check for cooperative mating opportunities
        cooperative_mating_opportunity = 0.0
        mating_orgs_nearby = 0
        for org in organisms:
            if org is self:
                continue
            dx = org.x - self.x
            dy = org.y - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.dna.vision_range * 0.6:  # Closer range for mating cooperation
                # Count organisms that might be seeking mates
                if hasattr(org, 'neural_mate') and org.neural_mate and org.energy > org.dna.reproduction_threshold * 0.8:
                    mating_orgs_nearby += 1
        cooperative_mating_opportunity = min(mating_orgs_nearby / 2.0, 1.0)  # Normalize to 0-1

        inputs[28] = cooperative_hunting_opportunity  # Cooperative hunting opportunity (0-1)
        inputs[29] = cooperative_mating_opportunity  # Cooperative mating opportunity (0-1)
        inputs[30] = self.dna.cooperative_hunting  # DNA cooperativeness for hunting (0-1)
        inputs[31] = self.dna.cooperative_mating  # DNA cooperativeness for mating (0-1)

        # Hazard Awareness Inputs
        nearest_hazard_dist = 1.0
        nearest_hazard_angle = 0.0
        
        if hasattr(self, 'detected_hazard') and self.detected_hazard:
             # Calculate distance and angle to known hazard
             dx = self.detected_hazard.x - self.x
             dy = self.detected_hazard.y - self.y
             dist = math.sqrt(dx * dx + dy * dy)
             normalized_dist = min(dist / self.dna.vision_range, 1.0)
             angle = math.atan2(dy, dx) - self.angle
             normalized_angle = angle / math.pi
             
             nearest_hazard_dist = normalized_dist
             nearest_hazard_angle = normalized_angle

        inputs[32] = nearest_hazard_dist # Distance to nearest hazard (1.0 = safe/far)
        inputs[33] = nearest_hazard_angle # Angle to nearest hazard

        # Remaining slots reserved/padding
        inputs[34] = 0.0
        inputs[35] = 0.0
        inputs[36] = 0.0
        inputs[37] = 0.0
        inputs[38] = 0.0
        inputs[39] = 0.0

        return inputs
        
        # Note: Energy level (inputs[6]) already provides information about fullness
        # The neural network can learn to avoid food when energy is high (close to 1.0)
        # The neural network can learn to seek mates with high energy, speed, and complexity
        # The neural network can learn to fight more when overcrowded (inputs[17] and inputs[18])
        # The neural network can learn pain avoidance and social behavior patterns (inputs[19])
        # The neural network can learn to move toward resource concentrations (inputs[20] and inputs[21])
        # The neural network can learn to respond to overall environmental density (inputs[22] and inputs[23])
        # The neural network can learn food quality preferences (inputs[24] and inputs[25])
        # The neural network can learn movement flow patterns (inputs[26] and inputs[27])
        

        
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
        
        # Reward for cooperative behaviors
        if hasattr(self, 'neural_cooperative_hunt') and self.neural_cooperative_hunt:
            # Reward for participating in cooperative hunting
            # Check if we're targeting prey that others are also targeting
            if self.target and hasattr(self.target, 'size'):
                prey_size = self.target.size
                self_size = self.size

                # Reward for hunting prey larger than self (cooperative advantage)
                if prey_size > self_size * 1.2:
                    # Count how many others are hunting the same prey
                    cooperative_partners = 0
                    for org in organisms:
                        if org is not self and hasattr(org, 'target') and org.target == self.target:
                            cooperative_partners += 1

                    # Reward increases with number of cooperative partners
                    if cooperative_partners > 0:
                        cooperation_reward = min(cooperative_partners * 0.3, 1.0)
                        reward += cooperation_reward

        if hasattr(self, 'neural_cooperative_mate') and self.neural_cooperative_mate:
            # Reward for participating in cooperative mating assistance
            # Check if we're facilitating mating by being near potential mates
            nearby_mating_orgs = 0
            for org in organisms:
                if org is not self and math.sqrt((org.x - self.x)**2 + (org.y - self.y)**2) < self.dna.vision_range * 0.7:
                    if hasattr(org, 'neural_mate') and org.neural_mate:
                        nearby_mating_orgs += 1

            # Small reward for being in position to assist mating
            if nearby_mating_orgs >= 2:  # At least two organisms seeking mates nearby
                reward += 0.2

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
        # Moderately permissive threshold - encourage mating but not excessively
        self.neural_mate = (mate_output + mate_bias) > -0.8  # Moderate threshold for balanced mating
        
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

        # Cooperative behavior decisions
        self.neural_cooperative_hunt = outputs[base_idx + 9] > 0.0  # Cooperative hunting decision
        self.neural_cooperative_mate = outputs[base_idx + 10] > 0.0  # Cooperative mating decision
        
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
        
        # Apply absolute speed cap to prevent excessive movement
        target_speed = min(target_speed, MAX_SPEED_CAP)
        
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
        
        # Apply absolute speed cap to prevent excessive movement speeds
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > MAX_SPEED_CAP:
            self.vx = (self.vx / speed) * MAX_SPEED_CAP
            self.vy = (self.vy / speed) * MAX_SPEED_CAP
        
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
                
                # Check if organisms can interact (fight or mate) before getting too close
                can_fight = False
                can_mate = False
                
                # Check if they can fight
                self._check_overcrowding(organisms)
                org._check_overcrowding(organisms)
                self_fight_probability = 1.0 if (hasattr(self, 'neural_fight') and self.neural_fight) else 0.0
                org_fight_probability = 1.0 if (hasattr(org, 'neural_fight') and org.neural_fight) else 0.0
                if hasattr(self, 'is_overcrowded') and self.is_overcrowded:
                    self_fight_probability = min(1.0, self_fight_probability + 0.3)
                if hasattr(org, 'is_overcrowded') and org.is_overcrowded:
                    org_fight_probability = min(1.0, org_fight_probability + 0.3)
                can_fight = (self_fight_probability > 0.7 and org_fight_probability > 0.7)
                
                # Check if they can mate
                min_energy_for_mating = 30.0
                can_mate = (self.age >= self.dna.min_mating_age and 
                           org.age >= org.dna.min_mating_age and
                           self.energy > min_energy_for_mating and
                           org.energy > min_energy_for_mating and
                           self.age - self.last_reproduction > 1.2 and
                           org.age - org.last_reproduction > 1.2)
                
                # If they can't interact at all, ignore each other and continue normal behavior
                if not can_fight and not can_mate:
                    # Clear any targeting of each other
                    if hasattr(self, 'target') and self.target == org:
                        self.target = None
                    if hasattr(org, 'target') and org.target == self:
                        org.target = None
                    # Only push apart if very close (collision distance)
                    if distance < self.size * 1.2:
                        if distance > 0:
                            push_force = 15.0
                            push_x = (dx / distance) * push_force * dt
                            push_y = (dy / distance) * push_force * dt
                            self.x -= push_x
                            self.y -= push_y
                            org.x += push_x
                            org.y += push_y
                    continue  # Ignore each other and continue looking for food/etc
                
                # Neural network decides: fight, mate, run, or chase
                # Both organisms must be close enough
                if distance < self.size * 1.8:  # Moderately close = potential interaction
                    if can_fight:
                        # Both want to fight - engage in combat
                        self.in_combat = True
                        self.combat_target = org
                        org.in_combat = True
                        org.combat_target = self
                        self.last_interaction = 0.0
                        org.last_interaction = 0.0
                        # Track fight (will be done in Simulation class)
                        return self._fight_organism(org, dt)
                    elif can_mate:
                        # Can mate - proceed with mating
                        # Calculate mating viability for quality assessment
                        age_ready_self = 1.0 if self.age >= self.dna.min_mating_age else 0.0
                        age_ready_org = 1.0 if org.age >= org.dna.min_mating_age else 0.0
                        energy_score_self = min(1.0, self.energy / min_energy_for_mating)
                        energy_score_org = min(1.0, org.energy / min_energy_for_mating)
                        cooldown_score_self = min(1.0, (self.age - self.last_reproduction) / 1.2)
                        cooldown_score_org = min(1.0, (org.age - org.last_reproduction) / 1.2)
                        mating_viability = (age_ready_self + age_ready_org + energy_score_self + energy_score_org +
                                          cooldown_score_self + cooldown_score_org) / 6.0

                        # Mate if viability is high enough, or with low probability for moderate viability
                        if mating_viability >= 0.9 or (mating_viability >= 0.6 and random.random() < mating_viability * 0.2):
                            # Mate - this is the default behavior when organisms meet
                            self._mate_with_organism(org, organisms)
                            self.last_interaction = 0.0
                            org.last_interaction = 0.0
                            # Track mating (will be done in Simulation class)
                            return None
                        else:
                            # Can mate but viability is low - gentle push and continue
                            if distance > 0:
                                push_force = 10.0
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

    def _perform_cooperative_behaviors(self, organisms: List['Organism'], dt: float):
        """Perform cooperative behaviors: hunting and mating assistance."""
        if not hasattr(self, 'neural_cooperative_hunt') or not hasattr(self, 'neural_cooperative_mate'):
            return

        # Cooperative hunting: help hunt larger prey
        if self.neural_cooperative_hunt and self.dna.cooperative_hunting > 0.3:
            self._assist_cooperative_hunt(organisms, dt)

        # Cooperative mating: help find mates or protect mating pairs
        if self.neural_cooperative_mate and self.dna.cooperative_mating > 0.3:
            self._assist_cooperative_mating(organisms, dt)

    def _assist_cooperative_hunt(self, organisms: List['Organism'], dt: float):
        """Assist in cooperative hunting by helping to surround or distract larger prey."""
        # Find potential prey that could benefit from cooperative hunting
        potential_prey = None
        helpers = []

        for org in organisms:
            if org is self:
                continue

            # Check if this organism is much larger (potential prey)
            if org.size > self.size * 1.5:
                # Count how many organisms are already targeting this prey
                targeting_count = 0
                for helper in organisms:
                    if helper is not self and helper is not org:
                        if (hasattr(helper, 'target') and helper.target == org) or \
                           (hasattr(helper, 'neural_cooperative_hunt') and helper.neural_cooperative_hunt):
                            targeting_count += 1

                # If there are helpers and prey is large enough, join the hunt
                if targeting_count >= 1 and org.energy > self.energy * 2:
                    potential_prey = org
                    # Find actual helpers
                    for helper in organisms:
                        if helper is not self and helper is not org and \
                           ((hasattr(helper, 'target') and helper.target == org) or \
                            (hasattr(helper, 'neural_cooperative_hunt') and helper.neural_cooperative_hunt)):
                            helpers.append(helper)
                    break

        if potential_prey and len(helpers) >= 1:
            # Join the cooperative hunt
            self.target = potential_prey

            # Position yourself strategically (try to flank or surround)
            dx = potential_prey.x - self.x
            dy = potential_prey.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 0:
                # Try to position at an angle to the prey, away from other helpers
                angle_to_prey = math.atan2(dy, dx)
                best_angle = angle_to_prey

                # Find angle with least helpers
                min_helpers_at_angle = float('inf')
                for test_angle in [angle_to_prey - math.pi/2, angle_to_prey + math.pi/2, angle_to_prey - math.pi, angle_to_prey]:
                    helpers_at_angle = 0
                    for helper in helpers:
                        hx = helper.x - potential_prey.x
                        hy = helper.y - potential_prey.y
                        helper_angle = math.atan2(hy, hx)
                        angle_diff = abs(helper_angle - test_angle)
                        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                        if angle_diff < math.pi/3:  # Within 60 degrees
                            helpers_at_angle += 1

                    if helpers_at_angle < min_helpers_at_angle:
                        min_helpers_at_angle = helpers_at_angle
                        best_angle = test_angle

                # Move toward flanking position
                flank_distance = potential_prey.size * 3
                target_x = potential_prey.x + math.cos(best_angle) * flank_distance
                target_y = potential_prey.y + math.sin(best_angle) * flank_distance

                # Adjust position gradually, respecting speed cap
                move_x = (target_x - self.x) * 0.1 * dt * 10
                move_y = (target_y - self.y) * 0.1 * dt * 10

                # Limit cooperative movement speed
                move_speed = math.sqrt(move_x * move_x + move_y * move_y) / dt
                if move_speed > MAX_SPEED_CAP:
                    move_x = (move_x / move_speed) * MAX_SPEED_CAP * dt
                    move_y = (move_y / move_speed) * MAX_SPEED_CAP * dt

                self.x += move_x
                self.y += move_y

    def _assist_cooperative_mating(self, organisms: List['Organism'], dt: float):
        """Assist in cooperative mating by helping find mates or protecting mating pairs."""
        # Look for organisms that are seeking mates
        for org in organisms:
            if org is self:
                continue

            # Check if this organism wants to mate but hasn't found a partner
            if hasattr(org, 'neural_mate') and org.neural_mate and org.energy > org.dna.reproduction_threshold * 0.8:
                # Check if there are potential mates nearby that this organism could help attract
                potential_mates = []
                for mate_candidate in organisms:
                    if mate_candidate is not self and mate_candidate is not org:
                        dx = mate_candidate.x - org.x
                        dy = mate_candidate.y - org.y
                        distance = math.sqrt(dx * dx + dy * dy)

                        if distance < self.dna.vision_range and \
                           hasattr(mate_candidate, 'neural_mate') and mate_candidate.neural_mate and \
                           mate_candidate.energy > mate_candidate.dna.reproduction_threshold * 0.8:
                            potential_mates.append(mate_candidate)

                if potential_mates:
                    # Help by positioning near the potential mate to encourage interaction
                    # or by moving the seeking organism toward potential mates
                    closest_mate = min(potential_mates, key=lambda m: math.sqrt((m.x - org.x)**2 + (m.y - org.y)**2))

                    # If the helper is closer to the potential mate, guide the seeker toward them
                    dist_helper_to_mate = math.sqrt((closest_mate.x - self.x)**2 + (closest_mate.y - self.y)**2)
                    dist_seeker_to_mate = math.sqrt((closest_mate.x - org.x)**2 + (closest_mate.y - org.y)**2)

                    if dist_helper_to_mate < dist_seeker_to_mate:
                        # Position between seeker and potential mate to facilitate introduction
                        mid_x = (org.x + closest_mate.x) / 2
                        mid_y = (org.y + closest_mate.y) / 2

                        # Move toward the midpoint
                        dx = mid_x - self.x
                        dy = mid_y - self.y
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > 5:
                            move_x = dx / dist * 20 * dt
                            move_y = dy / dist * 20 * dt

                            # Limit cooperative movement speed
                            move_speed = math.sqrt(move_x * move_x + move_y * move_y) / dt
                            if move_speed > MAX_SPEED_CAP:
                                move_x = (move_x / move_speed) * MAX_SPEED_CAP * dt
                                move_y = (move_y / move_speed) * MAX_SPEED_CAP * dt

                            self.x += move_x
                            self.y += move_y
                    break  # Only help one mating pair at a time

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
            mutation_mult = getattr(self, '_temp_mutation_mult', 1.0)
            new_dna = DNA(parent_dna=self.dna, parent2_dna=partner.dna, mutation_multiplier=mutation_mult)
            
            # Spread offspring around the mating location
            angle = (2 * math.pi * i) / num_offspring  # Distribute evenly in a circle
            distance = random.uniform(20, 50)  # Distance from center
            new_x = (self.x + partner.x) / 2 + math.cos(angle) * distance
            new_y = (self.y + partner.y) / 2 + math.sin(angle) * distance
            # Wrap around world bounds
            new_x = new_x % WORLD_WIDTH
            new_y = new_y % WORLD_HEIGHT
            new_org = Organism(new_x, new_y, new_dna)
            
            # Eject offspring with velocity to separate them from parents and siblings
            # Calculate direction away from parent center
            parent_center_x = (self.x + partner.x) / 2
            parent_center_y = (self.y + partner.y) / 2
            
            # Direction from parent center to offspring position
            dx = new_x - parent_center_x
            dy = new_y - parent_center_y
            dist_from_center = math.sqrt(dx * dx + dy * dy)
            
            if dist_from_center > 0:
                # Normalize direction
                dir_x = dx / dist_from_center
                dir_y = dy / dist_from_center
            else:
                # Fallback to random direction if at center
                angle = random.uniform(0, 2 * math.pi)
                dir_x = math.cos(angle)
                dir_y = math.sin(angle)
            
            # Strong ejection velocity to ensure separation
            # Minimum ejection speed of 50 pixels/second, up to 150 pixels/second
            parent_avg_max_speed = (self.dna.max_speed + partner.dna.max_speed) / 2.0
            base_ejection_speed = max(50.0, parent_avg_max_speed * 10.0)  # Strong base speed
            ejection_speed = base_ejection_speed * random.uniform(1.0, 2.0)  # 50-150 pixels/second
            
            # Add some random variation to direction to prevent all going same way
            angle_variation = random.uniform(-0.3, 0.3)  # ±17 degrees variation
            cos_var = math.cos(angle_variation)
            sin_var = math.sin(angle_variation)
            final_dir_x = dir_x * cos_var - dir_y * sin_var
            final_dir_y = dir_x * sin_var + dir_y * cos_var
            
            # Set ejection velocity
            new_org.vx = final_dir_x * ejection_speed
            new_org.vy = final_dir_y * ejection_speed
            
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

