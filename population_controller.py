"""AI controller for population management."""

import numpy as np
import random
from typing import Tuple

class PopulationControllerAI:
    """AI that learns to control simulation parameters for optimal population management."""

    def __init__(self):
        # Neural network for parameter control
        # State: [population, trend, metabolism, flagella, aggression, reproduction, food_density, avg_energy, avg_age, divine_energy]
        # Actions: [params... (12), spawn_manna, spawn_hazard]
        self.state_size = 10  # Added Divine Energy
        self.action_size = 14  # Added 2 Divine Powers

        # Neural network weights
        self.weights = np.random.randn(self.state_size, self.action_size) * 0.1

        # Learning parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

        # Experience replay
        self.memory = []
        self.max_memory = 1000

        # Previous state and action for learning
        self.prev_state = None
        self.prev_actions = None

    def get_state(self, simulation):
        """Get current state representation."""
        current_pop = len(simulation.organisms)
        max_pop = 150.0  # Normalize against expected max

        # Population trend (change over last minute)
        if len(simulation.population_history) >= 6:
            trend = current_pop - simulation.population_history[0]
        else:
            trend = 0

        # Food density (ratio of food to organisms, normalized)
        food_count = len(simulation.foods)
        food_density = min(2.0, food_count / max(1, current_pop)) / 2.0

        # Average energy level (normalized 0-1)
        avg_energy = 0.0
        avg_age = 0.0
        if current_pop > 0:
            avg_energy = sum(org.energy for org in simulation.organisms) / (current_pop * 150.0) # 150 is max_energy base
            avg_age = sum(org.age for org in simulation.organisms) / (current_pop * 100.0) # Normalize age against 100s

        # Normalize values
        pop_norm = min(1.0, current_pop / max_pop)
        trend_norm = np.tanh(trend / 20.0)  # Normalize trend
        divine_energy_norm = simulation.divine_energy / 100.0

        state = np.array([
            pop_norm,
            trend_norm,
            simulation.metabolism_multiplier / 2.0,
            simulation.flagella_impulse_multiplier / 2.0,
            simulation.aggression_multiplier / 2.0,
            simulation.reproduction_desire_multiplier / 2.0,
            food_density,
            avg_energy,
            avg_age,
            divine_energy_norm
        ])

        return state

    def choose_actions(self, state):
        """Choose actions for each parameter using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Random exploration
            actions = np.random.choice([-1, 0, 1], size=6)  # Parameters
            # Random powers
            power_action = random.choice([0, 1, 2]) # 0=None, 1=Manna, 2=Hazard
            
            # Combine
            full_actions = np.zeros(8, dtype=int)
            full_actions[:6] = actions
            if power_action == 1: full_actions[6] = 1 # Manna
            if power_action == 2: full_actions[7] = 1 # Hazard
            
            return full_actions
        else:
            # Greedy action selection
            q_values = np.dot(state, self.weights)

            # Convert Q-values to actions
            actions = np.zeros(8, dtype=int)
            
            # Parameters (indices 0-5)
            for i in range(6):  
                up_q = q_values[i*2]      
                down_q = q_values[i*2 + 1]

                if up_q > down_q:
                    actions[i] = 1   # up
                elif down_q > up_q:
                    actions[i] = -1  # down
                else:
                    actions[i] = 0   # stay
            
            # Powers (indices 6-7 mapped to weights 12, 13)
            # Threshold based trigger
            if q_values[12] > 0.5: actions[6] = 1 # Manna
            if q_values[13] > 0.5: actions[7] = 1 # Hazard

            return actions

    def apply_actions(self, simulation, actions):
        """Apply the chosen actions to simulation parameters."""
        adjustment = simulation.parameter_adjustment_rate

        # ... (Parameter adjustments handled below) ...
        # Actions 0-5 correspond to parameters
        # Actions 6 (Manna) and 7 (Hazard) are Divine Powers

        # Metabolism (index 0)
        if actions[0] == 1:
            simulation.metabolism_multiplier = min(1.5, simulation.metabolism_multiplier + adjustment)
        elif actions[0] == -1:
            simulation.metabolism_multiplier = max(0.5, simulation.metabolism_multiplier - adjustment)

        # Flagella (index 1)
        if actions[1] == 1:
            simulation.flagella_impulse_multiplier = min(3.0, simulation.flagella_impulse_multiplier + adjustment)
        elif actions[1] == -1:
            simulation.flagella_impulse_multiplier = max(0.1, simulation.flagella_impulse_multiplier - adjustment)

        # Aggression (index 2)
        if actions[2] == 1:
            simulation.aggression_multiplier = min(3.0, simulation.aggression_multiplier + adjustment)
        elif actions[2] == -1:
            simulation.aggression_multiplier = max(0.1, simulation.aggression_multiplier - adjustment)

        # Reproduction (index 3)
        if actions[3] == 1:
            simulation.reproduction_desire_multiplier = min(2.0, simulation.reproduction_desire_multiplier + adjustment)
        elif actions[3] == -1:
            simulation.reproduction_desire_multiplier = max(0.3, simulation.reproduction_desire_multiplier - adjustment)

        # Food Growth (index 4)
        if actions[4] == 1:
            simulation.food_growth_multiplier = min(3.0, simulation.food_growth_multiplier + adjustment)
        elif actions[4] == -1:
            simulation.food_growth_multiplier = max(0.1, simulation.food_growth_multiplier - adjustment)

        # Mutation Rate (index 5)
        if actions[5] == 1:
            simulation.mutation_rate_multiplier = min(3.0, simulation.mutation_rate_multiplier + adjustment)
        elif actions[5] == -1:
            simulation.mutation_rate_multiplier = max(0.1, simulation.mutation_rate_multiplier - adjustment)

        # Divine Powers
        # Manna Rain (Cost: 30)
        if actions[6] == 1 and simulation.divine_energy >= 30:
            simulation._spawn_manna_drop()
            simulation.divine_energy -= 30
            if simulation.show_terminal_output:
                print("✨ Divine Power: MANNA RAIN invoked!")

        # Hazard Zone (Cost: 20)
        if actions[7] == 1 and simulation.divine_energy >= 20:
            simulation._spawn_hazard_zone()
            simulation.divine_energy -= 20
            if simulation.show_terminal_output:
                print("⚠️ Divine Power: HAZARD ZONE invoked!")

        # Track changes for HUD
        param_names = ['metabolism', 'flagella_impulse', 'aggression', 'reproduction', 'food_growth', 'mutation_rate']
        for i, param in enumerate(param_names):
            if actions[i] != 0:
                simulation.last_parameter_changes[param] = simulation.total_time

    def calculate_reward(self, simulation, prev_population, new_population):
        """Calculate reward based on population change and divine energy usage."""
        reward = 0

        # Base reward for population in target range
        target_center = (simulation.target_population_min + simulation.target_population_max) / 2
        distance_from_target = abs(new_population - target_center)
        max_distance = max(target_center - simulation.target_population_min,
                          simulation.target_population_max - target_center)

        # Reward for being close to target (0-1 scale)
        population_reward = max(0, 1.0 - (distance_from_target / max_distance))
        reward += population_reward * 2.0  # Scale up

        # Reward for Divine Energy efficiency
        # We want the AI to hoard energy and only use it when necessary
        energy_reward = simulation.divine_energy / 200.0 # Small bonus for keeping energy high
        reward += energy_reward

        # Reward for population stability (not oscillating wildly)
        if prev_population > 0:
            stability_penalty = abs(new_population - prev_population) / max(prev_population, 1)
            reward -= stability_penalty * 0.5  # Penalty for large swings

        # Bonus for population growth when low
        if prev_population < simulation.target_population_min and new_population > prev_population:
            reward += 1.0

        # Penalty for population decline when high
        if prev_population > simulation.target_population_max and new_population < prev_population:
            reward += 0.5

        return reward

    def learn(self, prev_state, actions, reward, new_state):
        """Update neural network using Q-learning."""
        if prev_state is None:
            return

        # Current Q-values
        current_q = np.dot(prev_state, self.weights)

        # Next Q-values (target)
        next_q = np.dot(new_state, self.weights)
        
        # Calculate max next Q for parameters (pairs)
        # First 6 actions (12 weights)
        max_next_q_params = np.max([next_q[i*2:i*2+2] for i in range(6)], axis=1)
        
        # Target Q-values
        target_q = current_q.copy()
        
        # Update Parameter Q-Values
        for i in range(6):  
            if actions[i] != 0:  
                action_idx = i * 2 + (0 if actions[i] == 1 else 1)
                target_q[action_idx] = reward + self.discount_factor * max_next_q_params[i]
        
        # Update Power Q-Values (indices 12, 13)
        # Actions[6] corresponds to index 12 (Manna)
        if actions[6] == 1:
            target_q[12] = reward + self.discount_factor * max(next_q[12], 0) # Simple max
        
        # Actions[7] corresponds to index 13 (Hazard)
        if actions[7] == 1:
            target_q[13] = reward + self.discount_factor * max(next_q[13], 0)

        # Update weights
        q_error = target_q - current_q
        self.weights += self.learning_rate * np.outer(prev_state, q_error)

        # Store experience
        self.memory.append((prev_state, actions, reward, new_state))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def save_weights(self, filename="ai_controller_weights.npy"):
        """Save learned weights to file."""
        np.save(filename, self.weights)

    def load_weights(self, filename="ai_controller_weights.npy"):
        """Load learned weights from file."""
        try:
            loaded_weights = np.load(filename)
            # Check if loaded weights match current architecture
            if loaded_weights.shape == (self.state_size, self.action_size):
                self.weights = loaded_weights
                print(f"Loaded AI weights from {filename}")
            else:
                print(f"Saved weights shape {loaded_weights.shape} doesn't match current architecture ({self.state_size}, {self.action_size})")
                print("Reinitializing with random weights")
                self.weights = np.random.randn(self.state_size, self.action_size) * 0.1
        except FileNotFoundError:
            print(f"No saved weights found at {filename}, using random initialization")

    def train_on_experience(self):
        """Train on random experiences from memory."""
        if len(self.memory) < 10:
            return

        # Train on random batch
        batch_size = min(32, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        for prev_state, actions, reward, new_state in batch:
            self.learn(prev_state, actions, reward, new_state)


