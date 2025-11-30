"""Neural network for organism behavior control."""

import numpy as np
from typing import List, Optional
from constants import INPUT_SIZE


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

