"""
Virtual Life Simulation
A fluid environment where organisms evolve through DNA-based reproduction and mutation.
"""

import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from simulation import Simulation

# Initialize Pygame
pygame.init()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()

