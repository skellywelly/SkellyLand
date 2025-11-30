"""Constants for the Virtual Life Simulation."""

# Screen and World Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WORLD_WIDTH = 5000
WORLD_HEIGHT = 5000
FPS = 60

# Colors
BACKGROUND_COLOR = (20, 30, 50)
FOOD_COLOR = (100, 200, 100)

# Neural Network Constants
INPUT_SIZE = 40  # Increased from 38 to include hazard awareness
MAX_SPEED_CAP = 300.0  # Absolute maximum speed limit for all organisms (pixels per second)
# OUTPUT_SIZE is now variable: flagella_count + 3 (turn_left, turn_right, reproduction)

