# Virtual Life Simulation

A fluid environment simulation where organisms evolve through DNA-based reproduction, mutation, and natural selection.

## Features

- **Digital DNA System**: Organisms have DNA that controls:
  - Food preferences (herbivore/carnivore)
  - Color and appearance
  - Shape (polygon with variable points)
  - Propulsion strength and efficiency
  - Speed, vision, metabolism, and more

- **Evolution Mechanics**:
  - Sexual reproduction (combining two parents' DNA)
  - Asexual reproduction (cloning with mutation)
  - DNA mutations that can change any trait
  - Natural selection through predation and energy management

- **World System**:
  - Large world (5000x5000) that's larger than the screen
  - Camera system with zoom and follow
  - Food spawning system
  - Wrapping world edges

- **Organism Behavior**:
  - Vision-based target finding
  - Energy-based movement and metabolism
  - Eating (food and other organisms)
  - Mating and reproduction
  - Death from starvation

## Controls

- **SPACE**: Pause/Unpause simulation
- **F**: Spawn additional food
- **Mouse Wheel**: Zoom in/out
- **Close Window**: Exit simulation

## Running

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```

## Requirements

- Python 3.8+
- pygame
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

