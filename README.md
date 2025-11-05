# UAV-Assisted IoV Task Offloading Simulator

Implementation of "NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks"

## Project Structure
```
uav_task_offloading/
├── frontend/
│   └── app.py                          # GUI application using tkinter
├── backend/
│   ├── __init__.py                     # Backend package initialization
│   ├── config.py                       # System parameters configuration
│   ├── models/
│   │   ├── __init__.py                 # Models package
│   │   ├── task.py                     # Task model
│   │   └── position.py                 # Position model
│   ├── algorithms/
│   │   ├── __init__.py                 # Algorithms package
│   │   ├── genetic_algorithm.py        # GA implementation
│   │   └── particle_swarm_optimization.py  # PSO implementation
│   ├── utils/
│   │   ├── __init__.py                 # Utils package
│   │   └── calculations.py             # Helper calculation functions
│   └── experiments/
│       ├── __init__.py                 # Experiments package
│       └── comparison.py               # Experiment runner for figures
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Features

- **Genetic Algorithm (GA)**: Optimizes task offloading decisions using evolutionary computation
- **Particle Swarm Optimization (PSO)**: Alternative optimization approach using swarm intelligence
- **Interactive GUI**: User-friendly interface built with tkinter
- **Comprehensive Visualization**: Generates figures comparing algorithm performance
- **Export Functionality**: Save results in CSV or JSON format

## Installation

1. Create and navigate to the project directory:
```bash
./setup.sh
cd uav_task_offloading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the code files to their respective locations as per the structure above.

## Usage

### Run the GUI Application
```bash
python frontend/app.py
```

### Run Standalone Experiments
```bash
python -c "from backend.experiments.comparison import run_comparison_experiments; from backend.config import SystemParameters; import random; import numpy as np; random.seed(42); np.random.seed(42); run_comparison_experiments(SystemParameters())"
```

## GUI Features

1. **Algorithm Selection**: Choose between GA, PSO, or both
2. **Parameter Configuration**:
   - Number of tasks (5-50)
   - Number of UAVs (2-10)
   - UAV flight speed (5-50 m/s)
   - Population/swarm size (5-20)
   - Max iterations (50-200)
   - GA parameters (crossover rate, mutation rate)
   - PSO parameters (inertia weight, cognitive/social coefficients)

3. **Simulation Control**:
   - Run Simulation: Execute the selected algorithm(s)
   - Generate All Figures: Create comprehensive comparison plots (Figures 3-6)
   - Export Results: Save results as CSV or JSON

4. **Visualization**:
   - Real-time convergence plot
   - Detailed results display
   - Progress tracking

## Output Files

When generating all figures, the following files are created:
- `uav_figures_3_4_5_comparison.png`: Processing time, energy consumption, and UAV speed impact
- `uav_figure_6_comparison.png`: Task processing scheme analysis

## Algorithm Details

### Genetic Algorithm (GA)
- Tournament selection
- Single-point crossover
- Bit-flip mutation
- Binary encoding for offloading decisions

### Particle Swarm Optimization (PSO)
- Continuous position space with sigmoid transformation
- Personal and global best tracking
- Velocity and position updates with inertia

## System Model

The simulator models:
- Vehicle tasks with varying data sizes and processing requirements
- UAV-assisted Mobile Edge Computing (MEC)
- NOMA-based multiple access
- Local processing vs. offloading decisions
- Energy consumption and processing time optimization

## References

Based on the paper:
"NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks"
by Xiao et al., 2024 3rd International Conference on Computing, Communication, Perception and Quantum Technology (CCPQT)

## License

This implementation is for educational and research purposes.
