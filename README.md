# UAV-Assisted IoV Task Offloading Simulator

A Python-based implementation and simulation of the research paper: **"NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks"** by Xiao et al. (2024).

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [System Model](#system-model)
- [Simulation Parameters](#simulation-parameters)
- [Results and Visualization](#results-and-visualization)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This simulator implements a **NOMA-MEC (Non-Orthogonal Multiple Access - Mobile Edge Computing)** based task offloading system for **UAV-assisted IoV (Internet of Vehicles)** networks. The system optimizes task allocation between local vehicle processing and UAV-based MEC servers to minimize overall processing time while satisfying energy and computational constraints.

### Key Objectives

- **Minimize** total task processing time across all vehicle users
- **Optimize** energy consumption in the system
- **Balance** local processing vs. offloading decisions
- **Consider** UAV flight dynamics and channel conditions

### Performance Highlights

- ✅ **Up to 56% reduction** in task processing time compared to All-Offload strategy
- ✅ **Efficient energy management** with NOMA-based resource allocation
- ✅ **Adaptive** to varying numbers of tasks and UAV flight speeds
- ✅ **Real-time convergence** visualization and analysis

## ✨ Features

### Core Capabilities

- **Multiple Optimization Algorithms**
  - Genetic Algorithm (GA) - Primary optimization approach
  - Particle Swarm Optimization (PSO) - Alternative swarm intelligence method
  - Comparative analysis between algorithms

- **Interactive GUI Application**
  - Built with Python tkinter
  - Real-time parameter configuration
  - Live convergence plotting
  - Results visualization and export

- **Comprehensive System Modeling**
  - Local processing model for vehicle tasks
  - MEC offloading model with NOMA
  - UAV flight dynamics and positioning
  - Channel gain and SINR calculations
  - Energy consumption tracking

- **Advanced Visualization**
  - Processing time comparison charts (Figure 3)
  - Energy consumption analysis (Figure 4)
  - UAV speed impact study (Figure 5)
  - Task processing scheme evaluation (Figure 6)

- **Data Export**
  - CSV format for spreadsheet analysis
  - JSON format for data interchange
  - PNG figures for publication-ready charts

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    IoV Environment                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Vehicle 1│  │ Vehicle 2│  │ Vehicle J│  ...         │
│  │  Task q₁ │  │  Task q₂ │  │  Task qⱼ │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │             │             │                      │
│       └─────────────┴─────────────┘                     │
│                     │                                    │
│            ┌────────▼────────┐                          │
│            │  NOMA Channel   │                          │
│            └────────┬────────┘                          │
│                     │                                    │
│       ┌─────────────┴─────────────┐                    │
│       │                           │                      │
│  ┌────▼────┐               ┌────▼────┐                │
│  │  UAV 1  │      ...      │  UAV I  │                 │
│  │ (MEC)   │               │ (MEC)   │                 │
│  └─────────┘               └─────────┘                 │
│                                                          │
│  Decision: Local Processing (x=0) or Offload (x=1)     │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure
```
uav_task_offloading/
├── frontend/
│   └── app.py                          # GUI application (tkinter)
│
├── backend/
│   ├── __init__.py                     # Backend package initialization
│   ├── config.py                       # System parameters configuration
│   │
│   ├── models/
│   │   ├── __init__.py                 # Models package
│   │   ├── task.py                     # Task model (Task class)
│   │   └── position.py                 # Position model & generation
│   │
│   ├── algorithms/
│   │   ├── __init__.py                 # Algorithms package
│   │   ├── genetic_algorithm.py        # GA implementation (TOA)
│   │   └── particle_swarm_optimization.py  # PSO implementation
│   │
│   ├── utils/
│   │   ├── __init__.py                 # Utils package
│   │   └── calculations.py             # Helper calculation functions
│   │                                   # (distance, channel gain, SINR, 
│   │                                   #  rate, fitness, etc.)
│   └── experiments/
│       ├── __init__.py                 # Experiments package
│       └── comparison.py               # Generate Figures 3-6
│
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── run.sh                              # Quick start script
└── setup.sh                            # Project setup script
```

### Component Descriptions

| Component | Description |
|-----------|-------------|
| `frontend/app.py` | Main GUI application with parameter controls, algorithm selection, and visualization |
| `backend/config.py` | System parameters (UAV count, task numbers, channel parameters, etc.) |
| `backend/models/task.py` | Task model with data size, processing density, and deadline |
| `backend/models/position.py` | 3D position handling for UAVs and vehicles |
| `backend/algorithms/genetic_algorithm.py` | GA-based TOA (Task Offloading Algorithm) |
| `backend/algorithms/particle_swarm_optimization.py` | PSO alternative optimization |
| `backend/utils/calculations.py` | Mathematical utilities (distance, SINR, fitness) |
| `backend/experiments/comparison.py` | Batch experiments for research figures |

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) Virtual environment tool




### 1:Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy >= 1.20.0` - Numerical computations
- `matplotlib >= 3.3.0` - Visualization and plotting

### 2: Run
```bash
./run.sh
or 
python frontend/app.py
```




#### GUI Features

1. **Algorithm Selection**
   - Choose GA, PSO, or both for comparison
   - View algorithm-specific parameters

2. **Parameter Configuration**
   - **Number of Tasks**: 5-50 vehicle tasks
   - **Number of UAVs**: 2-10 MEC-enabled UAVs
   - **UAV Speed**: 5-50 m/s flight speed
   - **Population/Swarm Size**: 5-20 individuals
   - **Max Iterations**: 50-200 generations

3. **GA-Specific Parameters**
   - Crossover Rate (pc): 0.1-0.9
   - Mutation Rate (pm): 0.01-0.2

4. **PSO-Specific Parameters**
   - Inertia Weight (w): 0.1-0.9
   - Cognitive Coefficient (c1): 0.5-2.5
   - Social Coefficient (c2): 0.5-2.5

5. **Actions**
   - **Run Simulation**: Execute selected algorithm(s)
   - **Generate All Figures**: Create Figures 3-6 from paper
   - **Export Results**: Save as CSV or JSON

### Method 2: Programmatic Usage

#### Run Single Experiment
```python
from backend.algorithms.genetic_algorithm import genetic_algorithm
from backend.config import SystemParameters
from backend.models.task import Task
from backend.models.position import generate_positions
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize system
params = SystemParameters()
tasks = [Task(params) for _ in range(params.num_tasks)]
uav_positions, vehicle_positions = generate_positions(params)

# Run GA
best_solution, best_fitness, convergence = genetic_algorithm(
    tasks=tasks,
    uav_positions=uav_positions,
    vehicle_positions=vehicle_positions,
    params=params,
    population_size=10,
    max_iterations=100,
    crossover_rate=0.8,
    mutation_rate=0.05
)

print(f"Best Fitness: {best_fitness}")
print(f"Best Solution: {best_solution}")
```

#### Generate Research Figures
```python
from backend.experiments.comparison import run_comparison_experiments
from backend.config import SystemParameters
import random
import numpy as np

# Set seeds
random.seed(42)
np.random.seed(42)

# Run all experiments (Figures 3-6)
params = SystemParameters()
run_comparison_experiments(params)

# Output files created:
# - uav_figures_3_4_5_comparison.png
# - uav_figure_6_comparison.png
```

### Method 3: Command Line
```bash
# Quick experiment run
python -c "from backend.experiments.comparison import run_comparison_experiments; \
           from backend.config import SystemParameters; \
           import random; import numpy as np; \
           random.seed(42); np.random.seed(42); \
           run_comparison_experiments(SystemParameters())"
```

## 🧬 Algorithm Details

### Genetic Algorithm (GA) - TOA

The Task Offloading Algorithm (TOA) uses GA to optimize binary offloading decisions.

#### Encoding Scheme
```
Gene Sequence: [0, 1, 0, 1, 0, 1]
               │  │  │  │  │  │
Task:          1  2  3  4  5  6
Decision:      L  O  L  O  L  O

L = Local Processing (x=0)
O = Offload to UAV (x=1)
```

#### GA Operators

1. **Selection**: Tournament selection
2. **Crossover**: Single-point crossover
3. **Mutation**: Bit-flip mutation
4. **Fitness**: Total task processing time (minimize)

#### Fitness Function
```
F(X) = Σ(T_local + T_mec + T_tra)
     = Total processing time for all tasks
```

**Objective**: Minimize F(X)

### Particle Swarm Optimization (PSO)

Alternative optimization using swarm intelligence.

#### Components

- **Position**: Continuous values transformed via sigmoid
- **Velocity**: Momentum and direction of particles
- **Personal Best**: Best position found by each particle
- **Global Best**: Best position found by entire swarm

#### Update Equations
```
v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i(t)) + c2·r2·(gbest - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
decision = 1 if sigmoid(x_i) > 0.5 else 0
```

## 📊 System Model

### Task Model

Each vehicle j has a task:
```
q_j = {D_j, ω_j, T_j^max}
```

- **D_j**: Input data size (bits)
- **ω_j**: Processing density (CPU cycles/bit)
- **T_j^max**: Maximum tolerable delay (seconds)

### Local Processing

**Processing Time**:
```
T_j^loc = (D_j · ω_j) / C_j^loc
```

**Energy Consumption**:
```
E_j^loc = α · D_j · ω_j · (C_j^loc)²
```

Where:
- C_j^loc: Local CPU capacity
- α: CPU chip architecture constant

### MEC Offloading

**Total Time**:
```
T_j^mec = T_j^tra + (D_j / R_j) + (D_j · ω_j) / f^mec
        = Flight time + Transmission + Processing
```

**Energy Consumption**:
```
E_j^mec = p_u,i · (D_j / R_j) + p_j^le · (D_j · ω_j / f^mec)
```

### Channel Model

**Distance**:
```
d_i,j = √[(x_u,i - x_v,j)² + (y_u,i - y_v,j)²]
```

**Channel Gain**:
```
h_i,j = h_0 / (d_i,j² + z_u,i²)
```

**SINR**:
```
γ_i,j = (p_u,i · h_i,j) / (G_i,j + B·N_0)
```

**Transmission Rate**:
```
R_j = B · log₂(1 + γ_i,j)
```

### Optimization Problem

**Objective**:
```
minimize  T_total = Σ(T_j^loc + T_j^mec + T_j^tra)
   X
```

**Subject to**:
```
h_min ≤ z_u,i ≤ h_max           (UAV altitude)
0 ≤ C_j^loc ≤ C^loc_max         (Local capacity)
(D_j·ω_j)/(T_j^max·C_j^loc) < 1 (Feasibility)
x_j ∈ {0, 1}                    (Binary decision)
0 < p_u,i ≤ p_max               (Transmit power)
```

## ⚙️ Simulation Parameters

### Default System Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Number of UAVs | I | 5 | MEC-enabled UAVs |
| Number of Tasks | J | 20 | Vehicle tasks |
| UAV Altitude Range | [h_min, h_max] | [0, 100] m | Flight altitude |
| UAV Speed | υ | Variable | 5-50 m/s |
| Channel Bandwidth | B | 10 MHz | Available bandwidth |
| Noise PSD | N_0 | -174 dBm/Hz | AWGN |
| Reference Channel Gain | h_0 | -30 dB | At d_0 = 1m |
| Max Transmit Power | p_max | 1 W | UAV transmit power |
| MEC Capacity | f^mec | 10 GHz | Edge server CPU |
| Max Local Capacity | C^loc_max | 2 GHz | Vehicle CPU |

### GA Parameters (Paper Settings)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size (D) | 10 | Number of individuals |
| Max Iterations (K) | 100 | Generations |
| Crossover Rate (p_c) | 0.8 | Probability of crossover |
| Mutation Rate (p_m) | 0.05 | Probability of mutation |

### Task Characteristics

- **Data Size**: D_j ∈ [1, 10] MB (randomly generated)
- **Processing Density**: ω_j ∈ [500, 1500] cycles/bit
- **Deadline**: T_j^max ∈ [1, 10] seconds

## 📈 Results and Visualization

### Generated Figures

The simulator reproduces the key figures from the research paper:

#### Figure 3: Task Processing Time vs. Number of Tasks
- Comparison: TOA vs. All-Offload strategy
- **Result**: Up to 56% reduction in processing time
- X-axis: Number of tasks (5, 10, 15, 20, 25, 30)
- Y-axis: Total processing time (seconds)

#### Figure 4: Energy Consumption vs. Number of Tasks
- Comparison: TOA vs. All-Offload strategy
- **Result**: Significant energy savings with smart offloading
- X-axis: Number of tasks
- Y-axis: Total energy consumption (Joules)

#### Figure 5: Processing Time vs. UAV Flight Speed
- Analysis of UAV speed impact
- **Result**: TOA less sensitive to speed variations
- X-axis: UAV speed (5-50 m/s)
- Y-axis: Processing time (seconds)

#### Figure 6: Task Processing Scheme Comparison
- Local-only vs. TOA vs. All-Offload
- **Result**: TOA adapts to computational constraints
- Shows successful task completion rates

### Output Files
```
uav_task_offloading/
├── uav_figures_3_4_5_comparison.png    # Combined figures 3-5
├── uav_figure_6_comparison.png         # Task scheme analysis

```

## 📚 References

### Primary Paper

**Xiao, T., Du, P., Gou, H., & Zhang, G.** (2024). NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks. *2024 3rd International Conference on Computing, Communication, Perception and Quantum Technology (CCPQT)*, 190-194.





**Citation**: If you use this code in your research, please cite both the original paper and this implementation.
```bibtex
@inproceedings{xiao2024noma,
  title={NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks},
  author={Xiao, Tingyue and Du, Pengfei and Gou, Haosong and Zhang, Gaoyi},
  booktitle={2024 3rd International Conference on Computing, Communication, Perception and Quantum Technology (CCPQT)},
  pages={190--194},
  year={2024},
  organization={IEEE}
}
```

---

⭐ **Star this repository** if you find it helpful!

Last Updated: November 2025
