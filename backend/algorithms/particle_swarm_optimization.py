"""
Particle Swarm Optimization Implementation
UAV-assisted IoV Task Offloading using PSO
"""
import numpy as np
from typing import List, Dict
from backend.utils.calculations import calc_fitness


def particle_swarm_optimization(tasks: List, vehicle_positions: List,
                               uav_positions: List, params,
                               callback=None) -> Dict:
    """UAV-assisted IoV Task Offloading using PSO"""
    num_tasks = len(tasks)
    
    # Initialize particles with continuous positions
    particles = np.random.rand(params.D, num_tasks)
    velocities = np.random.rand(params.D, num_tasks) * 0.1
    
    # Personal best positions and fitness
    pbest = particles.copy()
    pbest_fitness = np.full(params.D, float('inf'))
    
    # Global best
    gbest = None
    gbest_fitness = float('inf')
    gbest_energy = float('inf')
    
    history = []
    
    for k in range(params.K):
        for i in range(params.D):
            # Convert continuous position to binary solution using sigmoid
            binary_solution = [1 if particles[i][j] > 0.5 else 0 for j in range(num_tasks)]
            
            # Calculate fitness
            time, energy = calc_fitness(binary_solution, tasks, vehicle_positions,
                                       uav_positions, params)
            
            # Update personal best
            if time < pbest_fitness[i]:
                pbest_fitness[i] = time
                pbest[i] = particles[i].copy()
            
            # Update global best
            if time < gbest_fitness:
                gbest_fitness = time
                gbest_energy = energy
                gbest = particles[i].copy()
        
        # Update velocities and positions
        if gbest is not None:
            for i in range(params.D):
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                
                # Velocity update
                velocities[i] = (params.w * velocities[i] + 
                               params.c1 * r1 * (pbest[i] - particles[i]) +
                               params.c2 * r2 * (gbest - particles[i]))
                
                # Position update with sigmoid to keep in [0,1]
                particles[i] = particles[i] + velocities[i]
                particles[i] = 1 / (1 + np.exp(-particles[i]))  # Sigmoid
        else:
            # Reset if no valid solution found
            particles = np.random.rand(params.D, num_tasks)
        
        history.append({'iteration': k, 'fitness': gbest_fitness, 'energy': gbest_energy})
        
        if callback:
            callback(k, params.K, gbest_fitness)
    
    # Convert final gbest to binary solution
    if gbest is not None:
        best_solution = [1 if gbest[j] > 0.5 else 0 for j in range(num_tasks)]
    else:
        # Fallback: all offload
        best_solution = [1] * num_tasks
        gbest_fitness, gbest_energy = calc_fitness(best_solution, tasks, vehicle_positions, 
                                                    uav_positions, params)
    
    return {
        'solution': best_solution,
        'fitness': gbest_fitness,
        'energy': gbest_energy,
        'history': history
    }