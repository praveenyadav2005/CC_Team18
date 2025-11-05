"""
Genetic Algorithm Implementation
UAV-assisted IoV Task Offloading Algorithm (TOA) using GA
"""
import random
from typing import List, Tuple, Dict
from backend.utils.calculations import calc_fitness


def initialize_population(pop_size: int, num_tasks: int) -> List[List[int]]:
    """Initialize population with random binary solutions"""
    return [[random.randint(0, 1) for _ in range(num_tasks)] 
            for _ in range(pop_size)]


def tournament_selection(population: List[List[int]], fitness_values: List[Tuple[float, float]], 
                        params) -> List[List[int]]:
    """Tournament selection"""
    new_population = []
    for _ in range(params.D):
        idx1, idx2 = random.sample(range(params.D), 2)
        fit1, fit2 = fitness_values[idx1][0], fitness_values[idx2][0]
        
        if fit1 < fit2:
            new_population.append(population[idx1][:])
        else:
            new_population.append(population[idx2][:])
    
    return new_population


def crossover(population: List[List[int]], params) -> List[List[int]]:
    """Single-point crossover operation"""
    num_tasks = len(population[0])
    for i in range(0, params.D - 1, 2):
        if random.random() < params.pc:
            point = random.randint(1, num_tasks - 1)
            temp = population[i][point:]
            population[i][point:] = population[i + 1][point:]
            population[i + 1][point:] = temp
    
    return population


def mutation(population: List[List[int]], params) -> List[List[int]]:
    """Bit-flip mutation operation"""
    num_tasks = len(population[0])
    for i in range(params.D):
        for j in range(num_tasks):
            if random.random() < params.pm:
                population[i][j] = 1 - population[i][j]
    
    return population


def genetic_algorithm(tasks: List, vehicle_positions: List,
                     uav_positions: List, params,
                     callback=None) -> Dict:
    """UAV-assisted IoV Task Offloading Algorithm (TOA) using GA"""
    num_tasks = len(tasks)
    population = initialize_population(params.D, num_tasks)
    
    best_solution = None
    best_fitness = float('inf')
    best_energy = float('inf')
    history = []
    
    for k in range(params.K):
        fitness_values = []
        for solution in population:
            time, energy = calc_fitness(solution, tasks, vehicle_positions, 
                                       uav_positions, params)
            fitness_values.append((time, energy))
        
        population = tournament_selection(population, fitness_values, params)
        population = crossover(population, params)
        population = mutation(population, params)
        
        for i, solution in enumerate(population):
            time, energy = calc_fitness(solution, tasks, vehicle_positions, 
                                       uav_positions, params)
            if time < best_fitness:
                best_fitness = time
                best_energy = energy
                best_solution = solution[:]
        
        history.append({'iteration': k, 'fitness': best_fitness, 'energy': best_energy})
        
        if callback:
            callback(k, params.K, best_fitness)
    
    return {
        'solution': best_solution,
        'fitness': best_fitness,
        'energy': best_energy,
        'history': history
    }