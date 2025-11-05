"""
Calculation Utilities
Helper functions for distance, channel gain, SINR, rate, and fitness calculations
"""
import numpy as np
from typing import List, Tuple


def calc_distance(pos1, pos2) -> float:
    """Calculate horizontal distance between two positions"""
    return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)


def calc_channel_gain(d: float, h: float, params) -> float:
    """Calculate channel gain (Equation 4)"""
    return params.h0 * h**2 / d**2 if d > 0 else params.h0 * h**2


def calc_sinr(pui: float, hij: float, G: float, params) -> float:
    """Calculate SINR (Equation 7)"""
    return (pui * hij) / (G + params.B * params.N0)


def calc_rate(sinr: float, params) -> float:
    """Calculate transmission rate (Equation 8)"""
    return params.B * np.log2(1 + sinr)


def calc_local_processing(task, params) -> Tuple[float, float]:
    """Calculate local processing time and energy (Equations 1-2)"""
    Tloc = (task.D * task.omega) / task.Cloc if task.Cloc > 0 else float('inf')
    Eloc = params.alpha * task.omega**2 * task.D * task.Cloc
    return Tloc, Eloc


def calc_mec_offloading(task, vehicle_pos, uav_pos, params) -> Tuple[float, float, float]:
    """Calculate MEC offloading time and energy (Equations 5, 9-10)"""
    d = calc_distance(vehicle_pos, uav_pos)
    h = calc_channel_gain(d, uav_pos.z, params)
    G = 0  # Simplified: no interference
    sinr = calc_sinr(params.pui, h, G, params)
    R = calc_rate(sinr, params)
    
    Ttra = d / params.uav_speed  # Flight time (Equation 5)
    Ttrans = task.D / R  # Transmission time
    Tcomp = (task.D * task.omega) / params.fmec  # Computation time
    Tmec = Ttrans + Tcomp  # Total MEC time (Equation 9)
    
    Etrans = params.pui * (task.D / R)
    Eidle = params.plej * (task.D * task.omega / params.fmec)
    Emec = Etrans + Eidle  # Total energy (Equation 10)
    
    return Tmec, Emec, Ttra


def calc_fitness(solution: List[int], tasks: List, 
                vehicle_positions: List, uav_positions: List,
                params) -> Tuple[float, float]:
    """Calculate fitness (total time and energy) for a solution"""
    total_time = 0
    total_energy = 0
    
    for j, decision in enumerate(solution):
        task = tasks[j]
        
        if decision == 0:  # Local processing
            can_process = (task.D * task.omega) / task.Cloc < task.Tmax if task.Cloc > 0 else False
            if not can_process:
                return float('inf'), float('inf')
            
            Tloc, Eloc = calc_local_processing(task, params)
            total_time += Tloc
            total_energy += Eloc
        else:  # Offload to UAV
            min_time = float('inf')
            min_energy = 0
            for uav_pos in uav_positions:
                Tmec, Emec, Ttra = calc_mec_offloading(task, vehicle_positions[j], 
                                                        uav_pos, params)
                task_time = Tmec + Ttra
                if task_time < min_time:
                    min_time = task_time
                    min_energy = Emec
            
            total_time += min_time
            total_energy += min_energy
    
    return total_time, total_energy