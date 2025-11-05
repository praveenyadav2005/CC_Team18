"""
Comparison Experiments
Run experiments comparing GA and PSO algorithms and generate figures
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import random

from backend.models.task import Task
from backend.models.position import generate_positions
from backend.algorithms.genetic_algorithm import genetic_algorithm
from backend.algorithms.particle_swarm_optimization import particle_swarm_optimization
from backend.utils.calculations import calc_fitness


def run_comparison_experiments(params) -> Dict:
    """Run experiments comparing GA and PSO algorithms"""
    
    print("Running UAV-Assisted IoV Task Offloading - GA vs PSO Comparison")
    print("=" * 70)
    
    # ========== Figure 3: Task Processing Time vs Number of Tasks ==========
    print("\nGenerating Figure 3: Task Processing Time vs Number of Tasks...")
    task_numbers = [5, 10, 15, 20, 25, 30]
    num_uavs = 5
    
    ga_times = []
    pso_times = []
    all_offload_times = []
    
    for num_tasks in task_numbers:
        print(f"  Processing {num_tasks} tasks...")
        
        tasks = [Task(i, params) for i in range(num_tasks)]
        vehicle_pos = generate_positions(num_tasks, False, params)
        uav_pos = generate_positions(num_uavs, True, params)
        
        ga_result = genetic_algorithm(tasks, vehicle_pos, uav_pos, params)
        ga_times.append(ga_result['fitness'])
        
        pso_result = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, params)
        pso_times.append(pso_result['fitness'])
        
        all_offload_solution = [1] * num_tasks
        all_offload_time, _ = calc_fitness(all_offload_solution, tasks, 
                                           vehicle_pos, uav_pos, params)
        all_offload_times.append(all_offload_time if all_offload_time != float('inf') else None)
    
    # ========== Figure 4: Energy Consumption vs Number of Tasks ==========
    print("\nGenerating Figure 4: Energy Consumption vs Number of Tasks...")
    ga_energy = []
    pso_energy = []
    all_offload_energy = []
    
    for num_tasks in task_numbers:
        tasks = [Task(i, params) for i in range(num_tasks)]
        vehicle_pos = generate_positions(num_tasks, False, params)
        uav_pos = generate_positions(num_uavs, True, params)
        
        ga_result = genetic_algorithm(tasks, vehicle_pos, uav_pos, params)
        ga_energy.append(ga_result['energy'])
        
        pso_result = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, params)
        pso_energy.append(pso_result['energy'])
        
        all_offload_solution = [1] * num_tasks
        _, energy = calc_fitness(all_offload_solution, tasks, vehicle_pos, uav_pos, params)
        all_offload_energy.append(energy if energy != float('inf') else None)
    
    # ========== Figure 5: Processing Time vs UAV Flight Speed ==========
    print("\nGenerating Figure 5: Processing Time vs UAV Flight Speed...")
    num_tasks = 20
    speeds = range(5, 55, 5)
    
    ga_speed_times = []
    pso_speed_times = []
    all_offload_speed_times = []
    
    for speed in speeds:
        params.uav_speed = speed
        
        tasks = [Task(i, params) for i in range(num_tasks)]
        vehicle_pos = generate_positions(num_tasks, False, params)
        uav_pos = generate_positions(num_uavs, True, params)
        
        ga_result = genetic_algorithm(tasks, vehicle_pos, uav_pos, params)
        ga_speed_times.append(ga_result['fitness'])
        
        pso_result = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, params)
        pso_speed_times.append(pso_result['fitness'])
        
        all_offload_solution = [1] * num_tasks
        all_offload_time, _ = calc_fitness(all_offload_solution, tasks, 
                                           vehicle_pos, uav_pos, params)
        all_offload_speed_times.append(all_offload_time)
    
    params.uav_speed = 20  # Reset to default
    
    # ========== Figure 6: Task Processing Scheme Analysis ==========
    print("\nGenerating Figure 6: Task Processing Scheme...")
    task_numbers_scheme = [10, 15, 20, 30, 50]
    
    ga_offload = []
    ga_local = []
    ga_incapacity = []
    pso_offload = []
    pso_local = []
    pso_incapacity = []
    all_local_offload = []
    all_local_local = []
    all_local_incapacity = []
    
    for num_tasks in task_numbers_scheme:
        tasks = [Task(i, params) for i in range(num_tasks)]
        vehicle_pos = generate_positions(num_tasks, False, params)
        uav_pos = generate_positions(num_uavs, True, params)
        
        # GA allocation
        ga_result = genetic_algorithm(tasks, vehicle_pos, uav_pos, params)
        ga_off_count = sum(ga_result['solution'])
        ga_loc_count = 0
        ga_incap_count = 0
        
        for j, decision in enumerate(ga_result['solution']):
            if decision == 0:
                can_process = (tasks[j].D * tasks[j].omega) / tasks[j].Cloc < tasks[j].Tmax if tasks[j].Cloc > 0 else False
                if can_process:
                    ga_loc_count += 1
                else:
                    ga_incap_count += 1
        
        ga_offload.append(ga_off_count)
        ga_local.append(ga_loc_count)
        ga_incapacity.append(ga_incap_count)
        
        # PSO allocation
        pso_result = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, params)
        pso_off_count = sum(pso_result['solution'])
        pso_loc_count = 0
        pso_incap_count = 0
        
        for j, decision in enumerate(pso_result['solution']):
            if decision == 0:
                can_process = (tasks[j].D * tasks[j].omega) / tasks[j].Cloc < tasks[j].Tmax if tasks[j].Cloc > 0 else False
                if can_process:
                    pso_loc_count += 1
                else:
                    pso_incap_count += 1
        
        pso_offload.append(pso_off_count)
        pso_local.append(pso_loc_count)
        pso_incapacity.append(pso_incap_count)
        
        # All-Local allocation
        all_local_solution = [0] * num_tasks
        al_off_count = 0
        al_loc_count = 0
        al_incap_count = 0
        
        for j in range(num_tasks):
            can_process = (tasks[j].D * tasks[j].omega) / tasks[j].Cloc < tasks[j].Tmax if tasks[j].Cloc > 0 else False
            if can_process:
                al_loc_count += 1
            else:
                al_incap_count += 1
        
        all_local_offload.append(al_off_count)
        all_local_local.append(al_loc_count)
        all_local_incapacity.append(al_incap_count)
    
    # ========== Final comparison ==========
    print("\nRunning final comparison...")
    num_tasks = 20
    tasks = [Task(i, params) for i in range(num_tasks)]
    vehicle_pos = generate_positions(num_tasks, False, params)
    uav_pos = generate_positions(num_uavs, True, params)
    
    ga_final = genetic_algorithm(tasks, vehicle_pos, uav_pos, params)
    pso_final = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, params)
    
    # Print comparison results
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON RESULTS:")
    print("=" * 70)
    print(f"\nGA Results:")
    print(f"  Best Processing Time: {ga_final['fitness']:.2f} s")
    print(f"  Total Energy: {ga_final['energy']:.2e} J")
    print(f"  Tasks Offloaded: {sum(ga_final['solution'])} / {num_tasks}")
    
    print(f"\nPSO Results:")
    print(f"  Best Processing Time: {pso_final['fitness']:.2f} s")
    print(f"  Total Energy: {pso_final['energy']:.2e} J")
    print(f"  Tasks Offloaded: {sum(pso_final['solution'])} / {num_tasks}")
    
    improvement = ((ga_final['fitness'] - pso_final['fitness']) / ga_final['fitness']) * 100
    winner = "PSO" if pso_final['fitness'] < ga_final['fitness'] else "GA"
    print(f"\nBetter Algorithm: {winner}")
    print(f"Performance Difference: {abs(improvement):.2f}%")
    
    # ========== Plotting Figures 3, 4, 5 ==========
    print("\nGenerating Figures 3, 4, 5...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Figure 3: Task Processing Time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(task_numbers, ga_times, 'o-', linewidth=2, markersize=8, 
             label='GA', color='#2563eb')
    ax1.plot(task_numbers, pso_times, 's-', linewidth=2, markersize=8,
             label='PSO', color='#16a34a')
    ax1.plot(task_numbers, all_offload_times, '^-', linewidth=2, markersize=8,
             label='All-Offload', color='#dc2626')
    ax1.set_xlabel('Number of Tasks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Processing Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 3: Processing Time vs Number of Tasks', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Figure 4: Energy Consumption
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(task_numbers, ga_energy, 'o-', linewidth=2, markersize=8,
             label='GA', color='#2563eb')
    ax2.plot(task_numbers, pso_energy, 's-', linewidth=2, markersize=8,
             label='PSO', color='#16a34a')
    ax2.plot(task_numbers, all_offload_energy, '^-', linewidth=2, markersize=8,
             label='All-Offload', color='#dc2626')
    ax2.set_xlabel('Number of Tasks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy Consumption (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 4: Energy Consumption vs Number of Tasks', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Figure 5: UAV Speed Impact
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(list(speeds), ga_speed_times, 'o-', linewidth=2, markersize=8,
             label='GA', color='#2563eb')
    ax3.plot(list(speeds), pso_speed_times, 's-', linewidth=2, markersize=8,
             label='PSO', color='#16a34a')
    ax3.plot(list(speeds), all_offload_speed_times, '^-', linewidth=2, markersize=8,
             label='All-Offload', color='#dc2626')
    ax3.set_xlabel('UAV Flight Speed (m/s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Processing Time (s)', fontsize=12, fontweight='bold')
    ax3.set_title('Figure 5: Processing Time vs UAV Flight Speed', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uav_figures_3_4_5_comparison.png', dpi=300, bbox_inches='tight')
    print("Figures 3, 4, 5 saved as 'uav_figures_3_4_5_comparison.png'")
    
    # ========== Plotting Figure 6 ==========
    print("Generating Figure 6...")
    
    fig6 = plt.figure(figsize=(16, 6))
    ax = plt.subplot(1, 1, 1)
    
    n_tasks = len(task_numbers_scheme)
    
    # Bar width and spacing
    bar_width = 0.25
    gap_between_tasks = 0.15
    gap_between_schemes = 1.5
    
    # Calculate x positions
    current_x = 0
    x_offload_ga = []
    x_offload_pso = []
    x_offload_al = []
    
    for i in range(n_tasks):
        x_offload_ga.append(current_x)
        x_offload_pso.append(current_x + bar_width)
        x_offload_al.append(current_x + bar_width * 2)
        current_x += bar_width * 3 + gap_between_tasks
    
    current_x += gap_between_schemes
    
    x_local_ga = []
    x_local_pso = []
    x_local_al = []
    for i in range(n_tasks):
        x_local_ga.append(current_x)
        x_local_pso.append(current_x + bar_width)
        x_local_al.append(current_x + bar_width * 2)
        current_x += bar_width * 3 + gap_between_tasks
    
    current_x += gap_between_schemes
    
    x_incap_ga = []
    x_incap_pso = []
    x_incap_al = []
    for i in range(n_tasks):
        x_incap_ga.append(current_x)
        x_incap_pso.append(current_x + bar_width)
        x_incap_al.append(current_x + bar_width * 2)
        current_x += bar_width * 3 + gap_between_tasks
    
    # Plot bars
    ax.bar(x_offload_ga, ga_offload, bar_width,
           label='GA', color='#ec4899', edgecolor='black', linewidth=1)
    ax.bar(x_offload_pso, pso_offload, bar_width,
           label='PSO', color='#6ee7b7', edgecolor='black', linewidth=1)
    ax.bar(x_offload_al, all_local_offload, bar_width,
           label='All-Local', color='#fbbf24', edgecolor='black', linewidth=1)
    
    ax.bar(x_local_ga, ga_local, bar_width,
           color='#ec4899', edgecolor='black', linewidth=1)
    ax.bar(x_local_pso, pso_local, bar_width,
           color='#6ee7b7', edgecolor='black', linewidth=1)
    ax.bar(x_local_al, all_local_local, bar_width,
           color='#fbbf24', edgecolor='black', linewidth=1)
    
    ax.bar(x_incap_ga, ga_incapacity, bar_width,
           color='#ec4899', edgecolor='black', linewidth=1)
    ax.bar(x_incap_pso, pso_incapacity, bar_width,
           color='#6ee7b7', edgecolor='black', linewidth=1)
    ax.bar(x_incap_al, all_local_incapacity, bar_width,
           color='#fbbf24', edgecolor='black', linewidth=1)
    
    # Add value labels
    def add_label(x_pos, value):
        if value > 0:
            ax.text(x_pos, value + 0.5, str(int(value)), 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i in range(n_tasks):
        add_label(x_offload_ga[i], ga_offload[i])
        add_label(x_offload_pso[i], pso_offload[i])
        add_label(x_offload_al[i], all_local_offload[i])
        add_label(x_local_ga[i], ga_local[i])
        add_label(x_local_pso[i], pso_local[i])
        add_label(x_local_al[i], all_local_local[i])
        add_label(x_incap_ga[i], ga_incapacity[i])
        add_label(x_incap_pso[i], pso_incapacity[i])
        add_label(x_incap_al[i], all_local_incapacity[i])
    
    # X-axis labels
    x_tick_positions = []
    x_tick_labels = []
    
    for i in range(n_tasks):
        x_tick_positions.append((x_offload_ga[i] + x_offload_al[i]) / 2)
        x_tick_positions.append((x_local_ga[i] + x_local_al[i]) / 2)
        x_tick_positions.append((x_incap_ga[i] + x_incap_al[i]) / 2)
        x_tick_labels.extend([str(task_numbers_scheme[i])] * 3)
    
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=10)
    
    # Category labels
    y_label_pos = -3
    offload_center = (x_offload_ga[0] + x_offload_al[-1]) / 2
    local_center = (x_local_ga[0] + x_local_al[-1]) / 2
    incap_center = (x_incap_ga[0] + x_incap_al[-1]) / 2
    
    ax.text(offload_center, y_label_pos, 'Offload', ha='center', 
           va='top', fontsize=12, fontweight='bold')
    ax.text(local_center, y_label_pos, 'Local', ha='center', 
           va='top', fontsize=12, fontweight='bold')
    ax.text(incap_center, y_label_pos, 'Incapacity', ha='center', 
           va='top', fontsize=12, fontweight='bold')
    
    # Separator lines
    sep1_x = (x_offload_al[-1] + x_local_ga[0]) / 2
    sep2_x = (x_local_al[-1] + x_incap_ga[0]) / 2
    ax.axvline(x=sep1_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=sep2_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_ylabel('Number of tasks', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of tasks\nProcessing scheme', fontsize=11, fontweight='bold')
    ax.set_title('Figure 6: Task Processing Scheme for Different Numbers of Tasks', 
                fontsize=13, fontweight='bold', pad=15)
    
    all_values = (ga_offload + ga_local + ga_incapacity + 
                  pso_offload + pso_local + pso_incapacity +
                  all_local_offload + all_local_local + all_local_incapacity)
    max_val = max(all_values) if all_values else 50
    ax.set_ylim(-4.5, max_val + 5)
    
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.set_facecolor('#fffef0')
    
    plt.tight_layout()
    plt.savefig('uav_figure_6_comparison.png', dpi=300, bbox_inches='tight')
    print("Figure 6 saved as 'uav_figure_6_comparison.png'")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("Two files generated:")
    print("  1. uav_figures_3_4_5_comparison.png")
    print("  2. uav_figure_6_comparison.png")
    print("=" * 70)
    
    return {'ga': ga_final, 'pso': pso_final}