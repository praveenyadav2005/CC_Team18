"""
Frontend GUI Application
NOMA-MEC UAV-IoV Task Offloading Simulator
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import numpy as np
import json
import csv
from datetime import datetime

from backend.config import SystemParameters
from backend.models.task import Task
from backend.models.position import generate_positions
from backend.algorithms.genetic_algorithm import genetic_algorithm
from backend.algorithms.particle_swarm_optimization import particle_swarm_optimization
from backend.utils.calculations import calc_fitness
from backend.experiments.comparison import run_comparison_experiments


class UAVSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NOMA-MEC UAV-IoV Task Offloading Simulator")
        self.root.geometry("1400x900")
        
        self.params = SystemParameters()
        self.results_ga = None
        self.results_pso = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root window grid
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="UAV-Assisted IoV Task Offloading Simulator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=(0,5))
        
        # Right panel - Results
        result_frame = ttk.LabelFrame(main_frame, text="Simulation Results", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=(0,5))
        
        # Make control/result frames resize nicely
        control_frame.columnconfigure(0, weight=0)
        control_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Parameters
        row = 0
        
        # Algorithm Selection
        ttk.Label(control_frame, text="Algorithm:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.algorithm_var = tk.StringVar(value="both")
        ttk.Radiobutton(control_frame, text="GA Only", variable=self.algorithm_var, value="ga").grid(row=row+1, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="PSO Only", variable=self.algorithm_var, value="pso").grid(row=row+1, column=1, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Both (Compare)", variable=self.algorithm_var, value="both").grid(row=row+2, column=0, columnspan=2, sticky=tk.W)
        row += 3
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        row += 1
        
        # Number of Tasks
        ttk.Label(control_frame, text="Number of Tasks:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.num_tasks_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=5, to=50, textvariable=self.num_tasks_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        # Number of UAVs
        ttk.Label(control_frame, text="Number of UAVs:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.num_uavs_var = tk.IntVar(value=5)
        ttk.Spinbox(control_frame, from_=2, to=10, textvariable=self.num_uavs_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        # UAV Speed
        ttk.Label(control_frame, text="UAV Speed (m/s):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.uav_speed_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=5, to=50, textvariable=self.uav_speed_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        row += 1
        
        # Common Parameters
        ttk.Label(control_frame, text="Population/Swarm Size:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.pop_size_var = tk.IntVar(value=10)
        ttk.Spinbox(control_frame, from_=5, to=20, textvariable=self.pop_size_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Label(control_frame, text="Max Iterations:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.max_iter_var = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=50, to=200, textvariable=self.max_iter_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        # GA Parameters
        ttk.Label(control_frame, text="GA - Crossover Rate:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.crossover_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(control_frame, from_=0.5, to=1.0, increment=0.05, textvariable=self.crossover_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Label(control_frame, text="GA - Mutation Rate:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.mutation_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(control_frame, from_=0.01, to=0.2, increment=0.01, textvariable=self.mutation_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        # PSO Parameters
        ttk.Label(control_frame, text="PSO - Inertia Weight:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.inertia_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(control_frame, from_=0.4, to=0.9, increment=0.1, textvariable=self.inertia_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Label(control_frame, text="PSO - Cognitive (c1):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.c1_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(control_frame, from_=1.0, to=2.5, increment=0.1, textvariable=self.c1_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Label(control_frame, text="PSO - Social (c2):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.c2_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(control_frame, from_=1.0, to=2.5, increment=0.1, textvariable=self.c2_var, width=10).grid(row=row, column=1, pady=3)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        row += 1
        
        # Buttons
        ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation).grid(row=row, column=0, columnspan=2, pady=6)
        row += 1
        
        ttk.Button(control_frame, text="Generate All Figures", command=self.generate_all_figures).grid(row=row, column=0, columnspan=2, pady=3)
        row += 1
        
        ttk.Button(control_frame, text="Export Results", command=self.export_results).grid(row=row, column=0, columnspan=2, pady=3)
        row += 1
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.progress_var, foreground='blue').grid(row=row, column=0, columnspan=2, pady=6)
        row += 1
        
        self.progress_bar = ttk.Progressbar(control_frame, mode='determinate', length=200)
        self.progress_bar.grid(row=row, column=0, columnspan=2, pady=3)
        
        # Results text
        self.results_text = tk.Text(result_frame, height=10, width=50, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text['yscrollcommand'] = scrollbar.set
        
        # Visualization area
        viz_frame = ttk.LabelFrame(main_frame, text="Convergence Visualization", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.fig.tight_layout(pad=2.5)
        
        # Embed the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=3)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def update_progress(self, iteration, total, fitness):
        progress = (iteration / total) * 100
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Iteration {iteration}/{total} - Best Fitness: {fitness:.4f}s")
        self.root.update_idletasks()
    
    def run_simulation(self):
        try:
            # Update parameters
            self.params.D = self.pop_size_var.get()
            self.params.K = self.max_iter_var.get()
            self.params.pc = self.crossover_var.get()
            self.params.pm = self.mutation_var.get()
            self.params.w = self.inertia_var.get()
            self.params.c1 = self.c1_var.get()
            self.params.c2 = self.c2_var.get()
            self.params.uav_speed = self.uav_speed_var.get()
            
            num_tasks = self.num_tasks_var.get()
            num_uavs = self.num_uavs_var.get()
            algorithm = self.algorithm_var.get()
            
            self.progress_var.set("Initializing...")
            self.progress_bar['value'] = 0
            
            # Generate scenario
            random.seed(42)
            np.random.seed(42)
            
            tasks = [Task(i, self.params) for i in range(num_tasks)]
            vehicle_pos = generate_positions(num_tasks, False, self.params)
            uav_pos = generate_positions(num_uavs, True, self.params)
            
            # Run algorithms
            if algorithm == "ga" or algorithm == "both":
                self.progress_var.set("Running GA...")
                self.results_ga = genetic_algorithm(tasks, vehicle_pos, uav_pos, self.params, 
                                               callback=self.update_progress)
            
            if algorithm == "pso" or algorithm == "both":
                self.progress_var.set("Running PSO...")
                self.results_pso = particle_swarm_optimization(tasks, vehicle_pos, uav_pos, self.params,
                                                          callback=self.update_progress)
            
            # Display results
            self.display_results(num_tasks, tasks, vehicle_pos, uav_pos)
            
            # Plot convergence
            self.plot_convergence()
            
            self.progress_var.set("Simulation Complete!")
            messagebox.showinfo("Success", "Simulation completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")
            self.progress_var.set("Error occurred")
    
    def display_results(self, num_tasks, tasks, vehicle_pos, uav_pos):
        self.results_text.delete(1.0, tk.END)
        
        result_text = f"""
SIMULATION RESULTS
{'='*50}

System Configuration:
  - Number of Tasks: {num_tasks}
  - Number of UAVs: {len(uav_pos)}
  - UAV Speed: {self.params.uav_speed} m/s
  - Population/Swarm Size: {self.params.D}
  - Max Iterations: {self.params.K}

"""
        
        if self.results_ga:
            result_text += f"""
GENETIC ALGORITHM (GA) Results:
  - Best Processing Time: {self.results_ga['fitness']:.4f} s
  - Total Energy: {self.results_ga['energy']:.4f} J
  - Tasks Offloaded: {sum(self.results_ga['solution'])} / {num_tasks}
  - Tasks Local: {num_tasks - sum(self.results_ga['solution'])} / {num_tasks}
"""
        
        if self.results_pso:
            result_text += f"""
PARTICLE SWARM OPTIMIZATION (PSO) Results:
  - Best Processing Time: {self.results_pso['fitness']:.4f} s
  - Total Energy: {self.results_pso['energy']:.4f} J
  - Tasks Offloaded: {sum(self.results_pso['solution'])} / {num_tasks}
  - Tasks Local: {num_tasks - sum(self.results_pso['solution'])} / {num_tasks}
"""
        
        if self.results_ga and self.results_pso:
            improvement = ((self.results_ga['fitness'] - self.results_pso['fitness']) / 
                          self.results_ga['fitness']) * 100
            winner = "PSO" if self.results_pso['fitness'] < self.results_ga['fitness'] else "GA"
            result_text += f"""
Algorithm Comparison:
  - Better Algorithm: {winner}
  - Time Difference: {abs(improvement):.2f}%
"""
        
        # Calculate baselines
        all_offload = [1] * num_tasks
        all_off_time, _ = calc_fitness(all_offload, tasks, vehicle_pos, uav_pos, self.params)
        
        if all_off_time != float('inf'):
            result_text += f"""
Comparison with Baselines:
  - All-Offload Time: {all_off_time:.4f} s
"""
            if self.results_ga:
                imp_ga = ((all_off_time - self.results_ga['fitness']) / all_off_time) * 100
                result_text += f"  - GA Improvement: {imp_ga:.2f}%\n"
            if self.results_pso:
                imp_pso = ((all_off_time - self.results_pso['fitness']) / all_off_time) * 100
                result_text += f"  - PSO Improvement: {imp_pso:.2f}%\n"
        
        self.results_text.insert(1.0, result_text)
    
    def plot_convergence(self):
        self.ax.clear()
        
        if self.results_ga:
            iterations = [h['iteration'] for h in self.results_ga['history']]
            fitness = [h['fitness'] for h in self.results_ga['history']]
            self.ax.plot(iterations, fitness, 'b-', linewidth=2, label='GA', marker='o', markersize=3)
        
        if self.results_pso:
            iterations = [h['iteration'] for h in self.results_pso['history']]
            fitness = [h['fitness'] for h in self.results_pso['history']]
            self.ax.plot(iterations, fitness, 'r-', linewidth=2, label='PSO', marker='s', markersize=3)
        
        self.ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Processing Time (s)', fontsize=11, fontweight='bold')
        self.ax.set_title('Algorithm Convergence Comparison', fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def generate_all_figures(self):
        try:
            self.progress_var.set("Generating all figures...")
            
            run_comparison_experiments(self.params)
            
            messagebox.showinfo("Success", "All figures generated successfully!\nCheck the output directory for PNG files.")
            self.progress_var.set("Figures generated!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Figure generation failed: {str(e)}")
    
    def export_results(self):
        if not self.results_ga and not self.results_pso:
            messagebox.showwarning("Warning", "No results to export. Run simulation first!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                if filename.endswith('.json'):
                    # Export as JSON
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'parameters': {
                            'num_tasks': self.num_tasks_var.get(),
                            'num_uavs': self.num_uavs_var.get(),
                            'uav_speed': self.params.uav_speed,
                            'population_size': self.params.D,
                            'max_iterations': self.params.K,
                            'crossover_rate': self.params.pc,
                            'mutation_rate': self.params.pm,
                            'inertia_weight': self.params.w,
                            'c1': self.params.c1,
                            'c2': self.params.c2
                        },
                        'results': {}
                    }
                    
                    if self.results_ga:
                        export_data['results']['GA'] = {
                            'best_time': self.results_ga['fitness'],
                            'best_energy': self.results_ga['energy'],
                            'solution': self.results_ga['solution'],
                            'history': self.results_ga['history']
                        }
                    
                    if self.results_pso:
                        export_data['results']['PSO'] = {
                            'best_time': self.results_pso['fitness'],
                            'best_energy': self.results_pso['energy'],
                            'solution': self.results_pso['solution'],
                            'history': self.results_pso['history']
                        }
                    
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                
                else:
                    # Export as CSV
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Algorithm', 'Iteration', 'Best Fitness (s)', 'Energy (J)'])
                        
                        if self.results_ga:
                            for h in self.results_ga['history']:
                                writer.writerow(['GA', h['iteration'], h['fitness'], h['energy']])
                        
                        if self.results_pso:
                            for h in self.results_pso['history']:
                                writer.writerow(['PSO', h['iteration'], h['fitness'], h['energy']])
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = UAVSimulatorGUI(root)
    root.mainloop()