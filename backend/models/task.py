"""
Task Model
Represents computational tasks for vehicles in the IoV network
"""
import numpy as np


class Task:
    """Task model for each vehicle"""
    def __init__(self, task_id: int, params):
        self.id = task_id
        self.D = np.random.uniform(1e6, 6e6)  # Data size (1-6 MB)
        self.omega = np.random.uniform(500, 1500)  # Processing density (cycles/bit)
        self.Tmax = np.random.uniform(5, 15)  # Max tolerable time (s)
        self.Cloc = np.random.uniform(0, params.Clocmax)  # Local processing capacity