"""
Position Model
Represents 3D positions for vehicles and UAVs
"""
import numpy as np
from typing import List


class Position:
    """3D position for vehicles and UAVs"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def generate_positions(n: int, is_uav: bool = False, params=None) -> List[Position]:
    """Generate random positions for vehicles or UAVs"""
    positions = []
    for _ in range(n):
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        z = np.random.uniform(params.hmin, params.hmax) if is_uav else 0
        positions.append(Position(x, y, z))
    return positions