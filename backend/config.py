"""
System Parameters Configuration
NOMA-MEC Based Task Offloading in UAV-assisted IoV Networks
"""

class SystemParameters:
    """System parameters for the UAV-MEC network"""
    def __init__(self):
        # Algorithm parameters
        self.K = 100  # Max iterations
        self.D = 10  # Population size
        self.pc = 0.8  # Crossover rate
        self.pm = 0.05  # Mutation rate
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        
        # Physical parameters
        self.alpha = 1e-28  # CPU chip architecture constant
        self.h0 = 1e-4  # Channel gain at reference distance
        self.B = 1e6  # Bandwidth (1 MHz)
        self.N0 = 1e-13  # Noise power spectral density
        self.fmec = 10e9  # MEC computational power (10 GHz)
        self.Clocmax = 5e9  # Max local processing capacity (5 GHz)
        self.pui = 1  # UAV transmit power (W)
        self.plej = 0.1  # Vehicle idle power (W)
        self.pmax = 2  # Max transmit power (W)
        self.hmin = 20  # Min UAV altitude (m)
        self.hmax = 100  # Max UAV altitude (m)
        self.uav_speed = 20  # UAV flight speed (m/s)