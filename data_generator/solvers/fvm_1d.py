import numpy as np
from typing import Dict, Tuple
from data_generator.boundary_conditions.bc_manager import BCManager
from data_generator.heat_sources.source_generator import SourceGenerator

class Solver1D:
    def __init__(self, material: dict, length: float, nodes: int, sim_time: float, 
                 initial_temp: float, bcs: Dict, source_config: Dict, cfl_factor: float = 0.9):
        if isinstance(material['alpha'], list):
            self.alpha = material['alpha'][0]
            self.k = material['k'][0]
        else:
            self.alpha = material['alpha']
            self.k = material['k']
        
        self.rho = material['rho']
        self.cp = material['cp']
        
        self.L = length
        self.nx = nodes
        self.dx = self.L / self.nx
        
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.nx)
        
        self.sim_time = sim_time
        
        # Stability dt
        self.dt = cfl_factor * (self.dx**2) / (2 * self.alpha)
        self.nt = int(np.ceil(self.sim_time / self.dt))
        self.dt = self.sim_time / self.nt # Recalculate to exactly match sim_time
        
        self.T = np.full(self.nx + 2, initial_temp)
        self.bcs = bcs
        
        self.q_dot = SourceGenerator.get_source_1d(source_config, self.nx, self.x)
        
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the 1D FVM simulation.
        Returns:
            t_array: time steps saved
            x_array: spatial coordinates
            T_history: Temperature history of shape (nt_saved, nx)
        """
        save_interval = max(1, self.nt // 100)
        
        T_history = []
        t_history = []
        
        BCManager.apply_1d_fvm_bc(self.T, self.bcs, self.dx, self.k)
        
        factor = self.alpha * self.dt / (self.dx**2)
        
        T_new = np.empty_like(self.T)
        for n in range(self.nt + 1):
            if n % save_interval == 0 or n == self.nt:
                T_history.append(self.T[1:-1].copy())
                t_history.append(n * self.dt)
                
            if n == self.nt:
                break
                
            T_new[:] = self.T[:]
            
            # Explicit FTCS step
            T_new[1:-1] = self.T[1:-1] + factor * (self.T[2:] - 2*self.T[1:-1] + self.T[:-2])
            T_new[1:-1] += (self.dt / (self.rho * self.cp)) * self.q_dot
            
            BCManager.apply_1d_fvm_bc(T_new, self.bcs, self.dx, self.k)
            
            self.T, T_new = T_new, self.T
            
        return np.array(t_history), self.x, np.array(T_history)
