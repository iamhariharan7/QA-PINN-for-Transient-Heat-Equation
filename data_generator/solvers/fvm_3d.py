import numpy as np
from typing import Dict, Tuple
from data_generator.boundary_conditions.bc_manager import BCManager
from data_generator.heat_sources.source_generator import SourceGenerator

class Solver3D:
    def __init__(self, material: dict, length_x: float, length_y: float, length_z: float,
                 nx: int, ny: int, nz: int, sim_time: float, 
                 initial_temp: float, bcs: Dict, source_config: Dict, cfl_factor: float = 0.9):
        self.k = material['k']
        self.rho = material['rho']
        self.cp = material['cp']
        
        if isinstance(material['alpha'], list):
            self.alpha_x = material['alpha'][0]
            self.alpha_y = material['alpha'][1]
            self.alpha_z = material['alpha'][2]
            self.k_avg = sum(material['k'])/3
        else:
            self.alpha_x = self.alpha_y = self.alpha_z = material['alpha']
            self.k_avg = material['k']
        
        self.Lx, self.Ly, self.Lz = length_x, length_y, length_z
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        self.x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)
        self.z = np.linspace(self.dz/2, self.Lz - self.dz/2, self.nz)
        
        self.sim_time = sim_time
        
        # Stability dt
        dt_max = 1.0 / (2 * self.alpha_x / (self.dx**2) + 2 * self.alpha_y / (self.dy**2) + 2 * self.alpha_z / (self.dz**2))
        self.dt = cfl_factor * dt_max
        self.nt = int(np.ceil(self.sim_time / self.dt))
        self.dt = self.sim_time / self.nt
        
        self.T = np.full((self.nz + 2, self.ny + 2, self.nx + 2), initial_temp)
        self.bcs = bcs
        
        self.q_dot = SourceGenerator.get_source_3d(source_config, (self.nz, self.ny, self.nx), self.x, self.y, self.z)
        
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        save_interval = max(1, self.nt // 20)
        
        T_history = []
        t_history = []
        
        BCManager.apply_3d_fvm_bc(self.T, self.bcs, self.dx, self.dy, self.dz, self.k_avg)
        
        fx = self.alpha_x * self.dt / (self.dx**2)
        fy = self.alpha_y * self.dt / (self.dy**2)
        fz = self.alpha_z * self.dt / (self.dz**2)
        
        T_new = np.empty_like(self.T)
        for n in range(self.nt + 1):
            if n % save_interval == 0 or n == self.nt:
                T_history.append(self.T[1:-1, 1:-1, 1:-1].copy())
                t_history.append(n * self.dt)
                
            if n == self.nt:
                break
                
            T_new[:] = self.T[:]
            
            T_new[1:-1, 1:-1, 1:-1] = self.T[1:-1, 1:-1, 1:-1] + \
                fx * (self.T[1:-1, 1:-1, 2:] - 2*self.T[1:-1, 1:-1, 1:-1] + self.T[1:-1, 1:-1, :-2]) + \
                fy * (self.T[1:-1, 2:, 1:-1] - 2*self.T[1:-1, 1:-1, 1:-1] + self.T[1:-1, :-2, 1:-1]) + \
                fz * (self.T[2:, 1:-1, 1:-1] - 2*self.T[1:-1, 1:-1, 1:-1] + self.T[:-2, 1:-1, 1:-1])
                
            T_new[1:-1, 1:-1, 1:-1] += (self.dt / (self.rho * self.cp)) * self.q_dot
            
            BCManager.apply_3d_fvm_bc(T_new, self.bcs, self.dx, self.dy, self.dz, self.k_avg)
            
            self.T, T_new = T_new, self.T
            
        return np.array(t_history), self.x, self.y, self.z, np.array(T_history)
