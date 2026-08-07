import numpy as np
from typing import Dict, Tuple

class SourceGenerator:
    """
    Generates internal heat sources for the domain.
    Volumetric heat generation: q_dot (W/m^3)
    Temperature equation: rho * cp * dT/dt = k * del^2 T + q_dot
    source term in explicit step: + (dt / (rho * cp)) * q_dot
    """

    @staticmethod
    def get_source_1d(config: Dict, grid_shape: int, x: np.ndarray) -> np.ndarray:
        """
        config = {'type': 'point' | 'distributed', 'position': float, 'width': float, 'power': float}
        power is total W, we need q_dot = power / Volume
        For 1D, assume cross sectional area A = 1m^2, so V = dx. q_dot = power / dx.
        """
        q_dot = np.zeros(grid_shape)
        if not config:
            return q_dot

        power = config.get('power', 0.0)
        source_type = config.get('type', 'distributed')
        
        if source_type == 'point':
            pos = config.get('position', x[-1]/2)
            # Find nearest node
            idx = (np.abs(x - pos)).argmin()
            # q_dot = Power / Volume. Assuming dx=x[1]-x[0], Volume=A*dx (A=1)
            dx = x[1] - x[0]
            q_dot[idx] = power / dx
        elif source_type == 'distributed':
            # Distributed over a region [start, end]
            pos = config.get('position', x[-1]/2)
            width = config.get('width', (x[-1]-x[0])*0.1)
            start = max(0, pos - width/2)
            end = min(x[-1], pos + width/2)
            
            mask = (x >= start) & (x <= end)
            vol = np.sum(mask) * (x[1]-x[0]) if np.sum(mask) > 0 else (x[1]-x[0])
            if np.sum(mask) > 0:
                q_dot[mask] = power / vol
                
        return q_dot

    @staticmethod
    def get_source_2d(config: Dict, grid_shape: Tuple[int, int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q_dot = np.zeros(grid_shape)
        if not config:
            return q_dot

        power = config.get('power', 0.0)
        source_type = config.get('type', 'distributed')
        X, Y = np.meshgrid(x, y)
        dx, dy = x[1]-x[0], y[1]-y[0]
        
        if source_type == 'point':
            pos_x = config.get('pos_x', x[-1]/2)
            pos_y = config.get('pos_y', y[-1]/2)
            idx_x = (np.abs(x - pos_x)).argmin()
            idx_y = (np.abs(y - pos_y)).argmin()
            q_dot[idx_y, idx_x] = power / (dx * dy) # Assuming depth = 1m
            
        elif source_type == 'distributed':
            pos_x = config.get('pos_x', x[-1]/2)
            pos_y = config.get('pos_y', y[-1]/2)
            width = config.get('width', (x[-1]-x[0])*0.1)
            height = config.get('height', (y[-1]-y[0])*0.1)
            
            mask_x = (X >= pos_x - width/2) & (X <= pos_x + width/2)
            mask_y = (Y >= pos_y - height/2) & (Y <= pos_y + height/2)
            mask = mask_x & mask_y
            
            vol = np.sum(mask) * (dx * dy) if np.sum(mask) > 0 else (dx * dy)
            if np.sum(mask) > 0:
                q_dot[mask] = power / vol
                
        return q_dot
        
    @staticmethod
    def get_source_3d(config: Dict, grid_shape: Tuple[int, int, int], x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        q_dot = np.zeros(grid_shape)
        if not config:
            return q_dot

        power = config.get('power', 0.0)
        source_type = config.get('type', 'distributed')
        # meshgrid in 3D: ij indexing to match typical Z,Y,X shapes
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
        
        if source_type == 'point':
            pos_x = config.get('pos_x', x[-1]/2)
            pos_y = config.get('pos_y', y[-1]/2)
            pos_z = config.get('pos_z', z[-1]/2)
            
            idx_x = (np.abs(x - pos_x)).argmin()
            idx_y = (np.abs(y - pos_y)).argmin()
            idx_z = (np.abs(z - pos_z)).argmin()
            
            q_dot[idx_z, idx_y, idx_x] = power / (dx * dy * dz)
            
        elif source_type == 'distributed':
            pos_x = config.get('pos_x', x[-1]/2)
            pos_y = config.get('pos_y', y[-1]/2)
            pos_z = config.get('pos_z', z[-1]/2)
            w_x = config.get('width_x', (x[-1]-x[0])*0.1)
            w_y = config.get('width_y', (y[-1]-y[0])*0.1)
            w_z = config.get('width_z', (z[-1]-z[0])*0.1)
            
            mask = ((X >= pos_x - w_x/2) & (X <= pos_x + w_x/2) &
                    (Y >= pos_y - w_y/2) & (Y <= pos_y + w_y/2) &
                    (Z >= pos_z - w_z/2) & (Z <= pos_z + w_z/2))
            
            vol = np.sum(mask) * (dx * dy * dz) if np.sum(mask) > 0 else (dx * dy * dz)
            if np.sum(mask) > 0:
                q_dot[mask] = power / vol
                
        return q_dot
