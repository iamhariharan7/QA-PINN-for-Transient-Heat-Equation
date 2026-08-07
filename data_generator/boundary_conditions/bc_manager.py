import numpy as np
from typing import Dict

class BCManager:
    """
    Manages boundary conditions for 1D, 2D, and 3D grids.
    Supports Dirichlet (fixed temp), Neumann (fixed flux), Adiabatic (zero flux), and Robin (convection).
    """

    @staticmethod
    def apply_1d_bc(T: np.ndarray, bcs: Dict, dx: float, k: float):
        """
        Applies BCs to a 1D temperature array inplace.
        bcs = {'left': {'type': 'dirichlet', 'value': 300}, 'right': {'type': 'adiabatic'}}
        """
        # Left boundary
        left_bc = bcs.get('left', {'type': 'adiabatic'})
        if left_bc['type'] == 'dirichlet':
            T[0] = left_bc['value']
        elif left_bc['type'] == 'neumann':
            # q = -k * dT/dx -> T[0] = T[1] + q*dx/k
            q = left_bc.get('value', 0.0)
            T[0] = T[1] + q * dx / k
        elif left_bc['type'] == 'adiabatic':
            T[0] = T[1]
        elif left_bc['type'] == 'convective':
            # -k * dT/dx = h * (T_inf - T_surface)
            h = left_bc.get('h', 10.0)
            T_inf = left_bc.get('T_inf', 293.15)
            # T[0] - T[1] = h*dx/k * (T_inf - T[0])
            # T[0] * (1 + h*dx/k) = T[1] + h*dx/k * T_inf
            factor = h * dx / k
            T[0] = (T[1] + factor * T_inf) / (1 + factor)

        # Right boundary
        right_bc = bcs.get('right', {'type': 'adiabatic'})
        if right_bc['type'] == 'dirichlet':
            T[-1] = right_bc['value']
        elif right_bc['type'] == 'neumann':
            # q = -k * dT/dx -> -q = k * (T[-1] - T[-2])/dx -> T[-1] = T[-2] - q*dx/k
            q = right_bc.get('value', 0.0)
            T[-1] = T[-2] - q * dx / k
        elif right_bc['type'] == 'adiabatic':
            T[-1] = T[-2]
        elif right_bc['type'] == 'convective':
            h = right_bc.get('h', 10.0)
            T_inf = right_bc.get('T_inf', 293.15)
            factor = h * dx / k
            T[-1] = (T[-2] + factor * T_inf) / (1 + factor)


    @staticmethod
    def apply_2d_bc(T: np.ndarray, bcs: Dict, dx: float, dy: float, k: float):
        """
        Applies BCs to a 2D temperature array inplace.
        bcs = {'top': ..., 'bottom': ..., 'left': ..., 'right': ...}
        """
        # Top (y=0)
        top = bcs.get('top', {'type': 'adiabatic'})
        if top['type'] == 'dirichlet':
            T[0, :] = top['value']
        elif top['type'] == 'adiabatic':
            T[0, :] = T[1, :]
        elif top['type'] == 'convective':
            h, T_inf = top.get('h', 10.0), top.get('T_inf', 293.15)
            factor = h * dy / k
            T[0, :] = (T[1, :] + factor * T_inf) / (1 + factor)
            
        # Bottom (y=-1)
        bottom = bcs.get('bottom', {'type': 'adiabatic'})
        if bottom['type'] == 'dirichlet':
            T[-1, :] = bottom['value']
        elif bottom['type'] == 'adiabatic':
            T[-1, :] = T[-2, :]
        elif bottom['type'] == 'convective':
            h, T_inf = bottom.get('h', 10.0), bottom.get('T_inf', 293.15)
            factor = h * dy / k
            T[-1, :] = (T[-2, :] + factor * T_inf) / (1 + factor)

        # Left (x=0)
        left = bcs.get('left', {'type': 'adiabatic'})
        if left['type'] == 'dirichlet':
            T[:, 0] = left['value']
        elif left['type'] == 'adiabatic':
            T[:, 0] = T[:, 1]
        elif left['type'] == 'convective':
            h, T_inf = left.get('h', 10.0), left.get('T_inf', 293.15)
            factor = h * dx / k
            T[:, 0] = (T[:, 1] + factor * T_inf) / (1 + factor)

        # Right (x=-1)
        right = bcs.get('right', {'type': 'adiabatic'})
        if right['type'] == 'dirichlet':
            T[:, -1] = right['value']
        elif right['type'] == 'adiabatic':
            T[:, -1] = T[:, -2]
        elif right['type'] == 'convective':
            h, T_inf = right.get('h', 10.0), right.get('T_inf', 293.15)
            factor = h * dx / k
            T[:, -1] = (T[:, -2] + factor * T_inf) / (1 + factor)

    @staticmethod
    def apply_3d_bc(T: np.ndarray, bcs: Dict, dx: float, dy: float, dz: float, k: float):
        """
        Applies BCs to a 3D temperature array inplace.
        Axes: 0=z, 1=y, 2=x
        bcs keys: z_front, z_back, y_top, y_bottom, x_left, x_right
        """
        # Z-axis (front, back)
        z_front = bcs.get('z_front', {'type': 'adiabatic'})
        if z_front['type'] == 'dirichlet': T[0, :, :] = z_front['value']
        elif z_front['type'] == 'adiabatic': T[0, :, :] = T[1, :, :]
        elif z_front['type'] == 'convective':
            f = z_front.get('h', 10) * dz / k
            T[0, :, :] = (T[1, :, :] + f * z_front.get('T_inf', 293.15)) / (1 + f)
            
        z_back = bcs.get('z_back', {'type': 'adiabatic'})
        if z_back['type'] == 'dirichlet': T[-1, :, :] = z_back['value']
        elif z_back['type'] == 'adiabatic': T[-1, :, :] = T[-2, :, :]
        elif z_back['type'] == 'convective':
            f = z_back.get('h', 10) * dz / k
            T[-1, :, :] = (T[-2, :, :] + f * z_back.get('T_inf', 293.15)) / (1 + f)

        # Y-axis (top, bottom)
        y_top = bcs.get('y_top', {'type': 'adiabatic'})
        if y_top['type'] == 'dirichlet': T[:, 0, :] = y_top['value']
        elif y_top['type'] == 'adiabatic': T[:, 0, :] = T[:, 1, :]
        elif y_top['type'] == 'convective':
            f = y_top.get('h', 10) * dy / k
            T[:, 0, :] = (T[:, 1, :] + f * y_top.get('T_inf', 293.15)) / (1 + f)
            
        y_bottom = bcs.get('y_bottom', {'type': 'adiabatic'})
        if y_bottom['type'] == 'dirichlet': T[:, -1, :] = y_bottom['value']
        elif y_bottom['type'] == 'adiabatic': T[:, -1, :] = T[:, -2, :]
        elif y_bottom['type'] == 'convective':
            f = y_bottom.get('h', 10) * dy / k
            T[:, -1, :] = (T[:, -2, :] + f * y_bottom.get('T_inf', 293.15)) / (1 + f)

        # X-axis (left, right)
        x_left = bcs.get('x_left', {'type': 'adiabatic'})
        if x_left['type'] == 'dirichlet': T[:, :, 0] = x_left['value']
        elif x_left['type'] == 'adiabatic': T[:, :, 0] = T[:, :, 1]
        elif x_left['type'] == 'convective':
            f = x_left.get('h', 10) * dx / k
            T[:, :, 0] = (T[:, :, 1] + f * x_left.get('T_inf', 293.15)) / (1 + f)
            
        x_right = bcs.get('x_right', {'type': 'adiabatic'})
        if x_right['type'] == 'dirichlet': T[:, :, -1] = x_right['value']
        elif x_right['type'] == 'adiabatic': T[:, :, -1] = T[:, :, -2]
        elif x_right['type'] == 'convective':
            f = x_right.get('h', 10) * dx / k
            T[:, :, -1] = (T[:, :, -2] + f * x_right.get('T_inf', 293.15)) / (1 + f)

    @staticmethod
    def apply_1d_fvm_bc(T: np.ndarray, bcs: Dict, dx: float, k: float):
        # Left boundary
        left = bcs.get('left', {'type': 'adiabatic'})
        if left['type'] == 'dirichlet':
            T[0] = 2 * left['value'] - T[1]
        elif left['type'] == 'neumann':
            T[0] = T[1] + left.get('value', 0.0) * dx / k
        elif left['type'] == 'adiabatic':
            T[0] = T[1]
        elif left['type'] == 'convective':
            h, T_inf = left.get('h', 10.0), left.get('T_inf', 293.15)
            T[0] = (h * T_inf + (k/dx - h/2) * T[1]) / (k/dx + h/2)

        # Right boundary
        right = bcs.get('right', {'type': 'adiabatic'})
        if right['type'] == 'dirichlet':
            T[-1] = 2 * right['value'] - T[-2]
        elif right['type'] == 'neumann':
            T[-1] = T[-2] - right.get('value', 0.0) * dx / k
        elif right['type'] == 'adiabatic':
            T[-1] = T[-2]
        elif right['type'] == 'convective':
            h, T_inf = right.get('h', 10.0), right.get('T_inf', 293.15)
            T[-1] = (h * T_inf + (k/dx - h/2) * T[-2]) / (k/dx + h/2)

    @staticmethod
    def apply_2d_fvm_bc(T: np.ndarray, bcs: Dict, dx: float, dy: float, k: float):
        # Top (y=0)
        top = bcs.get('top', {'type': 'adiabatic'})
        if top['type'] == 'dirichlet': T[0, :] = 2 * top['value'] - T[1, :]
        elif top['type'] == 'neumann': T[0, :] = T[1, :] + top.get('value', 0.0) * dy / k
        elif top['type'] == 'adiabatic': T[0, :] = T[1, :]
        elif top['type'] == 'convective':
            h, T_inf = top.get('h', 10.0), top.get('T_inf', 293.15)
            T[0, :] = (h * T_inf + (k/dy - h/2) * T[1, :]) / (k/dy + h/2)

        # Bottom (y=-1)
        bottom = bcs.get('bottom', {'type': 'adiabatic'})
        if bottom['type'] == 'dirichlet': T[-1, :] = 2 * bottom['value'] - T[-2, :]
        elif bottom['type'] == 'neumann': T[-1, :] = T[-2, :] - bottom.get('value', 0.0) * dy / k
        elif bottom['type'] == 'adiabatic': T[-1, :] = T[-2, :]
        elif bottom['type'] == 'convective':
            h, T_inf = bottom.get('h', 10.0), bottom.get('T_inf', 293.15)
            T[-1, :] = (h * T_inf + (k/dy - h/2) * T[-2, :]) / (k/dy + h/2)

        # Left (x=0)
        left = bcs.get('left', {'type': 'adiabatic'})
        if left['type'] == 'dirichlet': T[:, 0] = 2 * left['value'] - T[:, 1]
        elif left['type'] == 'neumann': T[:, 0] = T[:, 1] + left.get('value', 0.0) * dx / k
        elif left['type'] == 'adiabatic': T[:, 0] = T[:, 1]
        elif left['type'] == 'convective':
            h, T_inf = left.get('h', 10.0), left.get('T_inf', 293.15)
            T[:, 0] = (h * T_inf + (k/dx - h/2) * T[:, 1]) / (k/dx + h/2)

        # Right (x=-1)
        right = bcs.get('right', {'type': 'adiabatic'})
        if right['type'] == 'dirichlet': T[:, -1] = 2 * right['value'] - T[:, -2]
        elif right['type'] == 'neumann': T[:, -1] = T[:, -2] - right.get('value', 0.0) * dx / k
        elif right['type'] == 'adiabatic': T[:, -1] = T[:, -2]
        elif right['type'] == 'convective':
            h, T_inf = right.get('h', 10.0), right.get('T_inf', 293.15)
            T[:, -1] = (h * T_inf + (k/dx - h/2) * T[:, -2]) / (k/dx + h/2)

    @staticmethod
    def apply_3d_fvm_bc(T: np.ndarray, bcs: Dict, dx: float, dy: float, dz: float, k: float):
        # Z-axis (front, back)
        z_front = bcs.get('z_front', {'type': 'adiabatic'})
        if z_front['type'] == 'dirichlet': T[0, :, :] = 2 * z_front['value'] - T[1, :, :]
        elif z_front['type'] == 'neumann': T[0, :, :] = T[1, :, :] + z_front.get('value', 0.0) * dz / k
        elif z_front['type'] == 'adiabatic': T[0, :, :] = T[1, :, :]
        elif z_front['type'] == 'convective':
            h, T_inf = z_front.get('h', 10.0), z_front.get('T_inf', 293.15)
            T[0, :, :] = (h * T_inf + (k/dz - h/2) * T[1, :, :]) / (k/dz + h/2)
            
        z_back = bcs.get('z_back', {'type': 'adiabatic'})
        if z_back['type'] == 'dirichlet': T[-1, :, :] = 2 * z_back['value'] - T[-2, :, :]
        elif z_back['type'] == 'neumann': T[-1, :, :] = T[-2, :, :] - z_back.get('value', 0.0) * dz / k
        elif z_back['type'] == 'adiabatic': T[-1, :, :] = T[-2, :, :]
        elif z_back['type'] == 'convective':
            h, T_inf = z_back.get('h', 10.0), z_back.get('T_inf', 293.15)
            T[-1, :, :] = (h * T_inf + (k/dz - h/2) * T[-2, :, :]) / (k/dz + h/2)

        # Y-axis (top, bottom)
        y_top = bcs.get('y_top', {'type': 'adiabatic'})
        if y_top['type'] == 'dirichlet': T[:, 0, :] = 2 * y_top['value'] - T[:, 1, :]
        elif y_top['type'] == 'neumann': T[:, 0, :] = T[:, 1, :] + y_top.get('value', 0.0) * dy / k
        elif y_top['type'] == 'adiabatic': T[:, 0, :] = T[:, 1, :]
        elif y_top['type'] == 'convective':
            h, T_inf = y_top.get('h', 10.0), y_top.get('T_inf', 293.15)
            T[:, 0, :] = (h * T_inf + (k/dy - h/2) * T[:, 1, :]) / (k/dy + h/2)
            
        y_bottom = bcs.get('y_bottom', {'type': 'adiabatic'})
        if y_bottom['type'] == 'dirichlet': T[:, -1, :] = 2 * y_bottom['value'] - T[:, -2, :]
        elif y_bottom['type'] == 'neumann': T[:, -1, :] = T[:, -2, :] - y_bottom.get('value', 0.0) * dy / k
        elif y_bottom['type'] == 'adiabatic': T[:, -1, :] = T[:, -2, :]
        elif y_bottom['type'] == 'convective':
            h, T_inf = y_bottom.get('h', 10.0), y_bottom.get('T_inf', 293.15)
            T[:, -1, :] = (h * T_inf + (k/dy - h/2) * T[:, -2, :]) / (k/dy + h/2)

        # X-axis (left, right)
        x_left = bcs.get('x_left', {'type': 'adiabatic'})
        if x_left['type'] == 'dirichlet': T[:, :, 0] = 2 * x_left['value'] - T[:, :, 1]
        elif x_left['type'] == 'neumann': T[:, :, 0] = T[:, :, 1] + x_left.get('value', 0.0) * dx / k
        elif x_left['type'] == 'adiabatic': T[:, :, 0] = T[:, :, 1]
        elif x_left['type'] == 'convective':
            h, T_inf = x_left.get('h', 10.0), x_left.get('T_inf', 293.15)
            T[:, :, 0] = (h * T_inf + (k/dx - h/2) * T[:, :, 1]) / (k/dx + h/2)
            
        x_right = bcs.get('x_right', {'type': 'adiabatic'})
        if x_right['type'] == 'dirichlet': T[:, :, -1] = 2 * x_right['value'] - T[:, :, -2]
        elif x_right['type'] == 'neumann': T[:, :, -1] = T[:, :, -2] - x_right.get('value', 0.0) * dx / k
        elif x_right['type'] == 'adiabatic': T[:, :, -1] = T[:, :, -2]
        elif x_right['type'] == 'convective':
            h, T_inf = x_right.get('h', 10.0), x_right.get('T_inf', 293.15)
            T[:, :, -1] = (h * T_inf + (k/dx - h/2) * T[:, :, -2]) / (k/dx + h/2)
