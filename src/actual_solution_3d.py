import numpy as np
import os
import time

def solve_exact_3d(domain3d, output_dir=None):
    """
    Computes or retrieves the exact ground truth solution for the 3D heat equation.
    """
    inf_start = time.time()
    
    if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None:
        U_exact = domain3d.U_exact
    else:
        X4, Y4, Z4, T4 = domain3d.get_xyzt_grid()
        alpha = domain3d.alpha
        Lx, Ly, Lz = domain3d.Lx, domain3d.Ly, domain3d.Lz
        decay_rate = alpha * np.pi**2 * (1.0/Lx**2 + 1.0/Ly**2 + 1.0/Lz**2)
        U_exact = np.exp(-decay_rate * T4) * np.sin(np.pi * X4 / Lx) * np.sin(np.pi * Y4 / Ly) * np.sin(np.pi * Z4 / Lz)
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Actual", "3D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "Actual", "3D", "actual_temperature_3d.npz"), 
                 U=U_exact, x=domain3d.x, y=domain3d.y, z=domain3d.z, t=domain3d.t)
                 
    inference_time = time.time() - inf_start
    return U_exact, inference_time
