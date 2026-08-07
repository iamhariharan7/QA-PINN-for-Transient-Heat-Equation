import numpy as np
import os
import time

def solve_exact(domain, output_dir=None):
    """
    Computes or retrieves the exact ground truth solution for the 1D heat equation.
    """
    inf_start = time.time()
    
    if hasattr(domain, 'U_exact') and domain.U_exact is not None:
        U_exact = domain.U_exact
    else:
        X, T, x, t = domain.get_grid()
        U_exact = np.exp(- (np.pi / domain.L)**2 * domain.alpha * T) * np.sin(np.pi * X / domain.L)
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Actual", "1D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "Actual", "1D", "actual_temperature.npz"), U=U_exact, x=domain.x, t=domain.t)
        np.savetxt(os.path.join(output_dir, "Actual", "1D", "actual_temperature.csv"), U_exact, delimiter=",")
        
    inference_time = time.time() - inf_start
    return U_exact, inference_time
