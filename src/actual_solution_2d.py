import numpy as np
import os
import time

def solve_exact_2d(domain2d, output_dir=None):
    """
    Computes or retrieves the exact ground truth solution for the 2D heat equation.
    """
    inf_start = time.time()
    
    if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None:
        U_exact = domain2d.U_exact
    else:
        X3, Y3, T3 = domain2d.get_xyt_grid()
        lam = np.pi**2 * (1.0 / domain2d.Lx**2 + 1.0 / domain2d.Ly**2)
        U_exact = (
            np.exp(-domain2d.alpha * lam * T3)
            * np.sin(np.pi * X3 / domain2d.Lx)
            * np.sin(np.pi * Y3 / domain2d.Ly)
        )

    if output_dir:
        os.makedirs(os.path.join(output_dir, "Actual", "2D"), exist_ok=True)
        np.savez(
            os.path.join(output_dir, "Actual", "2D", "actual_temperature_2d.npz"),
            U=U_exact,
            x=domain2d.x,
            y=domain2d.y,
            t=domain2d.t,
        )
        mid_t = domain2d.Nt // 2
        np.savetxt(
            os.path.join(output_dir, "Actual", "2D", "actual_temperature_2d_midslice.csv"),
            U_exact[mid_t],
            delimiter=",",
        )

    inference_time = time.time() - inf_start
    return U_exact, inference_time
