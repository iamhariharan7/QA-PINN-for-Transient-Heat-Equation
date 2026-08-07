import numpy as np
import time
import os

def solve_cfd_3d(domain3d, output_dir=None):
    """
    Solves the 3D heat equation using Explicit Finite Difference Method.
    """
    start_time = time.time()
    
    alpha = domain3d.alpha
    Nx, Ny, Nz, Nt = domain3d.Nx, domain3d.Ny, domain3d.Nz, domain3d.Nt
    dx, dy, dz, dt = domain3d.dx, domain3d.dy, domain3d.dz, domain3d.dt
    
    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2
    rz = alpha * dt / dz**2
    
    if rx + ry + rz > 0.5:
        print(f"Warning: Explicit 3D CFD stability parameter rx+ry+rz = {rx+ry+rz:.4f} > 0.5")
    
    U = np.zeros((Nt, Nx, Ny, Nz))
    X3, Y3, Z3 = domain3d.get_xyz_grid()
    U[0] = domain3d.initial_condition(X3, Y3, Z3)
    
    if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None:
        U[:, 0, :, :] = domain3d.U_exact[:, 0, :, :]
        U[:, -1, :, :] = domain3d.U_exact[:, -1, :, :]
        U[:, :, 0, :] = domain3d.U_exact[:, :, 0, :]
        U[:, :, -1, :] = domain3d.U_exact[:, :, -1, :]
        U[:, :, :, 0] = domain3d.U_exact[:, :, :, 0]
        U[:, :, :, -1] = domain3d.U_exact[:, :, :, -1]

    for n in range(Nt - 1):
        if Nx > 2 and Ny > 2 and Nz > 2:
            U[n+1, 1:-1, 1:-1, 1:-1] = U[n, 1:-1, 1:-1, 1:-1] + \
                rx * (U[n, 2:, 1:-1, 1:-1] - 2*U[n, 1:-1, 1:-1, 1:-1] + U[n, :-2, 1:-1, 1:-1]) + \
                ry * (U[n, 1:-1, 2:, 1:-1] - 2*U[n, 1:-1, 1:-1, 1:-1] + U[n, 1:-1, :-2, 1:-1]) + \
                rz * (U[n, 1:-1, 1:-1, 2:] - 2*U[n, 1:-1, 1:-1, 1:-1] + U[n, 1:-1, 1:-1, :-2])
        else:
            # Handle degenerate spatial dimensions (e.g. Ny=1)
            U[n+1] = U[n]
            
    cfd_time = time.time() - start_time
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "CFD", "3D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "CFD", "3D", "cfd_temperature_3d.npz"), 
                 U=U, x=domain3d.x, y=domain3d.y, z=domain3d.z, t=domain3d.t)
                 
    return U, cfd_time, cfd_time
