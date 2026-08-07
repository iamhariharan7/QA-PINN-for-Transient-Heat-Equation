import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import os

def solve_cfd(domain, output_dir=None):
    """
    Solves the 1D heat equation using the Crank-Nicolson method.
    """
    start_time = time.time()
    
    Nx = domain.Nx
    Nt = domain.Nt
    dx = domain.dx
    dt = domain.dt
    alpha = domain.alpha
    
    r = alpha * dt / (2 * dx**2)
    
    U = np.zeros((Nt, Nx))
    
    U[0, :] = domain.initial_condition(domain.x)
    
    left_bc, right_bc = domain.boundary_conditions()
    U[:, 0] = left_bc
    U[:, -1] = right_bc
    
    diag_A = (1 + 2*r) * np.ones(Nx - 2)
    off_diag_A = -r * np.ones(Nx - 3)
    A = sp.diags([off_diag_A, diag_A, off_diag_A], [-1, 0, 1], format='csr')
    
    diag_B = (1 - 2*r) * np.ones(Nx - 2)
    off_diag_B = r * np.ones(Nx - 3)
    B = sp.diags([off_diag_B, diag_B, off_diag_B], [-1, 0, 1], format='csr')
    
    for n in range(0, Nt - 1):
        b = B.dot(U[n, 1:-1])
        b[0] += r * (U[n, 0] + U[n+1, 0])
        b[-1] += r * (U[n, -1] + U[n+1, -1])
        U[n+1, 1:-1] = spsolve(A, b)
        
    cfd_time = time.time() - start_time
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "CFD", "1D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "CFD", "1D", "cfd_temperature.npz"), U=U, x=domain.x, t=domain.t)
        
    return U, cfd_time, cfd_time
