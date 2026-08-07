"""
2D Heat Equation CFD Solver using the Alternating Direction Implicit (ADI) method.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import os

def _build_tridiagonal(n_inner, coeff_implicit, coeff_explicit):
    r = coeff_implicit
    r_e = coeff_explicit

    diag_A = (1.0 + 2.0 * r) * np.ones(n_inner)
    off_A = -r * np.ones(n_inner - 1)
    A = sp.diags([off_A, diag_A, off_A], [-1, 0, 1], format="csr")

    diag_B = (1.0 - 2.0 * r_e) * np.ones(n_inner)
    off_B = r_e * np.ones(n_inner - 1)
    B = sp.diags([off_B, diag_B, off_B], [-1, 0, 1], format="csr")

    return A, B

def solve_cfd_2d(domain2d, output_dir=None):
    start_time = time.time()

    alpha = domain2d.alpha
    Nx, Ny, Nt = domain2d.Nx, domain2d.Ny, domain2d.Nt
    dx, dy, dt = domain2d.dx, domain2d.dy, domain2d.dt

    rx = alpha * dt / (2.0 * dx**2)
    ry = alpha * dt / (2.0 * dy**2)

    Nx_inner = max(1, Nx - 2)
    Ny_inner = max(1, Ny - 2)

    Ax, Bx = _build_tridiagonal(Nx_inner, rx, ry)
    Ay, By = _build_tridiagonal(Ny_inner, ry, rx)

    U = np.zeros((Nt, Ny, Nx))
    X2, Y2 = domain2d.get_xy_grid()
    U[0] = domain2d.initial_condition(X2, Y2)

    if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None:
        U[:, 0, :]  = domain2d.U_exact[:, 0, :]
        U[:, -1, :] = domain2d.U_exact[:, -1, :]
        U[:, :, 0]  = domain2d.U_exact[:, :, 0]
        U[:, :, -1] = domain2d.U_exact[:, :, -1]

    for n in range(Nt - 1):
        U_n = U[n]

        U_star = np.zeros_like(U_n)
        U_star[0, :]  = U[n+1, 0, :]
        U_star[-1, :] = U[n+1, -1, :]
        U_star[:, 0]  = U[n+1, :, 0]
        U_star[:, -1] = U[n+1, :, -1]

        for j in range(1, Ny - 1):
            rhs = By.dot(U_n[j, 1:-1])
            rhs += ry * (U_n[j - 1, 1:-1] + U_n[j + 1, 1:-1])
            rhs[0] += rx * (U_star[j, 0] + U_n[j, 0])
            rhs[-1] += rx * (U_star[j, -1] + U_n[j, -1])
            U_star[j, 1:-1] = spsolve(Ax, rhs)

        U_next = np.zeros_like(U_n)
        U_next[0, :]  = U[n+1, 0, :]
        U_next[-1, :] = U[n+1, -1, :]
        U_next[:, 0]  = U[n+1, :, 0]
        U_next[:, -1] = U[n+1, :, -1]

        for i in range(1, Nx - 1):
            rhs = Bx.dot(U_star[1:-1, i])
            rhs += rx * (U_star[1:-1, i - 1] + U_star[1:-1, i + 1])
            rhs[0] += ry * (U_next[0, i] + U_star[0, i])
            rhs[-1] += ry * (U_next[-1, i] + U_star[-1, i])
            U_next[1:-1, i] = spsolve(Ay, rhs)

        U[n + 1] = U_next

    cfd2d_time = time.time() - start_time

    if output_dir:
        target_dir = os.path.join(output_dir, "CFD", "2D")
        os.makedirs(target_dir, exist_ok=True)
        np.savez(
            os.path.join(target_dir, "cfd_temperature_2d.npz"),
            U=U,
            x=domain2d.x,
            y=domain2d.y,
            t=domain2d.t,
        )

    return U, cfd2d_time, cfd2d_time
