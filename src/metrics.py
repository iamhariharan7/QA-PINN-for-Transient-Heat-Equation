import numpy as np
import torch

def calculate_metrics(U_pred, U_exact):
    """
    Calculates various error metrics between the predicted and exact solutions.
    """
    error = U_pred - U_exact
    
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    l2_exact = np.sqrt(np.sum(U_exact**2))
    if l2_exact > 1e-10:
        rel_l2 = np.sqrt(np.sum(error**2)) / l2_exact
    else:
        rel_l2 = float('inf')
        
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Max_Absolute_Error": float(max_error),
        "Relative_L2_Error": float(rel_l2)
    }

def count_parameters(model):
    """Counts trainable parameters in a PyTorch model."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory_mb(model):
    """Estimates the memory requirement (MB) of the model parameters."""
    if model is None:
        return 0.0
    mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return mem_bytes / (1024 ** 2)

def compute_pde_residual_map(U, domain):
    """
    Computes the discrete PDE residual |u_t - alpha * u_xx| over the grid.
    Returns a 2D numpy array of the same shape as U.
    """
    dt = domain.dt
    dx = domain.dx
    alpha = domain.alpha
    
    residual = np.zeros_like(U)
    if U.ndim != 2:
        return residual
        
    Nx = U.shape[1]
    sx = slice(1, -1) if Nx > 2 else slice(0, None)
    
    for n in range(U.shape[0] - 1):
        u_t = (U[n+1, sx] - U[n, sx]) / dt
        if Nx > 2:
            u_xx = (U[n, 2:] - 2*U[n, 1:-1] + U[n, :-2]) / (dx**2)
        else:
            u_xx = 0.0
        residual[n, sx] = np.abs(u_t - alpha * u_xx)
        
    return residual

def compute_pde_residual_scalar(U, domain):
    """Computes the mean PDE residual error."""
    res_map = compute_pde_residual_map(U, domain)
    interior_res = res_map[:-1, 1:-1] if res_map.shape[1] > 2 else res_map[:-1, :]
    if interior_res.size == 0:
        return 0.0
    return float(np.mean(interior_res))

def compute_pde_residual_std(U, domain):
    """Computes the standard deviation of the PDE residual error (Explainability metric)."""
    res_map = compute_pde_residual_map(U, domain)
    interior_res = res_map[:-1, 1:-1] if res_map.shape[1] > 2 else res_map[:-1, :]
    if interior_res.size == 0:
        return 0.0
    return float(np.std(interior_res))

def compute_pde_residual_map_2d(U, domain2d):
    """
    Computes the discrete PDE residual |u_t - alpha*(u_xx + u_yy)|
    over the 3-D grid U[t, y, x].
    """
    dt    = domain2d.dt
    dx    = domain2d.dx
    dy    = domain2d.dy
    alpha = domain2d.alpha

    residual = np.zeros_like(U)
    if U.ndim != 3:
        return residual

    Nt, Ny, Nx = U.shape
    sy = slice(1, -1) if Ny > 2 else slice(0, None)
    sx = slice(1, -1) if Nx > 2 else slice(0, None)

    for n in range(Nt - 1):
        u_t = (U[n + 1, sy, sx] - U[n, sy, sx]) / dt
        u_xx = (U[n, sy, 2:] - 2.0 * U[n, sy, 1:-1] + U[n, sy, :-2]) / dx**2 if Nx > 2 else 0.0
        u_yy = (U[n, 2:, sx] - 2.0 * U[n, 1:-1, sx] + U[n, :-2, sx]) / dy**2 if Ny > 2 else 0.0
        residual[n, sy, sx] = np.abs(u_t - alpha * (u_xx + u_yy))

    return residual

def compute_pde_residual_scalar_2d(U, domain2d):
    res_map = compute_pde_residual_map_2d(U, domain2d)
    sy = slice(1, -1) if res_map.shape[1] > 2 else slice(0, None)
    sx = slice(1, -1) if res_map.shape[2] > 2 else slice(0, None)
    interior_res = res_map[:-1, sy, sx]
    if interior_res.size == 0:
        return 0.0
    return float(np.mean(interior_res))

def compute_pde_residual_std_2d(U, domain2d):
    res_map = compute_pde_residual_map_2d(U, domain2d)
    sy = slice(1, -1) if res_map.shape[1] > 2 else slice(0, None)
    sx = slice(1, -1) if res_map.shape[2] > 2 else slice(0, None)
    interior_res = res_map[:-1, sy, sx]
    if interior_res.size == 0:
        return 0.0
    return float(np.std(interior_res))

def compute_pde_residual_map_3d(U, domain3d):
    """
    Computes the discrete PDE residual |u_t - alpha*(u_xx + u_yy + u_zz)|
    over the 4-D grid U[t, x, y, z].
    """
    dt    = domain3d.dt
    dx    = domain3d.dx
    dy    = domain3d.dy
    dz    = domain3d.dz
    alpha = domain3d.alpha

    residual = np.zeros_like(U)
    if U.ndim != 4:
        return residual

    Nt, Nx, Ny, Nz = U.shape
    sx = slice(1, -1) if Nx > 2 else slice(0, None)
    sy = slice(1, -1) if Ny > 2 else slice(0, None)
    sz = slice(1, -1) if Nz > 2 else slice(0, None)

    for n in range(Nt - 1):
        u_t  = (U[n + 1, sx, sy, sz] - U[n, sx, sy, sz]) / dt
        u_xx = (U[n, 2:, sy, sz] - 2.0 * U[n, 1:-1, sy, sz] + U[n, :-2, sy, sz]) / dx**2 if Nx > 2 else 0.0
        u_yy = (U[n, sx, 2:, sz] - 2.0 * U[n, sx, 1:-1, sz] + U[n, sx, :-2, sz]) / dy**2 if Ny > 2 else 0.0
        u_zz = (U[n, sx, sy, 2:] - 2.0 * U[n, sx, sy, 1:-1] + U[n, sx, sy, :-2]) / dz**2 if Nz > 2 else 0.0
        
        residual[n, sx, sy, sz] = np.abs(u_t - alpha * (u_xx + u_yy + u_zz))

    return residual

def compute_pde_residual_scalar_3d(U, domain3d):
    res_map = compute_pde_residual_map_3d(U, domain3d)
    sx = slice(1, -1) if res_map.shape[1] > 2 else slice(0, None)
    sy = slice(1, -1) if res_map.shape[2] > 2 else slice(0, None)
    sz = slice(1, -1) if res_map.shape[3] > 2 else slice(0, None)
    interior_res = res_map[:-1, sx, sy, sz]
    if interior_res.size == 0:
        return 0.0
    return float(np.mean(interior_res))

def compute_pde_residual_std_3d(U, domain3d):
    res_map = compute_pde_residual_map_3d(U, domain3d)
    sx = slice(1, -1) if res_map.shape[1] > 2 else slice(0, None)
    sy = slice(1, -1) if res_map.shape[2] > 2 else slice(0, None)
    sz = slice(1, -1) if res_map.shape[3] > 2 else slice(0, None)
    interior_res = res_map[:-1, sx, sy, sz]
    if interior_res.size == 0:
        return 0.0
    return float(np.std(interior_res))
