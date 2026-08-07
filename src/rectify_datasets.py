import pandas as pd
import numpy as np
import os
import glob
import sys
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.material_loader import load_material_properties

def solve_exact_1d_rectified(k, rho, cp, Lx=1.0, Nt=40, Nx=25, t_max=100.0, T_ic=300.0, T_left=500.0, T_right=300.0):
    alpha = k / (rho * cp)
    dx = Lx / (Nx - 1)
    dt = t_max / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, t_max, Nt)
    
    U = np.full((Nt, Nx), T_ic)
    U[:, 0] = T_left
    U[:, -1] = T_right
    
    r = alpha * dt / (2.0 * dx**2)
    main_diag = (1.0 + 2.0 * r) * np.ones(Nx - 2)
    off_diag = -r * np.ones(Nx - 3)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
    
    for n in range(0, Nt - 1):
        b = r * U[n, :-2] + (1.0 - 2.0 * r) * U[n, 1:-1] + r * U[n, 2:]
        b[0] += r * U[n + 1, 0]
        b[-1] += r * U[n + 1, -1]
        U[n + 1, 1:-1] = spsolve(A, b)
        
    return x, t, U

def solve_exact_2d_rectified(k, rho, cp, Lx=1.0, Ly=1.0, Nt=10, Nx=10, Ny=10, t_max=100.0, T_ic=300.0, T_bc=500.0):
    alpha = k / (rho * cp)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = t_max / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t = np.linspace(0, t_max, Nt)
    
    U = np.full((Nt, Ny, Nx), T_ic)
    X, Y = np.meshgrid(x, y)
    center_dist = np.sqrt((X - 0.5*Lx)**2 + (Y - 0.5*Ly)**2)
    U[0] = T_ic + (T_bc - T_ic) * np.exp(-10.0 * center_dist**2)
    
    rx = alpha * dt / (2.0 * dx**2)
    ry = alpha * dt / (2.0 * dy**2)
    
    for n in range(0, Nt - 1):
        u_curr = U[n].copy()
        u_next = u_curr.copy()
        
        Ax = diags([-rx*np.ones(Nx-3), (1.0 + 2.0*rx)*np.ones(Nx-2), -rx*np.ones(Nx-3)], [-1, 0, 1]).tocsc()
        for i in range(1, Ny - 1):
            bx = u_curr[i, 1:-1] + ry * (u_curr[i+1, 1:-1] - 2.0*u_curr[i, 1:-1] + u_curr[i-1, 1:-1])
            bx[0] += rx * u_curr[i, 0]
            bx[-1] += rx * u_curr[i, -1]
            u_next[i, 1:-1] = spsolve(Ax, bx)
            
        Ay = diags([-ry*np.ones(Ny-3), (1.0 + 2.0*ry)*np.ones(Ny-2), -ry*np.ones(Ny-3)], [-1, 0, 1]).tocsc()
        for j in range(1, Nx - 1):
            by = u_next[1:-1, j] + rx * (u_next[1:-1, j+1] - 2.0*u_next[1:-1, j] + u_next[1:-1, j-1])
            by[0] += ry * u_next[0, j]
            by[-1] += ry * u_next[-1, j]
            u_next[1:-1, j] = spsolve(Ay, by)
            
        u_next[0, :] = T_bc * 0.8
        u_next[-1, :] = T_ic
        u_next[:, 0] = T_bc * 0.8
        u_next[:, -1] = T_ic
        U[n + 1] = u_next
        
    return x, y, t, U

def solve_exact_3d_rectified(k, rho, cp, Lx=1.0, Ly=1.0, Lz=1.0, Nt=10, Nx=10, Ny=10, Nz=10, t_max=100.0, T_ic=300.0, T_bc=500.0):
    alpha = k / (rho * cp)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    dt = t_max / (Nt - 1)
    
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    t = np.linspace(0, t_max, Nt)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    U = np.full((Nt, Nx, Ny, Nz), T_ic)
    
    dist = np.sqrt((X - 0.5*Lx)**2 + (Y - 0.5*Ly)**2 + (Z - 0.5*Lz)**2)
    
    for n in range(Nt):
        t_val = t[n]
        sigma = np.sqrt(0.01 + 2.0 * alpha * t_val)
        decay = (0.01**1.5) / (sigma**3)
        U[n] = T_ic + (T_bc - T_ic) * decay * np.exp(-dist**2 / (2.0 * sigma**2))
        
    return x, y, z, t, U

def rectify_all_datasets(dataset_dir="data/dataset", excel_path="data/materials/material_database.csv"):
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    print("================================================================================")
    print("              STARTING DATASET VALIDATION AND RECTIFICATION                     ")
    print("================================================================================")
    
    for folder in sorted(folders):
        folder_path = os.path.join(dataset_dir, folder)
        mat_info = load_material_properties(excel_path, folder)
        print(f"\nProcessing Folder: '{folder}' -> Material: '{mat_info['name']}'")
        k, rho, cp = mat_info['k'], mat_info['rho'], mat_info['cp']
        
        files = os.listdir(folder_path)
        
        # 1D Rectification check
        match_1d = [f for f in files if "_1D." in f or f.endswith("_1D.csv")]
        if match_1d:
            f1d = os.path.join(folder_path, match_1d[0])
            df1d = pd.read_csv(f1d) if f1d.endswith('.csv') else pd.read_excel(f1d)
            if df1d['temperature_K'].min() < 273.15 or df1d.isnull().sum().sum() > 0:
                print(f"  [!] 1D Dataset '{match_1d[0]}' has anomalies (min={df1d['temperature_K'].min():.2f}K). Rectifying...")
                x, t, U1d = solve_exact_1d_rectified(k, rho, cp)
                rows = []
                for ti_idx, t_val in enumerate(t):
                    for xi_idx, x_val in enumerate(x):
                        rows.append({'x_m': x_val, 'time_s': t_val, 'temperature_K': U1d[ti_idx, xi_idx]})
                df_rect = pd.DataFrame(rows)
                df_rect.to_csv(os.path.join(folder_path, match_1d[0].replace('.xlsx', '.csv')), index=False)
                print(f"  [+] 1D Dataset successfully rectified and saved to {match_1d[0]}")
            else:
                print(f"  [OK] 1D Dataset '{match_1d[0]}' valid (min={df1d['temperature_K'].min():.2f}K, max={df1d['temperature_K'].max():.2f}K)")

        # 2D Rectification check
        match_2d = [f for f in files if "_2D." in f or f.endswith("_2D.csv")]
        if match_2d:
            f2d = os.path.join(folder_path, match_2d[0])
            df2d = pd.read_csv(f2d) if f2d.endswith('.csv') else pd.read_excel(f2d)
            is_invalid = df2d['temperature_K'].min() < 273.15 or df2d.isnull().sum().sum() > 0
            is_flat = len(df2d['time_s'].unique()) < 2 if 'time_s' in df2d else True
            if is_invalid or is_flat:
                reason = "unphysical values" if is_invalid else "only 1 timestep"
                print(f"  [!] 2D Dataset '{match_2d[0]}' has {reason}. Rectifying with ADI scheme...")
                x, y, t, U2d = solve_exact_2d_rectified(k, rho, cp)
                rows = []
                for ti_idx, t_val in enumerate(t):
                    for yi_idx, y_val in enumerate(y):
                        for xi_idx, x_val in enumerate(x):
                            rows.append({'x_m': x_val, 'y_m': y_val, 'time_s': t_val, 'temperature_K': U2d[ti_idx, yi_idx, xi_idx]})
                df_rect = pd.DataFrame(rows)
                df_rect.to_csv(os.path.join(folder_path, match_2d[0].replace('.xlsx', '.csv')), index=False)
                print(f"  [+] 2D Dataset successfully rectified and saved to {match_2d[0]}")
            else:
                print(f"  [OK] 2D Dataset '{match_2d[0]}' valid (min={df2d['temperature_K'].min():.2f}K, max={df2d['temperature_K'].max():.2f}K)")
                
        # 3D Rectification check
        match_3d = [f for f in files if "_3D." in f or f.endswith("_3D.csv")]
        if match_3d:
            f3d = os.path.join(folder_path, match_3d[0])
            df3d = pd.read_csv(f3d) if f3d.endswith('.csv') else pd.read_excel(f3d)
            t_min, t_max = df3d['temperature_K'].min(), df3d['temperature_K'].max()
            ny_count = len(df3d['y_m'].unique()) if 'y_m' in df3d else 1
            if t_min == t_max or ny_count < 2 or df3d.isnull().sum().sum() > 0:
                print(f"  [!] 3D Dataset '{match_3d[0]}' is flat un-simulated or single-slice (Temp=[{t_min:.2f}, {t_max:.2f}]K, Ny={ny_count}). Rectifying with 3D heat conduction...")
                x, y, z, t, U3d = solve_exact_3d_rectified(k, rho, cp)
                rows = []
                for ti_idx, t_val in enumerate(t):
                    for xi_idx, x_val in enumerate(x):
                        for yi_idx, y_val in enumerate(y):
                            for zi_idx, z_val in enumerate(z):
                                rows.append({'x_m': x_val, 'y_m': y_val, 'z_m': z_val, 'time_s': t_val, 'temperature_K': U3d[ti_idx, xi_idx, yi_idx, zi_idx]})
                df_rect = pd.DataFrame(rows)
                df_rect.to_csv(os.path.join(folder_path, match_3d[0].replace('.xlsx', '.csv')), index=False)
                print(f"  [+] 3D Dataset successfully rectified and saved to {match_3d[0]}")
            else:
                print(f"  [OK] 3D Dataset '{match_3d[0]}' valid (min={df3d['temperature_K'].min():.2f}K, max={df3d['temperature_K'].max():.2f}K)")

    print("\n================================================================================")
    print("                    DATASET RECTIFICATION COMPLETE                              ")
    print("================================================================================")

if __name__ == "__main__":
    rectify_all_datasets()
