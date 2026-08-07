import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

ErrorGYR = LinearSegmentedColormap.from_list('ErrorGYR', ['green', 'yellow', 'red'])
TempYOR = LinearSegmentedColormap.from_list('TempYOR', ['yellow', 'orange', 'red'])

def save_individual_heatmaps(domain, results, output_dir):
    x = domain.x
    t = domain.t
    
    U_exact = results['U_exact']
    U_cfd = results['U_cfd']
    U_pinn = results['U_pinn']
    U_qa = results['U_qa']
    
    err_pinn = np.abs(U_pinn - U_exact)
    err_qa = np.abs(U_qa - U_exact)
    
    vmin_u = min(U_exact.min(), U_cfd.min())
    vmax_u = max(U_exact.max(), U_cfd.max())
    
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6
    
    solutions = [
        ("Actual", U_exact, TempYOR, "Actual", vmin_u, vmax_u, "Temperature"),
        ("CFD", U_cfd, TempYOR, "CFD", vmin_u, vmax_u, "Temperature"),
        ("PINN Error", err_pinn, ErrorGYR, "PINN", 0, vmax_err, "Absolute Error"),
        ("QA-PINN Error", err_qa, ErrorGYR, "QA-PINN", 0, vmax_err, "Absolute Error"),
    ]
    
    for name, U, cmap, folder, vmin, vmax, title_suffix in solutions:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(x, t, U, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{name} Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        plt.tight_layout()
        
        target_dir = os.path.join(output_dir, folder, "1D")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, "Heat Map (1D).png"), dpi=150)
        plt.close(fig)

def generate_comparison_figure(domain, results, props, output_dir):
    fig = plt.figure(figsize=(25, 5))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    fig.suptitle(f"1D Heat Equation Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    x, t = domain.x, domain.t
    
    err_pinn = np.abs(results['U_pinn'] - results['U_exact'])
    err_qa = np.abs(results['U_qa'] - results['U_exact'])
    
    vmin_u = min(results['U_exact'].min(), results['U_cfd'].min())
    vmax_u = max(results['U_exact'].max(), results['U_cfd'].max())
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6
    
    solutions = [
        ("Actual", results['U_exact'], TempYOR, vmin_u, vmax_u),
        ("CFD", results['U_cfd'], TempYOR, vmin_u, vmax_u),
        ("PINN Error", err_pinn, ErrorGYR, 0, vmax_err),
        ("QA-PINN Error", err_qa, ErrorGYR, 0, vmax_err),
    ]
    
    for col, (name, U, cmap, vmin, vmax) in enumerate(solutions):
        ax = fig.add_subplot(gs[0, col])
        im = ax.contourf(x, t, U, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    target_dir = os.path.join(output_dir, "Comparison", "1D")
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, "Heat Map Comparison (1D).png"), dpi=200)
    plt.close(fig)

def generate_2d_comparison_figure(domain, results, props, output_dir):
    x, y = domain.x, domain.y
    Nt = domain.Nt
    t_indices = {"Start": 0, "Middle": Nt // 2, "End": -1}
    
    err_pinn = np.abs(results['U_pinn_2d'] - results['U_exact_2d'])
    err_qa = np.abs(results['U_qa_2d'] - results['U_exact_2d'])
    
    vmin_u = min(results['U_exact_2d'].min(), results['U_cfd_2d'].min())
    vmax_u = max(results['U_exact_2d'].max(), results['U_cfd_2d'].max())
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6
    
    solutions = [
        ("Actual", results['U_exact_2d'], TempYOR, vmin_u, vmax_u),
        ("CFD", results['U_cfd_2d'], TempYOR, vmin_u, vmax_u),
        ("PINN Error", err_pinn, ErrorGYR, 0, vmax_err),
        ("QA-PINN Error", err_qa, ErrorGYR, 0, vmax_err),
    ]
    
    target_dir = os.path.join(output_dir, "Comparison", "2D")
    os.makedirs(target_dir, exist_ok=True)

    for time_label, t_idx in t_indices.items():
        fig = plt.figure(figsize=(25, 5))
        gs = gridspec.GridSpec(1, 5, figure=fig)
        fig.suptitle(f"2D Heat Equation Comparison ({time_label}): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
        
        for col, (name, U, cmap, vmin, vmax) in enumerate(solutions):
            ax = fig.add_subplot(gs[0, col])
            U_slice = U[t_idx]
            im = ax.pcolormesh(x, y, U_slice, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name}\nt={domain.t[t_idx]:.3f}s", fontsize=14, fontweight='bold')
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(os.path.join(target_dir, f"Heat Map Comparison ({time_label}).png"), dpi=200)
        plt.close(fig)

def generate_3d_comparison_figure(domain, results, props, output_dir):
    z_mid = domain.Nz // 2
    t_mid = domain.Nt // 2
    x, y, z, t = domain.x, domain.y, domain.z, domain.t
    
    if len(y) == 1:
        y_plot = np.array([y[0] - 0.05, y[0] + 0.05])
        def make_u_plot(u_2d): return np.repeat(u_2d, 2, axis=1).T
    else:
        y_plot = y
        def make_u_plot(u_2d): return u_2d.T
        
    err_pinn = np.abs(results['U_pinn_3d'] - results['U_exact_3d'])
    err_qa = np.abs(results['U_qa_3d'] - results['U_exact_3d'])
    
    vmin_u = min(results['U_exact_3d'].min(), results['U_cfd_3d'].min())
    vmax_u = max(results['U_exact_3d'].max(), results['U_cfd_3d'].max())
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6

    solutions = [
        ("Actual", results['U_exact_3d'], TempYOR, vmin_u, vmax_u),
        ("CFD", results['U_cfd_3d'], TempYOR, vmin_u, vmax_u),
        ("PINN Error", err_pinn, ErrorGYR, 0, vmax_err),
        ("QA-PINN Error", err_qa, ErrorGYR, 0, vmax_err),
    ]
    
    target_dir = os.path.join(output_dir, "Comparison", "3D")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Slice Comparison
    fig = plt.figure(figsize=(25, 5))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    fig.suptitle(f"3D Slice Comparison (z-mid, t-mid): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    for col, (name, U, cmap, vmin, vmax) in enumerate(solutions):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(x, y_plot, make_u_plot(U[t_mid, :, :, z_mid]), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(name, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(target_dir, "Slice Heat Map Comparison.png"), dpi=200)
    plt.close(fig)

    # 2. Panel Comparison
    fig = plt.figure(figsize=(25, 8))
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.3)
    fig.suptitle(f"3D Panel Comparison (t-mid): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    z_indices = [0, z_mid, -1]
    z_labels = ["Bottom", "Middle", "Top"]
    for row, (z_idx, z_lbl) in enumerate(zip(z_indices, z_labels)):
        for col, (name, U, cmap, vmin, vmax) in enumerate(solutions):
            ax = fig.add_subplot(gs[row, col])
            im = ax.pcolormesh(x, y_plot, make_u_plot(U[t_mid, :, :, z_idx]), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            if col == 4: plt.colorbar(im, ax=ax)
            if row == 0: ax.set_title(name, fontsize=14, fontweight='bold')
            if col == 0: ax.set_ylabel(f"{z_lbl}\n(z={z[z_idx]:.2f})", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(target_dir, "Panel Heat Map Comparison.png"), dpi=200)
    plt.close(fig)

    # 3. Surface Comparison
    fig = plt.figure(figsize=(25, 6))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    fig.suptitle(f"3D Surface Comparison (y-mid, z-mid): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    X, T_grid = np.meshgrid(x, t)
    for col, (name, U, cmap, vmin, vmax) in enumerate(solutions):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        surf = ax.plot_surface(X, T_grid, U[:, :, len(y)//2, z_mid], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.view_init(elev=28, azim=-55)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(target_dir, "Surface Heat Map Comparison.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_2d_heatmaps(domain2d, results2d, output_dir):
    x  = domain2d.x
    y  = domain2d.y
    Nt = domain2d.Nt
    t_indices = {"Start": 0, "Middle": Nt // 2, "End": -1}

    err_pinn = np.abs(results2d['U_pinn_2d'] - results2d['U_exact_2d'])
    err_qa = np.abs(results2d['U_qa_2d'] - results2d['U_exact_2d'])
    
    vmin_u = min(results2d['U_exact_2d'].min(), results2d['U_cfd_2d'].min())
    vmax_u = max(results2d['U_exact_2d'].max(), results2d['U_cfd_2d'].max())
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6

    solutions = [
        ("Actual", results2d['U_exact_2d'], TempYOR, "Actual", vmin_u, vmax_u),
        ("CFD",    results2d['U_cfd_2d'],   TempYOR,  "CFD", vmin_u, vmax_u),
        ("PINN Error",   err_pinn,  ErrorGYR, "PINN", 0, vmax_err),
        ("QA-PINN Error", err_qa, ErrorGYR, "QA-PINN", 0, vmax_err),
    ]

    for label, U3, cmap, method_name, vmin, vmax in solutions:
        target_dir = os.path.join(output_dir, method_name, "2D")
        os.makedirs(target_dir, exist_ok=True)
        
        for time_label, t_idx in t_indices.items():
            U_slice = U3[t_idx]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(x, y, U_slice, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax, label="Temperature (u)" if "Error" not in label else "Absolute Error")
            t_val = domain2d.t[t_idx]
            ax.set_title(f"{label}  |  t = {t_val:.3f} s", fontsize=13, fontweight='bold')
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect('equal')
            plt.tight_layout()
            fname = f"Heat Map ({time_label}).png"
            plt.savefig(os.path.join(target_dir, fname), dpi=150)
            plt.close(fig)

def save_3d_heatmaps(domain3d, results3d, output_dir):
    z_mid = domain3d.Nz // 2
    t_mid = domain3d.Nt // 2
    
    x, y, z, t = domain3d.x, domain3d.y, domain3d.z, domain3d.t
    
    err_pinn = np.abs(results3d['U_pinn_3d'] - results3d['U_exact_3d'])
    err_qa = np.abs(results3d['U_qa_3d'] - results3d['U_exact_3d'])
    
    vmin_u = min(results3d['U_exact_3d'].min(), results3d['U_cfd_3d'].min())
    vmax_u = max(results3d['U_exact_3d'].max(), results3d['U_cfd_3d'].max())
    vmax_err = max(err_pinn.max(), err_qa.max())
    if vmax_err == 0: vmax_err = 1e-6
    
    solutions = [
        ("Actual", results3d['U_exact_3d'], TempYOR, "Actual", vmin_u, vmax_u),
        ("CFD",    results3d['U_cfd_3d'],   TempYOR, "CFD", vmin_u, vmax_u),
        ("PINN Error",   err_pinn,  ErrorGYR, "PINN", 0, vmax_err),
        ("QA-PINN Error", err_qa, ErrorGYR, "QA-PINN", 0, vmax_err),
    ]
    
    for label, U4, cmap, method_name, vmin, vmax in solutions:
        target_dir = os.path.join(output_dir, method_name, "3D")
        os.makedirs(target_dir, exist_ok=True)
        
        # 1. Slice Heat Map (z-mid, t-mid)
        fig, ax = plt.subplots(figsize=(6, 5))
        U_slice = U4[t_mid, :, :, z_mid]
        if len(y) == 1:
            y_plot = np.array([y[0] - 0.05, y[0] + 0.05])
            U_plot = np.repeat(U_slice, 2, axis=1).T
        else:
            y_plot = y
            U_plot = U_slice.T
        im = ax.pcolormesh(x, y_plot, U_plot, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, label="Temperature (u)" if "Error" not in label else "Absolute Error")
        ax.set_title(f"{label} (z-mid slice) | t = {t[t_mid]:.3f} s", fontsize=11, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "Slice Heat Map.png"), dpi=150)
        plt.close(fig)

        # 2. Panel Heat Map (multiple z-slices at t-mid)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        z_indices = [0, z_mid, -1]
        z_labels = ["Bottom", "Middle", "Top"]
        for ax, z_idx, z_lbl in zip(axes, z_indices, z_labels):
            U_panel = U4[t_mid, :, :, z_idx]
            U_plot_panel = np.repeat(U_panel, 2, axis=1).T if len(y) == 1 else U_panel.T
            im = ax.pcolormesh(x, y_plot, U_plot_panel, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{z_lbl} (z={z[z_idx]:.2f})", fontsize=10)
        fig.suptitle(f"{label} Panel View | t = {t[t_mid]:.3f} s", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "Panel Heat Map.png"), dpi=150)
        plt.close(fig)

        # 3. Surface Heat Map (x vs t surface at y-mid, z-mid)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, T_grid = np.meshgrid(x, t)
        U_surf = U4[:, :, len(y)//2, z_mid]
        surf = ax.plot_surface(X, T_grid, U_surf, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12, label="Temperature (u)" if "Error" not in label else "Absolute Error")
        ax.set_title(f"{label} — 3D Surface Heat Map", fontsize=13, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        ax.set_zlabel("u (temp)")
        ax.view_init(elev=28, azim=-55)
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "Surface Heat Map.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

def plot_training_loss_history(losses, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, label="Total Loss", linewidth=2, color="steelblue")
    transition_epoch = int(len(losses) * 0.8)
    if transition_epoch < len(losses):
        ax.axvline(x=transition_epoch, color='gray', linestyle=':', label="Adam -> L-BFGS")
    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training loss history — {title}")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_fourier_spectrum(U_exact, U_pred, x, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    final_exact = U_exact[-1, :]
    final_pred = U_pred[-1, :]
    freq = np.fft.fftfreq(len(x), d=(x[1]-x[0]))
    fft_exact = np.abs(np.fft.fft(final_exact))
    fft_pred = np.abs(np.fft.fft(final_pred))
    pos = freq > 0
    ax.plot(freq[pos], fft_exact[pos], 'k-', linewidth=2, label="Reference")
    ax.plot(freq[pos], fft_pred[pos], 'r--', linewidth=2, label="Model (PINN)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("|Fourier amplitude|")
    ax.set_title("Fourier spectrum recovery (final time slice)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_solution_snapshots(U_exact, U_pred, x, t, filename):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("Solution snapshots over time", fontweight='bold')
    
    indices = [max(0, int(len(t)*0.25)-1), max(0, int(len(t)*0.50)-1), max(0, int(len(t)*0.75)-1), len(t)-1]
    times = [0.25, 0.50, 0.75, 1.0]
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        ax.plot(x, U_exact[idx, :], 'k-', linewidth=2, label="Reference")
        ax.plot(x, U_pred[idx, :], 'r--', linewidth=2, label="Model (PINN)")
        ax.set_title(f"t = {times[i]:.2f} T_max")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Temperature (K)")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_fourier_spectrum_2d(U_exact, U_pred, x, y, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    final_exact = U_exact[-1, :, :]
    final_pred = U_pred[-1, :, :]
    
    mid_y = len(y) // 2
    slice_exact = final_exact[mid_y, :]
    slice_pred = final_pred[mid_y, :]
    
    freq = np.fft.fftfreq(len(x), d=(x[1]-x[0]))
    fft_exact = np.abs(np.fft.fft(slice_exact))
    fft_pred = np.abs(np.fft.fft(slice_pred))
    
    pos = freq > 0
    ax.plot(freq[pos], fft_exact[pos], 'k-', linewidth=2, label="Reference (Exact)")
    ax.plot(freq[pos], fft_pred[pos], 'r--', linewidth=2, label="Model (PINN)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Spatial frequency (x-direction)")
    ax.set_ylabel("|Fourier amplitude|")
    ax.set_title("2D Fourier Spectrum (y-mid slice, final time)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_fourier_spectrum_3d(U_exact, U_pred, x, y, z, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    final_exact = U_exact[-1, :, :, :]
    final_pred = U_pred[-1, :, :, :]
    
    mid_y = len(y) // 2
    mid_z = len(z) // 2
    slice_exact = final_exact[:, mid_y, mid_z]
    slice_pred = final_pred[:, mid_y, mid_z]
    
    freq = np.fft.fftfreq(len(x), d=(x[1]-x[0]))
    fft_exact = np.abs(np.fft.fft(slice_exact))
    fft_pred = np.abs(np.fft.fft(slice_pred))
    
    pos = freq > 0
    ax.plot(freq[pos], fft_exact[pos], 'k-', linewidth=2, label="Reference (Exact)")
    ax.plot(freq[pos], fft_pred[pos], 'r--', linewidth=2, label="Model (PINN)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Spatial frequency (x-direction)")
    ax.set_ylabel("|Fourier amplitude|")
    ax.set_title("3D Fourier Spectrum (y-mid, z-mid slice, final time)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_solution_snapshots_2d(U_exact, U_pred, x, y, t, filename):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("2D Solution Snapshots over time (y-mid slice)", fontweight='bold')
    
    indices = [max(0, int(len(t)*0.25)-1), max(0, int(len(t)*0.50)-1), max(0, int(len(t)*0.75)-1), len(t)-1]
    times = [0.25, 0.50, 0.75, 1.0]
    mid_y = len(y) // 2
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        ax.plot(x, U_exact[idx, mid_y, :], 'k-', linewidth=2, label="Reference")
        ax.plot(x, U_pred[idx, mid_y, :], 'r--', linewidth=2, label="Model (PINN)")
        ax.set_title(f"t = {times[i]:.2f} T_max")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Temperature (K)")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_solution_snapshots_3d(U_exact, U_pred, x, y, z, t, filename):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("3D Solution Snapshots over time (y-mid, z-mid slice)", fontweight='bold')
    
    indices = [max(0, int(len(t)*0.25)-1), max(0, int(len(t)*0.50)-1), max(0, int(len(t)*0.75)-1), len(t)-1]
    times = [0.25, 0.50, 0.75, 1.0]
    mid_y = len(y) // 2
    mid_z = len(z) // 2
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        ax.plot(x, U_exact[idx, :, mid_y, mid_z], 'k-', linewidth=2, label="Reference")
        ax.plot(x, U_pred[idx, :, mid_y, mid_z], 'r--', linewidth=2, label="Model (PINN)")
        ax.set_title(f"t = {times[i]:.2f} T_max")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Temperature (K)")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
