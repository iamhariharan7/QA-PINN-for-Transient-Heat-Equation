import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def save_individual_heatmaps(domain, results, output_dir):
    x = domain.x
    t = domain.t
    
    U_exact = results['U_exact']
    U_cfd = results['U_cfd']
    U_pinn = results['U_pinn']
    U_qa = results['U_qa']
    
    vmin = min(U_exact.min(), U_cfd.min(), U_pinn.min(), U_qa.min())
    vmax = max(U_exact.max(), U_cfd.max(), U_pinn.max(), U_qa.max())
    
    solutions = [
        ("Actual", U_exact, "Greens", "Actual", "Heat Map (1D).png"),
        ("CFD", U_cfd, "Blues", "CFD", "Heat Map (1D).png"),
        ("PINN", U_pinn, "Oranges", "PINN", "Heat Map (1D).png"),
        ("QA-PINN", U_qa, "Purples", "QA-PINN", "Heat Map (1D).png"),
    ]
    
    for name, U, cmap, folder, filename in solutions:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(x, t, U, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{name} Temperature Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        plt.tight_layout()
        
        target_dir = os.path.join(output_dir, folder, "1D")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, filename), dpi=150)
        plt.close(fig)



def build_bar_charts(fig, gs, start_row, results, dim_str="1d"):
    methods = ['CFD', 'PINN', 'QA-PINN']
    colors = ['steelblue', 'orange', 'purple', 'red']
    
    def _get_metric(m, k):
        return results.get(f'metrics_{m}_{dim_str.lower()}', {}).get(k, 0)
        
    def _get_time(m, type="time"):
        if dim_str.lower() == "1d":
            return results.get(f'{m}_{type}', 0)
        else:
            return results.get(f'{m}_{dim_str.lower()}_{type}', 0)
        
    metrics = [
        ("Training Time (s)", [_get_time('cfd', 'time'), _get_time('pinn', 'time'), _get_time('qa', 'time')], True, '.1f'),
        ("Inference Time (s)", [_get_time('cfd', 'inf_time'), _get_time('pinn', 'inf_time'), _get_time('qa', 'inf_time')], True, '.4f'),
        ("Trainable Parameters", [_get_metric('cfd','Parameters'), _get_metric('pinn','Parameters'), _get_metric('qa','Parameters')], True, 'd'),
        ("Memory (MB)", [_get_metric('cfd','Memory_MB'), _get_metric('pinn','Memory_MB'), _get_metric('qa','Memory_MB')], True, '.2f'),
        ("Relative L2 Error", [_get_metric('cfd','Relative_L2_Error'), _get_metric('pinn','Relative_L2_Error'), _get_metric('qa','Relative_L2_Error')], True, '.4f'),
        ("Standard RMSE", [_get_metric('cfd','RMSE'), _get_metric('pinn','RMSE'), _get_metric('qa','RMSE')], True, '.4f'),
        ("Unseen Domain RMSE", [_get_metric('cfd','Unseen_RMSE'), _get_metric('pinn','Unseen_RMSE'), _get_metric('qa','Unseen_RMSE')], True, '.4f'),
        ("PDE Residual Error", [_get_metric('cfd','PDE_Residual'), _get_metric('pinn','PDE_Residual'), _get_metric('qa','PDE_Residual')], True, '.2e'),
        ("Explainability (PDE Std)", [_get_metric('cfd','PDE_Residual_Std'), _get_metric('pinn','PDE_Residual_Std'), _get_metric('qa','PDE_Residual_Std')], True, '.2e')
    ]
    
    for i, (title, vals, log_opt, fmt) in enumerate(metrics):
        r = start_row + (i // 4)
        c = i % 4
        ax = fig.add_subplot(gs[r, c+1]) 
        bars = ax.bar(methods, vals, color=colors, alpha=0.8)
        ax.set_title(title + "\n(Lower is better)", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        if 'e' in fmt and any(v > 0 for v in vals):
            ax.set_yscale('log')
        for bar, val in zip(bars, vals):
            if val == 0 and fmt == 'd':
                text = "0"
                y = 0
            else:
                text = format(val, fmt)
                y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y, text, ha='center', va='bottom', fontsize=9)

def generate_comparison_figure(domain, results, props, output_dir):
    fig = plt.figure(figsize=(25, 5))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    fig.suptitle(f"1D Heat Equation Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    x, t = domain.x, domain.t
    
    solutions = [
        ("Actual", results['U_exact'], "Greens"),
        ("CFD", results['U_cfd'], "Blues"),
        ("PINN", results['U_pinn'], "Oranges"),
        ("QA-PINN", results['U_qa'], "Purples"),
    ]
    
    vmin = min([results[k].min() for k in ['U_exact', 'U_cfd', 'U_pinn', 'U_qa']])
    vmax = max([results[k].max() for k in ['U_exact', 'U_cfd', 'U_pinn', 'U_qa']])
    
    for col, (name, U, cmap) in enumerate(solutions):
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
    
    solutions = [
        ("Actual", results['U_exact_2d'], "Greens"),
        ("CFD", results['U_cfd_2d'], "Blues"),
        ("PINN", results['U_pinn_2d'], "Oranges"),
        ("QA-PINN", results['U_qa_2d'], "Purples"),
    ]
    
    vmin = min([U.min() for _, U, _ in solutions])
    vmax = max([U.max() for _, U, _ in solutions])
    
    target_dir = os.path.join(output_dir, "Comparison", "2D")
    os.makedirs(target_dir, exist_ok=True)

    for time_label, t_idx in t_indices.items():
        fig = plt.figure(figsize=(25, 5))
        gs = gridspec.GridSpec(1, 5, figure=fig)
        fig.suptitle(f"2D Heat Equation Comparison ({time_label}): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
        
        for col, (name, U, cmap) in enumerate(solutions):
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
        
    solutions = [
        ("Actual", results['U_exact_3d'], "Greens"),
        ("CFD", results['U_cfd_3d'], "Blues"),
        ("PINN", results['U_pinn_3d'], "Oranges"),
        ("QA-PINN", results['U_qa_3d'], "Purples"),
    ]
    
    vmin = min([U.min() for _, U, _ in solutions])
    vmax = max([U.max() for _, U, _ in solutions])
    
    target_dir = os.path.join(output_dir, "Comparison", "3D")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Slice Comparison
    fig = plt.figure(figsize=(25, 5))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    fig.suptitle(f"3D Slice Comparison (z-mid, t-mid): {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    for col, (name, U, cmap) in enumerate(solutions):
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
        for col, (name, U, cmap) in enumerate(solutions):
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
    for col, (name, U, cmap) in enumerate(solutions):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        surf = ax.plot_surface(X, T_grid, U[:, :, len(y)//2, z_mid], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.view_init(elev=28, azim=-55)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(target_dir, "Surface Heat Map Comparison.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_individual_3d_surfaces(domain, results, output_dir):
    x = domain.x
    t = domain.t
    X, T_grid = np.meshgrid(x, t)

    solutions = [
        ("Actual (Exact)", results['U_exact'],  "Greens",  "Actual"),
        ("CFD",            results['U_cfd'],     "Blues",   "CFD"),
        ("PINN",           results['U_pinn'],    "Oranges", "PINN"),
        ("QA-PINN",        results['U_qa'],      "Purples", "QA-PINN"),
    ]

    for name, U, cmap_name, method_name in solutions:
        fig = plt.figure(figsize=(9, 6))
        ax  = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            X, T_grid, U,
            cmap=cmap_name,
            edgecolor='none',
            alpha=0.9,
            rcount=50, ccount=50,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12, label="Temperature (u)")

        ax.set_title(f"{name} — 3-D Temperature Surface", fontsize=13, fontweight='bold')
        ax.set_xlabel("x (m)", labelpad=8)
        ax.set_ylabel("t (s)", labelpad=8)
        ax.set_zlabel("u (temp)", labelpad=8)
        ax.view_init(elev=28, azim=-55)

        plt.tight_layout()
        
        target_dir = os.path.join(output_dir, method_name, "1D")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, "3D Surface View.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

def save_2d_heatmaps(domain2d, results2d, output_dir):
    x  = domain2d.x
    y  = domain2d.y
    Nt = domain2d.Nt
    t_indices = {"Start": 0, "Middle": Nt // 2, "End": -1}

    solutions = [
        ("Actual", results2d['U_exact_2d'], "Greens", "Actual"),
        ("CFD",    results2d['U_cfd_2d'],   "Blues",  "CFD"),
        ("PINN",   results2d['U_pinn_2d'],  "Oranges", "PINN"),
        ("QA-PINN", results2d['U_qa_2d'], "Purples", "QA-PINN"),
    ]

    for label, U3, cmap, method_name in solutions:
        target_dir = os.path.join(output_dir, method_name, "2D")
        os.makedirs(target_dir, exist_ok=True)
        
        for time_label, t_idx in t_indices.items():
            U_slice = U3[t_idx]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(x, y, U_slice, cmap=cmap, shading='auto')
            plt.colorbar(im, ax=ax, label="Temperature (u)")
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
    
    solutions = [
        ("Actual", results3d['U_exact_3d'], "Greens"),
        ("CFD",    results3d['U_cfd_3d'],   "Blues"),
        ("PINN",   results3d['U_pinn_3d'],  "Oranges"),
        ("QA-PINN", results3d['U_qa_3d'], "Purples"),
    ]
    
    for label, U4, cmap in solutions:
        target_dir = os.path.join(output_dir, label, "3D")
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
        im = ax.pcolormesh(x, y_plot, U_plot, cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label="Temperature (u)")
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
            im = ax.pcolormesh(x, y_plot, U_plot_panel, cmap=cmap, shading='auto')
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
        surf = ax.plot_surface(X, T_grid, U_surf, cmap=cmap, edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12, label="Temperature (u)")
        ax.set_title(f"{label} — 3D Surface Heat Map", fontsize=13, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        ax.set_zlabel("u (temp)")
        ax.view_init(elev=28, azim=-55)
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "Surface Heat Map.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


