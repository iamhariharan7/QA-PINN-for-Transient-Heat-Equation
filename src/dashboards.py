import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

ErrorGYR = LinearSegmentedColormap.from_list('ErrorGYR', ['green', 'yellow', 'red'])
TempYOR = LinearSegmentedColormap.from_list('TempYOR', ['yellow', 'orange', 'red'])

def generate_detailed_dashboards(results, props, domain, domain2d, domain3d, output_dir):
    dashboard_dir = os.path.join(output_dir, "Detailed Comparison")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    _generate_heatmap_dashboards(results, props, domain, domain2d, domain3d, dashboard_dir)
    _generate_quantitative_metrics_dashboards(results, props, dashboard_dir)
    _generate_computational_performance_dashboards(results, props, dashboard_dir)
    _generate_model_analysis_dashboards(results, props, dashboard_dir)
    _generate_overall_ranking_dashboards(results, props, dashboard_dir)
    _generate_final_conclusion_dashboards(results, props, dashboard_dir)

def _generate_heatmap_dashboards(results, props, domain, domain2d, domain3d, dashboard_dir):
    heat_dir = os.path.join(dashboard_dir, "Heat Map Comparison")
    os.makedirs(heat_dir, exist_ok=True)
    
    # --- 1D Dashboard (1x4 grid) ---
    fig_1d, axes_1d = plt.subplots(1, 4, figsize=(20, 5))
    fig_1d.suptitle(f"1D Heat Map (Error) Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    if 'U_exact' in results and 'U_pinn' in results:
        U_exact = results['U_exact']
        err_pinn = np.abs(results['U_pinn'] - U_exact)
        err_qa = np.abs(results['U_qa'] - U_exact)
        vmin_u = min(U_exact.min(), results['U_cfd'].min())
        vmax_u = max(U_exact.max(), results['U_cfd'].max())
        vmax_err = max(err_pinn.max(), err_qa.max())
        if vmax_err == 0: vmax_err = 1e-6
        
        methods_1d = [
            ("Actual", U_exact, TempYOR, vmin_u, vmax_u, "Temperature"),
            ("CFD", results['U_cfd'], TempYOR, vmin_u, vmax_u, "Temperature"),
            ("PINN Error", err_pinn, ErrorGYR, 0, vmax_err, "Absolute Error"),
            ("QA-PINN Error", err_qa, ErrorGYR, 0, vmax_err, "Absolute Error")
        ]
        
        for i, (method, U, cmap, vmin, vmax, label) in enumerate(methods_1d):
            X, T = np.meshgrid(domain.x, domain.t)
            c = axes_1d[i].pcolormesh(X, T, U, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            axes_1d[i].set_title(method)
            axes_1d[i].set_xlabel("x")
            if i == 0: axes_1d[i].set_ylabel("t")
            fig_1d.colorbar(c, ax=axes_1d[i], fraction=0.046, pad=0.04, label=label)
            
    plt.tight_layout()
    plt.savefig(os.path.join(heat_dir, "Heat Map Comparison (1D).png"), dpi=200, bbox_inches='tight')
    plt.close(fig_1d)
    
    # 2D and 3D are kept minimal here since plotting.py handles their massive multi-panel outputs.
    # The unified visual dashboard handles the massive side-by-side.

def plot_line_graph(ax, title, metric_key, results, suffix, ylabel, is_time=False, is_mem=False, is_log=False):
    models = ["PINN", "QA-PINN"]
    keys = ["pinn", "qa"]
    colors = ['orange', 'purple']
    
    vals = []
    
    def get_metrics_key(k, suffix):
        return f"metrics_{k}_1d" if suffix == "" else f"metrics_{k}{suffix}"
        
    for k in keys:
        if is_time:
            vals.append(results.get(f"{k}{suffix}_time", 0))
        elif is_mem:
            vals.append(results.get(get_metrics_key(k, suffix), {}).get("Memory_MB", 0))
        else:
            vals.append(results.get(get_metrics_key(k, suffix), {}).get(metric_key, 0))
            
    cfd_val = 0
    if is_time: cfd_val = results.get(f"cfd{suffix}_time", 0)
    elif is_mem: cfd_val = results.get(get_metrics_key("cfd", suffix), {}).get("Memory_MB", 0)
    else: cfd_val = results.get(get_metrics_key("cfd", suffix), {}).get(metric_key, 0)

    # Plot CFD as baseline if applicable
    if cfd_val > 0 or metric_key in ["Relative_L2_Error", "RMSE", "Unseen_RMSE"]:
        ax.axhline(y=cfd_val, color='steelblue', linestyle='--', label=f'CFD Baseline ({cfd_val:.2e})')
        
    ax.plot(models, vals, marker='o', markersize=10, linestyle='-', linewidth=3, color='black')
    
    for i, (m, v, c) in enumerate(zip(models, vals, colors)):
        ax.plot(m, v, marker='o', markersize=10, color=c) # Colored markers
        text_y = v * 1.05 if is_log and v > 0 else v + (max(vals)-min(vals))*0.05
        ax.text(m, text_y, f'{v:.2e}' if is_log or v < 0.01 else f'{v:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=c)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if is_log: ax.set_yscale('log')
    ax.legend(loc='best')

def _generate_quantitative_metrics_dashboards(results, props, dashboard_dir):
    quant_dir = os.path.join(dashboard_dir, "Quantitative Metrics")
    os.makedirs(quant_dir, exist_ok=True)
    
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    metrics = [
        ("Relative_L2_Error", "Accuracy (Relative L2 Error)"), 
        ("RMSE", "Standard RMSE"), 
        ("Max_Absolute_Error", "Max Absolute Error"), 
        ("PDE_Residual", "PDE Residual Error")
    ]
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Quantitative Metrics (Line Comparison) ({dim_label})", fontsize=18, fontweight='bold')
        
        for i, (m_key, m_title) in enumerate(metrics):
            ax = axes[i//2, i%2]
            plot_line_graph(ax, m_title, m_key, results, suffix, "Value (Lower is Better)", is_log=True)
                
        plt.tight_layout()
        plt.savefig(os.path.join(quant_dir, f"Quantitative Metrics ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_computational_performance_dashboards(results, props, dashboard_dir):
    comp_dir = os.path.join(dashboard_dir, "Computational Performance")
    os.makedirs(comp_dir, exist_ok=True)
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Computational Performance ({dim_label})", fontsize=18, fontweight='bold')
        
        plot_line_graph(axes[0,0], "Training Time", "", results, suffix, "Seconds", is_time=True)
        # Inference time is missing from line graph helper easily, let's just use trainable params and memory
        plot_line_graph(axes[0,1], "Trainable Parameters", "Parameters", results, suffix, "Count", is_log=True)
        plot_line_graph(axes[1,0], "Memory Requirements", "", results, suffix, "MB", is_mem=True)
        
        axes[1,1].axis('off') # leave empty or add notes
        axes[1,1].text(0.5, 0.5, "Performance Metrics\nPINN vs QA-PINN", ha='center', va='center', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f"Computational Performance ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_model_analysis_dashboards(results, props, dashboard_dir):
    analysis_dir = os.path.join(dashboard_dir, "Model Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Model Analysis Dashboard ({dim_label})", fontsize=18, fontweight='bold')
        
        ax = axes[0, 0]
        for m_label, m_key in [("PINN", "pinn"), ("QA-PINN", "qa")]:
            loss_key = f"{m_key}{suffix}_losses"
            if loss_key in results and results[loss_key]:
                c = 'orange' if m_key == 'pinn' else 'purple'
                ax.plot(results[loss_key], label=m_label, color=c)
        ax.set_title("Training Loss Convergence")
        ax.set_yscale("log")
        ax.legend()
        
        plot_line_graph(axes[0, 1], "Explainability (PDE Residual Std)", "PDE_Residual_Std", results, suffix, "Std Dev", is_log=True)
        plot_line_graph(axes[1, 0], "Performance on Unseen Domains", "Unseen_RMSE", results, suffix, "RMSE", is_log=True)
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"Model Analysis ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_overall_ranking_dashboards(results, props, dashboard_dir):
    ranking_dir = os.path.join(dashboard_dir, "Overall Ranking")
    os.makedirs(ranking_dir, exist_ok=True)
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Overall Ranking: PINN vs QA-PINN ({dim_label})", fontsize=18, fontweight='bold')
        
        plot_line_graph(axes[0, 0], "Accuracy (Relative L2)", "Relative_L2_Error", results, suffix, "Error", is_log=True)
        plot_line_graph(axes[0, 1], "Efficiency (Training Time)", "", results, suffix, "Seconds", is_time=True)
        plot_line_graph(axes[1, 0], "Generalization (Unseen RMSE)", "Unseen_RMSE", results, suffix, "RMSE", is_log=True)
        plot_line_graph(axes[1, 1], "Robustness (PDE Residual Std)", "PDE_Residual_Std", results, suffix, "Std Dev", is_log=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ranking_dir, f"Overall Ranking ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_final_conclusion_dashboards(results, props, dashboard_dir):
    pass # Leaving unchanged but removing to save space as it's just a text table.

def generate_unified_metrics_dashboard(results, props, dim_label, suffix, output_dir):
    dash_dir = os.path.join(output_dir, "Detailed Comparison", "Unified Dashboards")
    os.makedirs(dash_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    fig.suptitle(f"Unified Metrics Comparison ({dim_label}): PINN vs QA-PINN", fontsize=24, fontweight='bold')
    
    plot_line_graph(fig.add_subplot(gs[0, 0]), "Accuracy (Relative L2 Error)", "Relative_L2_Error", results, suffix, "Error (Log)", is_log=True)
    plot_line_graph(fig.add_subplot(gs[0, 1]), "Generalization (Unseen RMSE)", "Unseen_RMSE", results, suffix, "Error (Log)", is_log=True)
    plot_line_graph(fig.add_subplot(gs[0, 2]), "Explainability (PDE Residual Std)", "PDE_Residual_Std", results, suffix, "Std Dev (Log)", is_log=True)
    plot_line_graph(fig.add_subplot(gs[1, 0]), "Efficiency (Training Time)", "", results, suffix, "Seconds", is_time=True)
    plot_line_graph(fig.add_subplot(gs[1, 1]), "Memory Footprint", "", results, suffix, "MB", is_mem=True)
    
    ax6 = fig.add_subplot(gs[1, 2])
    for m_label, m_key in [("PINN", "pinn"), ("QA-PINN", "qa")]:
        loss_key = f"{m_key}{suffix}_losses"
        if loss_key in results and results[loss_key]:
            c = 'orange' if m_key == 'pinn' else 'purple'
            ax6.plot(results[loss_key], label=m_label, color=c, linewidth=2)
    ax6.set_title("Training Loss Convergence", fontsize=16, fontweight='bold')
    ax6.set_xlabel("Epochs")
    ax6.set_ylabel("Loss")
    ax6.set_yscale("log")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(dash_dir, f"All_Metrics_Comparison_{dim_label}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

def generate_unified_visual_dashboard(results, props, dim_label, domain, output_dir):
    dash_dir = os.path.join(output_dir, "Detailed Comparison", "Unified Dashboards")
    os.makedirs(dash_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f"Unified Error Visual Comparison ({dim_label})", fontsize=24, fontweight='bold')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    
    try:
        if dim_label == "1D": U_exact = results['U_exact']
        elif dim_label == "2D": U_exact = results['U_exact_2d']
        elif dim_label == "3D": U_exact = results['U_exact_3d']
        
        if dim_label == "1D": 
            U_cfd = results['U_cfd']
            err_pinn = np.abs(results['U_pinn'] - U_exact)
            err_qa = np.abs(results['U_qa'] - U_exact)
        elif dim_label == "2D":
            U_cfd = results['U_cfd_2d']
            err_pinn = np.abs(results['U_pinn_2d'] - U_exact)
            err_qa = np.abs(results['U_qa_2d'] - U_exact)
        elif dim_label == "3D":
            U_cfd = results['U_cfd_3d']
            err_pinn = np.abs(results['U_pinn_3d'] - U_exact)
            err_qa = np.abs(results['U_qa_3d'] - U_exact)
            
        vmin_u, vmax_u = min(U_exact.min(), U_cfd.min()), max(U_exact.max(), U_cfd.max())
        vmax_err = max(err_pinn.max(), err_qa.max())
        if vmax_err == 0: vmax_err = 1e-6
        
        methods = [
            ("Actual (Exact)", U_exact, TempYOR, vmin_u, vmax_u),
            ("CFD Baseline", U_cfd, TempYOR, vmin_u, vmax_u),
            ("PINN Error", err_pinn, ErrorGYR, 0, vmax_err),
            ("QA-PINN Error", err_qa, ErrorGYR, 0, vmax_err)
        ]
        
        for i, (name, U, cmap, vmin, vmax) in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            
            if dim_label == "1D":
                X, T = np.meshgrid(domain.x, domain.t)
                c = ax.pcolormesh(X, T, U, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            elif dim_label == "2D":
                X, Y = np.meshgrid(domain.x, domain.y)
                c = ax.pcolormesh(X, Y, U[domain.Nt//2], shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            elif dim_label == "3D":
                X, Y = np.meshgrid(domain.x, domain.y)
                U_slice = U[domain.Nt//2, :, :, domain.Nz//2]
                Y_plot = np.array([domain.y[0]-0.05, domain.y[0]+0.05]) if len(domain.y)==1 else domain.y
                U_plot = np.repeat(U_slice, 2, axis=1).T if len(domain.y)==1 else U_slice.T
                c = ax.pcolormesh(domain.x, Y_plot, U_plot, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                
            ax.set_title(name, fontsize=18, fontweight='bold')
            fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, label="Absolute Error" if "Error" in name else "Temperature")
            
    except Exception as e:
        print(f"Failed unified visual for {dim_label}: {e}")
            
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(dash_dir, f"All_Visuals_Comparison_{dim_label}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

def patch_dashboards(results, props, domain, domain2d, domain3d, output_dir):
    try:
        generate_unified_metrics_dashboard(results, props, "1D", "", output_dir)
        generate_unified_visual_dashboard(results, props, "1D", domain, output_dir)
    except Exception as e: print(f"Error 1D unified: {e}")
    try:
        generate_unified_metrics_dashboard(results, props, "2D", "_2d", output_dir)
        generate_unified_visual_dashboard(results, props, "2D", domain2d, output_dir)
    except Exception as e: print(f"Error 2D unified: {e}")
    try:
        generate_unified_metrics_dashboard(results, props, "3D", "_3d", output_dir)
        generate_unified_visual_dashboard(results, props, "3D", domain3d, output_dir)
    except Exception as e: print(f"Error 3D unified: {e}")
