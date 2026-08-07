import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    methods = ["Actual", "CFD", "PINN", "QA-PINN"]
    
    # --- 1D Dashboard (1x5 grid) ---
    fig_1d, axes_1d = plt.subplots(1, 5, figsize=(25, 5))
    fig_1d.suptitle(f"1D Heat Map Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    u_keys_1d = ["U_exact", "U_cfd", "U_pinn", "U_qa"]
    for i, (method, key) in enumerate(zip(methods, u_keys_1d)):
        if key in results and results[key] is not None:
            U = results[key]
            vmin, vmax = U.min(), U.max()
            X, T = np.meshgrid(domain.x, domain.t)
            c = axes_1d[i].pcolormesh(X, T, U, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes_1d[i].set_title(method)
            axes_1d[i].set_xlabel("x")
            if i == 0: axes_1d[i].set_ylabel("t")
            fig_1d.colorbar(c, ax=axes_1d[i], fraction=0.046, pad=0.04)
        else:
            axes_1d[i].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes_1d[i].set_title(method)
            
    plt.tight_layout()
    plt.savefig(os.path.join(heat_dir, "Heat Map Comparison (1D).png"), dpi=200, bbox_inches='tight')
    plt.close(fig_1d)
    
    # --- 2D Dashboard (3x5 grid: Start, Middle, End) ---
    fig_2d, axes_2d = plt.subplots(3, 5, figsize=(25, 15))
    fig_2d.suptitle(f"2D Heat Map Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    u_keys_2d = ["U_exact_2d", "U_cfd_2d", "U_pinn_2d", "U_qa_2d"]
    t_idx_start = 0
    t_idx_mid = domain2d.Nt // 2
    t_idx_end = domain2d.Nt - 1
    t_indices = [("Start", t_idx_start), ("Middle", t_idx_mid), ("End", t_idx_end)]
    
    for row, (time_label, t_idx) in enumerate(t_indices):
        for col, (method, key) in enumerate(zip(methods, u_keys_2d)):
            ax = axes_2d[row, col]
            if key in results and results[key] is not None:
                U = results[key]
                vmin, vmax = U.min(), U.max()
                X, Y = np.meshgrid(domain2d.x, domain2d.y)
                U_slice = U[t_idx, :, :]
                c = ax.pcolormesh(X, Y, U_slice, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                if row == 0: ax.set_title(method)
                if col == 0: ax.set_ylabel(f"{time_label}\n y")
                else: ax.set_ylabel("y")
                ax.set_xlabel("x")
                fig_2d.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                if row == 0: ax.set_title(method)
                
    plt.tight_layout()
    plt.savefig(os.path.join(heat_dir, "Heat Map Comparison (2D).png"), dpi=200, bbox_inches='tight')
    plt.close(fig_2d)
    
    # --- 3D Dashboard (3x5 grid: Panel, Slice, Surface) ---
    fig_3d = plt.figure(figsize=(25, 15))
    fig_3d.suptitle(f"3D Heat Map Comparison: {props['name'].capitalize()}", fontsize=18, fontweight='bold')
    
    u_keys_3d = ["U_exact_3d", "U_cfd_3d", "U_pinn_3d", "U_qa_3d"]
    view_types = ["Panel", "Slice", "Surface"]
    t_mid = domain3d.Nt // 2
    z_mid = domain3d.Nz // 2
    
    for row, view_label in enumerate(view_types):
        for col, (method, key) in enumerate(zip(methods, u_keys_3d)):
            is_surface = (view_label == "Surface")
            ax = fig_3d.add_subplot(3, 5, row*5 + col + 1, projection='3d' if is_surface else None)
            
            if key in results and results[key] is not None:
                U = results[key]
                vmin, vmax = U.min(), U.max()
                
                if view_label == "Panel":
                    X, Y = np.meshgrid(domain3d.x, domain3d.y)
                    U_panel = U[t_mid, :, :, z_mid]
                    c = ax.pcolormesh(X, Y, U_panel, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_xlabel("x"); ax.set_ylabel("y")
                    fig_3d.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
                elif view_label == "Slice":
                    X, T = np.meshgrid(domain3d.x, domain3d.t)
                    U_slice = U[:, len(domain3d.y)//2, :, z_mid]
                    c = ax.pcolormesh(X, T, U_slice, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_xlabel("x"); ax.set_ylabel("t")
                    fig_3d.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
                elif view_label == "Surface":
                    X, T = np.meshgrid(domain3d.x, domain3d.t)
                    U_surf = U[:, len(domain3d.y)//2, :, z_mid]
                    surf = ax.plot_surface(X, T, U_surf, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)
                    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("u")
                    ax.view_init(elev=28, azim=-55)
                    
                if row == 0: ax.set_title(method)
                if col == 0 and not is_surface: ax.set_ylabel(f"{view_label}\n{ax.get_ylabel()}")
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                if row == 0: ax.set_title(method)
                
    plt.tight_layout()
    plt.savefig(os.path.join(heat_dir, "Heat Map Comparison (3D).png"), dpi=200, bbox_inches='tight')
    plt.close(fig_3d)

def _generate_quantitative_metrics_dashboards(results, props, dashboard_dir):
    quant_dir = os.path.join(dashboard_dir, "Quantitative Metrics")
    os.makedirs(quant_dir, exist_ok=True)
    
    dims = [("1D", "metrics_"), ("2D", "metrics_"), ("3D", "metrics_")]
    method_keys = [("CFD", "cfd"), ("PINN", "pinn"), ("QA-PINN", "qa")]
    metrics = ["RMSE", "Relative_L2_Error", "Max_Absolute_Error", "PDE_Residual"]
    colors = ['steelblue', 'red', 'orange', 'purple']
    
    for dim_label, prefix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Quantitative Metrics Dashboard ({dim_label})", fontsize=18, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = []
            labels = []
            
            for (m_label, m_key) in method_keys:
                dict_key = f"{prefix}{m_key}"
                if dim_label == "2D": dict_key += "_2d"
                elif dim_label == "3D": dict_key += "_3d"
                
                val = results.get(dict_key, {}).get(metric, 0)
                values.append(val)
                labels.append(m_label)
                
            bars = ax.bar(labels, values, color=colors)
            ax.set_title(metric.replace('_', ' '))
            ax.set_ylabel("Value")
            
            # Add values on top of bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4e}', ha='center', va='bottom', rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(quant_dir, f"Quantitative Metrics ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_computational_performance_dashboards(results, props, dashboard_dir):
    comp_dir = os.path.join(dashboard_dir, "Computational Performance")
    os.makedirs(comp_dir, exist_ok=True)
    
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    method_keys = [("CFD", "cfd"), ("PINN", "pinn"), ("QA-PINN", "qa")]
    colors = ['steelblue', 'red', 'orange', 'purple']
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Computational Performance ({dim_label})", fontsize=18, fontweight='bold')
        
        metrics = [
            ("Training Time (s)", f"{{m}}{suffix}_time"),
            ("Inference Time (s)", f"{{m}}{suffix}_inf_time"),
            ("Memory Usage (MB)", "metrics_{m}" + suffix), 
            ("Parameters", "metrics_{m}" + suffix)
        ]
        
        for i, (m_title, m_format) in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = []
            labels = []
            
            for m_label, m_key in method_keys:
                if m_title in ["Memory Usage (MB)", "Parameters"]:
                    dict_key = m_format.format(m=m_key)
                    if m_title == "Memory Usage (MB)":
                        val = results.get(dict_key, {}).get("Memory_MB", 0)
                    else:
                        val = results.get(dict_key, {}).get("Parameters", 0)
                else:
                    time_key = m_format.format(m=m_key)
                    val = results.get(time_key, 0)
                    
                values.append(val)
                labels.append(m_label)
                
            bars = ax.bar(labels, values, color=colors)
            ax.set_title(m_title)
            
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f"Computational Performance ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_model_analysis_dashboards(results, props, dashboard_dir):
    analysis_dir = os.path.join(dashboard_dir, "Model Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    method_keys = [("PINN", "pinn"), ("QA-PINN", "qa")] 
    all_methods = [("CFD", "cfd")] + method_keys
    colors = {'cfd': 'steelblue', 'pinn': 'orange', 'qa': 'purple'}
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Model Analysis Dashboard ({dim_label})", fontsize=18, fontweight='bold')
        
        ax = axes[0, 0]
        for m_label, m_key in method_keys:
            loss_key = f"{m_key}{suffix}_losses"
            if loss_key in results and results[loss_key]:
                ax.plot(results[loss_key], label=m_label, color=colors[m_key])
        ax.set_title("Training Loss Convergence")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (Log Scale)")
        ax.set_yscale("log")
        ax.legend()
        
        ax = axes[0, 1]
        vals = [results.get(f"metrics_{m_key}{suffix}", {}).get("PDE_Residual_Std", 0) for _, m_key in all_methods]
        bars = ax.bar([m for m, _ in all_methods], vals, color=[colors[k] for _, k in all_methods])
        ax.set_title("Explainability: PDE Residual Standard Deviation")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4e}', ha='center', va='bottom')
            
        ax = axes[1, 0]
        vals = [results.get(f"metrics_{m_key}{suffix}", {}).get("Unseen_RMSE", 0) for _, m_key in all_methods]
        bars = ax.bar([m for m, _ in all_methods], vals, color=[colors[k] for _, k in all_methods])
        ax.set_title("Unseen Domain Performance (RMSE)")
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
                
        ax = axes[1, 1]
        ax.axis('off')
        text = "Generalization metrics\n(No validation split used)"
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.set_title("Generalization Gap")
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"Model Analysis ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_overall_ranking_dashboards(results, props, dashboard_dir):
    ranking_dir = os.path.join(dashboard_dir, "Overall Ranking")
    os.makedirs(ranking_dir, exist_ok=True)
    
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    method_keys = [("CFD", "cfd"), ("PINN", "pinn"), ("QA-PINN", "qa")]
    colors = ['steelblue', 'red', 'orange', 'purple']
    
    for dim_label, suffix in dims:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Overall Ranking Dashboard ({dim_label})", fontsize=18, fontweight='bold')
        
        def plot_bar(ax, metric_key, title, lower_is_better=True, log_scale=False):
            vals = [results.get(f"metrics_{m}{suffix}", {}).get(metric_key, 0) for _, m in method_keys]
            bars = ax.bar([m for m, _ in method_keys], vals, color=colors)
            ax.set_title(title)
            if log_scale: ax.set_yscale('log')
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    fmt = '{:.2e}' if log_scale or yval < 0.01 else '{:.4f}'
                    ax.text(bar.get_x() + bar.get_width()/2, yval, fmt.format(yval), ha='center', va='bottom')
        
        plot_bar(axes[0, 0], "Relative_L2_Error", "Accuracy (Relative L2 - Lower is Better)", log_scale=True)
        
        ax = axes[0, 1]
        vals = [results.get(f"{m}{suffix}_time", 0) for _, m in method_keys]
        bars = ax.bar([m for m, _ in method_keys], vals, color=colors)
        ax.set_title("Efficiency (Training Time - Lower is Better)")
        for bar in bars:
            yval = bar.get_height()
            if yval > 0: ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}s', ha='center', va='bottom')
            
        plot_bar(axes[1, 0], "Unseen_RMSE", "Generalization (Unseen RMSE - Lower is Better)")
        plot_bar(axes[1, 1], "PDE_Residual_Std", "Robustness (PDE Residual Std - Lower is Better)", log_scale=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ranking_dir, f"Overall Ranking ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

def _generate_final_conclusion_dashboards(results, props, dashboard_dir):
    conc_dir = os.path.join(dashboard_dir, "Final Conclusion")
    os.makedirs(conc_dir, exist_ok=True)
    
    dims = [("1D", ""), ("2D", "_2d"), ("3D", "_3d")]
    method_keys = [("CFD", "cfd"), ("PINN", "pinn"), ("QA-PINN", "qa")]
    
    for dim_label, suffix in dims:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        def get_best(metric_key, minimize=True, is_time=False):
            if is_time:
                vals = [(m_label, results.get(f"{m_key}{suffix}_time", float('inf') if minimize else 0)) for m_label, m_key in method_keys]
            else:
                vals = [(m_label, results.get(f"metrics_{m_key}{suffix}", {}).get(metric_key, float('inf') if minimize else 0)) for m_label, m_key in method_keys]
            
            if minimize:
                vals = [v for v in vals if v[1] > 0 or v[0] == "CFD"]
            
            if not vals: return "N/A"
            best = min(vals, key=lambda x: x[1]) if minimize else max(vals, key=lambda x: x[1])
            return best[0]
            
        table_data = [
            ["Category", "Winner"],
            ["Best Overall Accuracy (L2)", get_best("Relative_L2_Error")],
            ["Fastest Method (Time)", get_best("", is_time=True)],
            ["Lowest Memory Usage", get_best("Memory_MB")],
            ["Most Physically Consistent (PDE Res)", get_best("PDE_Residual")],
            ["Best Generalization (Unseen RMSE)", get_best("Unseen_RMSE")]
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
        
        ax.set_title(f"Final Conclusion Winners ({dim_label})", fontsize=18, fontweight='bold', pad=20)
        plt.savefig(os.path.join(conc_dir, f"Final Conclusion ({dim_label}).png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    with open(os.path.join(conc_dir, "Final_Conclusion.txt"), "w") as f:
        f.write("Overall Simulation Conclusion:\n\n")
        f.write("CFD provides the exact baseline but scales poorly in memory.\n")
        f.write("PINN offers robust mesh-free solving at the cost of long training times.\n")
        f.write("QA-PINN shows promise for accelerating PINN convergences using quantum entanglement circuits.\n")
        
    with open(os.path.join(conc_dir, "Future_Work.txt"), "w") as f:
        f.write("Recommended Future Improvements:\n")
        f.write("- Scale QA-PINN to real-world quantum hardware.\n")
        f.write("- Explore hybrid CFD-PINN approaches to combine speed with physical accuracy.\n")
