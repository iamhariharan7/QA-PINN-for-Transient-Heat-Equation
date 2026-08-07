import os
import shutil
from datetime import datetime
from pathlib import Path

def archive_previous_output(base_dir: str, material_name: str) -> None:
    base_path = Path(base_dir)
    mat_dir = base_path / "outputs" / material_name
    history_dir = base_path / "outputs" / "history"

    if mat_dir.exists():
        history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_mat_dir = history_dir / f"{material_name}_{timestamp}"
        shutil.move(str(mat_dir), str(archived_mat_dir))
        print(f"Archived previous outputs for {material_name} to: {archived_mat_dir}")

def generate_report(results, props, config, output_dir):
    report_path = os.path.join(output_dir, "Final_Report.txt")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("================================================================================\n")
        f.write(f"Final Thermal Benchmark Report - Material: {props.get('name', 'Unknown')}\n")
        f.write("================================================================================\n\n")
        f.write(f"Material Properties:\n")
        f.write(f"  - Thermal Conductivity (k) : {props.get('k')} W/(m K)\n")
        f.write(f"  - Thermal Diffusivity (alpha): {props.get('alpha')} m^2/s\n")
        f.write(f"  - Density (rho)            : {props.get('rho')} kg/m^3\n")
        f.write(f"  - Specific Heat (Cp)       : {props.get('cp')} J/(kg K)\n")
        f.write(f"  - Application              : {props.get('application', '')}\n\n")
        
        methods = ["cfd", "pinn", "qa"]
        method_names = {"cfd": "CFD", "pinn": "PINN", "qa": "QA-PINN"}
        dims = ["1d", "2d", "3d"]
        
        for dim in dims:
            f.write("="*80 + "\n")
            f.write(f" {dim.upper()} SIMULATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for method in methods:
                metric_key = f"metrics_{method}_{dim}"
                if metric_key not in results:
                    continue
                
                f.write(f"Method: {method_names[method]}\n")
                f.write("-" * 30 + "\n")
                
                metrics = results[metric_key]
                for m, val in metrics.items():
                    if isinstance(val, float):
                        f.write(f"  {m:<20}: {val:.6g}\n")
                    else:
                        f.write(f"  {m:<20}: {val}\n")
                f.write("\n")
