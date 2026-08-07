import argparse
import yaml
import os
import datetime
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import copy
import shutil
import sys

from src.material_loader import discover_available_materials, load_material_properties, load_simulation_datasets
from src.rectify_datasets import rectify_all_datasets
from prepare_datasets import prepare_datasets
from src.heat_equation import HeatEquationDomain
from src.heat_equation_2d import HeatEquationDomain2D
from src.heat_equation_3d import HeatEquationDomain3D
from src.actual_solution_2d import solve_exact_2d
from src.actual_solution_3d import solve_exact_3d
from src.cfd_solver_2d import solve_cfd_2d
from src.cfd_solver_3d import solve_cfd_3d
from src.pinn_model_2d import solve_pinn_2d
from src.pinn_model_3d import solve_pinn_3d
from src.qa_pinn_model_2d import solve_qa_pinn_2d
from src.qa_pinn_model_3d import solve_qa_pinn_3d

from src.actual_solution import solve_exact
from src.cfd_solver import solve_cfd
from src.pinn_model import solve_pinn
from src.qa_pinn_model import solve_qa_pinn
from src.metrics import (
    calculate_metrics, count_parameters, estimate_memory_mb,
    compute_pde_residual_map, compute_pde_residual_scalar,
    compute_pde_residual_std,
    compute_pde_residual_map_2d, compute_pde_residual_scalar_2d, compute_pde_residual_std_2d,
    compute_pde_residual_map_3d, compute_pde_residual_scalar_3d, compute_pde_residual_std_3d
)
from src.plotting import (
    generate_comparison_figure, 
    save_individual_heatmaps, 
    generate_2d_comparison_figure,
    generate_3d_comparison_figure,
    save_2d_heatmaps,
    save_3d_heatmaps,
    plot_training_loss_history,
    plot_fourier_spectrum,
    plot_solution_snapshots,
    plot_fourier_spectrum_2d,
    plot_fourier_spectrum_3d,
    plot_solution_snapshots_2d,
    plot_solution_snapshots_3d
)
from src.dashboards import generate_detailed_dashboards, patch_dashboards
from src.report_generator import generate_report

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(description="Heat Equation Solvers Benchmark Pipeline")
    parser.add_argument("--material", type=str, default=None, help="Material name, alias, or folder name")
    parser.add_argument("--excel", type=str, default="data/materials/material_database.csv", help="Path to master materials excel/csv database")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset", help="Path to simulation datasets folder")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config yaml")
    parser.add_argument("--skip_training", action="store_true", help="Load pre-trained models instead of retraining")
    parser.add_argument("--dim", type=str, default=None, choices=["1D", "2D", "3D", "ALL"], help="Simulation dimension to run")
    args = parser.parse_args()
    
    print("================================================================================")
    print("      QUANTUM & PHYSICS-INFORMED NEURAL NETWORK MASTER BENCHMARK RUNNER      ")
    print("       Solving 1D, 2D, and 3D Thermal PDE Transport Across Solvers")
    print("================================================================================\n")
    
    prepare_datasets(args.excel, args.dataset_dir)
    rectify_all_datasets(args.dataset_dir, args.excel)
    
            
    discovered = discover_available_materials(args.dataset_dir, args.excel)
    
    selected_mat_info = None
    if args.material:
        target = args.material.lower().strip()
        for mat in discovered:
            folder = mat['folder_name'].lower()
            display = mat['display_name'].lower()
            aliases = mat['properties'].get('aliases', '').lower()
            if target == folder or target == display or target in aliases or folder in target or display in target:
                selected_mat_info = mat
                break

    if selected_mat_info is None:
        if sys.stdin.isatty():
            print("Select a Material to Simulate:")
            for idx, mat in enumerate(discovered):
                alias_info = f" (Folder: {mat['folder_name']})"
                if mat['properties'].get('aliases'):
                    alias_info += f" [Aliases: {mat['properties']['aliases']}]"
                print(f"  [{idx + 1}] {mat['display_name']}{alias_info}")
                
            try:
                sel = input(f"Enter the number corresponding to your material [1-{len(discovered)}] [Default: 1]: ").strip()
                if sel.isdigit() and (1 <= int(sel) <= len(discovered)):
                    selected_mat_info = discovered[int(sel) - 1]
                else:
                    selected_mat_info = discovered[0]
            except EOFError:
                selected_mat_info = discovered[0]
        else:
            selected_mat_info = discovered[0]


    sim_dim = args.dim
    if not sim_dim:
        if sys.stdin.isatty():
            print("\nSelect Simulation Dimension:")
            print("  [1] 1D Only")
            print("  [2] 2D Only")
            print("  [3] 3D Only")
            print("  [4] ALL (1D, 2D, 3D)")
            try:
                dim_sel = input("Enter the number corresponding to your choice [1-4] [Default: 4]: ").strip()
                dim_map = {'1': '1D', '2': '2D', '3': '3D', '4': 'ALL'}
                sim_dim = dim_map.get(dim_sel, 'ALL')
            except EOFError:
                sim_dim = 'ALL'
        else:
            sim_dim = 'ALL'
    else:
        sim_dim = sim_dim.upper()
        
    dims_to_run = ["1D", "2D", "3D"] if sim_dim == "ALL" else [sim_dim]
    print(f"\n[1/4] Loading Data Architecture for '{selected_mat_info['display_name']}'...")
    props = selected_mat_info['properties']
    datasets = load_simulation_datasets(selected_mat_info['folder_path'])

    print(f"  [+] Physical Properties loaded from: {args.excel}")
    print(f"      - Material Name             : {props['name']}")
    print(f"      - Thermal Conductivity (k) : {props['k']} W/(m K)")
    print(f"      - Thermal Diffusivity (alpha): {props['alpha']:.6e} m^2/s")
    print(f"      - Density (rho)            : {props['rho']} kg/m^3")
    print(f"      - Specific Heat (Cp)       : {props['cp']} J/(kg K)")
    print(f"      - Aliases                  : {props.get('aliases', 'None')}")
    print(f"      - Application              : {props.get('application', 'None')}")
    print(f"  [+] Simulation Datasets loaded from: {selected_mat_info['folder_path']}\n")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        

    base_outputs_dir = "outputs"
    current_out_dir = os.path.join(base_outputs_dir, "current_output")
    history_dir = os.path.join(base_outputs_dir, "history")
    
    os.makedirs(current_out_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    for item in os.listdir(current_out_dir):
        item_path = os.path.join(current_out_dir, item)
        if os.path.isdir(item_path):
            archive_dest = os.path.join(history_dir, item)
            if not os.path.exists(archive_dest):
                shutil.move(item_path, archive_dest)
            else:
                shutil.move(item_path, archive_dest + "_dup")
            print(f"\nArchived previous output to: {archive_dest}\n")
            
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{props['name'].lower().replace(' ', '_').replace('/', '_')}_{sim_dim}_{timestamp}"
    output_dir = os.path.join(current_out_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for method in ["Actual", "CFD", "PINN", "QA-PINN"]:
        for dim in dims_to_run:
            os.makedirs(os.path.join(output_dir, method, dim), exist_ok=True)
            
    print(f"Outputs will be saved to: {output_dir}\n")
    
    domain = HeatEquationDomain(dataset_dict=datasets['1D'], props=props)
    domain2d = HeatEquationDomain2D(dataset_dict=datasets['2D'], props=props)
    domain3d = HeatEquationDomain3D(dataset_dict=datasets['3D'], props=props)
    
    results = {}
    
    if "1D" in dims_to_run:
        print("\n" + "="*80 + "\n                         STARTING 1D SIMULATION PIPELINE\n" + "="*80 + "\n")
        print("1.1 Computing 1D Exact Solution...\n")
        U_exact, exact_inf = solve_exact(domain, output_dir)
        results['U_exact'] = U_exact
        results['exact_inf_time'] = exact_inf
        print(f"  -> Time taken: N/A (Inference: {exact_inf:.2f}s)\n")
    
        print("1.2 Solving 1D CFD (Crank-Nicolson)...\n")
        U_cfd, cfd_time, cfd_inf = solve_cfd(domain, output_dir)
        results['U_cfd'] = U_cfd
        results['cfd_time'] = cfd_time
        results['cfd_inf_time'] = cfd_inf
        print(f"  -> Time taken: {cfd_time:.2f}s (Inference: {cfd_inf:.2f}s)\n")
        results['metrics_cfd_1d'] = calculate_metrics(U_cfd, U_exact)
        results['res_map_cfd_1d'] = compute_pde_residual_map(U_cfd, domain)
        results['metrics_cfd_1d']['PDE_Residual'] = compute_pde_residual_scalar(U_cfd, domain)
        results['metrics_cfd_1d']['PDE_Residual_Std'] = compute_pde_residual_std(U_cfd, domain)
        results['metrics_cfd_1d']['Parameters'] = 0
        results['metrics_cfd_1d']['Memory_MB'] = 0.0
    
        print("1.3 Training 1D Classical PINN...")
        U_pinn, pinn_time, pinn_inf, pinn_losses, pinn_model = solve_pinn(domain, config['models']['pinn'], output_dir)
        torch.cuda.empty_cache()
        results['U_pinn'] = U_pinn
        results['pinn_time'] = pinn_time
        results['pinn_inf_time'] = pinn_inf
        print(f"  -> Time taken: {pinn_time:.2f}s (Inference: {pinn_inf:.2f}s)\n")
        results['pinn_losses'] = pinn_losses
        results['metrics_pinn_1d'] = calculate_metrics(U_pinn, U_exact)
        results['res_map_pinn_1d'] = compute_pde_residual_map(U_pinn, domain)
        results['metrics_pinn_1d']['PDE_Residual'] = compute_pde_residual_scalar(U_pinn, domain)
        results['metrics_pinn_1d']['PDE_Residual_Std'] = compute_pde_residual_std(U_pinn, domain)
        results['metrics_pinn_1d']['Parameters'] = count_parameters(pinn_model)
        results['metrics_pinn_1d']['Memory_MB'] = estimate_memory_mb(pinn_model)
    
        print("1.4 Training 1D Quantum-Assisted PINN (QA-PINN)...")
        U_qa, qa_time, qa_inf, qa_losses, qa_model = solve_qa_pinn(domain, config['models']['qa_pinn'], output_dir)
        torch.cuda.empty_cache()
        results['U_qa'] = U_qa
        results['qa_time'] = qa_time
        results['qa_inf_time'] = qa_inf
        print(f"  -> Time taken: {qa_time:.2f}s (Inference: {qa_inf:.2f}s)\n")
        results['qa_losses'] = qa_losses
        results['metrics_qa_1d'] = calculate_metrics(U_qa, U_exact)
        results['res_map_qa_1d'] = compute_pde_residual_map(U_qa, domain)
        results['metrics_qa_1d']['PDE_Residual'] = compute_pde_residual_scalar(U_qa, domain)
        results['metrics_qa_1d']['PDE_Residual_Std'] = compute_pde_residual_std(U_qa, domain)
        results['metrics_qa_1d']['Parameters'] = count_parameters(qa_model)
        results['metrics_qa_1d']['Memory_MB'] = estimate_memory_mb(qa_model)
    

    if "2D" in dims_to_run:
        print("\n" + "="*80 + "\n                         STARTING 2D SIMULATION PIPELINE\n" + "="*80 + "\n")
        print("2.1 Computing 2D Exact Solution...\n")
        U_exact_2d, exact_2d_inf = solve_exact_2d(domain2d, output_dir)
        results['U_exact_2d'] = U_exact_2d
        results['exact_2d_inf_time'] = exact_2d_inf
        print(f"  -> Time taken: N/A (Inference: {exact_2d_inf:.2f}s)\n")
    
        print("2.2 Solving 2D CFD (Alternating Direction Implicit - ADI)...\n")
        U_cfd_2d, cfd_2d_time, cfd_2d_inf = solve_cfd_2d(domain2d, output_dir)
        results['U_cfd_2d'] = U_cfd_2d
        results['cfd_2d_time'] = cfd_2d_time
        results['cfd_2d_inf_time'] = cfd_2d_inf
        print(f"  -> Time taken: {cfd_2d_time:.2f}s (Inference: {cfd_2d_inf:.2f}s)\n")
        results['metrics_cfd_2d'] = calculate_metrics(U_cfd_2d, U_exact_2d)
        results['res_map_cfd_2d'] = compute_pde_residual_map_2d(U_cfd_2d, domain2d)
        results['metrics_cfd_2d']['PDE_Residual'] = compute_pde_residual_scalar_2d(U_cfd_2d, domain2d)
        results['metrics_cfd_2d']['PDE_Residual_Std'] = compute_pde_residual_std_2d(U_cfd_2d, domain2d)
        results['metrics_cfd_2d']['Parameters'] = 0
        results['metrics_cfd_2d']['Memory_MB'] = 0.0
    
        print("2.3 Training 2D Classical PINN...")
        U_pinn_2d, pinn_2d_time, pinn_2d_inf, pinn_2d_losses, pinn_2d_model = solve_pinn_2d(domain2d, config['models_2d']['pinn_2d'], output_dir)
        torch.cuda.empty_cache()
        results['U_pinn_2d'] = U_pinn_2d
        results['pinn_2d_time'] = pinn_2d_time
        results['pinn_2d_inf_time'] = pinn_2d_inf
        print(f"  -> Time taken: {pinn_2d_time:.2f}s (Inference: {pinn_2d_inf:.2f}s)\n")
        results['pinn_2d_losses'] = pinn_2d_losses
        results['metrics_pinn_2d'] = calculate_metrics(U_pinn_2d, U_exact_2d)
        results['res_map_pinn_2d'] = compute_pde_residual_map_2d(U_pinn_2d, domain2d)
        results['metrics_pinn_2d']['PDE_Residual'] = compute_pde_residual_scalar_2d(U_pinn_2d, domain2d)
        results['metrics_pinn_2d']['PDE_Residual_Std'] = compute_pde_residual_std_2d(U_pinn_2d, domain2d)
        results['metrics_pinn_2d']['Parameters'] = count_parameters(pinn_2d_model)
        results['metrics_pinn_2d']['Memory_MB'] = estimate_memory_mb(pinn_2d_model)
    
        print("2.4 Training 2D Quantum-Assisted PINN (QA-PINN 2D)...")
        U_qa_2d, qa_2d_time, qa_2d_inf, qa_2d_losses, qa_2d_model = solve_qa_pinn_2d(domain2d, config['models_2d']['qa_pinn_2d'], output_dir)
        torch.cuda.empty_cache()
        results['U_qa_2d'] = U_qa_2d
        results['qa_2d_time'] = qa_2d_time
        results['qa_2d_inf_time'] = qa_2d_inf
        print(f"  -> Time taken: {qa_2d_time:.2f}s (Inference: {qa_2d_inf:.2f}s)\n")
        results['qa_2d_losses'] = qa_2d_losses
        results['metrics_qa_2d'] = calculate_metrics(U_qa_2d, U_exact_2d)
        results['res_map_qa_2d'] = compute_pde_residual_map_2d(U_qa_2d, domain2d)
        results['metrics_qa_2d']['PDE_Residual'] = compute_pde_residual_scalar_2d(U_qa_2d, domain2d)
        results['metrics_qa_2d']['PDE_Residual_Std'] = compute_pde_residual_std_2d(U_qa_2d, domain2d)
        results['metrics_qa_2d']['Parameters'] = count_parameters(qa_2d_model)
        results['metrics_qa_2d']['Memory_MB'] = estimate_memory_mb(qa_2d_model)
    

    if "3D" in dims_to_run:
        print("\n" + "="*80 + "\n                         STARTING 3D SIMULATION PIPELINE\n" + "="*80 + "\n")
        print("3.1 Computing 3D Exact Solution...\n")
        U_exact_3d, exact_3d_inf = solve_exact_3d(domain3d, output_dir)
        results['U_exact_3d'] = U_exact_3d
        results['exact_3d_inf_time'] = exact_3d_inf
        print(f"  -> Time taken: N/A (Inference: {exact_3d_inf:.2f}s)\n")
    
        print("3.2 Solving 3D CFD (Explicit Finite Difference)...\n")
        U_cfd_3d, cfd_3d_time, cfd_3d_inf = solve_cfd_3d(domain3d, output_dir)
        results['U_cfd_3d'] = U_cfd_3d
        results['cfd_3d_time'] = cfd_3d_time
        results['cfd_3d_inf_time'] = cfd_3d_inf
        print(f"  -> Time taken: {cfd_3d_time:.2f}s (Inference: {cfd_3d_inf:.2f}s)\n")
        results['metrics_cfd_3d'] = calculate_metrics(U_cfd_3d, U_exact_3d)
        results['res_map_cfd_3d'] = compute_pde_residual_map_3d(U_cfd_3d, domain3d)
        results['metrics_cfd_3d']['PDE_Residual'] = compute_pde_residual_scalar_3d(U_cfd_3d, domain3d)
        results['metrics_cfd_3d']['PDE_Residual_Std'] = compute_pde_residual_std_3d(U_cfd_3d, domain3d)
        results['metrics_cfd_3d']['Parameters'] = 0
        results['metrics_cfd_3d']['Memory_MB'] = 0.0
    
        print("3.3 Training 3D Classical PINN...")
        U_pinn_3d, pinn_3d_time, pinn_3d_inf, pinn_3d_losses, pinn_3d_model = solve_pinn_3d(domain3d, config['models_3d']['pinn_3d'], output_dir)
        torch.cuda.empty_cache()
        results['U_pinn_3d'] = U_pinn_3d
        results['pinn_3d_time'] = pinn_3d_time
        results['pinn_3d_inf_time'] = pinn_3d_inf
        print(f"  -> Time taken: {pinn_3d_time:.2f}s (Inference: {pinn_3d_inf:.2f}s)\n")
        results['pinn_3d_losses'] = pinn_3d_losses
        results['metrics_pinn_3d'] = calculate_metrics(U_pinn_3d, U_exact_3d)
        results['res_map_pinn_3d'] = compute_pde_residual_map_3d(U_pinn_3d, domain3d)
        results['metrics_pinn_3d']['PDE_Residual'] = compute_pde_residual_scalar_3d(U_pinn_3d, domain3d)
        results['metrics_pinn_3d']['PDE_Residual_Std'] = compute_pde_residual_std_3d(U_pinn_3d, domain3d)
        results['metrics_pinn_3d']['Parameters'] = count_parameters(pinn_3d_model)
        results['metrics_pinn_3d']['Memory_MB'] = estimate_memory_mb(pinn_3d_model)
    
        print("3.4 Training 3D Quantum-Assisted PINN (QA-PINN 3D)...")
        U_qa_3d, qa_3d_time, qa_3d_inf, qa_3d_losses, qa_3d_model = solve_qa_pinn_3d(domain3d, config['models_3d']['qa_pinn_3d'], output_dir)
        torch.cuda.empty_cache()
        results['U_qa_3d'] = U_qa_3d
        results['qa_3d_time'] = qa_3d_time
        results['qa_3d_inf_time'] = qa_3d_inf
        print(f"  -> Time taken: {qa_3d_time:.2f}s (Inference: {qa_3d_inf:.2f}s)\n")
        results['qa_3d_losses'] = qa_3d_losses
        results['metrics_qa_3d'] = calculate_metrics(U_qa_3d, U_exact_3d)
        results['res_map_qa_3d'] = compute_pde_residual_map_3d(U_qa_3d, domain3d)
        results['metrics_qa_3d']['PDE_Residual'] = compute_pde_residual_scalar_3d(U_qa_3d, domain3d)
        results['metrics_qa_3d']['PDE_Residual_Std'] = compute_pde_residual_std_3d(U_qa_3d, domain3d)
        results['metrics_qa_3d']['Parameters'] = count_parameters(qa_3d_model)
        results['metrics_qa_3d']['Memory_MB'] = estimate_memory_mb(qa_3d_model)
    

    if "1D" in dims_to_run:
        print("1.6 Evaluating Solvers on Unseen Domain (Complex Initial Condition)...")
        unseen_domain = copy.deepcopy(domain)
        unseen_domain.U_exact = None
        unseen_domain.initial_condition = lambda x: np.sin(np.pi * x / domain.L) + 0.5 * np.sin(3 * np.pi * x / domain.L)
    
        U_unseen_cfd, _, _ = solve_cfd(unseen_domain, None)
    
        X, T, _, _ = unseen_domain.get_grid()
        xt_test = torch.FloatTensor(np.column_stack([X.flatten(), T.flatten()])).to(device)
    
        with torch.no_grad():
            U_unseen_pinn = pinn_model(xt_test[:, 0:1], xt_test[:, 1:2]).cpu().numpy().reshape(domain.Nt, domain.Nx)
            U_unseen_qa = qa_model(xt_test[:, 0:1], xt_test[:, 1:2]).cpu().numpy().reshape(domain.Nt, domain.Nx)
        
        results['metrics_cfd_1d']['Unseen_RMSE'] = 0.0
        results['metrics_pinn_1d']['Unseen_RMSE'] = calculate_metrics(U_unseen_pinn, U_unseen_cfd)['RMSE']
        results['metrics_qa_1d']['Unseen_RMSE'] = calculate_metrics(U_unseen_qa, U_unseen_cfd)['RMSE']
    if "2D" in dims_to_run:
        print("2.5 Evaluating 2D Solvers on Unseen Domain (Complex Initial Condition)...")
        unseen_domain2d = copy.deepcopy(domain2d)
        unseen_domain2d.U_exact = None
        unseen_domain2d.initial_condition = lambda X, Y: np.sin(np.pi * X / domain2d.Lx) * np.sin(np.pi * Y / domain2d.Ly) + 0.5 * np.sin(3 * np.pi * X / domain2d.Lx) * np.sin(3 * np.pi * Y / domain2d.Ly)
        
        U_unseen_cfd_2d, _, _ = solve_cfd_2d(unseen_domain2d, None)
        
        X3, Y3, T3 = unseen_domain2d.get_xyt_grid()
        xyt_test = torch.FloatTensor(np.column_stack([X3.flatten(), Y3.flatten(), T3.flatten()])).to(device)
        with torch.no_grad():
            U_unseen_pinn_2d = pinn_2d_model(xyt_test[:, 0:1], xyt_test[:, 1:2], xyt_test[:, 2:3]).cpu().numpy().reshape(domain2d.Nt, domain2d.Ny, domain2d.Nx)
            U_unseen_qa_2d = qa_2d_model(xyt_test[:, 0:1], xyt_test[:, 1:2], xyt_test[:, 2:3]).cpu().numpy().reshape(domain2d.Nt, domain2d.Ny, domain2d.Nx)
            
        results['metrics_cfd_2d']['Unseen_RMSE'] = 0.0
        results['metrics_pinn_2d']['Unseen_RMSE'] = calculate_metrics(U_unseen_pinn_2d, U_unseen_cfd_2d)['RMSE']
        results['metrics_qa_2d']['Unseen_RMSE'] = calculate_metrics(U_unseen_qa_2d, U_unseen_cfd_2d)['RMSE']

    if "3D" in dims_to_run:
        print("3.5 Evaluating 3D Solvers on Unseen Domain (Complex Initial Condition)...")
        unseen_domain3d = copy.deepcopy(domain3d)
        unseen_domain3d.U_exact = None
        unseen_domain3d.initial_condition = lambda X, Y, Z: np.sin(np.pi * X / domain3d.Lx) * np.sin(np.pi * Y / domain3d.Ly) * np.sin(np.pi * Z / domain3d.Lz) + 0.5 * np.sin(3 * np.pi * X / domain3d.Lx) * np.sin(3 * np.pi * Y / domain3d.Ly) * np.sin(3 * np.pi * Z / domain3d.Lz)
        
        U_unseen_cfd_3d, _, _ = solve_cfd_3d(unseen_domain3d, None)
        
        X4, Y4, Z4, T4 = unseen_domain3d.get_xyzt_grid()
        xyzt_test = torch.FloatTensor(np.column_stack([X4.flatten(), Y4.flatten(), Z4.flatten(), T4.flatten()])).to(device)
        with torch.no_grad():
            U_unseen_pinn_3d = pinn_3d_model(xyzt_test[:, 0:1], xyzt_test[:, 1:2], xyzt_test[:, 2:3], xyzt_test[:, 3:4]).cpu().numpy().reshape(domain3d.Nt, domain3d.Nx, domain3d.Ny, domain3d.Nz)
            U_unseen_qa_3d = qa_3d_model(xyzt_test[:, 0:1], xyzt_test[:, 1:2], xyzt_test[:, 2:3], xyzt_test[:, 3:4]).cpu().numpy().reshape(domain3d.Nt, domain3d.Nx, domain3d.Ny, domain3d.Nz)
            
        results['metrics_cfd_3d']['Unseen_RMSE'] = 0.0
        results['metrics_pinn_3d']['Unseen_RMSE'] = calculate_metrics(U_unseen_pinn_3d, U_unseen_cfd_3d)['RMSE']
        results['metrics_qa_3d']['Unseen_RMSE'] = calculate_metrics(U_unseen_qa_3d, U_unseen_cfd_3d)['RMSE']
    print("\nGenerating Surface Plots & 3-Panel Comparisons (Viridis & Hot Error Maps)...")
    if "1D" in dims_to_run:
        generate_comparison_figure(domain, results, props, output_dir)
        save_individual_heatmaps(domain, results, output_dir)
        
        # New Advanced Visualizations
        plot_training_loss_history(results['pinn_losses'], "Model A (Classical PINN)", os.path.join(output_dir, "PINN", "1D", "training_loss.png"))
        plot_training_loss_history(results['qa_losses'], "Model B (QA-PINN)", os.path.join(output_dir, "QA-PINN", "1D", "training_loss.png"))
        
        plot_fourier_spectrum(results['U_exact'], results['U_pinn'], domain.x, os.path.join(output_dir, "PINN", "1D", "fourier_spectrum.png"))
        plot_fourier_spectrum(results['U_exact'], results['U_qa'], domain.x, os.path.join(output_dir, "QA-PINN", "1D", "fourier_spectrum.png"))
        
        plot_solution_snapshots(results['U_exact'], results['U_pinn'], domain.x, domain.t, os.path.join(output_dir, "PINN", "1D", "solution_snapshots.png"))
        plot_solution_snapshots(results['U_exact'], results['U_qa'], domain.x, domain.t, os.path.join(output_dir, "QA-PINN", "1D", "solution_snapshots.png"))
    if "2D" in dims_to_run:
        generate_2d_comparison_figure(domain2d, results, props, output_dir)
        save_2d_heatmaps(domain2d, results, output_dir)
        
        plot_training_loss_history(results['pinn_2d_losses'], "Model A (Classical PINN 2D)", os.path.join(output_dir, "PINN", "2D", "training_loss.png"))
        plot_training_loss_history(results['qa_2d_losses'], "Model B (QA-PINN 2D)", os.path.join(output_dir, "QA-PINN", "2D", "training_loss.png"))
        
        plot_fourier_spectrum_2d(results['U_exact_2d'], results['U_pinn_2d'], domain2d.x, domain2d.y, os.path.join(output_dir, "PINN", "2D", "fourier_spectrum.png"))
        plot_fourier_spectrum_2d(results['U_exact_2d'], results['U_qa_2d'], domain2d.x, domain2d.y, os.path.join(output_dir, "QA-PINN", "2D", "fourier_spectrum.png"))
        
        plot_solution_snapshots_2d(results['U_exact_2d'], results['U_pinn_2d'], domain2d.x, domain2d.y, domain2d.t, os.path.join(output_dir, "PINN", "2D", "solution_snapshots.png"))
        plot_solution_snapshots_2d(results['U_exact_2d'], results['U_qa_2d'], domain2d.x, domain2d.y, domain2d.t, os.path.join(output_dir, "QA-PINN", "2D", "solution_snapshots.png"))
        
    if "3D" in dims_to_run:
        generate_3d_comparison_figure(domain3d, results, props, output_dir)
        save_3d_heatmaps(domain3d, results, output_dir)
        
        plot_training_loss_history(results['pinn_3d_losses'], "Model A (Classical PINN 3D)", os.path.join(output_dir, "PINN", "3D", "training_loss.png"))
        plot_training_loss_history(results['qa_3d_losses'], "Model B (QA-PINN 3D)", os.path.join(output_dir, "QA-PINN", "3D", "training_loss.png"))
        
        plot_fourier_spectrum_3d(results['U_exact_3d'], results['U_pinn_3d'], domain3d.x, domain3d.y, domain3d.z, os.path.join(output_dir, "PINN", "3D", "fourier_spectrum.png"))
        plot_fourier_spectrum_3d(results['U_exact_3d'], results['U_qa_3d'], domain3d.x, domain3d.y, domain3d.z, os.path.join(output_dir, "QA-PINN", "3D", "fourier_spectrum.png"))
        
        plot_solution_snapshots_3d(results['U_exact_3d'], results['U_pinn_3d'], domain3d.x, domain3d.y, domain3d.z, domain3d.t, os.path.join(output_dir, "PINN", "3D", "solution_snapshots.png"))
        plot_solution_snapshots_3d(results['U_exact_3d'], results['U_qa_3d'], domain3d.x, domain3d.y, domain3d.z, domain3d.t, os.path.join(output_dir, "QA-PINN", "3D", "solution_snapshots.png"))
    
    print("\n" + "="*80 + "\n                    GENERATING DETAILED DASHBOARDS & REPORTS\n" + "="*80 + "\n\nGenerating Dashboards...\nGenerating Analytical Report...")
    
    generate_detailed_dashboards(results, props, domain, domain2d, domain3d, output_dir)
    patch_dashboards(results, props, domain, domain2d, domain3d, output_dir)
    generate_report(results, props, config, output_dir)
    
    print("\n" + "="*80 + "\n                               EXECUTION COMPLETE\n" + "="*80 + f"\n\n[+] ALL 1D, 2D, AND 3D SIMULATION OUTPUTS SUCCESSFULLY GENERATED AND SAVED TO:\n--> {output_dir}\n\nGenerated Artifact Summary:\n  * Classical & Quantum Architectures : PINN, QA-PINN (1D/2D/3D)\n  * Quantum Circuits                  : 1D, 2D, 3D PennyLane Variational Circuits\n  * Surface & Heatmap Diagrams        : Matplotlib viridis 3D surfaces & 2D slices\n  * Reports                           : Markdown & PDF Reports\n" + "="*80)
    print("\n" + "="*80)
    print("                [SUCCESS] RUN EXPERIMENT COMPLETED                      ")
    print("="*80 + "\n")
if __name__ == "__main__":
    main()
