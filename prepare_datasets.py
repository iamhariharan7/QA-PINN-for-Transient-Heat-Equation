import os
import shutil
import pandas as pd
import glob

def find_latest_simulation_csv(base_dir, material):
    mat_safe = material.replace(" ", "_").replace("/", "_")
    mat_dir = os.path.join(base_dir, mat_safe)
    
    if not os.path.exists(mat_dir):
        return None
        
    sim_dirs = [d for d in os.listdir(mat_dir) if d.startswith("Simulation_")]
    if not sim_dirs:
        return None
        
    # Pick the latest simulation directory by sorting (Simulation_000001, Simulation_000002, etc.)
    latest_sim = sorted(sim_dirs)[-1]
    csv_files = glob.glob(os.path.join(mat_dir, latest_sim, "*.csv"))
    
    if not csv_files:
        return None
        
    # Find temperature.csv if it exists, otherwise just pick the first csv
    for f in csv_files:
        if "temperature.csv" in os.path.basename(f):
            return f
            
    return csv_files[0]

def format_and_copy(src, dst):
    df = pd.read_csv(src)
    rename_map = {'time': 'time_s', 'x': 'x_m', 'y': 'y_m', 'z': 'z_m', 'temperature': 'temperature_K'}
    df.rename(columns=rename_map, inplace=True)
    df.to_csv(dst, index=False)

def prepare_datasets(materials_csv="data/materials/material_database.csv", dataset_dir="data/dataset"):
    print("================================================================================")
    print("                    BRIDGING AND PREPARING DATASETS                             ")
    print("================================================================================")
    
    if not os.path.exists(materials_csv):
        print(f"Warning: Material database {materials_csv} not found. Skipping preparation.")
        return
        
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(materials_csv)
    
    # Identify material name column
    name_col = None
    for c in df.columns:
        if c.lower() in ["name", "material_name", "material"]:
            name_col = c
            break
            
    if not name_col:
        print("Error: Could not find material name column in database.")
        return
        
    materials = df[name_col].unique()
    
    for mat in materials:
        target_dir = os.path.join(dataset_dir, mat)
        os.makedirs(target_dir, exist_ok=True)
        
        # Check 1D
        file_1d = find_latest_simulation_csv("data/1D", mat)
        target_1d = os.path.join(target_dir, f"{mat}_1D.csv")
        if file_1d:
            format_and_copy(file_1d, target_1d)
            print(f"  [+] 1D Data Formatted & Copied: {mat}")
        elif not os.path.exists(target_1d):
            with open(target_1d, "w") as f:
                f.write("x_m,time_s,temperature_K\n0.0,0.0,0.0\n")
            print(f"  [-] 1D Data Missing: {mat} (Created dummy for rectification)")
            
        # Check 2D
        file_2d = find_latest_simulation_csv("data/2D", mat)
        target_2d = os.path.join(target_dir, f"{mat}_2D.csv")
        if file_2d:
            format_and_copy(file_2d, target_2d)
            print(f"  [+] 2D Data Formatted & Copied: {mat}")
        elif not os.path.exists(target_2d):
            with open(target_2d, "w") as f:
                f.write("x_m,y_m,time_s,temperature_K\n0.0,0.0,0.0,0.0\n")
            print(f"  [-] 2D Data Missing: {mat} (Created dummy for rectification)")
            
        # Check 3D
        file_3d = find_latest_simulation_csv("data/3D", mat)
        target_3d = os.path.join(target_dir, f"{mat}_3D.csv")
        if file_3d:
            format_and_copy(file_3d, target_3d)
            print(f"  [+] 3D Data Formatted & Copied: {mat}")
        elif not os.path.exists(target_3d):
            with open(target_3d, "w") as f:
                f.write("x_m,y_m,z_m,time_s,temperature_K\n0.0,0.0,0.0,0.0,0.0\n")
            print(f"  [-] 3D Data Missing: {mat} (Created dummy for rectification)")
            
    print("Dataset preparation complete.\n")

if __name__ == "__main__":
    prepare_datasets()
