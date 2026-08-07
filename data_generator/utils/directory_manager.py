import os
import shutil
import json
from datetime import datetime
from data_generator.materials.material_manager import MaterialManager

class DirectoryManager:
    def __init__(self, output_base="outputs", data_base="data"):
        self.output_base = output_base
        self.data_base = data_base
        
        self.present_dir = os.path.join(self.output_base, "Present_Output")
        self.past_dir = os.path.join(self.output_base, "Past_Outputs")
        
        self.mat_mgr = MaterialManager()
        self.materials = self.mat_mgr.get_all_material_names()
        
    def setup_data_tree(self):
        """
        Pre-allocates the extensive data directory tree structure ensuring all 
        dimension and material folders exist.
        """
        os.makedirs(self.data_base, exist_ok=True)
        
        # Root directories
        subdirs = ["processed/1D", "processed/2D", "processed/3D",
                   "train/1D", "train/2D", "train/3D",
                   "validation/1D", "validation/2D", "validation/3D",
                   "test/1D", "test/2D", "test/3D"]
                   
        for sd in subdirs:
            os.makedirs(os.path.join(self.data_base, sd.replace("/", os.sep)), exist_ok=True)
            
        # Material directories for raw simulations
        for dim in [1, 2, 3]:
            dim_str = f"{dim}D"
            for mat in self.materials:
                mat_safe = mat.replace(" ", "_").replace("/", "_")
                os.makedirs(os.path.join(self.data_base, dim_str, mat_safe), exist_ok=True)
                
    def get_next_simulation_id(self, dimension: int, material: str) -> str:
        """
        Scans the target directory and returns the next available sequential Simulation_ID.
        """
        mat_safe = material.replace(" ", "_").replace("/", "_")
        dim_str = f"{dimension}D"
        path = os.path.join(self.data_base, dim_str, mat_safe)
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        existing_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("Simulation_")]
        
        if not existing_dirs:
            return "Simulation_000001"
            
        # Parse IDs and find max
        ids = []
        for d in existing_dirs:
            try:
                ids.append(int(d.split("_")[1]))
            except ValueError:
                continue
                
        next_id = max(ids) + 1 if ids else 1
        return f"Simulation_{next_id:06d}"

    def get_dataset_dir(self, dimension: int, material: str, sim_id: str) -> str:
        """
        Creates and returns the dataset path: data/<Dim>D/<Material>/<Sim_ID>
        """
        mat_safe = material.replace(" ", "_").replace("/", "_")
        dim_str = f"{dimension}D"
        path = os.path.join(self.data_base, dim_str, mat_safe, sim_id)
        os.makedirs(path, exist_ok=True)
        return path

    def setup_experiment_dir(self, run_name: str = None) -> str:
        """
        Archives the existing Present_Output and creates a new one with the complex dashboard folder structure.
        """
        if not os.path.exists(self.output_base): os.makedirs(self.output_base)
        if not os.path.exists(self.past_dir): os.makedirs(self.past_dir)
            
        # Archive Present_Output
        if os.path.exists(self.present_dir) and os.listdir(self.present_dir):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = os.path.join(self.past_dir, timestamp)
            shutil.move(self.present_dir, archive_path)
            
        os.makedirs(self.present_dir, exist_ok=True)
        
        if run_name is None:
            run_name = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            
        run_dir = os.path.join(self.present_dir, run_name)
        
        # Complex output structure for dashboards and logs (NO Reusable Datasets here)
        subdirs = [
            "Actual/1D", "Actual/2D", "Actual/3D",
            "CFD/1D", "CFD/2D", "CFD/3D",
            "CNN/1D", "CNN/2D", "CNN/3D",
            "PINN/1D", "PINN/2D", "PINN/3D",
            "QA-PINN/1D", "QA-PINN/2D", "QA-PINN/3D",
            "Comparison/1D", "Comparison/2D", "Comparison/3D",
            "Detailed Comparison/Heat Map Comparison",
            "Detailed Comparison/Quantitative Metrics",
            "Detailed Comparison/Computational Performance",
            "Detailed Comparison/Model Analysis",
            "Detailed Comparison/Overall Ranking",
            "Detailed Comparison/Final Conclusion",
            "Reports"
        ]
        
        for sd in subdirs:
            os.makedirs(os.path.join(run_dir, sd.replace("/", os.sep)), exist_ok=True)
            
        return run_dir
