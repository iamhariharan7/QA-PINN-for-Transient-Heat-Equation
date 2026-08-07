import os
import json
import pandas as pd

class ProjectIndexer:
    """
    Maintains project-wide catalogs mapping Data, Models, Experiments,
    and Outputs for easy searchability without hard-coded paths.
    """
    def __init__(self, root="."):
        self.root = root
        
    def generate_indexes(self):
        """Scans the project and creates the global JSON search index and master catalogs."""
        project_index = {
            "Datasets": [],
            "Models": [],
            "Experiments": [],
            "Outputs": [],
            "Materials": []
        }
        
        # 1. Models
        models_dir = os.path.join(self.root, "models")
        if os.path.exists(models_dir):
            for arch in ["CNN", "PINN", "QA-PINN"]:
                final_dir = os.path.join(models_dir, arch, "final_model")
                if os.path.exists(final_dir):
                    for m in os.listdir(final_dir):
                        meta_path = os.path.join(final_dir, m, "metadata.json")
                        if os.path.exists(meta_path):
                            with open(meta_path, "r") as f:
                                project_index["Models"].append(json.load(f))
                                
        # 2. Experiments
        exp_dir = os.path.join(self.root, "experiments")
        if os.path.exists(exp_dir):
            for e in os.listdir(exp_dir):
                if e.startswith("experiment_"):
                    meta_path = os.path.join(exp_dir, e, "experiment_metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            project_index["Experiments"].append(json.load(f))
                            
        # 3. Datasets (Pulling from data catalog if it exists)
        data_cat = os.path.join(self.root, "data", "dataset_catalog.csv")
        if os.path.exists(data_cat):
            df = pd.read_csv(data_cat)
            project_index["Datasets"] = df.to_dict('records')
            
        # 4. Materials
        mat_db = os.path.join(self.root, "data_generator", "materials", "material_db.json")
        if os.path.exists(mat_db):
            with open(mat_db, "r") as f:
                data = json.load(f)
                project_index["Materials"] = [m["name"] for m in data if "name" in m]

        # Write project search index
        with open(os.path.join(self.root, "project_index.json"), "w") as f:
            json.dump(project_index, f, indent=4)
            
        # Write flat CSV catalogs
        self._write_csv_catalog(project_index["Models"], "model_catalog.csv")
        self._write_csv_catalog(project_index["Experiments"], "experiment_catalog.csv")
        print("Project Indexing Complete.")

    def _write_csv_catalog(self, dict_list, filename):
        if dict_list:
            df = pd.DataFrame(dict_list)
            df.to_csv(os.path.join(self.root, filename), index=False)
        else:
            pd.DataFrame().to_csv(os.path.join(self.root, filename), index=False)

if __name__ == "__main__":
    indexer = ProjectIndexer()
    indexer.generate_indexes()
