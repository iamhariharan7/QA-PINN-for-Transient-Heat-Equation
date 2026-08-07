import os
import json
import random
import hashlib
from datetime import datetime
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pyarrow
except ImportError:
    pyarrow = None
    
from data_generator.materials.material_manager import MaterialManager

class DatasetManager:
    """
    Manages the global state of the dataset repository including catalog generation,
    summary statistics, dataset versioning, Parquet exports, and Material READMEs.
    """
    def __init__(self, data_base="data"):
        self.data_base = data_base
        self.version = "1.0.0"
        self.mat_mgr = MaterialManager()
        
    def generate_catalogs(self):
        print("Generating dataset catalog, summaries, and validation reports...")
        
        all_sims = []
        total_size_bytes = 0
        corrupted = 0
        passed = 0
        failed = 0
        missing_meta = 0
        
        materials = set()
        
        # Walk through 1D, 2D, 3D directories
        for dim in [1, 2, 3]:
            dim_dir = os.path.join(self.data_base, f"{dim}D")
            if not os.path.exists(dim_dir): continue
            
            for mat in os.listdir(dim_dir):
                mat_dir = os.path.join(dim_dir, mat)
                if not os.path.isdir(mat_dir): continue
                materials.add(mat)
                
                for sim in os.listdir(mat_dir):
                    if not sim.startswith("Simulation_"): continue
                    
                    sim_dir = os.path.join(mat_dir, sim)
                    meta_path = os.path.join(sim_dir, "metadata.json")
                    stats_path = os.path.join(sim_dir, "statistics.json")
                    geom_path = os.path.join(sim_dir, "geometry.json")
                    val_path = os.path.join(sim_dir, "validation_report.json")
                    
                    if not os.path.exists(meta_path):
                        corrupted += 1
                        missing_meta += 1
                        continue
                        
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    
                    try:
                        with open(stats_path, "r") as f:
                            stats = json.load(f)
                    except:
                        stats = {}
                        
                    try:
                        with open(geom_path, "r") as f:
                            geom = json.load(f)
                    except:
                        geom = {}
                        
                    try:
                        with open(val_path, "r") as f:
                            val = json.load(f)
                        if val.get("Overall Status") == "PASS":
                            passed += 1
                        else:
                            failed += 1
                    except:
                        failed += 1
                        
                    sim_size = sum(os.path.getsize(os.path.join(sim_dir, f)) for f in os.listdir(sim_dir) if os.path.isfile(os.path.join(sim_dir, f)))
                    total_size_bytes += sim_size
                    
                    all_sims.append({
                        "Simulation ID": meta.get("simulation_id", sim),
                        "UUID": meta.get("uuid", ""),
                        "Material": meta.get("material", mat.replace('_', ' ')),
                        "Dimension": f"{dim}D",
                        "Geometry": geom.get("Geometry Type", "Unknown"),
                        "Boundary Condition": "Mixed",
                        "Heat Source": "Yes" if meta.get("heat_source_config") else "No",
                        "Grid Size": meta.get("grid_resolution", "Unknown"),
                        "Simulation Time": meta.get("simulation_time", 0.0),
                        "Output Path": sim_dir,
                        "Creation Time": meta.get("generation_timestamp", ""),
                        "Validation Status": meta.get("validation_status", "UNKNOWN"),
                        "Quality Score": meta.get("quality_score", 0),
                        "max_temp": stats.get("max_temp", 0),
                        "min_temp": stats.get("min_temp", 0),
                        "avg_temp": stats.get("avg_temp", 0)
                    })

        # Save Catalogs
        catalog_path_csv = os.path.join(self.data_base, "dataset_catalog.csv")
        catalog_path_json = os.path.join(self.data_base, "dataset_catalog.json")
        catalog_path_parquet = os.path.join(self.data_base, "dataset_catalog.parquet")
        
        if pd is not None and all_sims:
            df = pd.DataFrame(all_sims)
            
            # Remove temp stats for the public catalog
            export_df = df.drop(columns=["max_temp", "min_temp", "avg_temp"])
            export_df.to_csv(catalog_path_csv, index=False)
            
            if pyarrow is not None:
                export_df.to_parquet(catalog_path_parquet, index=False)
            
            with open(catalog_path_json, "w") as f:
                json.dump(export_df.to_dict(orient="records"), f, indent=4)
                
            # Compile Global Summary
            summary_dict = {
                "Total Simulations": len(all_sims),
                "Total Materials": len(materials),
                "Simulation Count per Material": df["Material"].value_counts().to_dict(),
                "Simulation Count per Dimension": df["Dimension"].value_counts().to_dict(),
                "Average Temperature": float(df["avg_temp"].mean()) if not df.empty else 0,
                "Maximum Temperature": float(df["max_temp"].max()) if not df.empty else 0,
                "Minimum Temperature": float(df["min_temp"].min()) if not df.empty else 0,
                "Average Runtime": float(df["Simulation Time"].mean()) if not df.empty else 0,
                "Latest Simulation": df["Creation Time"].max() if not df.empty else "",
                "Dataset Size": f"{total_size_bytes / (1024*1024):.2f} MB"
            }
            
            with open(os.path.join(self.data_base, "dataset_statistics.json"), "w") as f:
                json.dump(summary_dict, f, indent=4)
                
            summary_df = pd.DataFrame([{
                "Total Simulations": len(all_sims),
                "Total Materials": len(materials),
                "Average Temperature": df["avg_temp"].mean() if not df.empty else 0,
                "Dataset Size (MB)": total_size_bytes / (1024*1024)
            }])
            summary_df.to_csv(os.path.join(self.data_base, "dataset_summary.csv"), index=False)
            
            if pyarrow is not None:
                summary_df.to_parquet(os.path.join(self.data_base, "dataset_summary.parquet"), index=False)
            
            # Generate Extras
            self._generate_material_summary(df)
            self._generate_material_readmes(df)
            
            # Keep all_sims available for manifest and splits
            self.total_sims = len(all_sims)
            
        # Global Validation Report
        val_report = {
            "Total Simulations": len(all_sims) + corrupted,
            "Passed Simulations": passed,
            "Failed Simulations": failed,
            "Validation Warnings": failed,
            "Duplicate Samples Removed": 0,
            "Corrupted Samples Removed": corrupted,
            "Missing Metadata": missing_meta,
            "Missing Files": corrupted,
            "Dataset Completeness": f"{(passed / max(1, len(all_sims))) * 100:.2f}%",
            "Validation Timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(self.data_base, "dataset_validation_report.json"), "w") as f:
            json.dump(val_report, f, indent=4)
            
        return all_sims
        
    def _generate_material_summary(self, df):
        if pd is None or df.empty: return
        
        mat_summary = []
        for mat in df["Material"].unique():
            mat_df = df[df["Material"] == mat]
            counts = mat_df["Dimension"].value_counts()
            
            mat_summary.append({
                "Material": mat,
                "Total Simulations": len(mat_df),
                "1D Count": counts.get("1D", 0),
                "2D Count": counts.get("2D", 0),
                "3D Count": counts.get("3D", 0),
                "Average Tmax": float(mat_df["max_temp"].mean()),
                "Average Tmin": float(mat_df["min_temp"].mean()),
                "Average Runtime": float(mat_df["Simulation Time"].mean())
            })
            
        ms_df = pd.DataFrame(mat_summary)
        mat_dir = os.path.join(self.data_base, "materials")
        os.makedirs(mat_dir, exist_ok=True)
        ms_df.to_csv(os.path.join(mat_dir, "material_summary.csv"), index=False)

    def _generate_material_readmes(self, df):
        """Generates a README.md inside every materialized dimension/material folder."""
        if pd is None or df.empty: return
        
        for mat in df["Material"].unique():
            mat_df = df[df["Material"] == mat]
            mat_info = self.mat_mgr.get_material(mat)
            
            for dim in mat_df["Dimension"].unique():
                dim_df = mat_df[mat_df["Dimension"] == dim]
                mat_folder_name = mat.replace(' ', '_')
                mat_dir = os.path.join(self.data_base, dim, mat_folder_name)
                if not os.path.exists(mat_dir): continue
                
                readme_path = os.path.join(mat_dir, "README.md")
                
                content = f"# {mat_info.get('name', mat)} ({mat_info.get('chemical_symbol', 'Unknown')})\n\n"
                content += f"**Material Category**: {mat_info.get('category', 'Engineering Material')}\n"
                content += f"**Isotropic**: {'Yes' if not isinstance(mat_info.get('k'), list) else 'No'}\n"
                content += f"**Number of Simulations**: {len(dim_df)}\n"
                content += f"**Dataset Version**: {self.version}\n"
                content += f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                content += "## Physical Properties\n"
                content += f"- Thermal Conductivity (k): {mat_info.get('k')} W/m·K\n"
                content += f"- Density (rho): {mat_info.get('rho')} kg/m³\n"
                content += f"- Specific Heat Capacity (Cp): {mat_info.get('cp')} J/kg·K\n"
                content += f"- Thermal Diffusivity (alpha): {mat_info.get('alpha')} m²/s\n\n"
                
                content += "## Typical Applications\n"
                content += f"{mat_info.get('applications', 'General engineering thermal applications.')}\n"
                
                with open(readme_path, "w") as f:
                    f.write(content)

    def generate_versioning(self):
        """Generates VERSION, CHANGELOG.md, and dataset_manifest.json"""
        timestamp_now = datetime.now().isoformat()
        
        with open(os.path.join(self.data_base, "VERSION"), "w") as f:
            f.write(self.version)
            
        changelog_path = os.path.join(self.data_base, "CHANGELOG.md")
        if not os.path.exists(changelog_path):
            with open(changelog_path, "w") as f:
                f.write(f"# Dataset Changelog\n\n## [{self.version}] - {datetime.now().strftime('%Y-%m-%d')}\n- Initial dataset generation.\n")
                
        manifest = {
            "Dataset Version": self.version,
            "Generator Version": "1.3.0",
            "Framework Version": "CFD_1.3.0",
            "Creation Date": timestamp_now,
            "Total Simulations": getattr(self, "total_sims", 0)
        }
        with open(os.path.join(self.data_base, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

    def generate_global_hash(self):
        """Generates a dataset_hash.sha256 representing the state of the entire repository."""
        print("Generating global dataset hash...")
        hasher = hashlib.sha256()
        
        # Sort files to ensure deterministic hashing
        all_files = []
        for root, dirs, files in os.walk(self.data_base):
            for file in files:
                if file == "dataset_hash.sha256": continue
                all_files.append(os.path.join(root, file))
        
        all_files.sort()
        
        for filepath in all_files:
            hasher.update(filepath.encode('utf-8'))
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
                    
        hash_val = hasher.hexdigest()
        with open(os.path.join(self.data_base, "dataset_hash.sha256"), "w") as f:
            f.write(f"{hash_val}  data/\n")

    def perform_splits(self, all_sims, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        print("Performing Train/Val/Test splits...")
        if not all_sims or pd is None:
            return
            
        random.shuffle(all_sims)
        n = len(all_sims)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_sims = all_sims[:n_train]
        val_sims = all_sims[n_train:n_train+n_val]
        test_sims = all_sims[n_train+n_val:]
        
        for split_name, split_data in zip(["train", "validation", "test"], [train_sims, val_sims, test_sims]):
            if not split_data: continue
            df = pd.DataFrame(split_data).drop(columns=["max_temp", "min_temp", "avg_temp"])
            out_path = os.path.join(self.data_base, f"{split_name}.csv")
            df.to_csv(out_path, index=False)
