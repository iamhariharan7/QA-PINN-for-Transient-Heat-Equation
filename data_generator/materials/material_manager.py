import os
import json
from typing import List, Dict, Union

class MaterialManager:
    def __init__(self, db_path: str = "data_generator/materials/material_db.json"):
        self.db_path = db_path
        self.materials: Dict[str, dict] = {}
        self._load_database()

    def _load_database(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Material database not found at {self.db_path}")
            
        with open(self.db_path, "r") as f:
            materials_list = json.load(f)
            
        for mat in materials_list:
            self.materials[mat["name"]] = mat
            
    def get_material(self, name: str) -> dict:
        """Retrieve material properties by name."""
        if name not in self.materials:
            raise ValueError(f"Material '{name}' not found in database.")
        return self.materials[name]

    def get_all_material_names(self) -> List[str]:
        """Return a list of all available material names."""
        return list(self.materials.keys())
        
    def filter_materials(self, names: List[str]) -> List[dict]:
        """Return properties for a filtered list of material names."""
        return [self.get_material(name) for name in names]

    def get_thermal_diffusivity(self, name: str) -> Union[float, List[float]]:
        """Helper to get thermal diffusivity (alpha)."""
        mat = self.get_material(name)
        return mat["alpha"]
        
    def export_to_formats(self, export_dir: str = "data/materials"):
        """Exports the current material database to CSV, JSON, and YAML formats."""
        os.makedirs(export_dir, exist_ok=True)
        
        # Export JSON
        json_path = os.path.join(export_dir, "material_database.json")
        with open(json_path, "w") as f:
            json.dump(list(self.materials.values()), f, indent=4)
            
        # Export YAML
        try:
            import yaml
            yaml_path = os.path.join(export_dir, "material_database.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(list(self.materials.values()), f)
        except ImportError:
            pass
            
        # Export CSV
        try:
            import pandas as pd
            csv_path = os.path.join(export_dir, "material_database.csv")
            df = pd.DataFrame(list(self.materials.values()))
            df.to_csv(csv_path, index=False)
        except ImportError:
            pass
