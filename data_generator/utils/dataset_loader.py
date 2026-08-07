import os
import json
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import torch
except ImportError:
    torch = None

class DatasetLoader:
    """
    A unified interface used by downstream models (CFD evaluation, CNN, PINN, QA-PINN)
    to cleanly load datasets from the repository.
    """
    def __init__(self, data_base="data", as_tensor=False):
        self.data_base = data_base
        self.as_tensor = as_tensor
        self.catalog_path = os.path.join(self.data_base, "dataset_catalog.csv")
        self.catalog = None
        if pd is not None and os.path.exists(self.catalog_path):
            self.catalog = pd.read_csv(self.catalog_path)
            
    def _convert(self, data):
        if self.as_tensor and torch is not None:
            if isinstance(data, dict):
                return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
            return torch.tensor(data, dtype=torch.float32)
        return data

    def _load_simulation_from_path(self, path: str):
        npz_path = os.path.join(path, "temperature.npz")
        csv_path = os.path.join(path, "temperature.csv")
        
        if os.path.exists(npz_path):
            data = dict(np.load(npz_path))
            return self._convert(data)
        elif pd is not None and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Depending on dimension, reconstruct. For 1D:
            if 'y' not in df.columns:
                t = df['time'].unique()
                x = df['x'].unique()
                T = df['temperature'].values.reshape(len(t), len(x))
                data = {'t': t, 'x': x, 'temperature': T}
                return self._convert(data)
        return None

    def _load_by_condition(self, condition):
        if self.catalog is None:
            raise ValueError("Catalog not loaded or pandas not installed.")
        filtered = self.catalog[condition]
        results = {}
        for _, row in filtered.iterrows():
            sim_id = row['Simulation ID']
            path = row['Output Path']
            results[sim_id] = self._load_simulation_from_path(path)
        return results

    def load_simulation(self, sim_id: str):
        if self.catalog is None: return None
        res = self.catalog[self.catalog['Simulation ID'] == sim_id]
        if res.empty: return None
        return self._load_simulation_from_path(res.iloc[0]['Output Path'])

    def load_by_uuid(self, uuid_str: str):
        if self.catalog is None: return None
        res = self.catalog[self.catalog['UUID'] == uuid_str]
        if res.empty: return None
        return self._load_simulation_from_path(res.iloc[0]['Output Path'])

    def load_dimension(self, dimension: str):
        """dimension format: '1D', '2D', '3D'"""
        return self._load_by_condition(self.catalog['Dimension'] == dimension)

    def load_material(self, material: str):
        return self._load_by_condition(self.catalog['Material'] == material)

    def load_by_boundary_condition(self, bc_type: str):
        return self._load_by_condition(self.catalog['Boundary Condition'].str.contains(bc_type, case=False, na=False))

    def load_by_geometry(self, geom_type: str):
        return self._load_by_condition(self.catalog['Geometry'] == geom_type)

    def _load_split(self, split_name: str):
        split_path = os.path.join(self.data_base, f"{split_name}.csv")
        if not os.path.exists(split_path) or pd is None:
            return {}
        split_df = pd.read_csv(split_path)
        results = {}
        for _, row in split_df.iterrows():
            sim_id = row['Simulation ID']
            path = row['Output Path']
            results[sim_id] = self._load_simulation_from_path(path)
        return results

    def load_training(self):
        return self._load_split("train")

    def load_validation(self):
        return self._load_split("validation")

    def load_testing(self):
        return self._load_split("test")
