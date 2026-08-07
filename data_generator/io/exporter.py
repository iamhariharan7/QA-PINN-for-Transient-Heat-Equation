import os
import json
import hashlib
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import h5py
except ImportError:
    h5py = None
try:
    import yaml
except ImportError:
    yaml = None

class Exporter:
    """
    Handles exporting simulation data to various formats into the reusable data/ directory.
    """
    def __init__(self):
        pass
        
    def _save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def export_metadata(self, dataset_dir: str, metadata: dict):
        self._save_json(os.path.join(dataset_dir, "metadata.json"), metadata)
            
    def export_statistics(self, dataset_dir: str, stats: dict):
        self._save_json(os.path.join(dataset_dir, "statistics.json"), stats)
        
    def export_geometry(self, dataset_dir: str, geometry: dict):
        self._save_json(os.path.join(dataset_dir, "geometry.json"), geometry)
        
    def export_boundary_conditions(self, dataset_dir: str, bcs: dict):
        self._save_json(os.path.join(dataset_dir, "boundary_conditions.json"), bcs)
        
    def export_heat_source(self, dataset_dir: str, source: dict):
        self._save_json(os.path.join(dataset_dir, "heat_source.json"), source)
        
    def export_validation_report(self, dataset_dir: str, validation: dict):
        self._save_json(os.path.join(dataset_dir, "validation_report.json"), validation)
        
    def export_seed(self, dataset_dir: str, seed_info: dict):
        self._save_json(os.path.join(dataset_dir, "seed.json"), seed_info)
        
    def export_status(self, dataset_dir: str, status_info: dict):
        self._save_json(os.path.join(dataset_dir, "status.json"), status_info)
        
    def export_provenance(self, dataset_dir: str, provenance_info: dict):
        self._save_json(os.path.join(dataset_dir, "provenance.json"), provenance_info)
        
    def export_units(self, dataset_dir: str, units_info: dict):
        self._save_json(os.path.join(dataset_dir, "units.json"), units_info)
        
    def export_config_snapshot(self, dataset_dir: str, config: dict):
        if yaml is None:
            return
        path = os.path.join(dataset_dir, "config_used.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)

    def export_npz(self, dataset_dir: str, data: dict):
        """data contains arrays like 't', 'x', 'y', 'z', 'temperature'"""
        path = os.path.join(dataset_dir, "temperature.npz")
        np.savez_compressed(path, **data)

    def export_hdf5(self, dataset_dir: str, data: dict, metadata: dict):
        if h5py is None:
            return
            
        path = os.path.join(dataset_dir, "temperature.h5")
        
        with h5py.File(path, "w") as f:
            for k, v in metadata.items():
                try:
                    f.attrs[k] = str(v)
                except:
                    pass
            for k, v in data.items():
                f.create_dataset(k, data=v, compression="gzip")
                
    def export_csv_1d(self, dataset_dir: str, t: np.ndarray, x: np.ndarray, T: np.ndarray, meta: dict):
        if pd is None:
            return
        path = os.path.join(dataset_dir, "temperature.csv")
        
        nt, nx = T.shape
        tt = np.repeat(t, nx)
        xx = np.tile(x, nt)
        TT = T.flatten()
        
        df = pd.DataFrame({
            'simulation_id': meta['simulation_id'],
            'material': meta['material'],
            'time': tt,
            'x': xx,
            'temperature': TT
        })
        df.to_csv(path, index=False)
        
    def export_csv_2d(self, dataset_dir: str, t: np.ndarray, x: np.ndarray, y: np.ndarray, T: np.ndarray, meta: dict):
        if pd is None:
            return
        path = os.path.join(dataset_dir, "temperature.csv")
        
        # Take final frame to prevent extreme file sizes
        T_final = T[-1]
        X, Y = np.meshgrid(x, y)
        
        df = pd.DataFrame({
            'simulation_id': meta['simulation_id'],
            'material': meta['material'],
            'time': t[-1],
            'x': X.flatten(),
            'y': Y.flatten(),
            'temperature': T_final.flatten()
        })
        df.to_csv(path, index=False)

    def export_checksums(self, dataset_dir: str):
        """Calculates SHA-256 for all generated files and saves to checksums.json"""
        checksums = {}
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file == "checksums.json": continue
                filepath = os.path.join(root, file)
                hasher = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                checksums[file] = hasher.hexdigest()
        
        self._save_json(os.path.join(dataset_dir, "checksums.json"), checksums)
        
    def export_archive(self, dataset_dir: str):
        """Compresses the completed simulation folder into a ZIP archive alongside it."""
        import shutil
        shutil.make_archive(dataset_dir, 'zip', dataset_dir)
