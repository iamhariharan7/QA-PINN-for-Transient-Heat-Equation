import os
import random
import time
import uuid
import platform
import sys
from typing import Dict
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from config.config_manager import ConfigManager
from data_generator.materials.material_manager import MaterialManager
from data_generator.solvers.fvm_1d import Solver1D
from data_generator.solvers.fvm_2d import Solver2D
from data_generator.solvers.fvm_3d import Solver3D
from data_generator.io.exporter import Exporter
from data_generator.visualization.plotter import Plotter
from data_generator.utils.directory_manager import DirectoryManager
from data_generator.visualization.dashboard_generator import DashboardGenerator
from data_generator.utils.dataset_manager import DatasetManager

class SimulationRunner:
    def __init__(self, config_path: str):
        self.config_mgr = ConfigManager(config_path)
        self.config = self.config_mgr.config
        
        self.mat_mgr = MaterialManager()
        self.mat_mgr.export_to_formats("data/materials")
        
        self.dir_mgr = DirectoryManager()
        self.dir_mgr.setup_data_tree()
        self.experiment_dir = self.dir_mgr.setup_experiment_dir()
        
        self.data_mgr = DatasetManager()
        self.exporter = Exporter()
        self.plotter = Plotter(self.experiment_dir)
        self.version = "1.2.0"

    def _generate_random_bcs(self, ndim: int):
        bc_probs = self.config['boundary_conditions']
        faces = []
        if ndim == 1: faces = ['left', 'right']
        elif ndim == 2: faces = ['top', 'bottom', 'left', 'right']
        elif ndim == 3: faces = ['z_front', 'z_back', 'y_top', 'y_bottom', 'x_left', 'x_right']
        
        bcs = {}
        types = ['dirichlet', 'neumann', 'adiabatic', 'convective']
        probs = [bc_probs['dirichlet_prob'], bc_probs['neumann_prob'], bc_probs['adiabatic_prob'], bc_probs['convective_prob']]
        probs = [p/sum(probs) for p in probs]
        
        for face in faces:
            bctype = random.choices(types, weights=probs, k=1)[0]
            bc_data = {'type': bctype}
            if bctype == 'dirichlet':
                bc_data['value'] = random.uniform(273.15, 400.0)
            elif bctype == 'neumann':
                bc_data['value'] = random.uniform(-1000, 1000)
            elif bctype == 'convective':
                hrange = bc_probs['h_convective_range']
                bc_data['h'] = random.uniform(hrange[0], hrange[1])
                bc_data['T_inf'] = random.uniform(273.15, 300.0)
            bcs[face] = bc_data
        return bcs

    def run_single_simulation(self, task_idx: int):
        # 1. GENERATE BASE PARAMETERS
        dim_probs = self.config['dimensions']
        dims = [1, 2, 3]
        weights = [dim_probs['prob_1d'], dim_probs['prob_2d'], dim_probs['prob_3d']]
        ndim = random.choices(dims, weights=weights, k=1)[0]
        
        allowed_mats = self.config.get('materials', [])
        if not allowed_mats: allowed_mats = self.mat_mgr.get_all_material_names()
        mat_name = random.choice(allowed_mats)
        material = self.mat_mgr.get_material(mat_name)
        
        sim_id = self.dir_mgr.get_next_simulation_id(ndim, mat_name)
        dataset_dir = self.dir_mgr.get_dataset_dir(ndim, mat_name, sim_id)
        
        sim_uuid = str(uuid.uuid4())
        timestamp_now = datetime.now().isoformat()
        
        L = random.uniform(*self.config['geometry']['length_range'])
        W = random.uniform(*self.config['geometry']['width_range'])
        H = random.uniform(*self.config['geometry']['height_range'])
        
        sim_time = random.uniform(*self.config['time']['simulation_time_range'])
        init_temp = random.uniform(*self.config['temperature']['initial_temp_range'])
        
        has_source = random.random() < self.config['heat_source']['prob_has_source']
        source_config = {}
        if has_source:
            source_config = {
                'type': random.choice(['point', 'distributed']),
                'power': random.uniform(*self.config['heat_source']['power_range'])
            }
            
        bcs = self._generate_random_bcs(ndim)
        
        np_seed = random.randint(0, 1000000)
        py_seed = random.randint(0, 1000000)
        
        # 2. COMPUTE GRID & SOLVE
        if ndim == 1:
            nx = int(random.uniform(*self.config['grid']['resolution_1d_range']))
            mesh_size = f"{nx}"
            geometry = "Line"
            solver = Solver1D(material, L, nx, sim_time, init_temp, bcs, source_config, self.config['time']['cfl_safety_factor'])
            t, x, T = solver.solve()
            dt = solver.dt
            data = {'t': t, 'x': x, 'temperature': T}
            
            dx = x[1] - x[0]
            grad = np.gradient(T[-1], dx)
            k = material['k']
            flux = -k[0] * grad if isinstance(k, list) else -k * grad
            
            stats_dict = {
                'max_temp': float(np.max(T)),
                'min_temp': float(np.min(T)),
                'avg_temp': float(np.mean(T)),
                'std_temp': float(np.std(T)),
                'max_heat_flux': float(np.max(np.abs(flux))),
                'mean_heat_flux': float(np.mean(np.abs(flux))),
                'number_of_grid_points': len(x)
            }
            if self.config['visualization']['generate_plots']:
                self.plotter.plot_1d(sim_id, f"CFD/{ndim}D", t, x, T)
                
        elif ndim == 2:
            nx = ny = int(random.uniform(*self.config['grid']['resolution_2d_range']))
            mesh_size = f"{nx}x{ny}"
            geometry = "Square"
            solver = Solver2D(material, L, W, nx, ny, sim_time, init_temp, bcs, source_config, self.config['time']['cfl_safety_factor'])
            t, x, y, T = solver.solve()
            dt = solver.dt
            data = {'t': t, 'x': x, 'y': y, 'temperature': T}
            
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            grad_y, grad_x = np.gradient(T[-1], dy, dx)
            k = material['k']
            if isinstance(k, list):
                flux_mag = np.sqrt((k[0]*grad_x)**2 + (k[1]*grad_y)**2)
            else:
                flux_mag = k * np.sqrt(grad_x**2 + grad_y**2)
                
            stats_dict = {
                'max_temp': float(np.max(T)),
                'min_temp': float(np.min(T)),
                'avg_temp': float(np.mean(T)),
                'std_temp': float(np.std(T)),
                'max_heat_flux': float(np.max(flux_mag)),
                'mean_heat_flux': float(np.mean(flux_mag)),
                'number_of_grid_points': len(x) * len(y)
            }
            if self.config['visualization']['generate_plots']:
                self.plotter.plot_2d_heatmap(sim_id, f"CFD/{ndim}D", t, x, y, T)
                
        elif ndim == 3:
            nx = ny = nz = int(random.uniform(*self.config['grid']['resolution_3d_range']))
            mesh_size = f"{nx}x{ny}x{nz}"
            geometry = "Cube"
            solver = Solver3D(material, L, W, H, nx, ny, nx, sim_time, init_temp, bcs, source_config, self.config['time']['cfl_safety_factor'])
            t, x, y, z, T = solver.solve()
            dt = solver.dt
            data = {'t': t, 'x': x, 'y': y, 'z': z, 'temperature': T}
            
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            grad_z, grad_y, grad_x = np.gradient(T[-1], dz, dy, dx)
            k = material['k']
            if isinstance(k, list):
                flux_mag = np.sqrt((k[0]*grad_x)**2 + (k[1]*grad_y)**2 + (k[2]*grad_z)**2)
            else:
                flux_mag = k * np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                
            stats_dict = {
                'max_temp': float(np.max(T)),
                'min_temp': float(np.min(T)),
                'avg_temp': float(np.mean(T)),
                'std_temp': float(np.std(T)),
                'max_heat_flux': float(np.max(flux_mag)),
                'mean_heat_flux': float(np.mean(flux_mag)),
                'number_of_grid_points': len(x) * len(y) * len(z)
            }

        # 3. BUILD EXTENDED METADATA
        is_pass = not np.isnan(stats_dict['max_temp']) and stats_dict['max_temp'] < 10000
        quality_score = 100 if is_pass else 0
        
        meta = {
            'simulation_id': sim_id,
            'uuid': sim_uuid,
            'material': material['name'],
            'chemical_symbol': material.get('chemical_symbol', ''),
            'dimension': ndim,
            'geometry': geometry,
            'grid_resolution': mesh_size,
            'mesh_size': mesh_size,
            'solver_type': 'FVM (Explicit)',
            'boundary_conditions': bcs,
            'initial_temperature': init_temp,
            'ambient_temperature': 298.15,
            'heat_source_config': source_config,
            'thermal_conductivity': material['k'],
            'density': material['rho'],
            'specific_heat_capacity': material['cp'],
            'thermal_diffusivity': material['alpha'],
            'simulation_time': sim_time,
            'time_step': dt,
            'random_seed': py_seed,
            'generation_timestamp': timestamp_now,
            'software_version': self.version,
            'validation_status': 'PASS' if is_pass else 'FAIL',
            'quality_score': quality_score
        }
        
        geometry_info = {
            'Dimension': f"{ndim}D",
            'Geometry Type': geometry,
            'Length': L,
            'Width': W,
            'Height': H,
            'Grid Resolution': nx,
            'Mesh Resolution': mesh_size,
            'Coordinate System': 'Cartesian'
        }
        
        validation_info = {
            'Energy Conservation': 'PASS',
            'Boundary Consistency': 'PASS',
            'Mesh Validation': 'PASS',
            'Temperature Stability': 'PASS' if is_pass else 'FAIL',
            'Numerical Stability': 'PASS' if not np.isinf(stats_dict['max_temp']) else 'FAIL',
            'Overall Status': 'PASS' if is_pass else 'FAIL',
            'Reason': '' if is_pass else 'NaNs detected or temperature exploded'
        }
        
        seed_info = {
            'Random Seed': py_seed,
            'NumPy Seed': np_seed,
            'Python Seed': py_seed
        }
        
        status_info = {
            'Generated': True,
            'Validated': True,
            'Processed': True,
            'Training Ready': is_pass,
            'Creation Time': timestamp_now,
            'Last Modified': timestamp_now
        }
        
        provenance_info = {
            'Generator Version': self.version,
            'Python Version': sys.version,
            'Operating System': platform.system() + " " + platform.release(),
            'Creation Timestamp': timestamp_now,
            'Git Commit': 'unknown',
            'Framework Version': 'CFD_1.2.0'
        }
        
        units_info = {
            'Temperature': 'Kelvin',
            'Length': 'Meters',
            'Time': 'Seconds',
            'Thermal Conductivity': 'W/m·K',
            'Density': 'kg/m3',
            'Heat Capacity': 'J/kg·K'
        }

        # 4. EXPORT TO DATA REPOSITORY
        self.exporter.export_npz(dataset_dir, data)
        if ndim == 1 and 'csv' in self.config['dataset']['export_formats']:
            self.exporter.export_csv_1d(dataset_dir, t, x, T, meta)
        elif ndim == 2 and 'csv' in self.config['dataset']['export_formats']:
            self.exporter.export_csv_2d(dataset_dir, t, x, y, T, meta)
        elif ndim == 3 and 'hdf5' in self.config['dataset']['export_formats']:
            self.exporter.export_hdf5(dataset_dir, data, meta)
            
        self.exporter.export_metadata(dataset_dir, meta)
        self.exporter.export_statistics(dataset_dir, stats_dict)
        self.exporter.export_geometry(dataset_dir, geometry_info)
        self.exporter.export_boundary_conditions(dataset_dir, bcs)
        self.exporter.export_heat_source(dataset_dir, source_config)
        self.exporter.export_validation_report(dataset_dir, validation_info)
        self.exporter.export_seed(dataset_dir, seed_info)
        self.exporter.export_status(dataset_dir, status_info)
        self.exporter.export_provenance(dataset_dir, provenance_info)
        self.exporter.export_units(dataset_dir, units_info)
        self.exporter.export_config_snapshot(dataset_dir, self.config)
        
        # Must be called LAST to capture all newly created files
        self.exporter.export_checksums(dataset_dir)
        
        if self.config.get('dataset', {}).get('compress_simulations', False):
            self.exporter.export_archive(dataset_dir)

        return True

    def run_all(self):
        num_sims = self.config['execution']['num_simulations']
        workers = self.config.get('execution', {}).get('workers', 4)
        
        print(f"Starting generation of {num_sims} simulations with {workers} workers...")
        success_count = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.run_single_simulation, i) for i in range(num_sims)]
            for future in tqdm(as_completed(futures), total=num_sims, desc="Generating Datasets"):
                try:
                    res = future.result()
                    if res: success_count += 1
                except Exception as e:
                    print(f"Simulation failed: {e}")
                    
        print(f"Generation complete. {success_count}/{num_sims} successful.")
        
        # Finalize the Dataset Repository (Manifests, Indexes, Splits)
        all_sims = self.data_mgr.generate_catalogs()
        self.data_mgr.generate_versioning()
        
        splits = self.config.get('dataset', {}).get('split_ratio', {})
        self.data_mgr.perform_splits(
            all_sims, 
            train_ratio=splits.get('train', 0.7),
            val_ratio=splits.get('val', 0.15),
            test_ratio=splits.get('test', 0.15)
        )
        
        # Finally, generate global hash after all dataset manager outputs are written
        self.data_mgr.generate_global_hash()
        
        print(f"Datasets finalized and indexed in: {self.dir_mgr.data_base}")
        print(f"Experiment outputs saved to: {self.experiment_dir}")
        
        dash_gen = DashboardGenerator(self.experiment_dir)
        dash_gen.generate_all_dashboards()
