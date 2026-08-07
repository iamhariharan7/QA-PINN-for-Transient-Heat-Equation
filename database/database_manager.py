import sqlite3
import os
import shutil
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

class DatabaseManager:
    """
    Object-Oriented Database Manager for the Results Database (SQLite).
    Strictly encapsulates all SQL execution.
    """
    def __init__(self, db_path="database/experiment.db"):
        self.db_path = db_path
        
    @contextmanager
    def get_connection(self):
        """Context manager for SQLite connections ensuring proper closure."""
        conn = sqlite3.connect(self.db_path)
        # Return dictionaries instead of tuples
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def execute_query(self, query: str, parameters: tuple = ()):
        """Executes a single query with optional parameters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            conn.commit()
            return cursor.lastrowid
            
    def execute_many(self, query: str, parameters_list: list):
        """Executes a query across a batch of parameters (transactions optimized)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, parameters_list)
            conn.commit()
            
    def fetch_all(self, query: str, parameters: tuple = ()):
        """Returns all matching records as a list of dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    def backup_database(self):
        """Creates a timestamped backup of the database in backups/."""
        if not os.path.exists(self.db_path):
            return
            
        backup_dir = os.path.join(os.path.dirname(self.db_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"experiment_backup_{timestamp}.db")
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to {backup_path}")

    # ==========================================================
    # INSERTION ABSTRACTIONS
    # ==========================================================
    
    def insert_material(self, data: dict):
        query = """
        INSERT OR IGNORE INTO Materials (name, chemical_symbol, category, thermal_conductivity, density, specific_heat, thermal_diffusivity, is_isotropic)
        VALUES (:name, :chemical_symbol, :category, :thermal_conductivity, :density, :specific_heat, :thermal_diffusivity, :is_isotropic)
        """
        self.execute_query(query, data)
        
    def insert_simulation(self, data: dict):
        query = """
        INSERT OR REPLACE INTO Simulations 
        (simulation_id, experiment_id, material, dimension, geometry, solver_type, grid_resolution, time_step, simulation_time, boundary_conditions, heat_source_info, random_seed, timestamp)
        VALUES (:simulation_id, :experiment_id, :material, :dimension, :geometry, :solver_type, :grid_resolution, :time_step, :simulation_time, :boundary_conditions, :heat_source_info, :random_seed, :timestamp)
        """
        self.execute_query(query, data)
        
    def insert_experiment(self, data: dict):
        query = """
        INSERT OR REPLACE INTO Experiments 
        (experiment_id, experiment_name, start_time, end_time, config_file, dataset_version, model_version, output_directory, status, notes)
        VALUES (:experiment_id, :experiment_name, :start_time, :end_time, :config_file, :dataset_version, :model_version, :output_directory, :status, :notes)
        """
        self.execute_query(query, data)

    def insert_metric(self, data: dict):
        query = """
        INSERT INTO Metrics 
        (experiment_id, model_id, rmse, mae, relative_l2_error, max_absolute_error, pde_residual_error, training_time, inference_time, memory_usage, cpu_time, gpu_time)
        VALUES (:experiment_id, :model_id, :rmse, :mae, :relative_l2_error, :max_absolute_error, :pde_residual_error, :training_time, :inference_time, :memory_usage, :cpu_time, :gpu_time)
        """
        self.execute_query(query, data)

    # ==========================================================
    # QUERY ABSTRACTIONS
    # ==========================================================
    
    def get_experiments_by_material(self, material_name: str):
        query = """
        SELECT DISTINCT e.* FROM Experiments e
        JOIN Simulations s ON e.experiment_id = s.experiment_id
        WHERE s.material = ?
        """
        return self.fetch_all(query, (material_name,))
        
    def get_fastest_model(self):
        query = """
        SELECT m.model_name, met.inference_time, met.rmse 
        FROM Models m
        JOIN Metrics met ON m.model_id = met.model_id
        ORDER BY met.inference_time ASC LIMIT 1
        """
        return self.fetch_all(query)
        
    def get_best_model_by_rmse(self):
        query = """
        SELECT m.model_name, met.rmse 
        FROM Models m
        JOIN Metrics met ON m.model_id = met.model_id
        ORDER BY met.rmse ASC LIMIT 1
        """
        return self.fetch_all(query)
        
    def export_summary(self, table_name: str, export_path: str):
        """Exports a table completely to CSV without pulling entire table into RAM."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(export_path, index=False)
