-- ==============================================================================
-- RESULTS DATABASE SCHEMA
-- Phase 7: SQLite Tracking System
-- ==============================================================================

-- 1. Materials
CREATE TABLE IF NOT EXISTS Materials (
    material_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    chemical_symbol TEXT,
    category TEXT,
    thermal_conductivity REAL,
    density REAL,
    specific_heat REAL,
    thermal_diffusivity REAL,
    is_isotropic BOOLEAN
);

-- 5. Experiments (Defined before Simulations due to Foreign Key)
CREATE TABLE IF NOT EXISTS Experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    config_file TEXT,
    dataset_version TEXT,
    model_version TEXT,
    output_directory TEXT,
    status TEXT,
    notes TEXT
);

-- 2. Simulations
CREATE TABLE IF NOT EXISTS Simulations (
    simulation_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    material TEXT NOT NULL,
    dimension TEXT NOT NULL,
    geometry TEXT NOT NULL,
    solver_type TEXT NOT NULL,
    grid_resolution TEXT,
    time_step REAL,
    simulation_time REAL,
    boundary_conditions TEXT,
    heat_source_info TEXT,
    random_seed INTEGER,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (material) REFERENCES Materials (name),
    FOREIGN KEY (experiment_id) REFERENCES Experiments (experiment_id)
);

-- 3. Datasets
CREATE TABLE IF NOT EXISTS Datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_version TEXT NOT NULL,
    simulation_id TEXT NOT NULL,
    dataset_type TEXT,
    split_type TEXT, -- Train / Validation / Test
    location TEXT NOT NULL,
    file_size INTEGER,
    sample_count INTEGER,
    generation_date DATETIME NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES Simulations (simulation_id)
);

-- 4. Models
CREATE TABLE IF NOT EXISTS Models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    material TEXT,
    dimension TEXT,
    training_date DATETIME NOT NULL,
    location TEXT NOT NULL,
    best_validation_score REAL
);

-- 6. Metrics
CREATE TABLE IF NOT EXISTS Metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    model_id TEXT,
    rmse REAL,
    mae REAL,
    relative_l2_error REAL,
    max_absolute_error REAL,
    pde_residual_error REAL,
    training_time REAL,
    inference_time REAL,
    memory_usage REAL,
    cpu_time REAL,
    gpu_time REAL,
    FOREIGN KEY (experiment_id) REFERENCES Experiments (experiment_id),
    FOREIGN KEY (model_id) REFERENCES Models (model_id)
);

-- 7. Outputs
CREATE TABLE IF NOT EXISTS Outputs (
    output_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    output_directory TEXT NOT NULL,
    report_location TEXT,
    dashboard_location TEXT,
    comparison_location TEXT,
    generated_time DATETIME NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES Experiments (experiment_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sim_material ON Simulations (material);
CREATE INDEX IF NOT EXISTS idx_sim_dimension ON Simulations (dimension);
CREATE INDEX IF NOT EXISTS idx_exp_dataset ON Experiments (dataset_version);
CREATE INDEX IF NOT EXISTS idx_metrics_rmse ON Metrics (rmse);
