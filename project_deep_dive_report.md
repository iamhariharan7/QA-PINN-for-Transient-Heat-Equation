# Complete Technical Deep-Dive Report
## Quantum & Classical Solver Benchmark for Heat Equation PDE
### Full reverse-engineering by Principal Software Architect / Research Scientist

---

## 1. Executive Summary

### What is this project?

A **research-grade computational framework** benchmarking five approaches to solving the
**transient heat conduction equation** (Fourier's law) in 1D, 2D, and 3D:

1. **Analytical/Exact** — closed-form Fourier mode solution
2. **CFD** — classical implicit finite difference (Crank-Nicolson / ADI)
3. **PINN** — Physics-Informed Neural Network (PyTorch + autograd PDE loss)
4. **QA-PINN** — Quantum-Assisted PINN (PennyLane variational quantum circuits)
5. **CNN** — data-driven surrogate trained on CFD output

It is split into two main pipelines:
- **Dataset Generator** (`generate_dataset.py` → `data_generator/`): Uses FDM to produce
  physically valid heat transfer simulations for 14 aerospace/engineering materials.
- **Benchmark Pipeline** (`run_experiment.py` → `src/`): Loads datasets, runs all five
  solvers in 1D/2D/3D, computes metrics, generates plots, dashboards, and a final report.

### Problem being solved

Can machine learning (classical or quantum-enhanced) replace or augment classical numerical
PDE solvers for heat conduction? This framework provides a reproducible, material-aware
experimental environment with real material properties (k, ρ, cp, α) to answer this question.

### Target users
- Computational physics researchers
- Quantum computing researchers
- Aerospace/thermal engineers
- ML researchers interested in physics-constrained networks

### Core technology stack
- PyTorch (neural networks, autograd)
- PennyLane (quantum circuits — MISSING from requirements.txt!)
- NumPy / SciPy (numerical solvers, sparse linear algebra)
- Pandas / PyArrow (data management)
- Matplotlib (visualization)
- PyYAML (config management)
- h5py, NPZ, CSV, Parquet (data I/O)

---

## 2. Complete Project Architecture

### 2.1 Folder Structure

```
project/
├── run_experiment.py           ← MAIN ENTRY POINT (benchmark pipeline)
├── generate_dataset.py         ← SECONDARY ENTRY POINT (dataset generator)
├── prepare_datasets.py         ← Data bridging helper
├── write_report.py             ← (temporary utility, can be deleted)
│
├── config/
│   ├── config_manager.py       ← YAML loader with validation
│   ├── default_config.yaml     ← Hyperparameters for all 9 model configs
│   └── generator_config.yaml   ← Dataset generation parameters
│
├── src/                        ← CORE MODEL LIBRARY
│   ├── heat_equation.py        ← 1D domain (grid, BCs, IC)
│   ├── heat_equation_2d.py     ← 2D domain
│   ├── heat_equation_3d.py     ← 3D domain
│   ├── actual_solution.py      ← 1D analytical solution
│   ├── actual_solution_2d.py   ← 2D analytical solution
│   ├── actual_solution_3d.py   ← 3D analytical solution
│   ├── cfd_solver.py           ← 1D CFD (Crank-Nicolson)
│   ├── cfd_solver_2d.py        ← 2D CFD (ADI)
│   ├── cfd_solver_3d.py        ← 3D CFD (Explicit FTCS)
│   ├── pinn_model.py           ← 1D PINN (5-layer MLP, Tanh)
│   ├── pinn_model_2d.py        ← 2D PINN
│   ├── pinn_model_3d.py        ← 3D PINN
│   ├── qa_pinn_model.py        ← 1D QA-PINN (Hybrid quantum-classical)
│   ├── qa_pinn_model_2d.py     ← 2D QA-PINN
│   ├── qa_pinn_model_3d.py     ← 3D QA-PINN
│   ├── cnn_model.py            ← 1D CNN surrogate
│   ├── cnn_model_2d.py         ← 2D CNN surrogate
│   ├── cnn_model_3d.py         ← 3D CNN surrogate
│   ├── material_loader.py      ← Material properties + dataset loading
│   ├── rectify_datasets.py     ← Validates and auto-corrects bad data
│   ├── metrics.py              ← RMSE, MAE, L2, PDE residuals
│   ├── plotting.py             ← Heatmaps, surface plots, comparisons
│   ├── dashboards.py           ← 6-category dashboard generator
│   └── report_generator.py     ← Final_Report.txt writer
│
├── data_generator/             ← CFD DATASET GENERATOR
│   ├── orchestrator/simulation_runner.py  ← Orchestrates all generation
│   ├── solvers/fdm_1d/2d/3d.py            ← Explicit FTCS solvers
│   ├── boundary_conditions/bc_manager.py  ← Dirichlet/Neumann/Adiabatic/Convective
│   ├── heat_sources/source_generator.py   ← Point and distributed sources
│   ├── materials/material_db.json         ← 14 materials (k, rho, cp, alpha)
│   ├── materials/material_manager.py      ← Load/filter from JSON DB
│   ├── io/exporter.py                     ← NPZ, HDF5, CSV, JSON exports
│   ├── visualization/plotter.py           ← Quick 1D/2D/3D plots
│   ├── visualization/dashboard_generator.py ← Mock dashboard generator
│   └── utils/directory_manager.py/dataset_manager.py/dataset_loader.py
│
├── data/                       ← GENERATED DATASETS
│   ├── dataset/<material>/     ← 14 material folders with 1D/2D/3D CSVs
│   ├── 1D/, 2D/, 3D/           ← Raw simulation outputs
│   ├── materials/              ← Exported material DB (csv/json/yaml)
│   ├── dataset_catalog.csv     ← Global simulation inventory
│   ├── dataset_hash.sha256     ← Repository integrity hash
│   └── train/val/test .csv     ← Split manifests
│
└── outputs/                    ← BENCHMARK RUN OUTPUTS
    ├── current_output/         ← Latest run (timestamped subfolder)
    └── history/                ← Archived previous runs
```

### 2.2 Module Dependency Graph

```
run_experiment.py
 ├── prepare_datasets.py
 ├── src/material_loader.py      (pandas, numpy, glob, re)
 ├── src/rectify_datasets.py     (scipy.sparse)
 ├── src/heat_equation*.py       (numpy, scipy.interpolate)
 ├── src/actual_solution*.py     (numpy)
 ├── src/cfd_solver*.py          (scipy.sparse.linalg.spsolve)
 ├── src/pinn_model*.py          (torch, torch.nn, torch.autograd)
 ├── src/qa_pinn_model*.py       (torch, pennylane)
 ├── src/cnn_model*.py           (torch, torch.nn)
 ├── src/metrics.py              (numpy, torch)
 ├── src/plotting.py             (matplotlib)
 ├── src/dashboards.py           (matplotlib)
 └── src/report_generator.py     (os, pathlib, datetime)

generate_dataset.py
 └── data_generator/orchestrator/simulation_runner.py
      ├── config/config_manager.py    → PyYAML
      ├── data_generator/materials/material_manager.py → material_db.json
      ├── data_generator/solvers/fdm_1d/2d/3d.py
      │    ├── bc_manager.py
      │    └── source_generator.py
      ├── data_generator/io/exporter.py  → h5py, pandas, yaml, hashlib
      └── data_generator/utils/dataset_manager.py → pandas, pyarrow, hashlib
```

---

## 3. Entry Point Traced Step-by-Step

### Entry: `run_experiment.py → main()`

**Step 1 — CLI Parsing**
```
--material, --excel, --dataset_dir, --config, --fast
Defaults: data/materials/material_database.csv, data/dataset, config/default_config.yaml
```

**Step 2 — Dataset Preparation**
- Reads data/materials/material_database.csv
- For each material, looks in data/1D/, data/2D/, data/3D/ for temperature CSV files
- Copies/renames to data/dataset/<material>/<material>_1D.csv (etc.)
- Missing files → creates a dummy single-row placeholder CSV

**Step 3 — Dataset Rectification**
- Scans data/dataset/ folders
- Validates: T >= 273.15 K, no NaNs, 2D/3D has multiple unique values
- Failures → re-generates data using built-in Crank-Nicolson (1D) or ADI (2D)
  with physically correct BCs for that material's k, rho, cp

**Step 4 — Fast Mode**
- If --fast or non-interactive terminal: reduces epochs drastically
  - PINN 1D: 150 (vs 5000), QA-PINN 1D: 30 (vs 600), CNN 1D: 50 (vs 2000)

**Step 5 — Material Selection**
- discover_available_materials() → fuzzy-matches dataset folders to material DB
- User selects interactively or via --material flag

**Step 6 — Domain Construction**
- Loads config/default_config.yaml
- Loads 1D/2D/3D datasets: {x, t, U} arrays
- Creates HeatEquationDomain, HeatEquationDomain2D, HeatEquationDomain3D

**Step 7 — Output Directory Setup**
- Creates outputs/current_output/<material_timestamp>/
- Archives existing contents to outputs/history/
- Creates: Actual/, CFD/, CNN/, PINN/, QA-PINN/ each with 1D/2D/3D subfolders

**Step 8 — 1D Solver Pipeline**
```
solve_exact(domain)         → U_exact[Nt, Nx]
solve_cfd(domain)           → U_cfd[Nt, Nx]
solve_pinn(domain, config)  → U_pinn[Nt, Nx] + losses + model
solve_qa_pinn(domain, cfg)  → U_qa[Nt, Nx] + losses + model
solve_cnn(domain, config)   → U_cnn[Nt, Nx] + train/val losses + model
```
After each: RMSE, MAE, Max Error, Relative L2, PDE Residual, Parameters, Memory MB

**Step 9 — 2D Pipeline** (same sequence, 2D variants)

**Step 10 — 3D Pipeline** (same sequence, 3D variants)

**Step 11 — Unseen Domain Test (1D only)**
- New IC: sin(πx/L) + 0.5·sin(3πx/L) — multi-harmonic, never seen during training
- Exact + CFD solved on new IC; PINN/QA-PINN/CNN infer without retraining
- Computes Unseen_RMSE — measures generalization to unseen data

**Step 12 — Plot + Dashboard + Report Generation**
- 5-panel comparison heatmaps (1D, 2D start/mid/end, 3D slice/panel/surface)
- 6 dashboard categories (Quantitative Metrics, Computational Performance,
  Model Analysis, Overall Ranking, Final Conclusion, Heat Map Comparison)
- Final_Report.txt with all metrics per method per dimension

---


## 4. File-by-File Explanation

### 4.1 `run_experiment.py`
Master orchestrator (~391 lines). Coordinates all 12 steps in sequence.

### 4.2 `src/heat_equation.py`
1D problem domain data container.
- Dual-mode: Works with real dataset OR synthetic sinusoidal setup
- Key attributes: x, t, U_exact, dx, dt, alpha, k, rho, cp, L, T
- initial_condition(x): Returns U[0,:] from dataset or sin(πx/L) fallback
- boundary_conditions(): Returns left/right BC values from dataset or zeros
- get_grid(): Returns 2D meshgrid arrays X, T

### 4.3 `src/heat_equation_2d.py`
2D domain container. Uses scipy.interpolate.RegularGridInterpolator for IC interpolation.
Handles both torch tensors and numpy arrays (checks hasattr(X, 'detach')).

### 4.4 `src/heat_equation_3d.py`
3D domain. IC uses 3-point interpolation (x, y, z). U_exact shape: (Nt, Nx, Ny, Nz).

### 4.5 `src/actual_solution.py / 2d / 3d`
1D: U = exp(-(π/L)² · α · t) · sin(πx/L)    [first Fourier mode]
2D: U = exp(-α·π²·(1/Lx²+1/Ly²)·t) · sin(πx/Lx) · sin(πy/Ly)
3D: Same, extended to three spatial dimensions

### 4.6 `src/cfd_solver.py` — Crank-Nicolson
r = α·dt/(2·dx²)
A = tridiag(-r, 1+2r, -r)   [implicit]
B = tridiag(+r, 1-2r, +r)   [explicit]
A·U[n+1] = B·U[n] + BC corrections, solved via spsolve
UNCONDITIONALLY STABLE.

### 4.7 `src/cfd_solver_2d.py` — ADI (Peaceman-Rachford)
Half step: implicit x, explicit y (row-by-row 1D tridiagonal solves)
Full step: implicit y, explicit x (column-by-column 1D tridiagonal solves)
rx = α·dt/(2·dx²), ry = α·dt/(2·dy²)
UNCONDITIONALLY STABLE.

### 4.8 `src/cfd_solver_3d.py` — Explicit FTCS
U[n+1,...] = U[n,...] + rx·Δ_x + ry·Δ_y + rz·Δ_z
CONDITIONALLY STABLE: requires rx+ry+rz ≤ 0.5
Warns but does NOT auto-fix dt. Weakest of the three CFD solvers.

### 4.9 `src/pinn_model.py` — 1D PINN
Architecture: Linear(2,128) → [Tanh, 128→128] × 5 → Linear(128,1)
Inputs: (x*, t*) normalized to [0,1]
Loss: L_pde + 10·L_ic + 10·L_bc
- L_pde = MSE(∂u*/∂t* − α*·∂²u*/∂x*²)   via autograd
- L_ic  = MSE(u*(x,0) − û_ic(x))
- L_bc  = MSE(u*(0,t) − bc_L) + MSE(u*(1,t) − bc_R)
α* = α·T_max/L² (non-dimensionalized)
Optimizer: Adam (lr=1e-3) + ReduceLROnPlateau

### 4.10 `src/pinn_model_2d.py` / `_3d.py`
2D: Linear(3,64) → [64→64, Tanh]×5 → Linear(64,1), inputs (x*,y*,t*)
3D: Linear(4,80) → [80→80, Tanh]×5 → Linear(80,1), inputs (x*,y*,z*,t*)
PDE loss uses full Laplacian (2D: 2 terms; 3D: 3 terms) via autograd

### 4.11 `src/qa_pinn_model.py` — THE QUANTUM LAYER
Architecture: pre → quantum circuit → post
- pre:  Linear(2, n_qubits)
- QC:   AngleEmbedding + BasicEntanglerLayers + PauliZ measurements
- post: Linear(n_qubits, 32) → Tanh → Linear(32, 1)

Quantum Circuit details:
1. AngleEmbedding(tanh(pre(x,t)), wires) → encodes into qubit rotations
2. BasicEntanglerLayers(weights, wires)  → Ry/Rz gates + CNOT ring entanglement
3. expval(PauliZ(i)) for each qubit → output in [-1, +1]

Device: qml.device("default.qubit") → classical simulation, NOT real quantum hardware
batch_size=5 (quantum simulation O(2^n) per sample)
Default: n_qubits=4, n_layers=3; full: n_qubits=8

### 4.12 `src/cnn_model.py` — CNN Surrogate
Encoder: Conv1d(1→32)→Conv1d(32→64)→Conv1d(64→128) [all ReLU]
Decoder: Flatten → Linear(128·Nx, 1024) → Linear(1024, Nx·Nt) → reshape (Nt,Nx)
Input: IC[1,1,Nx] — single temperature profile at t=0
Output: U[1,Nt,Nx] — full spatio-temporal field
Training: make_cnn_data() generates n_samples perturbed ICs solved by CFD
  Perturbation: IC_new = IC·U(0.8,1.2) + sin(m·πx/L)·U(-5,5), m∈{1,2,3}
Inference: single O(1) forward pass after training

### 4.13 `src/cnn_model_2d.py` / `_3d.py`
2D: Conv2d encoder → decode to (Nt,Ny,Nx)
3D: Conv3d encoder → decode to (Nt,Nx,Ny,Nz)
Training perturbation: adds 2D/3D spatial mode combinations

### 4.14 `src/metrics.py`
- calculate_metrics(U_pred, U_exact): RMSE, MAE, Max Abs Error, Relative L2
- count_parameters(model): sum of trainable parameter element counts
- estimate_memory_mb(model): sum(p.numel()×p.element_size()) / 1024²
- compute_pde_residual_map(U, domain): |∂U/∂t − α·∂²U/∂x²| via finite diffs
- compute_pde_residual_scalar(U, domain): mean of interior residual
- compute_pde_residual_std(U, domain): std of interior residual
All exist in 1D, 2D, 3D variants.

### 4.15 `src/material_loader.py`
- parse_scientific(): Handles superscript unicode, ×10 notation, bracket lists
- _find_column(): Fuzzy column name matching
- load_material_properties(): 3-tier matching (exact → token → substring)
- discover_available_materials(): Scans dataset dir, matches folders to material DB
- load_simulation_datasets(): Loads 1D/2D/3D CSVs, pivots to numpy arrays
  1D: pivot(index=time_s, columns=x_m) → (Nt, Nx)
  2D: reshape → (Nt, Ny, Nx)
  3D: reshape → (Nt, Nx, Ny, Nz)

### 4.16 `src/rectify_datasets.py`
- solve_exact_1d_rectified(): Crank-Nicolson, T_left=500K, T_right=300K, T_ic=300K
- solve_exact_2d_rectified(): ADI, Gaussian hot spot IC centered at domain center
- solve_exact_3d_rectified(): Gaussian plume: σ = √(0.01 + 2αt), natural diffusion
- rectify_all_datasets(): Validates T_min ≥ 273.15K, no NaN, ≥2 unique spatial values

### 4.17 `src/dashboards.py`
6 dashboard sub-generators:
- _generate_heatmap_dashboards(): 1D(1×5), 2D(3×5), 3D(3×5) pcolormesh grids
- _generate_quantitative_metrics_dashboards(): RMSE/MAE/L2/Max/PDE bar charts
- _generate_computational_performance_dashboards(): time/params/memory bars
- _generate_model_analysis_dashboards(): loss curves, parameter counts
- _generate_overall_ranking_dashboards(): composite scoring
- _generate_final_conclusion_dashboards(): text winner summary

### 4.18 `src/report_generator.py`
- generate_report(): Writes Final_Report.txt
  Material properties header + metrics per method per dimension
- archive_previous_output(): Archive utility

### 4.19 `data_generator/orchestrator/simulation_runner.py`
- _generate_random_bcs(): Randomly assigns BC types to each face using config probabilities
- run_single_simulation(): Full lifecycle:
  1. Pick random dim/material/geometry/BCs/source
  2. FDM solver → T[Nt,Nx[,Ny,Nz]]
  3. Heat flux statistics via np.gradient
  4. Build metadata dict (UUID, timestamp, provenance, validation status)
  5. Export NPZ + CSV/HDF5 + 11 JSON sidecar files + checksums.json
- run_all(): N simulations → catalogs → versioning → splits → global hash

### 4.20 `data_generator/solvers/fdm_1d.py`
Explicit FTCS: U_new[1:-1] = U[1:-1] + factor*(U[2:] - 2*U[1:-1] + U[:-2]) + q_source
Stability: dt = CFL × dx² / (2α); saves ~100 frames

### 4.21 `data_generator/solvers/fdm_2d.py`
Explicit FTCS anisotropic: separate alpha_x, alpha_y
Stability: dt = CFL / (2αx/dx² + 2αy/dy²); saves ~50 frames

### 4.22 `data_generator/solvers/fdm_3d.py`
Explicit FTCS anisotropic: alpha_x, alpha_y, alpha_z
Stability: dt = CFL / (2αx/dx² + 2αy/dy² + 2αz/dz²); saves ~20 frames

### 4.23 `data_generator/boundary_conditions/bc_manager.py`
Applies BCs in-place:
- Dirichlet: T[0] = value
- Neumann:   T[0] = T[1] + q·dx/k
- Adiabatic: T[0] = T[1]          (zero-flux ghost cell)
- Convective: T[0] = (T[1] + h·dx/k·T∞) / (1 + h·dx/k)  [Robin BC]
Implemented for 1D (2 boundaries), 2D (4 faces), 3D (6 faces)

### 4.24 `data_generator/heat_sources/source_generator.py`
- Point source: q̇[i] = P/dx (nearest node)
- Distributed: q̇[mask] = P / (N_nodes × vol_element) over a rectangular region
3D uses np.meshgrid with indexing='ij' for correct (z,y,x) axis orientation

### 4.25 `data_generator/materials/material_db.json`
14 materials: Copper, Aluminium 6061, Mild Steel, Stainless Steel 304,
Titanium Ti-6Al-4V, Inconel 718, LI-900 Silica Tile, Carbon-Carbon Composite,
Silicon, Alumina, Graphite, Silicon Carbide, Diamond, Tungsten

Properties per material: k, rho, cp, alpha, aliases, applications, category, type
ANISOTROPIC: Carbon-Carbon Composite and Graphite have k/alpha as [kx, ky, kz]

### 4.26 `data_generator/utils/dataset_manager.py`
- generate_catalogs(): Walks 1D/2D/3D dirs, reads metadata.json, produces catalog CSV/JSON/Parquet
- generate_versioning(): VERSION, CHANGELOG.md, dataset_manifest.json
- generate_global_hash(): Deterministic SHA-256 of all files (sorted walk)
- perform_splits(): Shuffle + split 70/15/15 → train/validation/test CSV

### 4.27 `data_generator/io/exporter.py`
Per-simulation: temperature.npz + temperature.csv (1D/2D) + temperature.h5 (3D)
11 JSON sidecars: metadata, statistics, geometry, boundary_conditions, heat_source,
validation_report, seed, status, provenance, units, config_used
checksums.json: SHA-256 hash of every generated file

### 4.28 `config/default_config.yaml`
Full epochs: PINN 5000, QA-PINN 600 (8 qubits), CNN 2000
2D: PINN 4000, QA-PINN 400, CNN 1000
3D: PINN 2000, QA-PINN 200, CNN 600
Grids: 1D (30×30), 2D (25×25×20), 3D (10×10×10×10)
material_tolerance: 1e-4 (for calculated vs Excel alpha verification)

### 4.29 `config/generator_config.yaml`
5 simulations default; 4 workers configured (but workers=1 hardcoded in code!)
BC probabilities: 40% Dirichlet, 20% each Neumann/Adiabatic/Convective
Heat source: 80% probability, power 100–10,000 W
Materials list: all 14 from material_db.json

---


## 5. Algorithm Analysis

### 5.1 The Governing PDE

Transient heat conduction (Fourier's Law):
    ∂u/∂t = α·∇²u + q̇/(ρ·cp)

Where:
- u(x,t) = temperature [K]
- α = k/(ρ·cp) = thermal diffusivity [m²/s]
- k = thermal conductivity [W/m·K]
- ρ = density [kg/m³]
- cp = specific heat [J/kg·K]
- q̇ = volumetric heat generation [W/m³]

### 5.2 Analytical Solution

For IC u(x,0)=sin(πx/L) with zero Dirichlet BCs:
1D: u(x,t) = exp(-(π/L)²·α·t) · sin(πx/L)
2D: u = exp(-α·π²·(1/Lx²+1/Ly²)·t) · sin(πx/Lx)·sin(πy/Ly)
3D: Same, extended to three sinusoidal modes

This is the first Fourier mode — exact only for sinusoidal IC/BC.

### 5.3 Crank-Nicolson (1D CFD)

r = α·dt/(2·dx²)
A = tridiag(-r, 1+2r, -r)    [implicit]
B = tridiag(+r, 1-2r, +r)    [explicit]
System: A·U[n+1] = B·U[n] + BC corrections
Solved: scipy.sparse.linalg.spsolve
Stability: UNCONDITIONAL (no CFL limit)

### 5.4 ADI Method (2D CFD)

Peaceman-Rachford splitting:
Half step: implicit in x, explicit in y → row-by-row 1D tridiagonal solves
Full step: implicit in y, explicit in x → column-by-column 1D tridiagonal solves
rx = α·dt/(2·dx²), ry = α·dt/(2·dy²)
Stability: UNCONDITIONAL for 2D heat equation

### 5.5 Explicit FTCS (3D CFD and Dataset Generator)

U[n+1,i,j,k] = U[n,...] + rx·(Δx) + ry·(Δy) + rz·(Δz)
Stability: CONDITIONAL — requires rx + ry + rz ≤ 0.5 (Von Neumann criterion)
Dataset generator enforces dt = CFL·(stability_limit); src/cfd_solver_3d.py warns only.

### 5.6 PINN Training

Network f_θ: (x*, t*) → u*(x*, t*)  [MLP with Tanh, inputs in [0,1]]

Loss composition:
  L_pde = E[(∂u*/∂t* − α*·∂²u*/∂x*²)²]       PDE physics residual (autograd)
  L_ic  = E[(u*(x,0) − û_ic(x))²]              initial condition
  L_bc  = E[(u*(0,t)−bc_L)² + (u*(1,t)−bc_R)²] boundary conditions
  TOTAL = L_pde + 10·L_ic + 10·L_bc

Key: autograd with create_graph=True → exact derivatives of network, not numerical approx.
α* = α·T_max/L² (non-dimensionalization)
λ=10 forces IC/BC compliance strongly (without it, the PDE loss is non-unique)

### 5.7 QA-PINN Circuit

Architecture: Input(x,t) → Linear(2→n_q) → Tanh → [QC] → Linear(n_q→32) → Tanh → Linear(32→1)

Quantum circuit:
1. AngleEmbedding(tanh(pre(x,t)), wires) → qubit rotation angles
2. BasicEntanglerLayers(weights, wires) → Ry/Rz + CNOT ring entanglement
3. expval(PauliZ(i)) for each qubit → output ∈ [-1, +1]

Device: qml.device("default.qubit") = classical simulator (NOT real quantum hardware)
q_weights = classical PyTorch parameters, optimized jointly via Adam
batch_size=5 because quantum simulation cost is O(2^n) per sample

### 5.8 CNN Surrogate

Learns G: IC ↦ U  (solution operator, purely data-driven)
Training pairs: (perturbed_IC, CFD_solution)
Perturbation: IC_new = IC·uniform(0.8,1.2) + sin(m·πx/L)·uniform(-5,5)
After training: inference = single forward pass (O(1), microseconds)
Limitation: trained on narrow IC family — may not generalize to very different ICs

### 5.9 PDE Residual Metric

Post-hoc physics quality metric:
residual[n,i] = |(U[n+1,i]-U[n,i])/dt - α·(U[n,i+1]-2U[n,i]+U[n,i-1])/dx²|

Model-agnostic: works on any solver output (CFD, PINN, CNN).
Low residual → satisfies heat equation. High residual → physics violated.

### 5.10 Optimizer Configuration

Adam: lr=1e-3 (PINN/CNN), lr=5e-3 (QA-PINN, higher due to fewer parameters)
ReduceLROnPlateau: factor=0.5, patience=max(10, epochs//20), min_lr=1e-6

---

## 6. Data Flow

### Dataset Generation Pipeline

    generator_config.yaml
           ↓
    SimulationRunner.__init__
      MaterialManager → material_db.json → 14 material dicts
      DirectoryManager → data/1D/2D/3D/<material>/<Simulation_ID>/ trees
           ↓
    run_single_simulation() × N
      → random: material, dim, geometry, BCs, source
      → FDM solver → T[Nt, Nx[, Ny, Nz]]
      → Exporter: NPZ + CSV/HDF5 + 11 JSON sidecars + checksums.json
           ↓
    run_all() post-processing
      → DatasetManager: catalog CSV/JSON/Parquet, VERSION, splits, global hash

### Benchmark Pipeline

    material_database.csv
           ↓ prepare_datasets() → data/dataset/<mat>/*_1D/2D/3D.csv
           ↓ rectify_all_datasets() → validates/replaces bad data
           ↓ discover_available_materials() → list of {folder, props}
           ↓ load_simulation_datasets() → {1D:{x,t,U}, 2D:{x,y,t,U}, 3D:{x,y,z,t,U}}
           ↓ HeatEquationDomain*(dataset_dict, props)
    ┌──────────────────────────────────────┐
    │  solve_exact → U_exact               │
    │  solve_cfd   → U_cfd                 │
    │  solve_pinn  → U_pinn + model        │
    │  solve_qa_pinn → U_qa + model        │
    │  solve_cnn   → U_cnn + model         │
    └──────────────────────────────────────┘
           ↓
    calculate_metrics + PDE residuals + model stats
           ↓
    plots + dashboards + Final_Report.txt

---

## 7. Training Pipeline

### PINN / QA-PINN
1. Seed everything (torch=42, numpy=42)
2. Sample N collocation points in [0,1]²
3. For each epoch:
   a. Forward pass → u_predicted
   b. Autograd → ∂u/∂t, ∂²u/∂x²
   c. L_pde, L_ic, L_bc → TOTAL = L_pde + 10·L_ic + 10·L_bc
   d. backward() → optimizer.step() → scheduler.step()
4. Infer on full grid → de-normalize → return U, losses, model

### CNN
1. make_cnn_data() → n_samples (IC, CFD_solution) pairs
2. Normalize by (u_min, u_scale)
3. Split 80/20 train/val
4. For each epoch: MSELoss, backward, optimizer
5. Infer on real IC → de-normalize → return U, losses, model

---

## 8. Evaluation Pipeline

For every method × dimension:
  - calculate_metrics(U_pred, U_exact): RMSE, MAE, Max Abs Error, Relative L2
  - compute_pde_residual_*(U, domain): mean and std of physics compliance
  - count_parameters(model), estimate_memory_mb(model)

Unseen Domain Test (1D):
  New IC: sin(πx/L) + 0.5·sin(3πx/L)  ← never seen during training
  Infer with existing models → Unseen_RMSE per method

---

## 9. Output Generation

Per solver per dimension:
  <method>/<dim>/<method>_temperature.npz
  <method>/<dim>/Heat Map (1D).png
  <method>/<dim>/Heat Map (Start|Middle|End).png  [2D/3D]

Comparison plots:
  Comparison/1D/Heat Map Comparison (1D).png
  Comparison/2D/Heat Map Comparison (Start|Middle|End).png
  Comparison/3D/Slice|Panel|Surface Heat Map Comparison.png

Detailed dashboards:
  Detailed Comparison/Heat Map Comparison/
  Detailed Comparison/Quantitative Metrics/
  Detailed Comparison/Computational Performance/
  Detailed Comparison/Model Analysis/
  Detailed Comparison/Overall Ranking/
  Detailed Comparison/Final Conclusion/

Report: Final_Report.txt (material properties + all metrics per method per dim)

---

## 10. External Dependencies

| Library   | Version  | Role                                          |
|-----------|----------|-----------------------------------------------|
| numpy     | >=1.24   | Core arrays, broadcasting, all math           |
| scipy     | >=1.10   | Sparse matrices, spsolve, interpolators       |
| matplotlib| >=3.7    | All plotting (contourf, pcolormesh, surface)  |
| pandas    | >=2.0    | CSV I/O, pivot tables, catalog management     |
| torch     | >=2.0    | Neural networks, autograd, Adam optimizer     |
| pennylane | MISSING! | Quantum circuits, qnodes, variational layers  |
| pyyaml    | >=6.0    | Config YAML parsing                           |
| h5py      | >=3.8    | HDF5 for 3D datasets                          |
| pyarrow   | >=12.0   | Parquet catalog exports                       |
| tqdm      | >=4.65   | Progress bars in dataset generator            |
| psutil    | >=5.9    | System memory/CPU reporting                   |

CRITICAL: PennyLane is NOT in requirements.txt — QA-PINN will crash without it!

---

## 11. Strengths

1. Comprehensive scope: 1D/2D/3D × 5 methods × 14 materials in a single pipeline
2. Physical correctness: Real material properties; temperature in Kelvin; T >= 273.15 K
3. Self-healing data: Rectification auto-generates physically correct data for bad datasets
4. Reproducibility: Seeds=42, checksums, provenance.json, config snapshots, SHA-256 hash
5. Hybrid quantum-classical: Real PennyLane circuits with gradient flow via autograd
6. Non-dimensionalization: PINN/QA-PINN operate in [0,1] normalized space
7. PDE residual metric: Physics compliance evaluated independently of reference solution
8. Unseen domain test: Measures generalization to unseen data distribution
9. Dual-mode domain classes: Work with real datasets OR synthetic sinusoidal setups
10. Output archiving: Previous runs automatically moved to history/
11. Anisotropic support: Solvers handle list-type k and alpha (Graphite, Carbon-Carbon)

---

## 12. Weaknesses

1. PennyLane not in requirements.txt — QA-PINN crashes on fresh install
2. 3D CFD is explicit (conditionally stable only) — warns but doesn't auto-fix
3. QA-PINN uses classical simulation — no real quantum hardware or noise models
4. PINN BC target uses IC at boundaries for all time — not exact for general BCs
5. CNN trained on narrow IC family — may fail on very different inputs
6. No gradient clipping — PINN may diverge in early epochs
7. 3D CNN decoder explodes for large grids (only works at 10×10×10)
8. Parallel workers disabled — workers=1 hardcoded despite config supporting 4
9. No model persistence — torch.save never called — retrains every run
10. ~25 fix_*.py debug scripts in root — code smell, should be cleaned up
11. src/ subfolders (cfd/, pinn/, etc.) contain only empty README stubs
12. DashboardGenerator in data_generator/visualization/ uses fake random data

---

## 13. Scientific Correctness Review

CORRECT:
✅ Heat equation formulation is physically correct
✅ Crank-Nicolson: r = α·dt/(2·dx²), unconditionally stable
✅ ADI: correct alternating x/y sweeps
✅ Analytical solutions for sinusoidal ICs are mathematically exact
✅ Von Neumann stability enforced for explicit solvers
✅ Temperature in Kelvin with physical validation >= 273.15 K
✅ PDE residual computed via finite differences — correct formulation
✅ PINN loss weighting (×10 for IC/BC) follows Raissi et al. best practice

QUESTIONABLE:
⚠️ PINN BC target: uses IC-interpolated values at boundary for ALL time
   → valid for zero Dirichlet only; incorrect for general transient BCs
⚠️ 3D CFD explicit stability: warning printed but no automatic dt correction
⚠️ QA-PINN "quantum advantage": default.qubit is classical — no quantum speedup verified
⚠️ Rectified 3D data uses Gaussian plume model (Green's function) — may not be
   consistent with the zero-Dirichlet BCs used in the benchmark pipeline
⚠️ CNN: BCs after t=0 are not captured since only IC is input

Material properties (spot-checked against literature):
- Copper k=398 W/mK [✓ correct ~400]
- Al 6061 k=167 [✓ correct]
- Ti-6Al-4V k=6.7 [✓ correct]
- Diamond k=2000 [✓ correct for natural diamond]
- LI-900 Silica k=0.048 [✓ correct for this space shuttle insulator]

---

## 14. Improvement Opportunities

Priority 1 (Blocking):
- Add pennylane to requirements.txt
- Add torch.save/load for trained model checkpoints
- Add --skip_training flag to load pre-trained models

Priority 2 (High):
- Enable parallel dataset generation (ProcessPoolExecutor, workers > 1)
- Add implicit 3D CFD (Douglas-Gunn ADI scheme) for unconditional stability
- Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
- Clean up ~25 fix_*.py scripts from root directory

Priority 3 (Medium):
- Add quantum noise models (qml device with shots parameter)
- Add FNO (Fourier Neural Operator) as a 6th solver
- Integrate TensorBoard or W&B for experiment logging
- Complete src/ module refactoring into planned subdirectories

Priority 4 (Long term):
- Streamlit web dashboard for interactive result exploration
- Support time-varying boundary conditions in all solvers
- GPU-accelerated CFD for large grids
- Radiation boundary conditions (Stefan-Boltzmann)
- Real quantum hardware integration (IBM Quantum via Qiskit)

---

## 15. Roadmap

Phase 1 (Stabilize):  Fix requirements.txt, add model save/load, unit tests, clean root
Phase 2 (Scale):      Enable parallel generation, implicit 3D CFD, batch inference
Phase 3 (Quantum):    Noise models, real hardware, IBM Quantum path
Phase 4 (Extend):     FNO/DeepONet solvers, non-homogeneous domains
Phase 5 (Production): Streamlit dashboard, MLflow tracking, GPU support, radiation BCs

---

Analysis complete. All 61 files and 22 subdirectories have been examined.
