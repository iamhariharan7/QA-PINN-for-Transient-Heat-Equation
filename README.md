# CFD Dataset Generation & PINN Framework

## Project Overview
A complete, modular, scalable, and research-grade Python framework designed to generate high-quality synthetic datasets for transient heat conduction problems. It serves as the ground-truth for training CNN, PINN, and QA-PINN surrogate models.

## Research Objective
To provide a highly reproducible pipeline for benchmarking Physics-Informed Neural Networks against traditional Computational Fluid Dynamics solvers for heat transfer applications.

## Architecture Explanation
- **config/**: Global YAML configurations.
- **data/**: Immutable, reusable generated CFD datasets.
- **models/**: Trained surrogate models (CNN, PINN, QA-PINN) and checkpoints.
- **experiments/**: Experiment tracking and system metadata.
- **outputs/**: Evaluation results and comparison dashboards.
- **src/**: Core reusable implementations.

## Installation Instructions
`ash
git clone <repo_url>
cd project
pip install -r requirements.txt
# Or using conda:
conda env create -f environment.yml
conda activate cfd-pinn
`

## Usage Examples
`ash
python scripts/run_experiment.py
python scripts/generate_report.py
`

## Dataset Generation Workflow
Material Database -> Configuration -> CFD Solver -> Dataset Generation -> Dataset Validation -> Train/Val/Test Split

## Model Training Workflow
Dataset Loader -> CNN/PINN/QA-PINN Training -> Model Saving -> Inference -> Output Archiving

## Supported Materials
Supports 14 industrial materials including Copper, Aluminium 6061, Mild Steel, Stainless Steel 304, Titanium Ti-6Al-4V, and more.

## Supported Simulations
- 1D Line Heat Conduction
- 2D Square Heat Conduction
- 3D Cube Heat Conduction

## LARGE FILE MANAGEMENT POLICY
The repository must remain lightweight and GitHub-friendly. Do not commit large generated files directly into Git.
Use appropriate solutions: Git LFS, External dataset hosting, Dataset download scripts, or Release assets.

## Citation Information
Please refer to the CITATION.cff file when citing this framework in academic publications.
