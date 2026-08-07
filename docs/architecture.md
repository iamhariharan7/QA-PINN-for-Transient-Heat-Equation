# Project Architecture

## Project modules
- **ml_core**: Contains PINN and QA-PINN architectures and Artifact Managers.
- **data_generator**: Contains the CFD numerical solvers and boundary condition handlers.
- **scripts**: Execution scripts for running experiments and generating reports.

## Data flow
Material Properties -> Numerical Solver -> Dataset Generation -> Preprocessing -> Model Training -> Output Generation.

## Solver architecture
Uses discrete finite-difference methods (FDM) applied to 1D, 2D, and 3D transient heat conduction equations.

## AI model pipeline
Loads processed .npz boundaries, converts to PyTorch tensors, optimizes via Adam/L-BFGS, and tracks via ArtifactManager.
