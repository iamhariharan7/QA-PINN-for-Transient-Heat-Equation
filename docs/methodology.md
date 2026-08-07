# Methodology

## Heat conduction equations
The project uses the standard heat diffusion equation: $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$

## CFD methodology
Employs explicit and implicit Finite Difference Methods (FDM) to generate ground-truth high-fidelity simulations over complex materials.

## Dataset generation process
Simulations are run across parameter sweeps (materials, boundary conditions). Results are saved to disk in HDF5 or NPZ formats.

## CNN methodology
Uses 2D and 3D convolutional layers to predict spatial heat distribution directly from boundary condition inputs.

## PINN methodology
Physics-Informed Neural Networks. The loss function contains both data-driven MSE loss and a physics-residual loss (PDE constraint).

## QA-PINN methodology
Quantum-Assisted PINNs integrate PennyLane parameterized quantum circuits (Ansatz) inside the classical neural network to explore complex Hilbert spaces during gradient descent.
