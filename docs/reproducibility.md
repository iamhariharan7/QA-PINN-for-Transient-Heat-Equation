# Reproducibility

## Environment setup
Strictly bound by 
equirements.txt and environment.yml. The ArtifactManager logs the exact pip freeze state upon every run.

## Random seeds
Global random seeds are strictly enforced across NumPy, PyTorch, and PennyLane at the beginning of 
un_experiment.py.

## Configuration management
JSON/YAML dictionaries control all hyperparameter sweeps and are dumped into the run folder for historical preservation.

## Experiment tracking
A local SQLite database tracks the exact git commit, python version, and metrics for every experiment, guaranteeing 100% traceability.
