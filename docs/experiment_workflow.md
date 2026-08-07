# Experiment Workflow

Details how ArtifactManager and ExperimentTracker isolate runs.

## DATABASE STORAGE POLICY

The experiment database must store only lightweight metadata.

The database must NOT store:
- Large datasets
- CSV files
- NPZ files
- HDF5 files
- Images
- Animations
- Videos
- Model weights

The database should store only:
- Experiment ID
- File paths
- Dataset location
- Model location
- Metrics
- Configuration references
- Runtime information
- Hardware information
- Version information

Large files must remain stored in their dedicated directories.
The database acts as an index and experiment tracking system, not as a file storage system.
