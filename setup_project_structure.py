import os
from pathlib import Path

base_dir = Path("e:/Quan/project")

# Project base structure
files_to_create = [
    ".vscode/settings.json",
    "config/default_config.yaml",
    "data/materials.xlsx",
    "docs/README.md",
    "scripts/run_experiment.py",
    "scripts/update_run_experiment.py",
    "src/actual_solution.py",
    "src/actual_solution_2d.py",
    "src/actual_solution_3d.py",
    "src/heat_equation.py",
    "src/heat_equation_2d.py",
    "src/heat_equation_3d.py",
    "src/cfd_solver.py",
    "src/cfd_solver_2d.py",
    "src/cfd_solver_3d.py",
    "src/cnn_model.py",
    "src/cnn_model_2d.py",
    "src/cnn_model_3d.py",
    "src/pinn_model.py",
    "src/pinn_model_2d.py",
    "src/pinn_model_3d.py",
    "src/qa_pinn_model.py",
    "src/qa_pinn_model_2d.py",
    "src/qa_pinn_model_3d.py",
    "src/material_loader.py",
    "src/metrics.py",
    "src/plotting.py",
    "src/report_generator.py",
    "requirements.txt",
    "columns.json",
    ".gitignore",
    "LICENSE",
]

for f in files_to_create:
    p = base_dir / f
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

# Output structure for a dummy material
mat_dir = base_dir / "outputs" / "material_name_placeholder"

methods = ["Actual", "CFD", "CNN", "PINN", "QA-PINN"]
dims = ["1D", "2D", "3D"]

for method in methods:
    for dim in dims:
        p = mat_dir / method / dim
        p.mkdir(parents=True, exist_ok=True)
        (p / f"Heat Map ({dim}).txt").touch(exist_ok=True)

comp_metrics = [
    "Heat Map Comparison",
    "Absolute Error Map",
    "PDE Residual Error",
    "Relative L2 Error",
    "Training Loss",
    "RMSE",
    "Training Time",
    "Number of Trainable Parameters",
    "Memory Requirements",
    "Explainability or Interpretability Metrics",
    "Performance on Unseen Domains"
]

for dim in dims:
    p = mat_dir / "Comparison" / dim
    p.mkdir(parents=True, exist_ok=True)
    for metric in comp_metrics:
        # creating as directories since they might contain multiple plots/files, or files. Let's do files.
        (p / f"{metric}.txt").touch(exist_ok=True)

final_comp = mat_dir / "Final Combined Comparison"
final_comp.mkdir(parents=True, exist_ok=True)

for dim in dims:
    p = final_comp / "Heat Map Comparison" / dim
    p.mkdir(parents=True, exist_ok=True)
    for method in methods:
        (p / f"{method}.txt").touch(exist_ok=True)

quant_metrics = ["RMSE", "Relative L2 Error", "PDE Residual Error", "Absolute Error"]
p = final_comp / "Quantitative Metrics"
p.mkdir(parents=True, exist_ok=True)
for metric in quant_metrics:
    (p / f"{metric}.txt").touch(exist_ok=True)

comp_perf = ["Training Time", "Inference Time", "Memory Requirements", "Number of Trainable Parameters"]
p = final_comp / "Computational Performance"
p.mkdir(parents=True, exist_ok=True)
for metric in comp_perf:
    (p / f"{metric}.txt").touch(exist_ok=True)

model_analysis = ["Training Loss", "Explainability or Interpretability", "Performance on Unseen Domains"]
p = final_comp / "Model Analysis"
p.mkdir(parents=True, exist_ok=True)
for metric in model_analysis:
    (p / f"{metric}.txt").touch(exist_ok=True)

overall_ranking = ["Accuracy", "Computational Efficiency", "Generalization", "Scalability (1D → 2D → 3D)", "Robustness"]
p = final_comp / "Overall Ranking"
p.mkdir(parents=True, exist_ok=True)
for metric in overall_ranking:
    # Handle the arrow symbol safely
    metric_safe = metric.replace("→", "-")
    (p / f"{metric_safe}.txt").touch(exist_ok=True)

final_conc = ["Best Overall Method", "Best Accuracy", "Fastest Method", "Lowest Memory Usage", "Most Physically Consistent", "Best Generalization", "Future Work"]
p = final_comp / "Final Conclusion"
p.mkdir(parents=True, exist_ok=True)
for metric in final_conc:
    (p / f"{metric}.txt").touch(exist_ok=True)

print("Project structure created successfully.")
