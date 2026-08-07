import json
import os
import pandas as pd

def generate_report(registry_path="artifacts/registry/experiment_registry.json", output_dir="outputs/Present_Output"):
    if not os.path.exists(registry_path):
        print(f"Registry not found: {registry_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    data = []
    for run_id, info in registry.items():
        data.append({
            "Run_ID": info.get("experiment_id", run_id),
            "Model": info.get("model", "Unknown"),
            "Dataset": info.get("dataset", "Unknown"),
            "Material": info.get("material", "Unknown"),
            "Dimension": info.get("dimension", "Unknown"),
            "Status": info.get("status", "Unknown"),
            "RMSE": info.get("rmse", "N/A")
        })
        
    if not data:
        print("No data to report.")
        return
        
    df = pd.DataFrame(data)
    
    # Generate CSV
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate LaTeX
    latex_path = os.path.join(output_dir, "experiment_summary.tex")
    tex_str = df.to_latex(index=False, caption="Experiment Results Summary", label="tab:exp_summary")
    with open(latex_path, "w") as f:
        f.write(tex_str)
        
    # Generate JSON
    json_path = os.path.join(output_dir, "experiment_summary.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Report generated successfully in {output_dir}.")
    print(f"Includes: CSV, LaTeX (.tex), and JSON formats.")

if __name__ == "__main__":
    generate_report()
