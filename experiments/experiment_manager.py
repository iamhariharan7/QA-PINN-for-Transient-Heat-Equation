import os
import json
import shutil
import platform
import subprocess
import psutil
from datetime import datetime

class ExperimentManager:
    """
    Manages the lifecycle of an experiment, handling system profiling,
    git tracking, configuration snapshotting, and final report generation.
    """
    def __init__(self, root_dir="experiments"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
        self._ensure_readme()
        
        self.exp_id = self._generate_next_id()
        self.exp_dir = os.path.join(self.root_dir, f"experiment_{self.exp_id:06d}")
        os.makedirs(self.exp_dir, exist_ok=True)

    def _ensure_readme(self):
        readme_path = os.path.join(self.root_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write("# Experiments Directory\n\nAutomatically manages isolated experiment tracking, system profiling, and report generation.\n")

    def _generate_next_id(self) -> int:
        existing = [d for d in os.listdir(self.root_dir) if d.startswith("experiment_") and os.path.isdir(os.path.join(self.root_dir, d))]
        if not existing:
            return 1
        ids = [int(d.split("_")[1]) for d in existing]
        return max(ids) + 1

    def capture_system_info(self):
        """Snapshots hardware and python environment info."""
        try:
            import torch
            pt_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pt_version = "Not Installed"
            cuda_available = False

        sys_info = {
            "OS": platform.system(),
            "OS_Release": platform.release(),
            "Python_Version": platform.python_version(),
            "CPU": platform.processor(),
            "RAM_GB": round(psutil.virtual_memory().total / (1024**3), 2),
            "PyTorch_Version": pt_version,
            "CUDA_Available": cuda_available
        }
        
        with open(os.path.join(self.exp_dir, "system_information.json"), "w") as f:
            json.dump(sys_info, f, indent=4)

    def capture_git_info(self):
        """Snapshots Git branch and commit if available."""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "Unknown"
            branch = "Unknown"
            
        git_info = {
            "Commit_Hash": commit,
            "Branch": branch
        }
        with open(os.path.join(self.exp_dir, "git_information.json"), "w") as f:
            json.dump(git_info, f, indent=4)

    def snapshot_config(self, config_path: str):
        """Copies the running configuration into the experiment dir."""
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(self.exp_dir, "config_snapshot.yaml"))

    def initialize_experiment(self, metadata: dict, config_path: str = None):
        """Starts the experiment tracking process."""
        self.capture_system_info()
        self.capture_git_info()
        
        if config_path:
            self.snapshot_config(config_path)
            
        # Basic notes
        with open(os.path.join(self.exp_dir, "notes.md"), "w") as f:
            f.write("# Experiment Notes\n\nAdd manual observations here.\n")
            
        # Save overarching metadata
        meta = {
            "Experiment_ID": f"experiment_{self.exp_id:06d}",
            "Timestamp": datetime.now().isoformat(),
            **metadata
        }
        with open(os.path.join(self.exp_dir, "experiment_metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

    def generate_report(self, summary_data: dict):
        """Generates MD and HTML reports out-of-the-box."""
        md_content = f"# Experiment Report: experiment_{self.exp_id:06d}\n\n"
        md_content += f"**Timestamp**: {datetime.now().isoformat()}\n\n"
        
        for k, v in summary_data.items():
            md_content += f"## {k}\n"
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    md_content += f"- **{sub_k}**: {sub_v}\n"
            else:
                md_content += f"{v}\n"
            md_content += "\n"
            
        md_path = os.path.join(self.exp_dir, "Experiment_Report.md")
        with open(md_path, "w") as f:
            f.write(md_content)
            
        # Basic HTML equivalent
        html_content = f"<html><body><h1>Experiment Report: experiment_{self.exp_id:06d}</h1>"
        html_content += md_content.replace("\n\n", "<br><br>").replace("\n", "<br>")
        html_content += "</body></html>"
        
        with open(os.path.join(self.exp_dir, "Experiment_Report.html"), "w") as f:
            f.write(html_content)

        # JSON Summary
        with open(os.path.join(self.exp_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary_data, f, indent=4)
            
        return self.exp_dir
