import os
import json
import pickle
import hashlib
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

class ArtifactManager:
    """
    Manages the lifecycle of Machine Learning assets (Models, Scalers, Preprocessors).
    Responsible for checkpointing, exporting, hashing, version control, and maintaining the central artifacts repository.
    """
    def __init__(self, root_dir: str = "artifacts"):
        self.root_dir = root_dir
        self.supported_architectures = ["cnn", "pinn", "qa_pinn"]
        
    def _get_arch_path(self, arch: str, base_folder: str, version: str = None) -> str:
        arch = arch.lower().replace('-', '_')
        if arch not in self.supported_architectures:
            raise ValueError(f"Architecture '{arch}' not supported. Choose from {self.supported_architectures}")
            
        if version and base_folder == "models":
            # Version control routing: artifacts/models/<arch>/<version>/
            path = os.path.join(self.root_dir, base_folder, arch, version)
        else:
            path = os.path.join(self.root_dir, base_folder, arch)
            
        os.makedirs(path, exist_ok=True)
        return path
        
    def _generate_hash(self, filepath: str) -> str:
        """Computes SHA-256 hash of a file for integrity verification."""
        if not os.path.exists(filepath):
            return ""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_checkpoint(self, arch: str, run_name: str, epoch: int, model, optimizer, scheduler=None, loss: float = 0.0):
        """Saves a training checkpoint and computes its hash."""
        if torch is None: return ""
        chkpt_dir = self._get_arch_path(arch, "checkpoints")
        run_chkpt_dir = os.path.join(chkpt_dir, run_name)
        os.makedirs(run_chkpt_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        target_file = os.path.join(run_chkpt_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, target_file)
        torch.save(checkpoint, os.path.join(run_chkpt_dir, "latest.pth"))
        
        return self._generate_hash(target_file)
        
    def load_checkpoint(self, arch: str, run_name: str, model, optimizer=None, scheduler=None, filename: str = "latest.pth"):
        """Loads a training checkpoint."""
        if torch is None: return 0, 0.0
        chkpt_path = os.path.join(self.root_dir, "checkpoints", arch.lower().replace('-','_'), run_name, filename)
        
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {chkpt_path}")
            
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)

    def save_final_model(self, arch: str, version: str, model, config: dict, metrics: dict, dataset_ref: dict, metadata: dict, is_best: bool = False):
        """
        Saves the model and its companion artifacts directly into the local version folder.
        artifacts/models/<arch>/<version>/
        """
        if torch is None: return ""
        
        # Route directly to artifacts/models/<ARCH>/<VERSION>/
        model_dir = self._get_arch_path(arch, "models", version=version)
        
        # 1. Save Model Weights
        target_file = os.path.join(model_dir, "model.pt")
        torch.save(model.state_dict(), target_file)
        model_hash = self._generate_hash(target_file)
        
        # 2. Save Companion Artifacts
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(model_dir, "dataset_reference.json"), "w") as f:
            json.dump(dataset_ref, f, indent=4)
            
        metadata['model_hash'] = model_hash
        metadata['creation_timestamp'] = datetime.now().isoformat()
        metadata['version'] = version
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        if is_best:
            self._update_latest_model(arch, target_file)
            
        return model_hash, target_file

    def save_scaler(self, scaler_obj, filename: str):
        """Saves scikit-learn or custom scaler objects."""
        scaler_dir = os.path.join(self.root_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        target = os.path.join(scaler_dir, filename)
        with open(target, "wb") as f:
            pickle.dump(scaler_obj, f)
        return self._generate_hash(target)

    def save_preprocessor(self, preprocessor_obj, filename: str):
        """Saves preprocessing pipelines."""
        prep_dir = os.path.join(self.root_dir, "preprocessors")
        os.makedirs(prep_dir, exist_ok=True)
        target = os.path.join(prep_dir, filename)
        with open(target, "wb") as f:
            pickle.dump(preprocessor_obj, f)
        return self._generate_hash(target)

    def export_model(self, arch: str, run_name: str, model, formats: list = ['onnx'], dummy_input=None):
        """Exports model to deployment formats."""
        if torch is None: return {}
        export_base = os.path.join(self.root_dir, "exports")
        hashes = {}
        
        if 'onnx' in formats and dummy_input is not None:
            onnx_dir = os.path.join(export_base, "onnx", arch)
            os.makedirs(onnx_dir, exist_ok=True)
            try:
                target = os.path.join(onnx_dir, f"{run_name}.onnx")
                torch.onnx.export(
                    model, dummy_input, target,
                    export_params=True, opset_version=14, do_constant_folding=True,
                    input_names=['input'], output_names=['output']
                )
                hashes['onnx'] = self._generate_hash(target)
            except Exception as e:
                print(f"Skipping ONNX export: {e}")
                
        if 'torchscript' in formats and dummy_input is not None:
            ts_dir = os.path.join(export_base, "torchscript", arch)
            os.makedirs(ts_dir, exist_ok=True)
            target = os.path.join(ts_dir, f"{run_name}.ts")
            traced = torch.jit.trace(model, dummy_input)
            traced.save(target)
            hashes['torchscript'] = self._generate_hash(target)
            
        return hashes

    def register_model(self, arch: str, run_name: str, metadata: dict, status: str = "Training"):
        """Central lifecycle registration (Training -> Validation -> Export -> Deployment Ready)."""
        reg_dir = os.path.join(self.root_dir, "registry")
        os.makedirs(reg_dir, exist_ok=True)
        registry_path = os.path.join(reg_dir, "model_registry.json")
        
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
                
        model_key = f"{arch}_{run_name}"
        if model_key not in registry:
            registry[model_key] = {
                "Architecture": arch,
                "Run_Name": run_name,
                "Creation_Date": datetime.now().isoformat()
            }
            
        registry[model_key]["Modification_Date"] = datetime.now().isoformat()
        registry[model_key]["Status"] = status
        registry[model_key].update(metadata)
        
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)

    def _update_latest_model(self, arch: str, target_file_path: str):
        """
        Updates the JSON pointer to the best performing model.
        This achieves zero-duplication without requiring OS-level symlinks.
        """
        reg_dir = os.path.join(self.root_dir, "registry")
        os.makedirs(reg_dir, exist_ok=True)
        latest_path = os.path.join(reg_dir, "latest_models.json")
        
        latest_data = {}
        if os.path.exists(latest_path):
            with open(latest_path, "r") as f:
                latest_data = json.load(f)
                
        # Use relative path dynamically pointed to by JSON
        latest_data[arch.lower()] = {
            "Latest_Model_Path": target_file_path.replace("\\", "/"),
            "Timestamp": datetime.now().isoformat()
        }
        
        with open(latest_path, "w") as f:
            json.dump(latest_data, f, indent=4)

    def save_reproducibility_metadata(self):
        """Snaps system profile and dependency data for strict reproducibility."""
        meta_dir = os.path.join(self.root_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        
        # OS Info
        env_info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": platform.system(),
            "OS_Release": platform.release(),
            "Python_Version": platform.python_version(),
            "CPU": platform.processor()
        }
        
        if psutil:
            env_info["RAM_GB"] = round(psutil.virtual_memory().total / (1024**3), 2)
            
        if torch:
            env_info["PyTorch_Version"] = torch.__version__
            env_info["CUDA_Available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["GPU_Name"] = torch.cuda.get_device_name(0)
                
        with open(os.path.join(meta_dir, "environment_info.json"), "w") as f:
            json.dump(env_info, f, indent=4)
            
        # Try Git
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            with open(os.path.join(meta_dir, "git_info.json"), "w") as f:
                json.dump({"Commit": commit, "Branch": branch}, f, indent=4)
        except Exception:
            pass

class ArtifactExperimentTracker:
    def __init__(self, root_dir: str = "artifacts"):
        self.root_dir = root_dir
        self.exp_dir = os.path.join(root_dir, "experiments")
        self.registry_dir = os.path.join(root_dir, "registry")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.registry_dir, exist_ok=True)
        
    def _get_next_run_id(self) -> str:
        runs = [d for d in os.listdir(self.exp_dir) if d.startswith("run_")]
        if not runs:
            return "run_001"
        ids = []
        for r in runs:
            try:
                ids.append(int(r.split("_")[1]))
            except:
                pass
        next_id = max(ids) + 1 if ids else 1
        return f"run_{next_id:03d}"
        
    def create_experiment(self, exp_name: str, config: dict, dataset_info: dict, model_info: dict, hardware_info: dict, quantum_config: dict = None) -> str:
        run_id = self._get_next_run_id()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{run_id}_{timestamp}"
        run_path = os.path.join(self.exp_dir, folder_name)
        os.makedirs(run_path, exist_ok=True)
        os.makedirs(os.path.join(run_path, "loss_curves"), exist_ok=True)
        
        # Save JSONs
        with open(os.path.join(run_path, "experiment_config.json"), "w") as f:
            json.dump({"Experiment_ID": run_id, "Name": exp_name, "Date": timestamp, "Status": "Running", **config}, f, indent=4)
        with open(os.path.join(run_path, "dataset_reference.json"), "w") as f:
            json.dump(dataset_info, f, indent=4)
        with open(os.path.join(run_path, "hyperparameters.json"), "w") as f:
            json.dump(model_info, f, indent=4)
            
        with open(os.path.join(run_path, "README.md"), "w") as f:
            f.write(f"# {exp_name}\nRun ID: {run_id}\nDate: {timestamp}\n")
            
        if quantum_config and model_info.get("Model Type") == "QA-PINN":
            with open(os.path.join(run_path, "quantum_configuration.json"), "w") as f:
                json.dump(quantum_config, f, indent=4)
                
        # Phase 12 Extensions
        self._capture_environment(run_path, hardware_info)
        self._capture_git_state(run_path)
        self._generate_source_reference(run_path, config.get("Project", "CFD_PINN_QAPINN_Framework"))
            
        self._update_registry(run_id, folder_name, exp_name, dataset_info, model_info, status="Running")
        return folder_name

    def _capture_environment(self, run_path: str, hardware_info: dict):
        env_dir = os.path.join(run_path, "environment")
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, "hardware_info.json"), "w") as f:
            json.dump(hardware_info, f, indent=4)
        import sys
        with open(os.path.join(env_dir, "python_version.txt"), "w") as f:
            f.write(sys.version)
        try:
            installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
            with open(os.path.join(env_dir, "installed_packages.txt"), "w") as f:
                f.write(installed)
            with open(os.path.join(env_dir, "requirements_snapshot.txt"), "w") as f:
                f.write(installed)
        except:
            pass
        if torch:
            with open(os.path.join(env_dir, "cuda_info.txt"), "w") as f:
                f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
                if torch.cuda.is_available():
                    f.write(f"CUDA Version: {torch.version.cuda}\n")
                    f.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")

    def _capture_git_state(self, run_path: str):
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            status = subprocess.check_output(["git", "status", "--short"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            with open(os.path.join(run_path, "git_commit.txt"), "w") as f:
                f.write(f"Commit: {commit}\nBranch: {branch}\nDate: {datetime.now().isoformat()}\n\nStatus:\n{status}")
        except:
            with open(os.path.join(run_path, "git_commit.txt"), "w") as f:
                f.write("Git not available. Tracking via timestamps only.\n")

    def _generate_source_reference(self, run_path: str, project_name: str):
        ref = {
            "project": project_name,
            "version": "1.0",
            "training_script": "train.py",
            "dataset_generator": "generator_v2",
            "git_branch": "unknown",
            "git_commit": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        try:
            ref["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            ref["git_branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except:
            pass
        with open(os.path.join(run_path, "source_code_reference.json"), "w") as f:
            json.dump(ref, f, indent=4)
        
    def _update_registry(self, run_id: str, folder_name: str, exp_name: str, dataset_info: dict, model_info: dict, status: str, metrics: dict = None):
        reg_path = os.path.join(self.registry_dir, "experiment_registry.json")
        registry = {}
        if os.path.exists(reg_path):
            with open(reg_path, "r") as f:
                registry = json.load(f)
                
        registry[run_id] = {
            "experiment_id": run_id,
            "folder": folder_name,
            "name": exp_name,
            "model": model_info.get("Model Type", "Unknown"),
            "dataset": dataset_info.get("Dataset Version", "Unknown"),
            "material": dataset_info.get("Material", "Unknown"),
            "dimension": dataset_info.get("Dimension", "Unknown"),
            "status": status,
            "last_updated": datetime.now().isoformat()
        }
        if metrics:
            registry[run_id].update(metrics)
            
        with open(reg_path, "w") as f:
            json.dump(registry, f, indent=4)
            
    def update_status(self, folder_name: str, status: str, metrics: dict = None):
        run_id = folder_name.split("_")[0] + "_" + folder_name.split("_")[1]
        reg_path = os.path.join(self.registry_dir, "experiment_registry.json")
        if os.path.exists(reg_path):
            with open(reg_path, "r") as f:
                registry = json.load(f)
            if run_id in registry:
                registry[run_id]["status"] = status
                registry[run_id]["last_updated"] = datetime.now().isoformat()
                if metrics:
                    registry[run_id].update(metrics)
                with open(reg_path, "w") as f:
                    json.dump(registry, f, indent=4)
                    
        # Update local config status
        conf_path = os.path.join(self.exp_dir, folder_name, "experiment_config.json")
        if os.path.exists(conf_path):
            with open(conf_path, "r") as f:
                conf = json.load(f)
            conf["Status"] = status
            with open(conf_path, "w") as f:
                json.dump(conf, f, indent=4)

    def log_metrics(self, folder_name: str, metrics: dict):
        run_path = os.path.join(self.exp_dir, folder_name)
        with open(os.path.join(run_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def link_artifacts(self, folder_name: str, model_path: str, checkpoint_dir: str = ""):
        run_path = os.path.join(self.exp_dir, folder_name)
        with open(os.path.join(run_path, "model_reference.json"), "w") as f:
            json.dump({"Final_Model_Path": model_path}, f, indent=4)
        if checkpoint_dir:
            with open(os.path.join(run_path, "checkpoints_reference.json"), "w") as f:
                json.dump({"Checkpoint_Directory": checkpoint_dir}, f, indent=4)
