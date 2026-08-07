import os
import yaml

class ConfigManager:
    def __init__(self, config_path: str = "config/generator_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML file: {exc}")
                
        self._validate_config(config)
        return config

    def _validate_config(self, config: dict):
        # Basic validation of probabilities
        dim_probs = config.get("dimensions", {})
        total_dim_prob = dim_probs.get("prob_1d", 0) + dim_probs.get("prob_2d", 0) + dim_probs.get("prob_3d", 0)
        if abs(total_dim_prob - 1.0) > 1e-5:
            raise ValueError(f"Dimension probabilities must sum to 1.0, got {total_dim_prob}")

        split = config.get("dataset", {})
        total_split = split.get("train_split", 0) + split.get("val_split", 0) + split.get("test_split", 0)
        if abs(total_split - 1.0) > 1e-5:
            raise ValueError(f"Dataset splits must sum to 1.0, got {total_split}")
            
    def get(self, section: str, key: str = None):
        """Retrieve a section or a specific key from a section."""
        sec = self.config.get(section, {})
        if key:
            return sec.get(key)
        return sec
