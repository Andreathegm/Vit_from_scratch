import yaml
from pathlib import Path
from box import ConfigBox

def _load(file_name):
    try:
        with open(file_name, "r") as f:
            content = yaml.safe_load(f)
            return ConfigBox(content)
        
    except Exception as e:
        raise RuntimeError(f"Impossible to load the conf. file {e}")

# instantiate a global variable
Hyperparams = _load("hyperparameters.yaml")