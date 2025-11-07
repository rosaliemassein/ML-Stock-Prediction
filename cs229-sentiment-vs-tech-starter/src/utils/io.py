
import os, yaml

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
