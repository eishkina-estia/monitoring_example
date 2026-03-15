import os
import yaml

# project root (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# full config path
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yml')

def get_full_path(rel_path):
    return os.path.normpath(os.path.join(ROOT_DIR, rel_path))

def resolve_paths(obj):
    """
    Recursively convert all relative paths inside the config
    to absolute paths based on ROOT_DIR.
    """
    if isinstance(obj, dict):
        return {k: resolve_paths(v) for k, v in obj.items()}

    if isinstance(obj, str):
        return get_full_path(obj)

    return obj


with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

CONFIG["paths"] = resolve_paths(CONFIG["paths"])
