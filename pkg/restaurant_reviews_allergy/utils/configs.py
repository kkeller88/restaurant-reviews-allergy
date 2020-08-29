from pathlib import Path
import os
import json

REPO_BASE = Path(__file__).parents[3]
MODEL_CONFIG_BASE = os.path.join(REPO_BASE, 'model_config')

def read_model_config(name, dir=MODEL_CONFIG_BASE):
    path = os.path.join(dir, name)
    with open(path, 'r') as file:
        model_config = json.load(file)
    return model_config
