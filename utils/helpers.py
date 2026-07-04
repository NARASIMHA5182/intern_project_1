import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]

def ensure_folder(path: Path) -> None:
    """Create a folder if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
