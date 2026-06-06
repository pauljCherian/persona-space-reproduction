#turns .pt files into numpy arrays
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#how we laod a pt file into a numpy array. check if its a dictionary or not and resolve differently
def load_pt(path: Path | str) -> np.ndarray:
    """Load a single activation vector as a 1-D float32 ndarray."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("vector", obj.get("mean", next(iter(obj.values()))))
    #return the object as a numpy array 
    return obj.float().squeeze().numpy()

#load the matrix of all the roles in their activation space in a N by d matrix (276 by 4096 for llama 3.1 8B). 
#we can optionally get rid of the default assistant if we want too 
def load_role_matrix(vec_dir: Path | str, drop_default: bool = False):
    vec_dir = Path(vec_dir)
    files = sorted(f for f in vec_dir.glob("*.pt") if not (drop_default and f.stem == "default"))
    if not files:
        raise FileNotFoundError(f"No .pt role vectors in {vec_dir}")
    roles = [f.stem for f in files]
    X = np.stack([load_pt(f) for f in files])
    return X, roles
