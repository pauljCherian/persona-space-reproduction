"""Shared activation-vector loaders for the figure scripts.

Consolidates the per-figure loaders that were previously duplicated across the
three source projects. Behaviour is preserved exactly:

  * Figure 1 stacks all 276 vectors (275 roles + ``default``) and separates the
    default by name  -> ``load_role_matrix(dir, drop_default=False)``.
  * Figure 2 stacks all 276 vectors and runs PCA over the whole set
    -> ``load_role_matrix(dir, drop_default=False)``.
  * Figures 3 / 5 build the baseline cloud from the 275 roles only, loading the
    (baseline or steered) default separately
    -> ``load_role_matrix(dir, drop_default=True)`` + ``load_pt(default_path)``.

Every .pt may store either a bare tensor or a dict like ``{"vector": tensor}``
(also seen: ``{"mean": tensor}``); ``load_pt`` normalises both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_pt(path: Path | str) -> np.ndarray:
    """Load a single activation vector as a 1-D float32 ndarray."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("vector", obj.get("mean", next(iter(obj.values()))))
    return obj.float().squeeze().numpy()


def load_role_matrix(vec_dir: Path | str, drop_default: bool = False):
    """Stack every .pt in ``vec_dir`` into an (N, d) matrix.

    Returns ``(X, roles)`` with ``roles`` sorted alphabetically and the rows of
    ``X`` aligned to it. With ``drop_default=True`` the ``default.pt`` vector is
    excluded (used for the baseline 275-role clouds in figures 3 and 5).
    """
    vec_dir = Path(vec_dir)
    files = sorted(f for f in vec_dir.glob("*.pt") if not (drop_default and f.stem == "default"))
    if not files:
        raise FileNotFoundError(f"No .pt role vectors in {vec_dir}")
    roles = [f.stem for f in files]
    X = np.stack([load_pt(f) for f in files])
    return X, roles
