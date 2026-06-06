#utilities for steering
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent

#define our models 
MODELS: dict[str, dict] = {
    "llama8b": {
        "model_id":   "meta-llama/Llama-3.1-8B-Instruct",
        "path":       "llama-3.1-8b",
        "layer":      16,
        "hidden_dim": 4096,
    },
    "qwen7b": {
        "model_id":   "Qwen/Qwen2.5-7B-Instruct",
        "path":       "qwen2.5-7b",
        "layer":      14,
        "hidden_dim": 3584,
    },
    "dolphin8b": {
        "model_id":   "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        "path":       "dolphin3-llama3.1-8b",
        "layer":      16,
        "hidden_dim": 4096,
    },
}

# Traits from the canonical persona_vectors' trait list; each matches a JSON in the third_party/persona_vectors/data_generation/trait_data_extract/
# These are the traits that were actually computed in the Chen et al paper
TRAITS: list[str] = [
    "evil", "sycophantic", "hallucinating", "humorous",
    "apathetic", "impolite", "optimistic",
    "self_aggrandizement", "theatricality",
    "caution", "verbosity", "open_mindedness",
    "agency", "intellectual_honesty", "warmth",
    "anti_assistant",
]

_REPO = REPO_ROOT.parents[1]
DATA_ROOT = Path(os.environ.get("PHASE_K_DATA_ROOT", _REPO / "data" / "steering")).resolve()

#this was the previous directory that i had used for steering
def baseline_dir(tag: str) -> Path:
    """Return the symlinked baseline dir (read-only, lives in prior repo)."""
    return DATA_ROOT / "baseline" / tag

#we save steering vectors in two ways so need to index for both (addition addition is default additive steering, but we can save any vector as a type of steering vector in this format)
def steered_dir(tag: str, vector_name: str, alpha: float, intervention: str = "addition") -> Path:
    """data/<DATA_ROOT>/steered/<tag>/<vector_name>/alpha_<a>/         (addition; default)
    data/<DATA_ROOT>/steered/<tag>/<vector_name>/<intervention>_coef_<a>/  (ablation, etc.)"""
    if intervention == "addition":
        return DATA_ROOT / "steered" / tag / vector_name / f"alpha_{alpha:g}"
    return DATA_ROOT / "steered" / tag / vector_name / f"{intervention}_coef_{alpha:g}"

#get the paths to all the trait vectors
def trait_vector_path(tag: str, trait: str) -> Path:
    return DATA_ROOT / "vectors_steering" / tag / f"{trait}.pt"

#path to control random vectors
def control_vector_path(tag: str, vector_name: str) -> Path:
    return DATA_ROOT / "vectors_steering" / tag / "controls" / f"{vector_name}.pt"

#path to the persona space (pca basis) if we save it
def pca_basis_path(tag: str) -> Path:
    return DATA_ROOT / "pca_basis" / f"{tag}_basis.pt"

# extract a tensor from an object if its a dict, or just the raw tensor
def _extract_tensor(obj) -> torch.Tensor:
    if isinstance(obj, dict):
        obj = obj.get("vector", obj.get("mean", next(iter(obj.values()))))
    return obj

#load project per model role vectors (all 276 as a matrix of (276, d) and return a list of the sorted names for each of the roels
def load_baseline_role_matrix(tag: str) -> tuple[np.ndarray, list[str]]:
    vec_dir = baseline_dir(tag) / "vectors"
    files = sorted(vec_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No role vectors found in {vec_dir}")
    roles, vecs = [], []
    for f in files:
        roles.append(f.stem)
        v = _extract_tensor(torch.load(f, map_location="cpu", weights_only=False))
        vecs.append(v.float().squeeze().numpy())
    return np.stack(vecs), roles


# load the default assistant vector
def load_baseline_default(tag: str) -> np.ndarray:
    v = _extract_tensor(torch.load(baseline_dir(tag) / "default.pt",
                                   map_location="cpu", weights_only=False))
    return v.float().squeeze().numpy()
