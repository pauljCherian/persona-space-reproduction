"""Config + paths for the role-vector extraction pipeline.

PHASE_H_DATA_ROOT overrides the output root (default: <repo>/data/comparison).
pipeline.sh shells out to this for params.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

MODELS: dict[str, dict] = {
    "llama8b": {
        "model_id":   "meta-llama/Llama-3.1-8B-Instruct",
        "path":       "llama-3.1-8b",
        "layer":      16,           # N/2 for 32-layer Llama-3.1
        "hidden_dim": 4096,
    },
    "qwen7b": {
        "model_id":   "Qwen/Qwen2.5-7B-Instruct",
        "path":       "qwen2.5-7b",
        "layer":      14,           # N/2 for 28-layer Qwen-2.5-7B
        "hidden_dim": 3584,
    },
    "dolphin8b": {
        "model_id":   "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        "path":       "dolphin3-llama3.1-8b",
        "layer":      16,           # N/2 for 32-layer Llama-3.1 base
        "hidden_dim": 4096,
    },
}

DATA_ROOT = Path(os.environ.get("PHASE_H_DATA_ROOT", REPO_ROOT / "data" / "comparison")).resolve()


def model_dir(tag: str) -> Path:
    if tag not in MODELS:
        raise KeyError(f"Unknown model tag {tag!r}; configured: {list(MODELS)}")
    return DATA_ROOT / MODELS[tag]["path"]
