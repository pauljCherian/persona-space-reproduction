#config file for getting models and the paths to models that we want to test
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

#these are the models we want to test and the filepaths to get to them
# we also mark the canonical middle layer we want to extract at
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

#this is where we'll keep the paths to all of our models
DATA_ROOT = Path(os.environ.get("PHASE_H_DATA_ROOT", REPO_ROOT / "data" / "comparison")).resolve()

#takes a model and ges the path to the model
def model_dir(tag: str) -> Path:
    if tag not in MODELS:
        raise KeyError(f"Unknown model tag {tag!r}; configured: {list(MODELS)}")
    return DATA_ROOT / MODELS[tag]["path"]
