#!/bin/bash
# Extract per-role activation vectors for one model via the assistant-axis
# pipeline. NO judge step (filter rate ~100% for these instruct models).
#
# Usage:   PHASE_H_DATA_ROOT=<out_root> ./pipeline.sh <model_tag>
#   <model_tag> is a key in _common.py's MODELS dict.
#   Output goes to $PHASE_H_DATA_ROOT/<MODELS[tag]['path']>/.
#
# Steps:
#   1. vLLM generation: 276 roles × (prompt variants × 240 questions) → responses/<role>.jsonl
#   2. Activation extraction at layer L (mean over assistant tokens) → activations/<role>.pt
#   3. Unfiltered role vectors (mean over rollouts) → vectors/<role>.pt
#   4. Lu-style assistant axis (default − mean(roles)) → axis.pt
#   5. Default activation vector → default.pt
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: PHASE_H_DATA_ROOT=<out> $0 <model_tag>" >&2
    exit 2
fi

TAG="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"   # vLLM fraction of GPU memory (lower on shared cards)

read MODEL_ID OUTPUT LAYER HIDDEN_DIM <<<"$(
    cd "$REPO_ROOT" && "$PYTHON" -c "
import sys
from _common import MODELS, model_dir
tag = '$TAG'
if tag not in MODELS:
    sys.exit(f'Unknown model tag {tag!r}; configured tags: {list(MODELS)}')
m = MODELS[tag]
print(m['model_id'], model_dir(tag), m['layer'], m['hidden_dim'])
"
)"

# Locate the assistant-axis library (pip-installed editable from third_party/assistant_axis).
PIPELINE_DIR="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/pipeline")')"
ROLES_DIR="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/roles/instructions")')"
QUESTIONS_FILE="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/extraction_questions.jsonl")')"

mkdir -p "$OUTPUT"

echo "=== Pipeline: tag=$TAG ==="
echo "  model_id:   $MODEL_ID"
echo "  output:     $OUTPUT"
echo "  layer:      $LAYER   hidden_dim: $HIDDEN_DIM"

echo "=== Step 1/5: Generate responses (vLLM) ==="
"$PYTHON" "$PIPELINE_DIR/1_generate.py" \
    --model "$MODEL_ID" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT/responses" \
    --question_count 240 \
    --max_tokens 512 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM_UTIL"

echo "=== Step 2/5: Extract activations at layer $LAYER ==="
"$PYTHON" "$PIPELINE_DIR/2_activations.py" \
    --model "$MODEL_ID" \
    --responses_dir "$OUTPUT/responses" \
    --output_dir "$OUTPUT/activations" \
    --layers "$LAYER" \
    --batch_size 32

echo "=== Step 3/5: Compute unfiltered role vectors (no judge) ==="
"$PYTHON" - "$OUTPUT/activations" "$OUTPUT/vectors" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
n = 0
for af in sorted(act_dir.glob("*.pt")):
    role = af.stem
    activations = torch.load(af, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()])
    mean_vec = stacked.mean(dim=0)
    torch.save({"vector": mean_vec, "type": "mean", "role": role}, out_dir / f"{role}.pt")
    n += 1
print(f"Wrote {n} unfiltered role vectors → {out_dir}")
PYEOF

echo "=== Step 4/5: Compute Lu-style assistant axis ==="
"$PYTHON" - "$OUTPUT/vectors" "$OUTPUT/axis.pt" <<'PYEOF'
import sys
import torch
from pathlib import Path

vec_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
default_vec = None
role_vecs = []
for vf in sorted(vec_dir.glob("*.pt")):
    data = torch.load(vf, map_location="cpu", weights_only=False)
    v = data["vector"] if isinstance(data, dict) else data
    v = v.squeeze() if v.ndim > 1 else v
    if vf.stem == "default":
        default_vec = v
    else:
        role_vecs.append(v)
if default_vec is None:
    sys.exit("FATAL: no vectors/default.pt found")
axis = default_vec - torch.stack(role_vecs).mean(dim=0)
torch.save(axis, out_path)
print(f"Wrote axis {tuple(axis.shape)}, norm={axis.norm().item():.4f}, n_roles={len(role_vecs)} → {out_path}")
PYEOF

echo "=== Step 5/5: Save default activation vector ==="
"$PYTHON" - "$OUTPUT/activations/default.pt" "$OUTPUT/default.pt" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
activations = torch.load(act_path, map_location="cpu", weights_only=False)
default_vec = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()]).mean(dim=0)
torch.save(default_vec, out_path)
print(f"Wrote default vector {tuple(default_vec.shape)} → {out_path}")
PYEOF

echo "=== Done. Outputs in $OUTPUT ==="
