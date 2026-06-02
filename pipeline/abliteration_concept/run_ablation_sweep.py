#!/usr/bin/env python3
"""Concept-abliteration sweep: does persona space survive removing a concept subspace?

For each concept (evil / humor / mysticism) we project out the top-k columns of its
concept subspace (from build_directions.py) and measure where the concept's coded roles
+ the default assistant land in the EXISTING 275-role PCA basis. A matched random
subspace (same k) is the scale reference.

Method (cheap, no 275-role re-run, no weight surgery):
  1. Generate the subset probes ONCE with the unmodified model (coded roles + default).
  2. For each condition, re-extract the SAME fixed conversations under an
     assistant_axis.ActivationSteering(intervention_type="ablation", coeff=0) hook over
     the top-k subspace columns -> mean over assistant-turn tokens at layer 16
     (identical convention to the baseline cloud) -> project into the basis.
  Displacement of a role = || pos(baseline) - pos(ablated) || in the PC1xPC2 plane.

Inference-time directional ablation (h <- h - QQ^T h at the hooked layers) is the
per-token equivalent of Arditi weight-orthogonalisation; fixing the text isolates the
representational effect of the removal.

Output:  data/abliteration_concept/runs/<run_name>/positions.csv  (+ meta.json)
           columns: condition, family, k, role, group, pc1, pc2
Usage:   CUDA_VISIBLE_DEVICES=4 python run_ablation_sweep.py [--n_questions 8]
             [--n_system_prompts 2] [--layers 16|all] [--max_roles 0] [--run_name sweep]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import assistant_axis as _aa
from assistant_axis import ActivationSteering
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
HIDDEN = 4096
N_LAYERS = 32

_AA_DATA = Path(_aa.__file__).resolve().parent.parent / "data"
ROLES_DIR = _AA_DATA / "roles" / "instructions"
QUESTIONS_FILE = _AA_DATA / "extraction_questions.jsonl"

DIR_DIR = ROOT / "data" / "abliteration_concept" / "directions"
BASIS_FILE = ROOT / "data" / "steering" / "pca_basis" / "llama8b_basis.pt"
RUNS_DIR = ROOT / "data" / "abliteration_concept" / "runs"

CONCEPTS = ["evil", "humor", "mysticism"]
CODED_ROLES = {
    "evil":      ["demon", "vampire", "criminal", "predator", "saboteur"],
    "humor":     ["comedian", "jester", "fool", "trickster", "absurdist"],
    "mysticism": ["mystic", "sage", "oracle", "ascetic", "prophet"],
}
K_SWEEP = [1, 2, 3, 5, 8, 10]
MAX_NEW_TOKENS = 256
GEN_BATCH = 16


# ---------------------------------------------------------------- generation
def load_questions(n: int) -> list[str]:
    qs = [json.loads(l)["question"] for l in QUESTIONS_FILE.read_text().splitlines() if l.strip()]
    return qs[:n]


def role_instructions(role: str, n_system: int) -> list[str]:
    if role == "default":
        return [""]
    data = json.loads((ROLES_DIR / f"{role}.json").read_text())
    return [i["pos"] for i in data["instruction"]][:n_system]


def generate_conversations(pm, roles, questions, n_system):
    """Greedy-generate one response per (role, system_variant, question); return
    {role: [conversation,...]} where each conversation is a chat-message list."""
    tok = pm.tokenizer
    convs = {r: [] for r in roles}
    for role in roles:
        items = [(s, q) for s in role_instructions(role, n_system) for q in questions]
        for i in range(0, len(items), GEN_BATCH):
            batch = items[i:i + GEN_BATCH]
            texts = []
            for sys_p, user_q in batch:
                msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + \
                       [{"role": "user", "content": user_q}]
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(pm.model.device)
            with torch.no_grad():
                out = pm.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, (sys_p, user_q) in enumerate(batch):
                resp = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                conv = ([{"role": "system", "content": sys_p}] if sys_p else []) + \
                       [{"role": "user", "content": user_q}, {"role": "assistant", "content": resp}]
                convs[role].append(conv)
        print(f"  generated {role}: {len(convs[role])} probes", flush=True)
    return convs


# ---------------------------------------------------------------- extraction
def extract_role_vector(extractor, encoder, span_mapper, conversations, batch_size=16):
    """Mean-over-assistant-tokens layer-LAYER vector, meaned over a role's probes.
    Mirrors assistant_axis pipeline/2_activations.py exactly (so it matches the cloud)."""
    vecs = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i + batch_size]
        acts, meta = extractor.batch_conversations(batch, layer=[LAYER], max_length=2048)
        _, spans, _smeta = encoder.build_batch_turn_spans(batch)
        per_conv = span_mapper.map_spans(acts, spans, meta)   # list of (n_turns, n_layers, hidden)
        for conv_acts in per_conv:
            if conv_acts.numel() == 0:
                continue
            assistant = conv_acts[1::2]                       # assistant turns
            if assistant.shape[0] > 0:
                vecs.append(assistant.mean(dim=0)[0].float().cpu())   # (hidden,) — single layer
    return torch.stack(vecs).mean(dim=0)


def ablation_ctx(model, Q_cols, layers):
    """ActivationSteering that projects out the given orthonormal columns at `layers`.
    Q_cols: list of (hidden,) tensors (orthonormal). coeff=0 => pure projection-out."""
    if not Q_cols:
        return nullcontext()
    dev = model.device
    vecs = [col.to(dev).to(torch.bfloat16) for L in layers for col in Q_cols]
    layer_idx = [L for L in layers for _ in Q_cols]
    return ActivationSteering(model, steering_vectors=vecs, coefficients=[0.0] * len(vecs),
                              layer_indices=layer_idx, intervention_type="ablation", positions="all")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_questions", type=int, default=8)
    ap.add_argument("--n_system_prompts", type=int, default=2)
    ap.add_argument("--layers", default="16", help='"16" (read layer) or "all" (Arditi-style)')
    ap.add_argument("--k_sweep", default=",".join(map(str, K_SWEEP)))
    ap.add_argument("--max_roles", type=int, default=0, help=">0: cap subset roles (smoke test)")
    ap.add_argument("--roles", default="", help="comma list to override the subset (e.g. demon,default)")
    ap.add_argument("--run_name", default="sweep")
    args = ap.parse_args()

    layers = list(range(N_LAYERS)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    ks = [int(x) for x in args.k_sweep.split(",")]

    basis = torch.load(BASIS_FILE, map_location="cpu", weights_only=False)
    role_mean = np.asarray(basis["role_mean"], dtype=np.float64)
    pc1 = np.asarray(basis["pc1"], dtype=np.float64); pc1 /= np.linalg.norm(pc1)
    pc2 = np.asarray(basis["pc2"], dtype=np.float64); pc2 /= np.linalg.norm(pc2)

    def project(v: torch.Tensor):
        c = v.double().numpy() - role_mean
        return float(c @ pc1), float(c @ pc2)

    subspaces = {c: torch.load(DIR_DIR / f"{c}_concept_subspace.pt", weights_only=False)["U"]
                 for c in CONCEPTS}

    subset = [r for c in CONCEPTS for r in CODED_ROLES[c]] + ["default"]
    role_group = {r: c for c in CONCEPTS for r in CODED_ROLES[c]}
    role_group["default"] = "default"
    if args.max_roles > 0:
        subset = subset[:args.max_roles]
    if args.roles:
        subset = [r.strip() for r in args.roles.split(",") if r.strip()]
    role_group = {r: role_group.get(r, "other") for r in subset}

    print(f"=== concept-abliteration sweep | layers={args.layers} | ks={ks} ===", flush=True)
    print(f"  loading {MODEL_ID} ...", flush=True)
    pm = ProbingModel(MODEL_ID)
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    questions = load_questions(args.n_questions)
    print(f"  generating probes for {len(subset)} roles ...", flush=True)
    convs = generate_conversations(pm, subset, questions, args.n_system_prompts)

    rows = []   # (condition, family, k, role, group, pc1, pc2)

    def run_condition(condition, family, k, Q_cols):
        with ablation_ctx(pm.model, Q_cols, layers):
            for role in subset:
                v = extract_role_vector(extractor, encoder, span_mapper, convs[role])
                p1, p2 = project(v)
                rows.append((condition, family, k, role, role_group[role], p1, p2))
        print(f"  [{condition}] done", flush=True)

    # baseline (no ablation)
    run_condition("baseline", "baseline", 0, [])
    # concept sweeps
    for c in CONCEPTS:
        U = subspaces[c]
        for k in ks:
            if k > U.shape[1]:
                continue
            cols = [U[:, i].clone() for i in range(k)]
            run_condition(f"{c}_k{k}", c, k, cols)
    # matched random controls (seed-matched per k)
    for k in ks:
        g = torch.Generator().manual_seed(1000 + k)
        Qr, _ = torch.linalg.qr(torch.randn(HIDDEN, k, generator=g))
        cols = [Qr[:, i].clone() for i in range(k)]
        run_condition(f"random_k{k}", "random", k, cols)

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "family", "k", "role", "group", "pc1", "pc2"])
        w.writerows(rows)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": MODEL_ID, "layer": LAYER, "ablate_layers": args.layers,
        "n_questions": args.n_questions, "n_system_prompts": args.n_system_prompts,
        "k_sweep": ks, "subset": subset, "concepts": CONCEPTS, "coded_roles": CODED_ROLES,
    }, indent=2))
    print(f"wrote {out_dir/'positions.csv'} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
