#!/usr/bin/env python3
# Run the assistant axis pipeline on a steered model.
#Usage:
# python run_steered.py <tag> <vector_name> <alpha> [--n_questions 32] [--batch_size 16]

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MODELS, steered_dir, trait_vector_path, control_vector_path
import assistant_axis as _aa
from assistant_axis import ActivationSteering

#Get the role instructions + extraction questions that the assistant axis repo provides
_AA_DATA = Path(_aa.__file__).resolve().parent.parent / "data"
ROLES_DIR = _AA_DATA / "roles" / "instructions"
QUESTIONS_FILE = _AA_DATA / "extraction_questions.jsonl"
MAX_NEW_TOKENS = 256
DEFAULT_N_QUESTIONS = 32
DEFAULT_BATCH_SIZE = 16

# get a steering vector that we have created
def load_steering_vector(tag: str, vector_name: str) -> torch.Tensor:
    # check if we have a role vector
    if vector_name.startswith("role_"):
        role_name = vector_name[len("role_"):]
        role_dir = Path(__file__).resolve().parents[1] / "data" / "vectors_steering" / tag / "role"
        d = torch.load(role_dir / f"{role_name}.pt", map_location="cpu", weights_only=False)
        return d["vector"].float()
    # check if it is canonical
    if vector_name in {"evil", "sycophantic", "hallucinating", "humorous"}:
        d = torch.load(trait_vector_path(tag, vector_name), map_location="cpu", weights_only=False)
        return d["vector_at_canonical_layer"].float()
    # otherwise it might not be wrapped in anything
    d = torch.load(control_vector_path(tag, vector_name), map_location="cpu", weights_only=False)
    return d["vector"].float()

# load the questions we need to ask the role playing model
def load_questions(n: int) -> list[dict]:
    qs = [json.loads(l) for l in QUESTIONS_FILE.read_text().splitlines() if l.strip()]
    return qs[:n]

# build a chat prompt to use
def build_chat_text(tokenizer, system: str, user: str) -> str:
    if system:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# make a forward pass through full (system + user + assistant response), then return hidden state at the last token of the reponse
def get_activation_for_response(model, tokenizer, layer_module, system: str, user: str, response: str):
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    else:
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(full, return_tensors="pt", add_special_tokens=False).to(model.device)
    captured = {}
    def cap_hook(module, ins, out):
        captured["o"] = (out[0] if isinstance(out, tuple) else out).detach()
    h_handle = layer_module.register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        h_handle.remove()
    h = captured["o"][:, -1, :].squeeze(0).to(torch.bfloat16).cpu()
    return h.unsqueeze(0)

# go through and ask the questions and generate the respones and get the activations
def process_role(role_file: Path, model, tokenizer, layer_module,
                 questions: list[dict], out_resp: Path, out_act: Path,
                 out_vec: Path, batch_size: int):

    # the name of the role we are generating for
    role = role_file.stem
    if out_resp.exists() and out_act.exists() and out_vec.exists():
        print(f"  [{role}] already complete, skipping")
        return

    # get the information about the roel
    role_data = json.loads(role_file.read_text())

    # get the instructors for that role
    instructions = [i["pos"] for i in role_data["instruction"]]
    if role == "default":
        instructions = [""]   # default has empty system prompt
    # build full prompt list (prompt_idx, q_idx, system, user)
    items = []

    # create a list of tules that have all the information needed for role instructions and role questions
    for p_idx, instr in enumerate(instructions):
        for q_idx, q in enumerate(questions):
            items.append((p_idx, q_idx, instr, q["question"]))
    # Generate responses in batched format
    responses = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        chat_texts = [build_chat_text(tokenizer, s, u) for _, _, s, u in batch]
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j in range(len(batch)):
            r = tokenizer.decode(out[j, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(r)
    # Write responses and extract activations
    resp_records = []
    act_dict = {}
    for (p_idx, q_idx, instr, user_q), resp in zip(items, responses):
        if instr:
            conv = [{"role": "system", "content": instr},
                    {"role": "user", "content": user_q},
                    {"role": "assistant", "content": resp}]
        else:
            conv = [{"role": "user", "content": user_q},
                    {"role": "assistant", "content": resp}]
        resp_records.append({
            "system_prompt": instr, "prompt_index": p_idx, "question_index": q_idx,
            "question": user_q, "conversation": conv, "label": "pos",
        })
        act = get_activation_for_response(model, tokenizer, layer_module, instr, user_q, resp)
        act_dict[f"pos_p{p_idx}_q{q_idx}"] = act
    # save the responses and teh activations
    out_resp.parent.mkdir(parents=True, exist_ok=True)
    out_act.parent.mkdir(parents=True, exist_ok=True)
    out_vec.parent.mkdir(parents=True, exist_ok=True)
    with open(out_resp, "w") as f:
        for r in resp_records:
            f.write(json.dumps(r) + "\n")
    torch.save(act_dict, out_act)
    # Take an average of the activations and save that
    stacked = torch.stack([v.squeeze(0) for v in act_dict.values()])  # (n_rollouts, hidden_dim)
    mean_vec = stacked.float().mean(dim=0)
    torch.save({"vector": mean_vec, "type": "mean", "role": role}, out_vec)
    print(f"  [{role}] {len(resp_records)} rollouts, ||mean||={mean_vec.norm().item():.3f}", flush=True)


# main loop to run everything
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=list(MODELS))
    ap.add_argument("vector_name", help="trait name (evil/sycophantic/hallucinating/humorous) or control (random_unit_s42 etc)")
    ap.add_argument("alpha", type=float, help="multiplier of v_raw (the vector is applied as alpha * v_raw)")
    ap.add_argument("--n_questions", type=int, default=DEFAULT_N_QUESTIONS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max_roles", type=int, default=0, help="if >0, only process this many roles (for smoke tests)")
    ap.add_argument("--intervention_type", choices=["addition", "ablation"], default="addition",
                    help="addition: h += coef*v.  ablation: project v out of h, then add coef*v "
                         "(coef=0 = pure ablation).  Output dir suffix changes for non-addition.")
    args = ap.parse_args()

    info = MODELS[args.tag]
    out_dir = steered_dir(args.tag, args.vector_name, args.alpha, intervention=args.intervention_type)
    print(f"=== run_steered: {args.tag} | {args.vector_name} | {args.intervention_type} coef={args.alpha} ===", flush=True)
    print(f"  layer={info['layer']}  n_questions={args.n_questions}  batch_size={args.batch_size}", flush=True)
    print(f"  output: {out_dir}", flush=True)

    print(f"  loading vector...", flush=True)
    v = load_steering_vector(args.tag, args.vector_name)
    assert v.shape == (info["hidden_dim"],), f"vector shape {v.shape} != ({info['hidden_dim']},)"
    print(f"  ||v||={v.norm().item():.4f}  α·||v||={args.alpha * v.norm().item():.4f}", flush=True)

    print(f"  loading model {info['model_id']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  

    #get the model
    model = AutoModelForCausalLM.from_pretrained(
        info["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )

    # put into inference mode
    model.eval()

    # load the questions
    questions = load_questions(args.n_questions)
    role_files = sorted(ROLES_DIR.glob("*.json"))
    if args.max_roles > 0:
        role_files = role_files[:args.max_roles]
        print(f"  SMOKE TEST: capping to {args.max_roles} roles", flush=True)
    print(f"  {len(role_files)} role files; will also process 'default' (empty system).", flush=True)

    # put the model on our gpu
    v_device = v.to(model.device).to(torch.bfloat16)
    layer_module = model.model.layers[info["layer"]]

    #use the activation steering part of the assistant axis repo
    with ActivationSteering(
        model,
        steering_vectors=[v_device],
        coefficients=[float(args.alpha)],
        layer_indices=[info["layer"]],
        intervention_type=args.intervention_type,
        positions="all",
    ):
        t0 = time.time()
        # process roles and time them
        for i, rf in enumerate(role_files):
            t = time.time()
            process_role(
                rf, model, tokenizer, layer_module, questions,
                out_dir / "responses" / f"{rf.stem}.jsonl",
                out_dir / "activations" / f"{rf.stem}.pt",
                out_dir / "vectors" / f"{rf.stem}.pt",
                args.batch_size,
            )
            elapsed = time.time() - t
            total = time.time() - t0
            eta = total / (i + 1) * (len(role_files) + 1 - (i + 1))
            print(f"  [{i+1}/{len(role_files)+1}] {rf.stem}: {elapsed:.1f}s  (ETA {eta/60:.1f} min)", flush=True)
        default_role_data = {"instruction": [{"pos": ""}]}
        tmp_default = out_dir / "_default_role.json"
        tmp_default.write_text(json.dumps(default_role_data))
        process_role(
            tmp_default, model, tokenizer, layer_module, questions,
            out_dir / "responses" / "default.jsonl",
            out_dir / "activations" / "default.pt",
            out_dir / "vectors" / "default.pt",
            args.batch_size,
        )
        tmp_default.unlink()
        import shutil
        shutil.copyfile(out_dir / "vectors" / "default.pt", out_dir / "default.pt")
        print(f"  TOTAL: {(time.time()-t0)/60:.1f} min", flush=True)



if __name__ == "__main__":
    main()
