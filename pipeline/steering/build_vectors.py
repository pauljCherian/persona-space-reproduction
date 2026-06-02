#!/usr/bin/env python3
"""Build trait steering vectors per (model, trait).

Adapts persona_vectors' `response_avg_diff` methodology but SKIPS the OpenAI judge
filter — uses every generated (system_prompt → response) pair, no trait-score gating.
Phase 2 calibration empirically validates that the resulting vectors actually steer.

For each (model, trait):
  - Load third_party/persona_vectors/data_generation/trait_data_extract/<trait>.json
    → 5 (pos, neg) system-instruction pairs × 20 user questions = 100 conditions/pole.
  - Generate one response per (instruction, question) under pos system prompt;
    same under neg system prompt. HF batched generation, greedy decode.
  - For each (prompt, response) pair, run a forward pass with output_hidden_states=True;
    take mean response-token activation at every layer.
  - trait_vector_at_layer_L = mean(pos response activations at L) − mean(neg ... at L).
  - Save full all-layers diff + the canonical-layer slice + metadata.

Usage:  python build_vectors.py <tag> <trait>
        tag ∈ {llama8b, qwen7b, dolphin8b};  trait ∈ {evil, sycophantic, hallucinating, humorous}
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent          # pipeline/steering (has _common.py)
_REPO = _HERE.parents[1]                          # repo root (has third_party/)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "third_party" / "persona_vectors"))

from _common import MODELS, TRAITS, trait_vector_path
from generate_vec import get_hidden_p_and_r

TRAIT_DATA_DIR = _REPO / "third_party" / "persona_vectors" / "data_generation" / "trait_data_extract"
MAX_NEW_TOKENS = 100   # short responses sufficient for trait extraction
BATCH_SIZE = 8


def build_chat_prompt(tokenizer, system_prompt: str, user_question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch)):
            r = tokenizer.decode(out[j, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(r)
        print(f"    generated {len(responses)}/{len(prompts)}", flush=True)
    return responses


def build_trait_vector(tag: str, trait: str):
    info = MODELS[tag]
    out_path = trait_vector_path(tag, trait)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[{tag} {trait}] already exists at {out_path}; skipping.")
        return

    trait_data = json.loads((TRAIT_DATA_DIR / f"{trait}.json").read_text())
    instructions = trait_data["instruction"]
    questions = trait_data["questions"]
    n_total = len(instructions) * len(questions)
    print(f"=== {tag} | {trait}: {len(instructions)} instructions × {len(questions)} questions = {n_total} conditions/pole ===", flush=True)

    print(f"  loading model {info['model_id']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # required for batched generate
    model = AutoModelForCausalLM.from_pretrained(
        info["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    pos_prompts = [build_chat_prompt(tokenizer, instr["pos"], q) for instr in instructions for q in questions]
    neg_prompts = [build_chat_prompt(tokenizer, instr["neg"], q) for instr in instructions for q in questions]

    print(f"  generating {len(pos_prompts)} POSITIVE responses ...", flush=True)
    pos_responses = generate_responses(model, tokenizer, pos_prompts, MAX_NEW_TOKENS, BATCH_SIZE)
    print(f"  generating {len(neg_prompts)} NEGATIVE responses ...", flush=True)
    neg_responses = generate_responses(model, tokenizer, neg_prompts, MAX_NEW_TOKENS, BATCH_SIZE)

    # Need right padding for the activation extraction step's tokenize path
    tokenizer.padding_side = "right"
    print(f"  extracting hidden states (POS) ...", flush=True)
    _, _, response_avg_pos = get_hidden_p_and_r(model, tokenizer, pos_prompts, pos_responses, layer_list=None)
    print(f"  extracting hidden states (NEG) ...", flush=True)
    _, _, response_avg_neg = get_hidden_p_and_r(model, tokenizer, neg_prompts, neg_responses, layer_list=None)

    n_layers_plus_1 = len(response_avg_pos)
    diff = torch.stack(
        [response_avg_pos[l].mean(0).float() - response_avg_neg[l].mean(0).float() for l in range(n_layers_plus_1)],
        dim=0,
    )
    canonical = info["layer"]
    v_canonical = diff[canonical]
    torch.save({
        "vector_all_layers": diff,                          # (n_layers+1, hidden_dim)
        "vector_at_canonical_layer": v_canonical,           # (hidden_dim,)
        "canonical_layer": canonical,
        "trait": trait, "model_tag": tag,
        "n_pos": len(pos_responses), "n_neg": len(neg_responses),
        "n_instructions": len(instructions), "n_questions": len(questions),
    }, out_path)
    print(f"  saved: {out_path}")
    print(f"  norm at L={canonical}: {v_canonical.norm().item():.4f}  hidden_dim={v_canonical.shape[0]}", flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", choices=list(MODELS))
    parser.add_argument("trait", choices=TRAITS)
    args = parser.parse_args()
    build_trait_vector(args.tag, args.trait)


if __name__ == "__main__":
    main()
