#!/usr/bin/env python3
# Build trait steering vectors for each (model, trait)


#For each (model, trait):
  # Load third_party/persona_vectors/data_generation/trait_data_extract/<trait>.json
        # This gets 5 (pos, neg) system-instruction pairs x 20 user questions = 100 conditions per pole
  # Generate one response per (instruction, question) under positive system prompt, then do the same under each negative system prompt
  # For each (prompt, response) pair do a forward pass with outputting the hidden states then take mean response-token activation at every layer.
  # Then calculate the trait_vector_at_layer_L = mean(positive response activations at L) − mean(negative response activations at L).
  # Then save all the layers trait vectors
# Usage:  python build_vectors.py <tag> <trait>
# tag in  {llama8b, qwen7b, dolphin8b}
# trait in {evil, sycophantic, hallucinating, humorous}

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent          
_REPO = _HERE.parents[1]                        
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "third_party" / "persona_vectors"))

from _common import MODELS, TRAITS, trait_vector_path
from generate_vec import get_hidden_p_and_r

TRAIT_DATA_DIR = _REPO / "third_party" / "persona_vectors" / "data_generation" / "trait_data_extract"
MAX_NEW_TOKENS = 100  
BATCH_SIZE = 8

#helper function to build a chat prompt
def build_chat_prompt(tokenizer, system_prompt: str, user_question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#generate reponses for a model with prompts and batch it
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

#build the trait vector by forward passing positive and negative prompts then taking the difference 
def build_trait_vector(tag: str, trait: str):

    # get the model and where to write results
    info = MODELS[tag]
    out_path = trait_vector_path(tag, trait)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[{tag} {trait}] already exists at {out_path}; skipping.")
        return

    # load the questions to ask for the trait that we are creating a vector for
    trait_data = json.loads((TRAIT_DATA_DIR / f"{trait}.json").read_text())
    instructions = trait_data["instruction"]
    questions = trait_data["questions"]
    n_total = len(instructions) * len(questions)

    # pretty print
    print(f"=== {tag} | {trait}: {len(instructions)} instructions × {len(questions)} questions = {n_total} conditions/pole ===", flush=True)

    print(f"  loading model {info['model_id']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])

    #tokenizing for easier batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    #get the model
    model = AutoModelForCausalLM.from_pretrained(
        info["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )

    #put in inference mode
    model.eval()

    #ge the positive and negative prompts that we will sent for completion
    pos_prompts = [build_chat_prompt(tokenizer, instr["pos"], q) for instr in instructions for q in questions]
    neg_prompts = [build_chat_prompt(tokenizer, instr["neg"], q) for instr in instructions for q in questions]

    #generate responses for the positive rollouts and the negative rollouts
    print(f"  generating {len(pos_prompts)} POSITIVE responses ...", flush=True)
    pos_responses = generate_responses(model, tokenizer, pos_prompts, MAX_NEW_TOKENS, BATCH_SIZE)
    print(f"  generating {len(neg_prompts)} NEGATIVE responses ...", flush=True)
    neg_responses = generate_responses(model, tokenizer, neg_prompts, MAX_NEW_TOKENS, BATCH_SIZE)
    tokenizer.padding_side = "right"
    
    #extract the hidden states for the postive and the negative rollouts. Then average
    print(f"  extracting hidden states (POS) ...", flush=True)
    _, _, response_avg_pos = get_hidden_p_and_r(model, tokenizer, pos_prompts, pos_responses, layer_list=None)
    print(f"  extracting hidden states (NEG) ...", flush=True)
    _, _, response_avg_neg = get_hidden_p_and_r(model, tokenizer, neg_prompts, neg_responses, layer_list=None)

    n_layers_plus_1 = len(response_avg_pos)

    # get the difference for each of the layers 
    diff = torch.stack(
        [response_avg_pos[l].mean(0).float() - response_avg_neg[l].mean(0).float() for l in range(n_layers_plus_1)],
        dim=0,
    )
    canonical = info["layer"]
    v_canonical = diff[canonical]

    #save the vectors we created to disc. Save all of them in a value AND save just the canonical L//2 vector as well
    torch.save({
        "vector_all_layers": diff,                          
        "vector_at_canonical_layer": v_canonical, 
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
