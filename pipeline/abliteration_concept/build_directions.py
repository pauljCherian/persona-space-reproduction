#!/usr/bin/env python3
"""Build a CONCEPT SUBSPACE to abliterate, for evil / humor / mysticism.

NO GPU.  Pure linear algebra over already-extracted per-word concept vectors.

For each concept we take a set of synonym words that ALREADY have a layer-16
"vector_at_canonical_layer" on disk (built earlier by pipeline/steering/build_vectors.py,
the persona_vectors contrastive convention — same layer/pooling as the 275-role cloud),
unit-normalise them, stack them as columns, and SVD.  The left singular vectors U
give an ORTHONORMAL, importance-ORDERED basis of the concept's representational
subspace, so an abliteration sweep just projects out the top-k columns U[:, :k].

  data/abliteration_concept/directions/{concept}_concept_subspace.pt
      {"U": (hidden_dim, n) float32 orthonormal, "svals": (n,), "names": [...], "concept": str}

The synonym vectors live in the steering-persona-space project (build-time only); the
saved .pt files are self-contained for the run/figure steps.
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]                     # persona-space-reproduction
SPS = Path("/jumbo/lisp/paulcherian/steering-persona-space")   # build-time source of per-word vectors
TRAIT_DIR = SPS / "data" / "vectors_steering" / "llama8b"          # evil.pt, humorous.pt, ...
SYN_DIR = SPS / "data" / "vectors_steering_synonyms" / "llama8b"   # malevolent.pt, witty.pt, mystical.pt, ...
OUT_DIR = ROOT / "data" / "abliteration_concept" / "directions"

# Synonym words per concept — only words with a pre-built vector on disk (verified).
SYNONYMS = {
    "evil":      ["evil", "malevolent", "wicked", "cruel", "sadistic", "ruthless",
                  "corrupt", "hateful", "immoral", "treacherous", "demonic", "sinful"],
    "humor":     ["humorous", "funny", "witty", "playful", "comedic", "satirical",
                  "facetious", "jocular", "sarcastic", "absurdist", "irreverent"],
    "mysticism": ["mystical", "occult", "spiritual", "prophetic", "sacred", "transcendent",
                  "divine", "devotional", "shamanic", "ascetic"],
}


def load_word_vector(name: str) -> torch.Tensor:
    """Layer-16 concept vector for one word; evil/humorous live in TRAIT_DIR, rest in SYN_DIR."""
    for d in (SYN_DIR, TRAIT_DIR):
        p = d / f"{name}.pt"
        if p.exists():
            obj = torch.load(p, map_location="cpu", weights_only=False)
            v = obj["vector_at_canonical_layer"] if isinstance(obj, dict) else obj
            return v.float().flatten()
    raise FileNotFoundError(f"no pre-built vector for {name!r} in {SYN_DIR} or {TRAIT_DIR}")


def build_subspace(concept: str) -> dict:
    names = SYNONYMS[concept]
    cols = [(v := load_word_vector(n)) / v.norm() for n in names]  # equal weight per synonym
    M = torch.stack(cols, dim=1).double()               # (hidden_dim, n)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)  # U: (hidden_dim, n), ordered by S
    return {"U": U.float(), "svals": S.float(), "names": names, "concept": concept}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for concept in ("evil", "humor", "mysticism"):
        payload = build_subspace(concept)
        out = OUT_DIR / f"{concept}_concept_subspace.pt"
        torch.save(payload, out)
        sv = payload["svals"]
        print(f"{out.name:34s} U={tuple(payload['U'].shape)} "
              f"svals[0:3]={[round(x, 2) for x in sv[:3].tolist()]} n={len(payload['names'])}")


if __name__ == "__main__":
    main()
