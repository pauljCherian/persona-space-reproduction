#!/usr/bin/env python3
#build a CONCEPT SUBSPACE to abliterate, for evil / humor / mysticism. We do this over word-concept vectors that we have already extracted
# For a set of concept vectors, we normalize them, put them into a matrix, calaculate their orthonormal basis, and then figure out which ones represent most of the variance

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]                     # persona-space-reproduction
SPS = Path("/jumbo/lisp/paulcherian/steering-persona-space")   # build-time source of per-word vectors
TRAIT_DIR = SPS / "data" / "vectors_steering" / "llama8b"          # evil.pt, humorous.pt, ...
SYN_DIR = SPS / "data" / "vectors_steering_synonyms" / "llama8b"   # malevolent.pt, witty.pt, mystical.pt, ...
OUT_DIR = ROOT / "data" / "abliteration_concept" / "directions"

# Synonym words for each concept subspace we want to build.
SYNONYMS = {
    "evil":      ["evil", "malevolent", "wicked", "cruel", "sadistic", "ruthless",
                  "corrupt", "hateful", "immoral", "treacherous", "demonic", "sinful"],
    "humor":     ["humorous", "funny", "witty", "playful", "comedic", "satirical",
                  "facetious", "jocular", "sarcastic", "absurdist", "irreverent"],
    "mysticism": ["mystical", "occult", "spiritual", "prophetic", "sacred", "transcendent",
                  "divine", "devotional", "shamanic", "ascetic"],
}

# load the concept vector for a word, extracted at layer 16
def load_word_vector(name: str) -> torch.Tensor:
    for d in (SYN_DIR, TRAIT_DIR):
        p = d / f"{name}.pt"
        if p.exists():
            obj = torch.load(p, map_location="cpu", weights_only=False)
            v = obj["vector_at_canonical_layer"] if isinstance(obj, dict) else obj
            return v.float().flatten()
    raise FileNotFoundError(f"no pre-built vector for {name!r} in {SYN_DIR} or {TRAIT_DIR}")

# build a subspace by stacking several word concept vectors into a matrix, them doing SVD on them to get the orthonormal basis
# with the most important vectors spannign that basis sorted
def build_subspace(concept: str) -> dict:
    names = SYNONYMS[concept]
    cols = [(v := load_word_vector(n)) / v.norm() for n in names]  # equal weight per synonym
    M = torch.stack(cols, dim=1).double()               
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return {"U": U.float(), "svals": S.float(), "names": names, "concept": concept}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for concept in ("evil", "humor", "mysticism"):
        # build the subspace (just doing SVD)
        payload = build_subspace(concept)
        out = OUT_DIR / f"{concept}_concept_subspace.pt"
        torch.save(payload, out)
        sv = payload["svals"]
        print(f"{out.name:34s} U={tuple(payload['U'].shape)} "
              f"svals[0:3]={[round(x, 2) for x in sv[:3].tolist()]} n={len(payload['names'])}")


if __name__ == "__main__":
    main()
