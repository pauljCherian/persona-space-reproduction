# persona-space-reproduction

Repository to investigate the effects of model interventions on Persona Space (Lu et. al, 2026). We examine two types of model interventions: steering (Chen et. al, 2025) and abltieration (Arditi et. al, 2024)

## Repository Structure

| Path | What |
|---|---|
| `figures/` | the 6 poster figure scripts (+ compose) → `gallery/` |
| `common.py`, `persona_plot_style.py` | shared figure loaders + matplotlib style |
| `data/` | committed activation vectors, PCA basis, ablation results |
| `pipeline/` | GPU scripts that regenerate `data/` (manual; not in the Makefile) |
| `third_party/` | pinned submodules: `assistant_axis`, `persona_vectors` |

## Regenerating data (GPU)

`pip install -e third_party/assistant_axis` first, then run by hand:

1. **Role cloud** — `pipeline/extract_role_vectors/pipeline.sh <tag>`: role-play ~275
   characters, extract mid-layer activations → `data/comparison/`.
2. **Steering** — `pipeline/steering/`: `build_vectors.py` (trait vectors) ->
   `compute_baseline_pca.py` (PCA basis) → `run_steered.py` (steered clouds).
3. **Abliteration** — `pipeline/abliteration_concept/`: `build_directions.py` (concept
   subspaces) -> `run_ablation_sweep.py` (ablation positions).

## Notes
- Figures render on CPU from committed data; the pipeline is GPU-only.
- `.pt` vectors are dicts (`{"vector": tensor}`, etc.). We process these in our extraction
- The upstream judge/filter step is intentionally skipped - this is where we differ from the Assistant Axis Pipeline. OpenAI judging was too expensive

## AI Usage
- I used Claude Code with Opus 4.8 to code this repository. I made the experimetnal design and research setup all myself, but used Claude to implement the experiments, using plan files, workflows, and lots of audits. I read through every line of code, had Claude explain the PyTorch to me, and wrote commments. I also made sure that every experimental output was reviewed wholly by me. I also used Claude Code for creating the figures.

