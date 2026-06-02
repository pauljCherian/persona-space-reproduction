# Rebuild the 6 poster figures from committed data (CPU, seconds):
#   make poster PYTHON=/path/to/venv/bin/python   # render into gallery/
#   make clean                                     # remove intermediate PNGs
# GPU data regeneration is driven manually from pipeline/.

PYTHON ?= python
export MPLBACKEND := Agg

# Rendered into figures/ then copied to gallery/ ; vs rendered straight to gallery/.
FIG_TO_GALLERY := figure_1_pc1_replicable steering_flow abliteration_flow figure_3_steering_trajectory
GALLERY_DIRECT := assistant_axis_simple evil_structure_only beauty_panel

.PHONY: poster clean

poster:
	@for f in $(FIG_TO_GALLERY); do echo ">>> $$f"; PSR_POSTER=1 $(PYTHON) figures/$$f.py && cp figures/$$f.png gallery/$$f.png; done
	@for f in $(GALLERY_DIRECT); do echo ">>> $$f"; PSR_POSTER=1 $(PYTHON) figures/$$f.py; done
	@echo ">>> compose (Figure 1)"; $(PYTHON) figures/compose_assistant_axis_combined.py
	@echo "Done — 6 poster figures in gallery/."

clean:
	rm -f figures/*.png
