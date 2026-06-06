#file to make the 6 poster figures from committed data
#make poster PYTHON=/path/to/venv/bin/python   
#make clean                                    

PYTHON ?= python
export MPLBACKEND := Agg

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
