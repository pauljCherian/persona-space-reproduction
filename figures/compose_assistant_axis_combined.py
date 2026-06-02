#!/usr/bin/env python3
"""Compose Figure 1: assistant-axis pipeline (left) + Llama-3.1-8B persona space (right),
side by side at a common height. Run AFTER assistant_axis_simple.py and
figure_1_pc1_replicable.py. Output: gallery/assistant_axis_combined.png"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
flow = plt.imread(ROOT / "gallery" / "assistant_axis_simple.png")
scat = plt.imread(ROOT / "gallery" / "figure_1_pc1_replicable.png")
fa = flow.shape[1] / flow.shape[0]; sa = scat.shape[1] / scat.shape[0]
H = 6.0; gap = 0.45; W = H * fa + gap + H * sa
fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
ax1 = fig.add_axes([0, 0, H * fa / W, 1]); ax1.imshow(flow, aspect="auto"); ax1.axis("off")
ax2 = fig.add_axes([(H * fa + gap) / W, 0.02, H * sa / W, 0.96]); ax2.imshow(scat, aspect="auto"); ax2.axis("off")
fig.savefig(ROOT / "gallery" / "assistant_axis_combined.png", dpi=150, facecolor="white")
print("wrote gallery/assistant_axis_combined.png  (%.1f x %.1f in)" % (W, H))
