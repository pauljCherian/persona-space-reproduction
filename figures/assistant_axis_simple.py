#!/usr/bin/env python3
"""Simplified 'Building Persona Space' (assistant-axis) flow — 4 steps, uniform arrows.
Left panel of the combined Figure 1. Output: gallery/assistant_axis_simple.png"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "figure.facecolor": "white", "savefig.facecolor": "white"})

ROOT = Path(__file__).resolve().parents[1]
INK = "#1f2a37"; PANELBG = "#eef1f5"
ROLE_F, ROLE_S = "#e6dffa", "#8064cf"; PROC_F, PROC_S = "#e8ecf2", "#7a8699"; RES_F, RES_S = "#f0d2cb", "#b0432f"; ARR = "#5f6b7a"

fig = plt.figure(figsize=(5.6, 6.8)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax.set_xlim(0, 64); ax.set_ylim(8, 77); ax.axis("off")

def box(cy, h, fc, ec, title, body="", tfs=13, bfs=10, tcol=INK, lw=1.8):
    ax.add_patch(FancyBboxPatch((32 - 27, cy - h / 2), 54, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw, mutation_aspect=1, zorder=3))
    if body:
        ax.text(32, cy + h * 0.20, title, ha="center", va="center", fontsize=tfs, weight="bold", color=tcol, zorder=4)
        ax.text(32, cy - h * 0.22, body, ha="center", va="center", fontsize=bfs, color=INK, zorder=4, linespacing=1.25)
    else:
        ax.text(32, cy, title, ha="center", va="center", fontsize=tfs, weight="bold", color=tcol, zorder=4)

def arrow(y1, y2):
    ax.add_patch(FancyArrowPatch((32, y1), (32, y2), arrowstyle="-|>", mutation_scale=16, lw=2, color=ARR, zorder=2, shrinkA=0, shrinkB=0))

box(70, 6, PANELBG, INK, "MODEL", tfs=15)
box(55, 10, ROLE_F, ROLE_S, "Role-play ≈ 275 characters",
    "demon · sage · scientist · jester · oracle · …")
box(34.5, 11, PROC_F, PROC_S, "Extract & average mid-layer activations",
    "post-MLP residual, averaged over response\ntokens  →  one vector per role in ℝᵈ", tfs=12)
box(14.5, 9, RES_F, RES_S, "Run PCA  →  PC1 = Assistant Axis",
    "on the mean-centered ≈ 275-vector cloud", tfs=12.5, tcol=RES_S)
for y1, y2 in [(67, 60), (50, 40), (29, 19)]: arrow(y1, y2)

fig.savefig(ROOT / "gallery" / "assistant_axis_simple.png", dpi=150, bbox_inches="tight", pad_inches=0.08)
print("wrote gallery/assistant_axis_simple.png")
