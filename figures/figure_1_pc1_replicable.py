#!/usr/bin/env python3
"""Figure 1 — PC1 is replicable across large (7–8B) models.

3-panel PCA scatter: Llama-3.1-8B / Qwen-2.5-7B / Dolphin-3.0-Llama-3.1-8B.
275 roles per panel; default highlighted; Assistant Axis (sign-aligned PC1) overlaid as dashed line.

Inputs:  data/comparison/<tag>/vectors/*.pt  + data/comparison/<tag>/default.pt
Output:  figures/figure_1_pc1_replicable.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import persona_plot_style as style
from common import load_role_matrix, load_pt

style.apply()

# Data root defaults to the committed data/; set PSR_DATA to render from a
# regenerated tree (e.g. PSR_DATA=data_regen).
DATA = Path(os.environ.get("PSR_DATA", ROOT / "data")) / "comparison"

MODELS = [
    ("llama-3.1-8b", "Llama-3.1-8B", 16),
]

# Roles to label across every panel — expanded set covering more archetypes.
LABEL_ROLES = {
    # evil / dark
    "demon", "vampire", "criminal", "predator", "saboteur", "narcissist",
    # comedic / playful
    "comedian", "jester", "fool", "trickster", "absurdist",
    # mystic / wise
    "mystic", "sage", "prophet", "oracle", "shaman", "ascetic",
    # neutral
    "philosopher",
    # extreme
    "toddler", "caveman", "infant",
}

# High (assistant) end of PC1 — professional / analytic personas at or beyond the
# default on the Assistant Axis: the "better assistant" exemplars. Distinct colour,
# exempt from the default keep-out.
ASSISTANT_POLE_ROLES = {
    "evaluator", "planner", "strategist",
    "analyst", "researcher", "consultant",
}
ASSIST_LABEL = "#0f766e"   # teal-green — the assistant-pole cluster


def pca_project(tag: str):
    model_dir = DATA / tag
    X, roles = load_role_matrix(model_dir / "vectors")
    default = load_pt(model_dir / "default.pt")
    non_default = np.array([r != "default" for r in roles])

    role_mean = X[non_default].mean(axis=0)
    centered = X[non_default] - role_mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]

    lu_axis = default - role_mean
    if pc1 @ lu_axis < 0:
        pc1 = -pc1

    var_frac = (S ** 2) / (S ** 2).sum()

    centered_all = X - role_mean
    px = centered_all @ pc1
    py = centered_all @ pc2

    default_idx = roles.index("default")
    return px, py, default_idx, var_frac, non_default, roles


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.4),
                            gridspec_kw=dict(left=0.115, right=0.97, top=0.95, bottom=0.10))

    for ax, (tag, display, layer) in zip([ax], MODELS):
        px, py, default_idx, var_frac, non_default, roles = pca_project(tag)
        x0, y0 = px[default_idx], py[default_idx]

        ax.scatter(
            px[non_default], py[non_default],
            s=16, c=style.CLOUD, edgecolor=style.CLOUD_EDGE, linewidth=0.35,
            alpha=0.85, zorder=2,
        )

        # Don't label any role near the default — keeps the Assistant-pole region
        # uncluttered around the gold star.
        d_from_default = np.sqrt((px - x0) ** 2 + (py - y0) ** 2)
        xlim = ax.get_xlim()
        xlim = (xlim[0], xlim[1] + 0.14 * (xlim[1] - xlim[0]))   # headroom for assistant-pole labels
        ax.set_xlim(xlim)
        ylim = ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        keepout_radius = 0.08 * np.sqrt(x_span ** 2 + y_span ** 2)

        texts = []
        for r, x, y, dd in zip(roles, px, py, d_from_default):
            if r in ASSISTANT_POLE_ROLES:
                texts.append(ax.text(x, y, r, fontsize=7.0, color=ASSIST_LABEL, weight="bold",
                                       alpha=0.99, zorder=4,
                                       bbox=dict(boxstyle="square,pad=0.12",
                                                  fc="white", ec="none", alpha=0.85)))
            elif r in LABEL_ROLES and dd > keepout_radius:
                texts.append(ax.text(x, y, r, fontsize=7.0, color=style.CLOUD_LABEL,
                                       alpha=0.98, zorder=3,
                                       bbox=dict(boxstyle="square,pad=0.12",
                                                  fc="white", ec="none",
                                                  alpha=0.8)))
        if texts:
            adjust_text(
                texts, ax=ax,
                expand=(1.5, 1.7),
                arrowprops=dict(arrowstyle="-", color=style.CLOUD_EDGE, lw=0.45, alpha=0.75),
                force_text=(1.4, 1.6),
                force_static=(1.0, 1.1),
                force_explode=(0.7, 0.8),
                iter_lim=250,
                only_move={"text": "xy", "static": "xy",
                            "explode": "xy", "pull": "xy"},
            )

        x_line = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(x_line, np.zeros_like(x_line), ls="--", color=style.AXIS_LINE,
                lw=1.0, alpha=0.6, zorder=2.5, label="Assistant Axis (PC1)")

        # Gold star + "default" label anchored ABOVE the star
        # so it stays out of the dense Assistant-pole cluster.
        ax.scatter([x0], [y0], s=260, c=style.GOLD, marker="*",
                   edgecolor="black", linewidth=0.9, zorder=5)
        ax.annotate(
            "default", (x0, y0),
            xytext=(-40, 22), textcoords="offset points",
            fontsize=10, weight="bold", color=style.FG,
            ha="center", va="bottom", zorder=20,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                       edgecolor=style.GOLD, linewidth=0.8, alpha=0.97),
            arrowprops=dict(arrowstyle="-", color=style.FG, lw=0.7, alpha=0.7),
        )

        ax.set_title(f"{display}  ·  L={layer}", pad=8, fontsize=12.5)
        ax.set_xlabel(f"PC1  ({var_frac[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2  ({var_frac[1]*100:.1f}% var)")
        style.finalize_axes(ax)
        ax.set_xlim(xlim)

    # (figure title omitted — poster supplies the heading)
    ax.legend(loc="lower right", fontsize=10)

    out = ROOT / "figures" / "figure_1_pc1_replicable.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
