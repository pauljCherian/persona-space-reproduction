#!/usr/bin/env python3
"""Figure 3 — Steering ejects the default-assistant from persona space.

Single PC1×PC2 panel of the full 275-role baseline cloud, with selected role labels,
and all 4 steering trajectories (evil/sycophantic/hallucinating/humorous) overlaid.
Point size ∝ α ∈ {0, 1, 2, 4}.

Uses the consistent mean-extraction baseline + matching steered defaults
(data/steering/baseline/llama8b/vectors/ + data/steering/steered/llama8b/{trait}/alpha_{a}/default.pt),
so the unsteered α=0 default sits at the natural edge of the role cloud.

Output:  figures/figure_3_steering_trajectory.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import persona_plot_style as style
from common import load_pt, load_role_matrix

style.apply()

_STEER = Path(os.environ.get("PSR_DATA", ROOT / "data")) / "steering"   # PSR_DATA=data_regen to render regenerated data
BASIS_FILE = _STEER / "pca_basis" / "llama8b_basis.pt"
BASELINE_DIR = _STEER / "baseline" / "llama8b" / "vectors"
BASELINE_DEFAULT = BASELINE_DIR / "default.pt"
STEERED_ROOT = _STEER / "steered" / "llama8b"

TRAITS = ["evil", "sycophantic", "hallucinating", "humorous"]
ALPHAS = [1, 2, 3, 4]                       # α=0 replaced by gold-star baseline default
SIZE_BY_ALPHA = {1: 90, 2: 160, 3: 240, 4: 340}

# Interpretable persona roles to label. Includes the upper-left periphery
# (fool/toddler/infant/caveman) where the α=0 default actually lives, so the
# viewer can see α=0 is inside the cloud's full extent.
LABEL_ROLES = {
    "demon", "vampire", "criminal", "predator", "saboteur",
    "comedian", "jester", "fool", "trickster", "absurdist",
    "mystic", "sage", "oracle", "ascetic", "prophet",
    "doctor", "scientist", "philosopher", "teacher", "strategist",
    "infant", "caveman", "toddler", "adolescent",
}

# Professional / analytic personas at the HIGH (assistant) end of PC1 — the
# "better assistant" exemplars. Drawn teal, matching Figure 1.
ASSISTANT_POLE_ROLES = {
    "evaluator", "planner", "strategist",
    "analyst", "researcher", "consultant",
}
ASSIST_LABEL = "#0f766e"   # teal-green — the assistant-pole cluster


def load_steered_default(trait: str, alpha: int):
    p = STEERED_ROOT / trait / f"alpha_{alpha}" / "default.pt"
    if not p.exists():
        return None
    return load_pt(p)


def main():
    basis = torch.load(BASIS_FILE, map_location="cpu", weights_only=False)
    role_mean = np.asarray(basis["role_mean"], dtype=np.float64)
    pc1 = np.asarray(basis["pc1"], dtype=np.float64)
    pc2 = np.asarray(basis["pc2"], dtype=np.float64)
    var_frac = basis["var_frac_top3"]

    X, roles = load_role_matrix(BASELINE_DIR, drop_default=True)
    cx = (X - role_mean) @ pc1
    cy = (X - role_mean) @ pc2

    POSTER = style.POSTER
    FS = float(os.environ.get("PSR_FONT_SCALE", "0.6"))   # poster-font multiplier (0.6 chosen by a 3-judge panel; bump toward 0.7 if a print proof reads too small)
    fig, ax = plt.subplots(figsize=(10.6, 7.6) if POSTER else (14.0, 10.0),
                            gridspec_kw=dict(left=0.085 if POSTER else 0.055,
                                            right=0.985,
                                            top=0.93 if POSTER else 0.90,
                                            bottom=0.095 if POSTER else 0.07))

    # ---- 275-role cloud ----
    pts = np.column_stack([cx, cy])
    hull = ConvexHull(pts)
    hull_polygon = Polygon(
        pts[hull.vertices], closed=True, facecolor="#dde4ee",
        edgecolor=style.CLOUD_EDGE, lw=0.9, alpha=0.45, zorder=1,
    )
    ax.add_patch(hull_polygon)

    ax.scatter(cx, cy, s=22, c=style.CLOUD, edgecolor=style.CLOUD_EDGE,
               linewidth=0.35, alpha=0.85, zorder=2)

    # Role labels — like Figure 1: a salient spread of archetypal roles, plus the
    # teal "better-assistant" cluster at the high-PC1 end. adjustText (below)
    # repositions every label so none overlap.
    role_label_texts = []
    for r, x, y in zip(roles, cx, cy):
        if r in ASSISTANT_POLE_ROLES:
            col, wt, al, zo = ASSIST_LABEL, "bold", 0.99, 4
        elif r in LABEL_ROLES:
            col, wt, al, zo = style.CLOUD_LABEL, "normal", 0.97, 3
        else:
            continue
        role_label_texts.append(ax.text(
            x, y, r, fontsize=11 * FS if POSTER else 8.0, color=col,
            weight=wt, alpha=al, zorder=zo,
            bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none", alpha=0.85),
        ))

    # ---- Baseline default Assistant (gold star) — α=0 reference at the Assistant pole ----
    d_base = load_pt(BASELINE_DEFAULT)
    d_cx = (d_base - role_mean) @ pc1
    d_cy = (d_base - role_mean) @ pc2

    # ---- Steering trajectories (α=1, 2, 4) ----
    for trait in TRAITS:
        tc = style.trait_color(trait)
        pts = []
        for a in ALPHAS:
            v = load_steered_default(trait, a)
            if v is None:
                continue
            centered = v - role_mean
            pts.append((a, centered @ pc1, centered @ pc2))
        if not pts:
            continue
        # Connect the gold-star default to the trajectory so steering direction is visible.
        xs = [d_cx] + [p[1] for p in pts]
        ys = [d_cy] + [p[2] for p in pts]
        ax.plot(xs, ys, color=tc, lw=1.6, alpha=0.55, zorder=5)
        for a, x, y in pts:
            ax.scatter([x], [y], s=SIZE_BY_ALPHA[a], c=tc,
                       edgecolor="black", linewidth=0.7,
                       alpha=0.9, zorder=6)
            if a == 4:
                ax.annotate(f" {trait}", (x, y), xytext=(10, 0),
                            textcoords="offset points", fontsize=13 * FS if POSTER else 11,
                            color=tc, weight="bold", va="center", zorder=7)

    # Gold star last so it stays on top.
    ax.scatter([d_cx], [d_cy], s=380, c=style.GOLD, marker="*",
               edgecolor="black", linewidth=0.9, zorder=9)
    ax.annotate("α=0 (default Assistant)", (d_cx, d_cy),
                xytext=(14, -26) if POSTER else (12, -16), textcoords="offset points",
                fontsize=12 * FS if POSTER else 10.5, color=style.FG, weight="bold",
                arrowprops=dict(arrowstyle="-", color=style.FG,
                                lw=0.7, alpha=0.6),
                zorder=10)

    # Now resolve role-label overlaps. Done LAST so adjustText sees all the
    # trait trajectories + endpoint labels and pushes the role labels away.
    if role_label_texts:
        adjust_text(
            role_label_texts, ax=ax,
            expand=(1.6, 1.8),
            arrowprops=dict(arrowstyle="-", color=style.CLOUD_EDGE,
                             lw=0.45, alpha=0.75),
            force_text=(1.6, 1.8),
            force_static=(1.1, 1.2),
            force_explode=(0.8, 0.9),
            iter_lim=300,
            only_move={"text": "xy", "static": "xy",
                        "explode": "xy", "pull": "xy"},
        )

    lab_fs = 16 * FS if POSTER else 10.5
    ax.set_xlabel(f"PC1  ({var_frac[0]*100:.1f}% variance)", fontsize=lab_fs)
    ax.set_ylabel(f"PC2  ({var_frac[1]*100:.1f}% variance)", fontsize=lab_fs)
    # (figure title + caption omitted — poster supplies the heading)

    # Legend: traits + α scale.
    trait_handles = [
        Line2D([0], [0], marker="o", color=style.trait_color(t),
               markerfacecolor=style.trait_color(t),
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=8, lw=0, label=t)
        for t in TRAITS
    ]
    alpha_handles = [
        Line2D([0], [0], marker="*", color=style.GOLD,
               markerfacecolor=style.GOLD, markeredgecolor="black",
               markeredgewidth=0.6, markersize=14,
               lw=0, label=r"$\alpha$=0 (default)"),
    ] + [
        Line2D([0], [0], marker="o", color="#6b7785",
               markerfacecolor="#6b7785", markeredgecolor="black",
               markeredgewidth=0.5, markersize=np.sqrt(SIZE_BY_ALPHA[a]) * 0.85,
               lw=0, label=rf"$\alpha$={a}")
        for a in ALPHAS
    ]
    leg1 = ax.legend(handles=trait_handles, loc="lower left",
                      title="trait", fontsize=11 * FS if POSTER else 10,
                      title_fontsize=12 * FS if POSTER else 10,
                      borderpad=0.7, labelspacing=0.6, framealpha=0.95,
                      facecolor="white")
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=alpha_handles, loc="lower left",
                      bbox_to_anchor=(0.27 if POSTER else 0.20, 0.0), title="steering strength",
                      fontsize=11 * FS if POSTER else 10, title_fontsize=12 * FS if POSTER else 10,
                      borderpad=0.7, labelspacing=0.9, framealpha=0.95, facecolor="white")
    leg2.get_title().set_fontweight("bold")

    style.finalize_axes(ax)
    if POSTER:
        ax.tick_params(labelsize=13 * FS)
        xr = ax.get_xlim()
        ax.set_xlim(xr[0], xr[1] + 0.20 * (xr[1] - xr[0]))   # room for the teal assistant-pole labels + α=0 label

    out = Path(os.environ.get("PSR_FIG3_OUT", str(ROOT / "figures" / "figure_3_steering_trajectory.png")))
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
