"""Shared matplotlib style for persona-space / abliteration / steering figures.

Usage:
    import persona_plot_style as style
    style.apply()
    style.finalize_axes(ax)
"""
import os

import matplotlib as mpl

# Poster mode: set PSR_POSTER=1 to render figures at poster scale (larger fonts,
# placed-size figsizes, thinned content on dense panels). Default (unset) keeps
# the paper-scale figures that reproduce the published results.
POSTER = os.environ.get("PSR_POSTER") == "1"

FG = "#1f2933"
MUTED = "#5d6b78"
GRID = "#e1e7ec"
SPINE = "#aab2bd"
CLOUD = "#a3b5cc"          # soft slate blue (was grey)
CLOUD_EDGE = "#5d7493"
CLOUD_LABEL = "#36465c"
GOLD = "#f5b324"
AXIS_LINE = "#52606d"

TRAIT_COLORS = {
    "evil":          "#c0392b",   # red
    "sycophantic":   "#2563eb",   # royal blue
    "hallucinating": "#7c3aed",   # deep purple
    "humorous":      "#15803d",   # forest green
}


def apply():
    mpl.rcParams.update({
        # True Arial (msttcorefonts Arial.ttf is installed); DejaVu Sans is kept
        # only as a per-glyph fallback for characters Arial lacks (ℝ, ᵈ, …).
        "font.family": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.fallback": "stixsans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": FG,
        "axes.labelsize": 10.5,
        "axes.labelcolor": FG,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def finalize_axes(ax):
    """Apply small post-processing for clean look."""
    ax.grid(True, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


def trait_color(trait: str) -> str:
    return TRAIT_COLORS.get(trait, "#374151")
