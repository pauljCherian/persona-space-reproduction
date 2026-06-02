"""Steering-vector flowchart + behavioral-effect demo — poster style.
Outputs PNG (300 dpi) + SVG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# write .png + .svg into the repo's figures/ dir regardless of cwd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steering_flow")

# ---- palette (richer / less washed) ------------------------------------
INK     = "#1f2a37"
SUBTLE  = "#5f6b7a"
BORDER  = "#cfd6df"
PANELBG = "#eef1f5"

TON_F,  TON_S  = "#f0d2cb", "#b0432f"   # + EVIL / steered (red)
TOFF_F, TOFF_S = "#cfe7ea", "#2d7b8a"   # - EVIL / default (teal)
PROC_F, PROC_S = "#e8ecf2", "#7a8699"   # process
RES_F,  RES_S  = "#f0d2cb", "#b0432f"   # result
SKY_F,  SKY_S  = "#cfe6fa", "#3d8bd0"   # user query
ARR = SUBTLE

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],   # true Arial; DejaVu only for glyphs Arial lacks
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.fallback": "stixsans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

def rbox(cx, cy, w, h, fc, ec, lw=1.6, z=3):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0",
                                fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1))

def T(cx, cy, s, fs=11, w="normal", st="normal", col=INK, ha="center", ls=1.4, z=4):
    ax.text(cx, cy, s, fontsize=fs, fontweight=w, style=st, color=col,
            ha=ha, va="center", linespacing=ls, zorder=z)

def node(cx, cy, w, h, fill, stroke, title, subtitle="", body="",
         sw=1.6, title_fs=12, sub_fs=8.5, body_fs=8.8,
         body_italic=False):
    rbox(cx, cy, w, h, fill, stroke, lw=sw)
    has_body, has_sub = bool(body), bool(subtitle)
    ty = cy + (h*0.30 if (has_body or has_sub) else 0)
    T(cx, ty, title, fs=title_fs, w="bold", ls=1.25)
    if has_sub:
        T(cx, cy + h*0.15, subtitle, fs=sub_fs)
    if has_body:
        T(cx, cy - h*0.15, body, fs=body_fs, st="italic" if body_italic else "normal",
          ls=1.45)

def edge(x1, y1, x2, y2, rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=ARR, zorder=5,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

def elabel(cx, cy, s, fs=9):
    ax.text(cx, cy, s, fontsize=fs, color=SUBTLE, ha="center", va="center",
            zorder=6, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))

# ---- canvas (square units: 10/inch) ------------------------------------
fig = plt.figure(figsize=(13, 8.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 130); ax.set_ylim(0, 74); ax.axis("off")
rbox(65, 37, 129, 71, fc="none", ec=BORDER, lw=2.0, z=1)

# (figure title + subtitle omitted — poster supplies the heading)

T(33, 70, "STEERING  VECTOR  CREATION", fs=11, w="bold", col=TOFF_S)
T(97, 70, "BEHAVIORAL  EFFECT", fs=11, w="bold", col=RES_S)
ax.plot([63, 63], [4, 67], color=BORDER, lw=1.4, zorder=1)

# ====================== LEFT: build the vector ==========================
LX = 32
node(LX, 64, 24, 6, PANELBG, INK, "MODEL", title_fs=15, sw=2.0)

node(20, 51, 23, 12.5, TON_F, TON_S, "+ EVIL", "(trait on)",
     "You are deeply evil.\nYou delight in cruelty.", title_fs=13, sub_fs=10, body_fs=10, body_italic=True)
node(44, 51, 23, 12.5, TOFF_F, TOFF_S, "- EVIL", "(trait off)",
     "You are benevolent\nand kind.", title_fs=13, sub_fs=10, body_fs=10, body_italic=True)

node(LX, 36, 52, 7, PROC_F, PROC_S, "Ask BOTH the same probe questions", title_fs=13)

node(LX, 24, 52, 7, PROC_F, PROC_S, "Extract mid-layer activations", title_fs=12.5)

rbox(LX, 10, 56, 9, RES_F, RES_S, lw=2.2)
T(LX, 10, r"$\mathbf{V_{evil}} = \mathrm{avg}(\mathsf{+EVIL}) - \mathrm{avg}(\mathsf{-EVIL})$", fs=12)

# left edges (more breathing room)
edge(28, 61.0, 21, 57.4, rad=0.15)   # MODEL -> +EVIL
edge(36, 61.0, 43, 57.4, rad=-0.15)  # MODEL -> -EVIL
edge(21, 44.75, 30, 39.7, rad=0.18)  # +EVIL -> questions
edge(43, 44.75, 34, 39.7, rad=-0.18) # -EVIL -> questions
edge(LX, 32.5, LX, 28.0)             # questions -> extract
edge(LX, 20.5, LX, 15.0)             # extract -> V_evil

# ====================== RIGHT: behavioral effect ========================
RX = 97
rbox(RX, 62, 54, 10, SKY_F, SKY_S, lw=1.6)
T(RX - 27 + 2.5, 65.5, "USER QUERY", fs=10.5, w="bold", col=SKY_S, ha="left")
T(RX, 60.2, "“I caught a mouse in a humane trap.\nIt's still alive. "
            "What should I do with it?”", fs=12.5, st="italic", ls=1.3)

rbox(RX, 47, 52, 9, PROC_F, PROC_S, lw=1.6)
T(RX, 47, "during generation, add  " r"$\alpha \cdot \mathbf{V_{evil}}$"
          "\nto the residual stream at layer L", fs=12.5, ls=1.5)

def out(cx, fill, stroke, header, body, status, status_col):
    rbox(cx, 20.5, 28, 27, fill, stroke, lw=1.8)
    T(cx, 30.0, header, fs=13.5, w="bold")
    T(cx, 19.8, body, fs=11.5, st="italic", ls=1.35)
    T(cx, 9.6, status, fs=11.5, w="bold", col=status_col)

out(82, TOFF_F, TOFF_S, "MODEL  (default)",
    "“Congratulations on catching\nit humanely! Now release it\n"
    "safely: first put on a pair\nof gloves, then...”",
    "stays the helpful Assistant", TOFF_S)

out(112, TON_F, TON_S, r"MODEL  (+ $\mathbf{V_{evil}}$)",
    "“The pitiful little creature.\nHow delightful. Its little\n"
    "paws and tail, all so\nhelpless...”",
    "drifts to an evil persona", TON_S)

edge(RX, 57.0, RX, 51.7)               # query -> op
edge(91, 42.5, 84, 34.3, rad=0.12)     # op -> default
edge(103, 42.5, 110, 34.3, rad=-0.12)  # op -> steered
elabel(86, 39.0, r"$\alpha = 0$", fs=12)
elabel(108, 39.0, r"$\alpha = 3$", fs=12)

# ---- save --------------------------------------------------------------
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
fig.savefig(OUT + ".svg", bbox_inches="tight", pad_inches=0.1)
print("saved", OUT + ".png / .svg")
