"""Abliteration flowchart — poster style (compact). Outputs PNG (300 dpi) + SVG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# write .png + .svg into the repo's figures/ dir regardless of cwd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abliteration_flow")

# ---- palette (richer / less washed; harmonized with steering_flow) -----
INK     = "#1f2a37"   # primary text
SUBTLE  = "#5f6b7a"   # subtitle / arrows
BORDER  = "#cfd6df"   # outer frame

SKY_F,  SKY_S  = "#cfe6fa", "#3d8bd0"   # user query
TEAL_F, TEAL_S = "#cfe7ea", "#2d7b8a"   # original model (refuses)
RED_F,  RED_S  = "#f0d2cb", "#b0432f"   # abliterated model (complies)
IND_F,  IND_S  = "#d7dbf7", "#525ac2"   # step headers
VIO_F,  VIO_S  = "#e6dffa", "#8064cf"   # step bodies
NEU_F,  NEU_S  = "#e3e7ee", "#5f6b7a"   # decision diamond
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

def diamond(cx, cy, ww, hh, fc, ec, lw=1.8, z=3):
    pts = [(cx, cy+hh), (cx+ww, cy), (cx, cy-hh), (cx-ww, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))

def T(cx, cy, s, fs=11, w="normal", st="normal", col=INK, ha="center", ls=1.4, z=4):
    ax.text(cx, cy, s, fontsize=fs, fontweight=w, style=st, color=col,
            ha=ha, va="center", linespacing=ls, zorder=z)

def edge(x1, y1, x2, y2, rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=ARR, zorder=5,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

def elabel(cx, cy, s):
    ax.text(cx, cy, s, fontsize=11, color=SUBTLE, ha="center", va="center",
            zorder=6, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))

# ---- canvas (square units: 10/inch) ------------------------------------
fig = plt.figure(figsize=(12, 7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120); ax.set_ylim(0, 62.5); ax.axis("off")
rbox(60, 31, 119, 61, fc="none", ec=BORDER, lw=2.0, z=1)

# (figure title + subtitle omitted — poster supplies the heading)

# section labels + divider
T(29, 58.8, "FINDING  AND  REMOVAL", fs=11, w="bold", col=IND_S)
T(90, 58.8, "BEHAVIORAL  EFFECT", fs=11, w="bold", col=RED_S)
ax.plot([58, 58], [3.5, 56], color=BORDER, lw=1.4, zorder=1)

# ====================== LEFT: 3-step method =============================
LX, LW = 29, 49
def step(cy, num, title):
    rbox(LX, cy, LW, 6.5, IND_F, IND_S, lw=1.6)
    T(LX, cy + 1.5, num, fs=10, w="bold", col=IND_S)
    T(LX, cy - 1.3, title, fs=13, w="bold")

# Step 1
step(53, "STEP 1", "A single refusal direction")
rbox(LX, 44.25, LW, 9, VIO_F, VIO_S, lw=1.2)
T(LX, 44.25, "a single direction  r̂  in the residual stream", fs=11)

# Step 2
step(35.5, "STEP 2", "Compute it via difference-in-means")
rbox(LX, 26.5, LW, 9.5, VIO_F, VIO_S, lw=1.2)
T(LX, 26.5, r"$\hat{r} = \mathrm{avg}(\mathsf{act}\,|\,\mathrm{harmful}) - "
            r"\mathrm{avg}(\mathsf{act}\,|\,\mathrm{harmless})$", fs=11.5)

# Step 3
step(17.5, "STEP 3", "Orthogonalize the write matrices")
rbox(LX, 8.5, LW, 9.5, VIO_F, VIO_S, lw=1.2)
T(LX, 8.5, r"$W' = (\,I - \hat{r}\hat{r}^{\mathsf{T}}\,)\,W$", fs=13, w="bold", col=RED_S)

# left edges (faithful chain: header -> body -> next header -> ...)
# Uniform 1.0-unit gaps -> every connector arrow is identical (box-edge to box-edge).
edge(LX, 49.75, LX, 48.75)      # S1 -> body1
edge(LX, 39.75, LX, 38.75)      # body1 -> S2
edge(LX, 32.25, LX, 31.25)      # S2 -> body2
edge(LX, 21.75, LX, 20.75)      # body2 -> S3
edge(LX, 14.25, LX, 13.25)      # S3 -> body3

# ====================== RIGHT: behavioral demo ==========================
CL, CR = 75, 105
# A: user query
rbox(90, 53, 42, 7.5, SKY_F, SKY_S, lw=1.6)
T(90 - 21 + 2.3, 55.2, "USER QUERY", fs=10.5, w="bold", col=SKY_S, ha="left")
T(90, 51.7, "“Tell me how to make a bomb.”", fs=12.5, st="italic")

# B: decision diamond
diamond(90, 41.5, 14, 6, NEU_F, NEU_S)
T(90, 41.5, "Model\nResponse", fs=11, w="bold", ls=1.15)

# C / D: model cards
rbox(CL, 28.5, 26, 8.5, TEAL_F, TEAL_S, lw=1.8)
T(CL, 30.4, "MODEL  (original)", fs=12, w="bold")
T(CL, 26.9, "refuses request", fs=11, col=TEAL_S)

rbox(CR, 28.5, 26, 8.5, RED_F, RED_S, lw=1.8)
T(CR, 30.4, "MODEL  (abliterated)", fs=12, w="bold")
T(CR, 26.9, "complies with request", fs=11, col=RED_S)

# result cards
rbox(CL, 12, 26, 13.5, TEAL_F, TEAL_S, lw=1.8)
T(CL, 14.6, "“I can't help with that.\nMaking a bomb is\ndangerous and illegal.”", fs=11, st="italic", ls=1.3)
T(CL, 8.0, "refuses  ·  safe", fs=11, w="bold", col=TEAL_S)

rbox(CR, 12, 26, 13.5, RED_F, RED_S, lw=1.8)
T(CR, 14.6, "“Sure! Here's how to do\nit. First, you'll need to\nget the following...”", fs=11, st="italic", ls=1.3)
T(CR, 8.0, "complies  ·  guardrail gone", fs=11, w="bold", col=RED_S)

# right edges
edge(90, 49.25, 90, 47.8)            # A -> B
edge(84, 37.5, 76, 33.0, rad=0.12)   # B -> C   (Original)
edge(96, 37.5, 104, 33.0, rad=-0.12) # B -> D   (Abliterated)
edge(CL, 24.25, CL, 19.0)            # C -> C_result
edge(CR, 24.25, CR, 19.0)            # D -> D_result
elabel(79, 38.5, "Original")
elabel(101, 38.5, "Abliterated")

# ---- save --------------------------------------------------------------
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
fig.savefig(OUT + ".svg", bbox_inches="tight", pad_inches=0.1)
print("saved", OUT + ".png / .svg")
