#!/usr/bin/env python3
"""Evil abliteration — structure only (re-centered), evil-coded roles, 2x2: k=2/4/8/12.
Arrow = displacement minus the shared shift of the 5 evil roles (structural residual);
per-role residual in a corner box. Output: gallery/evil_structure_only.png

Reads: data/steering/pca_basis/llama8b_basis.pt, data/comparison/llama-3.1-8b/vectors/*.pt,
       data/abliteration_concept/runs/evil_struct4/positions.csv
"""
import csv, glob, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_pt
b = torch.load(ROOT / "data/steering/pca_basis/llama8b_basis.pt", map_location="cpu", weights_only=False)
mu = np.asarray(b["role_mean"], float); p1 = np.asarray(b["pc1"], float); p1 /= np.linalg.norm(p1)
p2 = np.asarray(b["pc2"], float); p2 /= np.linalg.norm(p2)
def proj(v): c = np.asarray(v, float) - mu; return np.array([c @ p1, c @ p2])
cloud = np.array([proj(load_pt(f)) for f in glob.glob(str(ROOT / "data/comparison/llama-3.1-8b/vectors/*.pt"))])
cloud_std = cloud.std(0).mean()
def committed(r): return proj(load_pt(ROOT / f"data/comparison/llama-3.1-8b/vectors/{r}.pt"))
rows = list(csv.DictReader(open(ROOT / "data/abliteration_concept/runs/evil_struct4/positions.csv")))
base = {r["role"]: np.array([float(r["pc1"]), float(r["pc2"])]) for r in rows if r["condition"] == "baseline"}
def disp(cond, role):
    for r in rows:
        if r["condition"] == cond and r["role"] == role: return np.array([float(r["pc1"]), float(r["pc2"])]) - base[role]
    return np.zeros(2)
ROLES = ["demon", "vampire", "criminal", "predator", "saboteur"]; tc = "#b0432f"
KS = [2, 4, 8, 12]

fig, axes = plt.subplots(2, 2, figsize=(11, 10.2))
for ax, k in zip(axes.flat, KS):
    cond = f"evil_k{k}"; shift = np.mean([disp(cond, r) for r in ROLES], axis=0)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, c="#c8d0da", alpha=.5, zorder=1, linewidths=0)
    resid = {}
    for role in ROLES:
        c0 = committed(role); d = disp(cond, role) - shift; resid[role] = np.hypot(*d)
        ax.annotate("", xy=c0 + d, xytext=c0, zorder=4, arrowprops=dict(arrowstyle="-|>", color=tc, lw=2.2, alpha=.9))
        ax.scatter(*c0, s=58, c=tc, edgecolor="black", lw=.8, zorder=5)
        ax.annotate(role, c0, xytext=(4, 5), textcoords="offset points", fontsize=8.3, color="#333", zorder=6)
    tbl = "residual\n" + "\n".join(f"{r:<9s}{resid[r]:.2f}" for r in sorted(resid, key=resid.get, reverse=True))
    ax.text(0.03, 0.035, tbl, transform=ax.transAxes, fontsize=8.3, family="monospace", va="bottom", ha="left",
            color=tc, zorder=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tc, lw=1, alpha=.95))
    ax.set_xlim(-3.2, 2.2); ax.set_ylim(-2.6, 2.8); ax.axhline(0, color="#eee", lw=.6); ax.axvline(0, color="#eee", lw=.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    mr = np.mean(list(resid.values()))
    ax.set_title(f"evil abliteration · k={k}", fontsize=13, weight="bold", color=tc)
    print(f"k={k}: mean residual {mr:.2f}  ({mr/cloud_std:.0%} of spread)")
fig.suptitle("Evil concept abliteration, evil-coded roles, re-centered (shared shift removed = normalized). All-layers.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(ROOT / "gallery" / "evil_structure_only.png", dpi=150)
print("wrote gallery/evil_structure_only.png  cloud_std=%.2f" % cloud_std)
