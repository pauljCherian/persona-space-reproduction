#!/usr/bin/env python3
#Compute PC1/2/3 of the role matrix for each model. This is exactly what they do in the assistant axis paper.
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MODELS, load_baseline_role_matrix, load_baseline_default, pca_basis_path

#
def compute_one(tag: str):
    # get the matrix and the roles
    X, roles = load_baseline_role_matrix(tag)
    #get the defeault
    default = load_baseline_default(tag)

    # det all the non default
    non_default = np.array([r != "default" for r in roles])

    # don't include the deafult assistant
    X_roles = X[non_default]
    
    #get the mean and then center, and do SVD on the mean centered matrix to get PC1,2,3
    role_mean = X_roles.mean(axis=0)
    centered = X_roles - role_mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1, pc2, pc3 = Vt[0], Vt[1], Vt[2]

    # Compute the assistant axis with average like Christina lu does. Figure out which direction to have pc1 point in 
    lu_axis_raw = default - role_mean
    if pc1 @ lu_axis_raw < 0:
        pc1 = -pc1

    #compare assistanat axis average with pc1
    var_frac = (S ** 2) / (S ** 2).sum()
    lu_unit = lu_axis_raw / (np.linalg.norm(lu_axis_raw) + 1e-12)
    cos_pc1_lu = float(pc1 @ lu_unit)

    # output dictionary to write to disc
    out = {
        "role_mean": role_mean.astype(np.float32),
        "pc1": pc1.astype(np.float32),
        "pc2": pc2.astype(np.float32),
        "pc3": pc3.astype(np.float32),
        "var_frac_top3": (float(var_frac[0]), float(var_frac[1]), float(var_frac[2])),
        "cum_var_top3": float(var_frac[:3].sum()),
        "cos_pc1_lu": cos_pc1_lu,
        "n_roles_used": int(non_default.sum()),
        "hidden_dim": int(X.shape[1]),
        "source": f"data/baseline/{tag}/vectors  (prior project, mean-over-assistant-token extraction)",
    }

    # write it to disc
    path = pca_basis_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"  {tag}: var_frac top3 = {out['var_frac_top3']}  cumulative={out['cum_var_top3']*100:.1f}%  "
          f"cos(PC1, Lu)={cos_pc1_lu:+.3f}  n={out['n_roles_used']}  saved {path}")


def main():
    for tag in MODELS:
        try:
            compute_one(tag)
        except FileNotFoundError as e:
            print(f"  {tag}: SKIPPED — no baseline available ({e})")


if __name__ == "__main__":
    main()
