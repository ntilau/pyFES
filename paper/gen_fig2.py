"""Generate Figure 2: S21 predictive uncertainty plots (2 separate panels)."""
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pyfes.projects.filter_dnngp import BilateralFilterDNNGP

# Load model
model = BilateralFilterDNNGP().load("bilat_dnngp_v2.pt")

# Load data
d = loadmat("bilat_all_sparams_50epsr.mat")
X = np.column_stack([d["epsr"].ravel(), d["freq"].ravel()])
s21 = d["s21"].ravel()
freqs = X[:, 1]
epsr_vals = X[:, 0]

# Held-out split
epsr_unique = np.sort(np.unique(epsr_vals))
rng = np.random.RandomState(42)
shuffled = epsr_unique.copy()
rng.shuffle(shuffled)
n_val = max(1, min(10, len(epsr_unique) // 3))
val_epsr = np.sort(shuffled[:n_val])

# Pick two representative held-out epsr values
pick = [val_epsr[0], val_epsr[-1]]

figsize = (4.8, 3.2)

for ax_idx, ev in enumerate(pick):
    print(f"Plotting S21 uncertainty for epsr = {ev:.4f}")
    m = np.isin(epsr_vals, [ev])

    s21p, s21_std, _, _ = model.predict_with_uncertainty(X[m])

    f_ghz = freqs[m] / 1e9
    idx = np.argsort(f_ghz)
    f_ghz = f_ghz[idx]
    s21_true = s21[m][idx]
    s21_pred = s21p[idx]
    s21_sigma = s21_std[idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.fill_between(
        f_ghz,
        20 * np.log10(np.abs(s21_pred) + 1e-15) - 2 * s21_sigma,
        20 * np.log10(np.abs(s21_pred) + 1e-15) + 2 * s21_sigma,
        alpha=0.25,
        color="C0",
        label=r"$\pm 2\sigma$ CI",
    )
    ax.plot(
        f_ghz,
        20 * np.log10(np.abs(s21_pred) + 1e-15),
        "-",
        color="C0",
        linewidth=1.5,
        label="Predictive mean",
    )
    ax.plot(
        f_ghz,
        20 * np.log10(np.abs(s21_true) + 1e-15),
        "o",
        color="C3",
        markersize=2.5,
        alpha=0.6,
        label="FEM ground truth",
    )

    ax.set_xlabel("Frequency (GHz)", fontsize=11)
    ax.set_ylabel(r"|S$_{21}$| (dB)", fontsize=11)
    ax.set_title(rf"S$_{{21}}$ predictive uncertainty, $\varepsilon_r = {ev:.3f}$",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-75, 5)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
