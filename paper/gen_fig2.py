"""Generate uncertainty figures: S11 + S21 overlaid in single panel each."""
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pyfes.projects.filter_dnngp import BilateralFilterDNNGP

model = BilateralFilterDNNGP().load("bilat_dnngp_v2.pt")

d = loadmat("bilat_all_sparams_50epsr.mat")
X = np.column_stack([d["epsr"].ravel(), d["freq"].ravel()])
s11 = d["s11"].ravel()
s21 = d["s21"].ravel()
freqs = X[:, 1]
epsr_vals = X[:, 0]

epsr_unique = np.sort(np.unique(epsr_vals))
rng = np.random.RandomState(42)
shuffled = epsr_unique.copy()
rng.shuffle(shuffled)
n_val = max(1, min(10, len(epsr_unique) // 3))
val_epsr = np.sort(shuffled[:n_val])

pick = [val_epsr[0], val_epsr[-1]]

figsize = (4.8, 3.2)

for ax_idx, ev in enumerate(pick):
    print(f"Plotting S11 + S21 uncertainty for epsr = {ev:.4f}")
    m = np.isin(epsr_vals, [ev])

    s11p, s11_std, s21p, s21_std = model.predict_with_uncertainty(X[m])

    f_ghz = freqs[m] / 1e9
    idx = np.argsort(f_ghz)
    f_ghz = f_ghz[idx]
    s11_true = s11[m][idx]
    s11_pred = s11p[idx]
    s11_sigma = s11_std[idx]
    s21_true = s21[m][idx]
    s21_pred = s21p[idx]
    s21_sigma = s21_std[idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # S11 uncertainty band + mean + truth
    ax.fill_between(
        f_ghz,
        20 * np.log10(np.abs(s11_pred) + 1e-15) - 3 * s11_sigma,
        20 * np.log10(np.abs(s11_pred) + 1e-15) + 3 * s11_sigma,
        alpha=0.15, color="C0", label=r"S$_{11}$ $\pm 3\sigma$",
    )
    ax.plot(f_ghz, 20 * np.log10(np.abs(s11_pred) + 1e-15),
            "-", color="C0", linewidth=1.5, label=r"S$_{11}$ pred")
    ax.plot(f_ghz, 20 * np.log10(np.abs(s11_true) + 1e-15),
            "o", color="C0", markersize=2.5, alpha=0.5,
            label=r"S$_{11}$ FEM")

    # S21 uncertainty band + mean + truth
    ax.fill_between(
        f_ghz,
        20 * np.log10(np.abs(s21_pred) + 1e-15) - 3 * s21_sigma,
        20 * np.log10(np.abs(s21_pred) + 1e-15) + 3 * s21_sigma,
        alpha=0.15, color="C3", label=r"S$_{21}$ $\pm 3\sigma$",
    )
    ax.plot(f_ghz, 20 * np.log10(np.abs(s21_pred) + 1e-15),
            "-", color="C3", linewidth=1.5, label=r"S$_{21}$ pred")
    ax.plot(f_ghz, 20 * np.log10(np.abs(s21_true) + 1e-15),
            "s", color="C3", markersize=2.5, alpha=0.5,
            label=r"S$_{21}$ FEM")

    ax.set_xlabel("Frequency (GHz)", fontsize=11)
    ax.set_ylabel(r"|S$_{11}$|, |S$_{21}$| (dB)", fontsize=11)
    ax.set_title(rf"Predictive uncertainty, $\varepsilon_r = {ev:.3f}$", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_ylim(-75, 10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
