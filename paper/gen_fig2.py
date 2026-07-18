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

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax1.twinx()

    # S11 on left axis (ax1)
    ax1.fill_between(
        f_ghz,
        20 * np.log10(np.abs(s11_pred) + 1e-15) - 3 * s11_sigma,
        20 * np.log10(np.abs(s11_pred) + 1e-15) + 3 * s11_sigma,
        alpha=0.15, color="C0",
    )
    ax1.plot(f_ghz, 20 * np.log10(np.abs(s11_pred) + 1e-15),
             "-", color="C0", linewidth=1.5, label=r"S$_{11}$ pred")
    ax1.plot(f_ghz, 20 * np.log10(np.abs(s11_true) + 1e-15),
             "o", color="C0", markersize=2.5, alpha=0.5,
             label=r"S$_{11}$ FEM")
    ax1.set_ylabel(r"|S$_{11}$| (dB)", fontsize=11, color="C0")
    ax1.set_ylim(-60, 10)
    ax1.tick_params(axis="y", labelcolor="C0", labelsize=9)

    # S21 on right axis (ax2)
    ax2.fill_between(
        f_ghz,
        20 * np.log10(np.abs(s21_pred) + 1e-15) - 3 * s21_sigma,
        20 * np.log10(np.abs(s21_pred) + 1e-15) + 3 * s21_sigma,
        alpha=0.15, color="C3",
    )
    ax2.plot(f_ghz, 20 * np.log10(np.abs(s21_pred) + 1e-15),
             "-", color="C3", linewidth=1.5, label=r"S$_{21}$ pred")
    ax2.plot(f_ghz, 20 * np.log10(np.abs(s21_true) + 1e-15),
             "s", color="C3", markersize=2.5, alpha=0.5,
             label=r"S$_{21}$ FEM")
    ax2.set_ylabel(r"|S$_{21}$| (dB)", fontsize=11, color="C3")
    ax2.set_ylim(-75, 10)
    ax2.tick_params(axis="y", labelcolor="C3", labelsize=9)

    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.set_title(rf"Predictive uncertainty, $\varepsilon_r = {ev:.3f}$", fontsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="lower left", ncol=2)

    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
