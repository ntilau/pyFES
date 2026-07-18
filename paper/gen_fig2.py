"""Generate uncertainty figures: each shows S11 (left) + S21 (right) panels."""
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

# Held-out split
epsr_unique = np.sort(np.unique(epsr_vals))
rng = np.random.RandomState(42)
shuffled = epsr_unique.copy()
rng.shuffle(shuffled)
n_val = max(1, min(10, len(epsr_unique) // 3))
val_epsr = np.sort(shuffled[:n_val])

pick = [val_epsr[0], val_epsr[-1]]

for ax_idx, ev in enumerate(pick):
    print(f"Plotting uncertainty for epsr = {ev:.4f}")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.2))

    for ax, st, sp, sig, label, ylim, color in [
        (ax1, s11_true, s11_pred, s11_sigma, r"S$_{11}$", (-40, 5), "C0"),
        (ax2, s21_true, s21_pred, s21_sigma, r"S$_{21}$", (-75, 5), "C0"),
    ]:
        ax.fill_between(
            f_ghz,
            20 * np.log10(np.abs(sp) + 1e-15) - 2 * sig,
            20 * np.log10(np.abs(sp) + 1e-15) + 2 * sig,
            alpha=0.25, color=color, label=r"$\pm 2\sigma$ CI",
        )
        ax.plot(f_ghz, 20 * np.log10(np.abs(sp) + 1e-15),
                "-", color=color, linewidth=1.5, label="Predictive mean")
        ax.plot(f_ghz, 20 * np.log10(np.abs(st) + 1e-15),
                "o", color="C3", markersize=2.5, alpha=0.6, label="FEM")
        ax.set_xlabel("Frequency (GHz)", fontsize=11)
        ax.set_ylabel(f"|{label}| (dB)", fontsize=11)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    ax1.set_title(rf"S$_{{11}}$ uncertainty, $\varepsilon_r = {ev:.3f}$", fontsize=10)
    ax2.set_title(rf"S$_{{21}}$ uncertainty, $\varepsilon_r = {ev:.3f}$", fontsize=10)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
