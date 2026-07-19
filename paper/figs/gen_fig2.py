"""Generate figures: Sxx on left axis, RMSE on right axis."""
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
figsize = (5.2, 3.4)

for ax_idx, ev in enumerate(pick):
    print(f"Plotting Sxx + RMSE for epsr = {ev:.4f}")
    m = np.isin(epsr_vals, [ev])
    s11p, s21p = model.predict(X[m])

    f_ghz = freqs[m] / 1e9
    idx = np.argsort(f_ghz)
    f_ghz = f_ghz[idx]

    s11_dB = 20 * np.log10(np.abs(s11p[idx]) + 1e-15)
    s21_dB = 20 * np.log10(np.abs(s21p[idx]) + 1e-15)
    s11_t_dB = 20 * np.log10(np.abs(s11[m][idx]) + 1e-15)
    s21_t_dB = 20 * np.log10(np.abs(s21[m][idx]) + 1e-15)

    # Absolute error in dB: 20·log10(|ΔS|)
    s11_re = 20 * np.log10(
        np.maximum(np.abs(s11p[idx] - s11[m][idx]), 1e-15)
    )
    s21_re = 20 * np.log10(
        np.maximum(np.abs(s21p[idx] - s21[m][idx]), 1e-15)
    )

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax1.twinx()

    # ── Left axis: S-parameters ──
    ax1.plot(f_ghz, s11_dB, "-", color="C0", lw=1.5, label=r"S$_{11}$ pred")
    ax1.plot(f_ghz, s11_t_dB, "o", color="C0", ms=3, alpha=0.5, label=r"S$_{11}$ FEM")
    ax1.plot(f_ghz, s21_dB, "-", color="C3", lw=1.5, label=r"S$_{21}$ pred")
    ax1.plot(f_ghz, s21_t_dB, "s", color="C3", ms=3, alpha=0.5, label=r"S$_{21}$ FEM")
    ax1.set_ylabel(r"|S$_{11}$|, |S$_{21}$| (dB)", fontsize=11)
    ax1.set_ylim(-75, 10)
    ax1.tick_params(labelsize=9)

    # ── Right axis: RMSE ──
    ax2.plot(f_ghz, s11_re, ":", color="C0", lw=1.2, label=r"S$_{11}$ abs. error")
    ax2.plot(f_ghz, s21_re, ":", color="C3", lw=1.2, label=r"S$_{21}$ abs. error")
    ax2.set_ylabel(r"Absolute error (dB)", fontsize=11)
    ax2.set_ylim(-60, 5)
    ax2.tick_params(labelsize=9)

    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.set_title(rf"S-parameter predictions and absolute error, $\varepsilon_r = {ev:.3f}$",
                  fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, loc="lower left", ncol=2)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    path = f"paper/figs/fig_uncertainty_{ax_idx + 1}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
