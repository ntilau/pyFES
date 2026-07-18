"""Generate uncertainty figures: S11 + S21 overlaid, twin axes, proper dB CI."""
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
    print(f"Plotting S11 + S21 uncertainty for epsr = {ev:.4f}")
    m = np.isin(epsr_vals, [ev])
    s11p, s11_std, s21p, s21_std = model.predict_with_uncertainty(X[m])

    f_ghz = freqs[m] / 1e9
    idx = np.argsort(f_ghz)
    f_ghz = f_ghz[idx]

    s11_abs = np.abs(s11p[idx])
    s21_abs = np.abs(s21p[idx])
    s11_dB = 20 * np.log10(s11_abs + 1e-15)
    s21_dB = 20 * np.log10(s21_abs + 1e-15)
    s11_true_dB = 20 * np.log10(np.abs(s11[m][idx]) + 1e-15)
    s21_true_dB = 20 * np.log10(np.abs(s21[m][idx]) + 1e-15)
    s11_sig = s11_std[idx]
    s21_sig = s21_std[idx]

    # S11: linear-magnitude std → dB CI: 20·log10(|S| ± 3σ)
    s11_lo = 20 * np.log10(np.maximum(s11_abs - 2 * s11_sig, 1e-15))
    s11_hi = 20 * np.log10(s11_abs + 2 * s11_sig + 1e-15)

    # S21: model trained on log10|S|.  σ_y = s21_std / (|S|·ln(10))
    #       dB CI = 20·(log10|S| ± 3·σ_y) = s21_dB ± 60·σ_y
    s21_sig_log10 = s21_sig / (np.maximum(s21_abs, 1e-15) * np.log(10))
    s21_lo = s21_dB - 40 * s21_sig_log10
    s21_hi = s21_dB + 40 * s21_sig_log10

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax1.twinx()

    # ── S11 on left axis ──
    ax1.fill_between(f_ghz, s11_lo, s11_hi, alpha=0.3, color="C0",
                     label=r"S$_{11}$ $\pm 2\sigma$")
    ax1.plot(f_ghz, s11_dB, "-", color="C0", lw=1.8, label=r"S$_{11}$ pred")
    ax1.plot(f_ghz, s11_true_dB, "o", color="C0", ms=3, alpha=0.5, label=r"S$_{11}$ FEM")
    ax1.set_ylabel(r"|S$_{11}$| (dB)", fontsize=11, color="C0")
    ax1.set_ylim(-60, 10)
    ax1.tick_params(axis="y", labelcolor="C0", labelsize=9)

    # ── S21 on right axis ──
    ax2.fill_between(f_ghz, s21_lo, s21_hi, alpha=0.3, color="C3",
                     label=r"S$_{21}$ $\pm 2\sigma$")
    ax2.plot(f_ghz, s21_dB, "-", color="C3", lw=1.8, label=r"S$_{21}$ pred")
    ax2.plot(f_ghz, s21_true_dB, "s", color="C3", ms=3, alpha=0.5, label=r"S$_{21}$ FEM")
    ax2.set_ylabel(r"|S$_{21}$| (dB)", fontsize=11, color="C3")
    ax2.set_ylim(-75, 5)
    ax2.tick_params(axis="y", labelcolor="C3", labelsize=9)

    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.set_title(rf"Predictive uncertainty, $\varepsilon_r = {ev:.3f}$", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, loc="lower left", ncol=2)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
