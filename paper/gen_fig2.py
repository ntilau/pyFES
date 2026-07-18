"""Generate RMSE curves: S11 and S21 error vs frequency at held-out epsr."""
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
val_mask = np.isin(epsr_vals, val_epsr)

# Predict on all held-out data
s11p, s21p = model.predict(X[val_mask])
s11t = s11[val_mask]
s21t = s21[val_mask]
freqs_val = X[val_mask, 1]
epsr_val = X[val_mask, 0]

figsize = (5.2, 3.4)

# S11: absolute error in dB ← 20·log10(|S_pred/S_true|)
rel_err_s11 = np.abs(s11p - s11t) / np.maximum(np.abs(s11t), 1e-15)
rmse_db_s11 = 20 * np.log10(1 + rel_err_s11)  # approx dB error

# S21: absolute error in dB
rel_err_s21 = np.abs(s21p - s21t) / np.maximum(np.abs(s21t), 1e-15)
rmse_db_s21 = 20 * np.log10(1 + rel_err_s21)

for ax_idx, ev in enumerate([val_epsr[0], val_epsr[-1]]):
    print(f"Plotting RMSE for epsr = {ev:.4f}")
    m = np.isin(epsr_val, [ev])
    f_ghz = freqs_val[m] / 1e9
    idx = np.argsort(f_ghz)
    f_ghz = f_ghz[idx]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax1.twinx()

    # S11 RMSE (dB) on left axis
    s11_err = rmse_db_s11[m][idx]
    ax1.plot(f_ghz, s11_err, "-", color="C0", lw=1.5, label=r"S$_{11}$ RMSE")
    ax1.set_ylabel(r"S$_{11}$ error (dB)", fontsize=11, color="C0")
    ax1.set_ylim(-60, 5)
    ax1.tick_params(axis="y", labelcolor="C0", labelsize=9)

    # S21 RMSE (dB) on right axis
    s21_err = rmse_db_s21[m][idx]
    ax2.plot(f_ghz, s21_err, "-", color="C3", lw=1.5, label=r"S$_{21}$ RMSE")
    ax2.set_ylabel(r"S$_{21}$ error (dB)", fontsize=11, color="C3")
    ax2.set_ylim(-60, 5)
    ax2.tick_params(axis="y", labelcolor="C3", labelsize=9)

    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.set_title(rf"Prediction error, $\varepsilon_r = {ev:.3f}$", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    path = f"paper/fig_uncertainty_{ax_idx + 1}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("Done.")
