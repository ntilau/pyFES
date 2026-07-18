"""Generate 6 separate panel figures for the paper."""
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
s11 = d["s11"].ravel()
s21 = d["s21"].ravel()
freqs = X[:, 1]
epsr_vals = X[:, 0]

# Held-out split (matches model training)
epsr_unique = np.sort(np.unique(epsr_vals))
rng = np.random.RandomState(42)
shuffled = epsr_unique.copy()
rng.shuffle(shuffled)
n_val = max(1, min(10, len(epsr_unique) // 3))
val_epsr = np.sort(shuffled[:n_val])
val_mask = np.isin(epsr_vals, val_epsr)

print(f"Held-out epsr values: {val_epsr}")

# Predict on all data
s11_pred, s21_pred = model.predict(X)

epsr_lo, epsr_hi = 2.0, 2.2
cm = plt.cm.viridis


def plot_panel(ax, freqs, st, sp, epsr_vals, val_epsr,
               ylabel, ylim, sfx, is_mag=False):
    """Plot magnitude or phase for one S-param, single panel."""
    for ev in np.sort(np.unique(epsr_vals)):
        m = np.abs(epsr_vals - ev) < 1e-6
        if m.sum() == 0:
            continue
        is_val = val_epsr is not None and np.any(
            np.abs(np.asarray(val_epsr) - ev) < 1e-6
        )
        alpha = 0.8 if is_val else 0.2
        lw = 1.5 if is_val else 0.6
        color = cm((ev - epsr_lo) / (epsr_hi - epsr_lo))
        f_ghz = freqs[m] / 1e9
        if is_mag:
            ax.plot(f_ghz, 20 * np.log10(np.abs(st[m]) + 1e-15),
                    "-", lw=lw, alpha=alpha, color=color)
            ax.plot(f_ghz, 20 * np.log10(np.abs(sp[m]) + 1e-15),
                    "--", lw=lw, alpha=alpha, color=color)
        else:
            # Only show phase where BOTH true and predicted |S| > 0.05
            # to avoid phase noise near reflection nulls (|S11| ~ 0 at ~150 GHz)
            mask_h = (np.abs(st[m]) > 0.05) & (np.abs(sp[m]) > 0.05)
            if mask_h.sum() == 0:
                continue
            f_ghz_h = f_ghz[mask_h]
            ax.plot(f_ghz_h, np.angle(st[m][mask_h], deg=True),
                    "-", lw=lw, alpha=alpha, color=color)
            ax.plot(f_ghz_h, np.angle(sp[m][mask_h], deg=True),
                    "--", lw=lw, alpha=alpha, color=color)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Frequency (GHz)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    # Colorbar for epsr
    norm = matplotlib.colors.Normalize(vmin=epsr_lo, vmax=epsr_hi)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(r"$\varepsilon_r$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)


def plot_error_scatter(ax, freqs, epsr_vals, st, sp, title):
    """Error scatter plot."""
    rel = np.abs(sp - st) / (np.abs(st) + 1e-15) * 100
    sc = ax.scatter(freqs / 1e9, epsr_vals,
                    c=np.log10(np.clip(rel, 1e-4, 100)),
                    s=6, cmap="plasma", alpha=0.6)
    ax.set_xlabel("Frequency (GHz)", fontsize=11)
    ax.set_ylabel(r"$\varepsilon_r$", fontsize=11)
    ax.tick_params(labelsize=9)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}$(error %)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)


# ── 6 individual figures, each sized to fit IEEE column width ──
figsize = (4.8, 3.2)  # roughly column-width at 150 DPI
col = "black"

idx = 0
for st, sp, sfx, prefix in [
    (s11, s11_pred, "S$_{11}$", "s11"),
    (s21, s21_pred, "S$_{21}$", "s21"),
]:
    # Panel 1: magnitude
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_panel(ax, freqs, st, sp, epsr_vals, val_epsr,
               r"|" + sfx + r"| (dB)", (-80, 5), sfx, is_mag=True)
    ax.set_title(sfx + r" magnitude", fontsize=11)
    fig.tight_layout()
    path = f"paper/fig_{prefix}_mag.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Panel 2: phase
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_panel(ax, freqs, st, sp, epsr_vals, val_epsr,
               r"$\angle$" + sfx + r" (deg)", None, sfx, is_mag=False)
    ax.set_title(sfx + r" phase", fontsize=11)
    fig.tight_layout()
    path = f"paper/fig_{prefix}_phase.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Panel 3: error scatter
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_error_scatter(ax, freqs, epsr_vals, st, sp,
                       sfx + r" relative error")
    ax.set_title(sfx + r" relative error", fontsize=11)
    fig.tight_layout()
    path = f"paper/fig_{prefix}_error.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    idx += 1

print("\nAll 6 figures generated.")
