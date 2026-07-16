"""DNN-GP surrogate model for bilateral filter S-parameters.

Trains a Deep Neural Network Gaussian Process to predict complex
S-parameters as a function of substrate permittivity (εᵣ) and frequency.

Uses four independent DNN-GP models for Re(S₁₁), Im(S₁₁), Re(S₂₁), Im(S₂₁).
Includes harmonic frequency features (sin/cos up to 3rd harmonic) to help
the model capture the wave-like nature of the S-parameter response.
"""

import numpy as np
from scipy import sparse

import torch
import gpytorch

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear, assemble_waveguide_port
from ..projects._utils import scattering_parameters

# ──────────────────────────────────────────────
# DNN feature extractor
# ──────────────────────────────────────────────


class FeatureExtractor(torch.nn.Module):
    """Deep neural network mapping features → learned features for GP."""

    def __init__(self, in_dim=8, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DNN_GP(gpytorch.models.ExactGP):
    """Exact GP with DNN-based feature extraction."""

    def __init__(self, train_x, train_y, likelihood, in_dim=8, feat_dim=128):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = FeatureExtractor(in_dim=in_dim, out_dim=feat_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────


def generate_data(epsr_values=None, n_freqs=81, freq_range=(138e9, 158e9),
                  data_dir="./data", verbose=True):
    """Run FEM solves to build training dataset.

    Parameters
    ----------
    epsr_values : array-like, optional
        Substrate permittivity values to sweep. Defaults to 50 values
        from 2.0 to 2.2.
    n_freqs : int
        Number of frequency points.
    freq_range : tuple
        Frequency range (Hz).
    data_dir : str
        Path to mesh data files.
    verbose : bool
        Print progress.

    Returns
    -------
    X : ndarray, shape (N, 2)
        Input features: (εᵣ, freq_Hz).
    s11, s21 : ndarray, shape (N,)
        Complex S-parameters.
    """
    if epsr_values is None:
        epsr_values = np.linspace(2.0, 2.2, 50)

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_total = len(epsr_values) * n_freqs

    X_e = np.empty(n_total)
    X_f = np.empty(n_total)
    s11_arr = np.empty(n_total, dtype=complex)
    s21_arr = np.empty(n_total, dtype=complex)

    idx = 0
    for epsr in epsr_values:
        sys = {"pOrd": 2, "hOrd": 1}
        mesh = read_poly("BilatFilter", data_dir=data_dir, scale=1e-6)
        mesh["epsr"] = [1, epsr]
        mesh["BC"] = {"Dir": 1, "WP": [11, 12]}
        sys["Height"] = 1.651e-3 / 2
        sys["WPnModes"] = 15
        sys["WPportPlot"] = 1
        sys["WPmodePlot"] = 1
        sys["WPpow"] = 1

        sys, mesh = assemble_linear(sys, mesh)

        for kf, freq in enumerate(freqs):
            sys = assemble_waveguide_port(sys, freq)
            X = sparse.linalg.spsolve(sys["A"], sys["B"])
            sp = scattering_parameters(X, sys)
            s11_arr[idx] = sp[0, (sys["WPportPlot"] - 1) * sys["WPnModes"]
                              + sys["WPmodePlot"] - 1]
            s21_arr[idx] = sp[15, (sys["WPportPlot"] - 1) * sys["WPnModes"]
                              + sys["WPmodePlot"] - 1]
            X_e[idx] = epsr
            X_f[idx] = freq
            idx += 1

        if verbose:
            print(f"  εᵣ={epsr:.3f}  ({idx}/{n_total})")

    X = np.column_stack([X_e, X_f])
    return X, s11_arr, s21_arr


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


class BilateralFilterDNNGP:
    """DNN-GP surrogate model for bilateral filter S-parameters.

    Trains 4 independent DNN-GP models mapping (εᵣ, freq) → S-params,
    all using Re/Im representation:

      Re(S₁₁), Im(S₁₁), Re(S₂₁), Im(S₂₁)

    Input features include harmonic frequency expansions
    (sin/cos up to 3rd harmonic) to help capture the wave-like
    response of the waveguide.
    """

    def __init__(self, feat_dim=128, n_epochs=500, lr=0.005,
                 weight_decay=1e-4, verbose=True):
        self.feat_dim = feat_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.models = [None] * 4  # DNN-GP instances
        self.likelihoods = [None] * 4
        self.scalers_x = None
        self.scalers_y = [None] * 4
        self.is_trained = False
        self._n_feat = 8  # base + harmonic features

        self._target_names = ["Re(S₁₁)", "Im(S₁₁)",
                              "Re(S₂₁)", "Im(S₂₁)"]

    @staticmethod
    def _add_harmonics(X, f_scale=10e9):
        """Augment (εᵣ, f) with sin/cos harmonic features."""
        epsr = X[:, [0]]
        f = X[:, [1]]
        w = 2 * np.pi * f / f_scale
        harm = np.column_stack([
            np.sin(w), np.cos(w),
            np.sin(2 * w), np.cos(2 * w),
            np.sin(3 * w), np.cos(3 * w),
        ])
        return np.column_stack([epsr, f / f_scale, harm])

    def _prepare_targets(self, s11, s21, epsr_vals=None, freq_vals=None):
        """Convert complex S-params to 4 Re/Im training targets."""
        return [s11.real, s11.imag, s21.real, s21.imag]

    def _reconstruct_s11(self, pred_re, pred_im):
        return pred_re + 1j * pred_im

    def _reconstruct_s21(self, pred_re, pred_im):
        return pred_re + 1j * pred_im

    def train(self, X, s11, s21, val_epsr=None, random_seed=42):
        """Train the 4 DNN-GP models.

        Parameters
        ----------
        X : ndarray, shape (N, 2)
            Inputs: (εᵣ, freq_Hz).
        s11, s21 : ndarray, shape (N,)
            Complex S-parameters.
        val_epsr : array-like, optional
            εᵣ values to hold out for validation. If None, 10 values
            are randomly selected.
        random_seed : int
            Seed for validation split.

        Returns
        -------
        dict
            Validation metrics.
        """
        from sklearn.preprocessing import StandardScaler

        epsr_vals = X[:, 0]
        epsr_unique = np.sort(np.unique(epsr_vals))

        # Validation split (adaptive: at most 1/3 of epsr values)
        rng = np.random.RandomState(random_seed)
        shuffled = epsr_unique.copy()
        rng.shuffle(shuffled)
        n_val = max(1, min(10, len(epsr_unique) // 3))
        if val_epsr is None:
            val_epsr = np.sort(shuffled[:n_val])
        train_epsr = np.sort(
            epsr_unique[~np.isin(epsr_unique, val_epsr)]
        )

        train_mask = np.isin(epsr_vals, train_epsr)
        val_mask = np.isin(epsr_vals, val_epsr)

        if self.verbose:
            n_train = np.sum(train_mask)
            n_val = np.sum(val_mask)
            print(f"Training: {n_train} samples "
                  f"({len(train_epsr)} εᵣ values)")
            print(f"Validation: {n_val} samples "
                  f"({len(val_epsr)} εᵣ values)")

        # Normalize inputs with harmonic features
        X_harm = self._add_harmonics(X[train_mask])
        self.scalers_x = StandardScaler().fit(X_harm)
        X_train = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )
        X_val = torch.tensor(
            self.scalers_x.transform(
                self._add_harmonics(X[val_mask])
            ), dtype=torch.float32
        )

        # Prepare targets for full dataset (for S21 phase detrending)
        Y_full = self._prepare_targets(
            s11, s21,
            epsr_vals=X[:, 0], freq_vals=X[:, 1]
        )

        # Train the 4 models — all use DNN-GP (ExactGP)
        for i in range(4):
            name = self._target_names[i]
            if self.verbose:
                print(f"\n── Training {name} ──")

            y_train = Y_full[i][train_mask]
            y_val_true = Y_full[i][val_mask]

            ys = StandardScaler().fit(y_train.reshape(-1, 1))
            self.scalers_y[i] = ys
            Yt = torch.tensor(
                ys.transform(y_train.reshape(-1, 1))[:, 0],
                dtype=torch.float32,
            )

            # DNN-GP (ExactGP) for all outputs
            lik = gpytorch.likelihoods.GaussianLikelihood()
            model = DNN_GP(X_train, Yt, lik,
                           in_dim=self._n_feat, feat_dim=self.feat_dim)
            model.train()
            lik.train()
            opt = torch.optim.AdamW(
                model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.n_epochs
            )
            for epoch in range(self.n_epochs):
                opt.zero_grad()
                out = model(X_train)
                loss = -mll(out, Yt)
                loss.backward()
                opt.step()
                sched.step()
                if self.verbose and (epoch + 1) % 250 == 0:
                    print(f"  Epoch {epoch + 1:4d}  loss={loss.item():.2f}")

            # Evaluate
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_val))
            pred_n = p.mean.numpy()
            pred = ys.inverse_transform(pred_n.reshape(-1, 1))[:, 0]
            rmse = np.sqrt(np.mean((pred - y_val_true)**2))
            if self.verbose:
                print(f"  Val RMSE = {rmse:.4f}")

            self.models[i] = model
            self.likelihoods[i] = lik

        self.is_trained = True

        # Compute aggregate metrics
        s11_true = s11[val_mask]
        s21_true = s21[val_mask]
        s11_pred, s21_pred = self.predict(X[val_mask])
        return self._metrics(s11_true, s21_true, s11_pred, s21_pred)

    def predict(self, X):
        """Predict complex S-parameters at new (εᵣ, freq) points.

        Parameters
        ----------
        X : ndarray, shape (M, 2)
            Input points: (εᵣ, freq_Hz).

        Returns
        -------
        s11_pred, s21_pred : ndarray, shape (M,)
            Predicted complex S-parameters.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. "
                               "Call .train() first.")

        X_harm = self._add_harmonics(X)
        X_norm = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )

        preds = []
        for i in range(4):
            model = self.models[i]
            lik = self.likelihoods[i]
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_norm))
            pred_n = p.mean.numpy()
            pred = self.scalers_y[i].inverse_transform(
                pred_n.reshape(-1, 1)
            )[:, 0]
            preds.append(pred)

        s11_pred = self._reconstruct_s11(preds[0], preds[1])
        s21_pred = self._reconstruct_s21(preds[2], preds[3])
        return s11_pred, s21_pred

    @staticmethod
    def _metrics(s11_true, s21_true, s11_pred, s21_pred):
        """Compute RMSE, relative error, and phase error."""
        def _compute(st, sp):
            rmse_c = np.sqrt(np.mean(np.abs(st - sp)**2))
            if np.mean(np.abs(st)**2) > 0:
                rel = 100 * rmse_c / np.sqrt(np.mean(np.abs(st)**2))
            else:
                rel = float("inf")
            mask = np.abs(st) > 0.02
            if mask.sum() > 0:
                ph = np.mean(
                    np.abs(np.angle(st[mask] / sp[mask]))
                ) * 180 / np.pi
            else:
                ph = 0.0
            return rmse_c, rel, ph

        m11 = _compute(s11_true, s11_pred)
        m21 = _compute(s21_true, s21_pred)
        return {
            "s11_rmse": m11[0], "s11_rel_pct": m11[1],
            "s11_phase_deg": m11[2],
            "s21_rmse": m21[0], "s21_rel_pct": m21[1],
            "s21_phase_deg": m21[2],
        }

    def save(self, path="bilat_dnngp.pt"):
        """Save trained model weights and scalers."""
        state = {
            "scalers_x": self.scalers_x,
            "scalers_y": self.scalers_y,
            "feat_dim": self.feat_dim,
            "model_states": [],
            "lik_states": [],
        }
        for i in range(4):
            state["model_states"].append(self.models[i].state_dict())
            state["lik_states"].append(self.likelihoods[i].state_dict())
        torch.save(state, path)

    def load(self, path="bilat_dnngp.pt"):
        """Load trained model weights and scalers."""
        from sklearn.preprocessing import StandardScaler

        state = torch.load(path, weights_only=False)
        self.feat_dim = state["feat_dim"]
        self.scalers_x = state["scalers_x"]
        self.scalers_y = state["scalers_y"]

        dummy_x = torch.zeros((10, self._n_feat), dtype=torch.float32)
        dummy_y = torch.zeros(10, dtype=torch.float32)

        for i in range(4):
            lik = gpytorch.likelihoods.GaussianLikelihood()
            model = DNN_GP(
                dummy_x, dummy_y, lik,
                in_dim=self._n_feat, feat_dim=self.feat_dim
            )
            model.load_state_dict(state["model_states"][i])
            lik.load_state_dict(state["lik_states"][i])
            self.models[i] = model
            self.likelihoods[i] = lik

        self.is_trained = True
        return self

    def predict_with_uncertainty(self, X):
        """Predict with 2σ confidence intervals.

        Returns
        -------
        s11_pred, s11_std, s21_pred, s21_std : ndarray
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X_harm = self._add_harmonics(X)
        X_norm = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )

        means = []
        stds = []
        for i in range(4):
            model = self.models[i]
            lik = self.likelihoods[i]
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_norm))
            pred_n = p.mean.numpy()
            std_n = np.maximum(p.stddev.numpy(), 0.0)
            pred = self.scalers_y[i].inverse_transform(
                pred_n.reshape(-1, 1)
            )[:, 0]
            means.append(pred)
            std = std_n * np.sqrt(self.scalers_y[i].var_[0])
            stds.append(std)

        s11_pred = self._reconstruct_s11(means[0], means[1])
        s21_pred = self._reconstruct_s21(means[2], means[3])
        s11_std = 0.5 * (stds[0] + stds[1])
        s21_std = 0.5 * (stds[2] + stds[3])
        return s11_pred, s11_std, s21_pred, s21_std


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────


def plot_results(freqs, s11_true, s21_true, s11_pred, s21_pred,
                 test_epsr_values, epsr_test_mask, path=None,
                 title_suffix=""):
    """Plot magnitude and phase of true vs predicted S-parameters.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector (Hz).
    s11_true, s21_true : ndarray
        Ground truth complex S-parameters.
    s11_pred, s21_pred : ndarray
        Predicted complex S-parameters.
    test_epsr_values : ndarray
        εᵣ values corresponding to each test point.
    epsr_test_mask : ndarray or None
        Boolean mask for test set (same length as freqs).
    path : str, optional
        Save path. If None, displays interactively.
    title_suffix : str
        Appended to figure title.
    """
    import matplotlib.pyplot as plt

    from ..post.plot import _build_grid  # mesh helper unused here
    import matplotlib  # noqa: F401 — ensures backend is loaded

    if epsr_test_mask is None:
        epsr_test_mask = np.ones(len(freqs), dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    cm = plt.cm.viridis

    for col, (st, sp, sfx) in enumerate([
        (s11_true, s11_pred, "S₁₁"),
        (s21_true, s21_pred, "S₂₁"),
    ]):
        # Magnitude
        ax = axes[0, col]
        for j, ev in enumerate(np.sort(np.unique(test_epsr_values))):
            m = (np.abs(test_epsr_values - ev)
                 < 1e-6) & epsr_test_mask
            if m.sum() == 0:
                continue
            f_ghz = freqs[m] / 1e9
            ax.plot(f_ghz, 20 * np.log10(np.abs(st[m]) + 1e-15),
                    "k-", lw=0.8, alpha=0.3)
            ax.plot(f_ghz, 20 * np.log10(np.abs(sp[m]) + 1e-15),
                    "--", lw=1.5, color=cm((ev - 2.0) / 0.2))
        ax.set_ylabel(f"|{sfx}| (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-80, 5)

        # Phase
        ax = axes[1, col]
        for j, ev in enumerate(np.sort(np.unique(test_epsr_values))):
            m = (np.abs(test_epsr_values - ev)
                 < 1e-6) & epsr_test_mask
            if m.sum() == 0:
                continue
            mask_hires = np.abs(st[m]) > 0.02
            f_ghz = freqs[m][mask_hires] / 1e9
            if len(f_ghz) == 0:
                continue
            ax.plot(f_ghz, np.angle(st[m][mask_hires], deg=True),
                    "k-", lw=0.8, alpha=0.3)
            ax.plot(f_ghz, np.angle(sp[m][mask_hires], deg=True),
                    "--", lw=1.5, color=cm((ev - 2.0) / 0.2))
        ax.set_ylabel(f"∠{sfx} (deg)")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Frequency (GHz)")

    fig.suptitle(
        f"DNN-GP Bilateral Filter Surrogate{title_suffix}",
        fontsize=13,
    )
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    else:
        plt.show()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def bilateral_filter_dnngp(n_epsr=10, n_freqs=81, n_epochs=500,
                           plot=True, save_model="bilat_dnngp.pt",
                           generate=True, data_dir="./data",
                           mat_file=None):
    """Train a DNN-GP surrogate for the bilateral filter.

    Trains 4 models: DNN-GP (ExactGP) for S11 Re/Im, DNN-only for
    S21 Re/Im. Data can be loaded from a .mat file, generated via
    FEM solves, or loaded from a prior .npz cache.

    Parameters
    ----------
    n_epsr : int
        Number of εᵣ training values in [2.0, 2.2] (used when
        ``generate=True`` and ``mat_file=None``).
    n_freqs : int
        Frequency points per εᵣ value.
    n_epochs : int
        Training epochs per model.
    plot : bool
        Save validation plots.
    save_model : str or None
        Path to save trained model checkpoint.
    generate : bool
        If True and no mat_file, generate FEM data.
        If False and no mat_file, load from ``bilat_dnngp_data.npz``.
    data_dir : str
        Path to mesh data files.
    mat_file : str or None
        Path to a .mat file with ``epsr``, ``freq``, ``s11``, ``s21``
        arrays. Overrides ``generate`` when provided.

    Returns
    -------
    model : BilateralFilterDNNGP
        Trained surrogate model.
    metrics : dict
        Validation metrics.
    """
    from scipy.io import loadmat

    if mat_file is not None:
        print(f"Loading data from {mat_file}...")
        d = loadmat(mat_file)
        # Handle both .mat (scipy.io) and .npz formats
        if "epsr" in d:
            X = np.column_stack([d["epsr"].ravel(), d["freq"].ravel()])
            s11 = d["s11"].ravel()
            s21 = d["s21"].ravel()
            n_epsr = len(np.unique(d["epsr"]))
            print(f"  Loaded {len(X)} samples ({n_epsr} εᵣ values)")
        else:
            raise KeyError(
                ".mat file must contain 'epsr', 'freq', 's11', 's21'"
            )
    elif generate:
        epsr_vals = np.linspace(2.0, 2.2, n_epsr)
        print(f"Generating FEM data ({n_epsr} εᵣ × {n_freqs} freq"
              f" = {n_epsr * n_freqs} samples)...")
        X, s11, s21 = generate_data(
            epsr_values=epsr_vals, n_freqs=n_freqs,
            data_dir=data_dir,
        )
        np.savez(
            "bilat_dnngp_data.npz",
            X=X, s11=s11, s21=s21,
        )
        print("Saved bilat_dnngp_data.npz")
    else:
        print("Loading precomputed data...")
        d = np.load("bilat_dnngp_data.npz")
        X, s11, s21 = d["X"], d["s11"], d["s21"]

    model = BilateralFilterDNNGP(
        feat_dim=64, n_epochs=n_epochs, verbose=True
    )
    metrics = model.train(X, s11, s21)

    if save_model:
        model.save(save_model)

    print(f"\n{'=' * 55}")
    print("Validation metrics:")
    for key in ["s11_rmse", "s11_rel_pct", "s11_phase_deg",
                 "s21_rmse", "s21_rel_pct", "s21_phase_deg"]:
        print(f"  {key}: {metrics[key]:.3g}")
    print(f"{'=' * 55}")
    s11_ok = metrics["s11_rel_pct"] < 1.0
    s21_ok = metrics["s21_rel_pct"] < 1.0
    print(f"S11 {'✓ < 1%' if s11_ok else '✗ ≥ 1%'}  |  "
          f"S21 {'✓ < 1%' if s21_ok else '✗ ≥ 1%'}")

    if plot:
        # Reconstruct predictions for held-out set
        epsr_vals = X[:, 0]
        epsr_unique = np.sort(np.unique(epsr_vals))
        rng = np.random.RandomState(42)
        shuffled = epsr_unique.copy()
        rng.shuffle(shuffled)
        val_epsr = np.sort(shuffled[:10])
        val_mask = np.isin(epsr_vals, val_epsr)

        s11_pred, s21_pred = model.predict(X[val_mask])
        plot_results(
            X[val_mask, 1], s11[val_mask], s21[val_mask],
            s11_pred, s21_pred,
            X[val_mask, 0], None,
            path="bilat_dnngp_results.png",
            title_suffix=f" ({n_epsr} εᵣ, {n_epochs} epochs)",
        )

    return model, metrics
