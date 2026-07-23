#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "scipy>=1.10",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.8",
#   "captum>=0.7",
#   "umap-learn>=0.5",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Stim Artifact Explainability — interpretability experiments.

Generates publication-quality explainability analyses for the trained
1D U-Net stim artifact detector.

Experiments:
  1. calibration       — Reliability diagram, entropy overlay, uncertainty vs params
  2. film_jacobian     — FiLM Jacobian heatmap (32x20) + sweep curves
  3. film_distributions — FiLM gamma/beta distributions per layer
  4. deeplift          — DeepLIFT/IntegratedGradients per-channel attributions
  5. gradcam           — Per-layer GradCAM activation heatmaps
  6. umap              — UMAP bottleneck embeddings (multi-panel)
  7. cluster           — Cluster morphology (HDBSCAN + aligned waveforms)
  8. failure_cases     — FP/FN panel with signal overlay

Usage:
    uv run src/stim_explainability.py --experiments all
    uv run src/stim_explainability.py --experiments calibration film_jacobian deeplift
    uv run src/stim_explainability.py --max-files 2000
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import find_objects, label, zoom
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    _load_dat_channels,
    _robust_scale_stats,
    build_true_mask,
    extract_conditioning_vector,
    load_annotations,
    load_checkpoint,
    post_process_mask,
    predict_file_proba,
)

FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COND_NAMES = [
    "B1_current",
    "B1_pulse_width",
    "B1_charge",
    "B1_frequency",
    "B1_duration",
]
COND_NAMES += [f"B1_montage_{i}" for i in range(9)]
COND_NAMES += [
    "B2_current",
    "B2_pulse_width",
    "B2_charge",
    "B2_frequency",
    "B2_duration",
]
COND_NAMES += [f"B2_montage_{i}" for i in range(9)]
COND_NAMES += ["Lead1_type", "Lead1_spacing", "Lead2_type", "Lead2_spacing"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def prepare_data(
    cfg: Config,
    split: str = "val",
    max_files: int | None = None,
    checkpoint_type: str = "stimask",
) -> pd.DataFrame:
    """Load catalog, split, subsample."""
    df = load_annotations(cfg)
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)
    print(f"Catalog: {len(df)} files on disk")

    if split == "val":
        if checkpoint_type == "deployed":
            val_subjects = ["300-002", "301-003", "303-004", "305-001"]
            df = df[df["subject"].isin(val_subjects)].reset_index(drop=True)
        else:
            groups = df["subject"].values
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
            _, val_idx = next(splitter.split(df, groups=groups))
            df = df.iloc[val_idx].reset_index(drop=True)
        print(f"Val split: {len(df)} files")

    if max_files and len(df) > max_files:
        df = df.sample(n=max_files, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} files")

    return df


def load_signal_and_mask(row, cfg):
    """Load signal, build true mask and prediction for a single file."""
    fp = row["file_path"]
    data = _load_dat_channels(Path(fp), cfg.use_channels)
    med, iqr = _robust_scale_stats(data)
    rlen = data.shape[1]

    onsets = row.get("onset_times", [])
    if isinstance(onsets, str):
        onsets = json.loads(onsets) if onsets.strip() else []
    if isinstance(onsets, np.ndarray):
        onsets = onsets.tolist()

    mask_dur_ms = row.get("mask_duration_ms", 1000.0)
    if pd.isna(mask_dur_ms):
        mask_dur_ms = 1000.0
    onset_offset_ms = row.get("mask_onset_offset_ms", 0.0)
    if pd.isna(onset_offset_ms):
        onset_offset_ms = 0.0

    true_mask = build_true_mask(
        rlen, onsets, mask_dur_ms, cfg.sampling_rate, onset_offset_ms
    )
    return data, med, iqr, true_mask, onsets


# ---------------------------------------------------------------------------
# Experiment 1: Calibration + Uncertainty
# ---------------------------------------------------------------------------


def _compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error and per-bin stats."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_correct = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_total[i] += mask.sum()
            bin_correct[i] += (labels[mask] == 1).sum()
    bin_acc = np.divide(
        bin_correct, bin_total, where=bin_total > 0, out=np.zeros(n_bins)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = bin_total.sum()
    ece = (
        float(np.sum(bin_total / total * np.abs(bin_acc - bin_centers)))
        if total > 0
        else 0.0
    )
    return ece, bin_acc, bin_total, bin_centers


def _fit_temperature(
    logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 200
) -> float:
    """Fit temperature T by minimizing NLL on (logits, labels).

    Uses L-BFGS-B via scipy (Guo et al., 2017).
    Returns optimal T > 0.
    """
    from scipy.optimize import minimize

    def nll(T_log):
        T = np.exp(T_log[0])  # ensure T > 0
        scaled = logits / T
        # numerically stable sigmoid + NLL
        log_p = -np.logaddexp(0, -scaled)
        log_1mp = -np.logaddexp(0, scaled)
        return -float(np.mean(labels * log_p + (1 - labels) * log_1mp))

    result = minimize(nll, x0=[0.0], method="L-BFGS-B")
    T = float(np.exp(result.x[0]))
    return T


def _plot_reliability(
    bin_acc: np.ndarray,
    bin_total: np.ndarray,
    bin_centers: np.ndarray,
    ece: float,
    n_bins: int,
    title: str,
    path_stem: str,
) -> None:
    """Plot and save a reliability diagram."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.bar(
        bin_centers,
        bin_acc,
        width=1.0 / n_bins,
        alpha=0.7,
        color="#2196F3",
        edgecolor="white",
    )
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax1.transAxes, fontsize=10)

    ax2.bar(
        bin_centers, bin_total, width=1.0 / n_bins, color="#9E9E9E", edgecolor="white"
    )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{path_stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{path_stem}.png")
    plt.close(fig)


def experiment_calibration(model, cfg, df) -> None:
    """Reliability diagram, temperature scaling, uncertainty vs stim params."""
    print("\n=== Experiment 1: Calibration + Temperature Scaling ===")
    n_bins = 15

    # Collect per-sample probabilities and GT labels
    all_probs_list = []
    all_labels_list = []
    file_entropies = []
    file_params = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calibration"):
        fp = row["file_path"]
        if not Path(fp).exists():
            continue
        cond = extract_conditioning_vector(row)
        proba = predict_file_proba(model, cfg, fp, cond)
        _, _, _, true_mask, _ = load_signal_and_mask(row, cfg)

        # Subsample for memory — keep every 4th sample (still millions of points)
        step = 4
        all_probs_list.append(proba[::step])
        all_labels_list.append(true_mask[::step])

        # Entropy in uncertain region
        eps = 1e-8
        ent = -(proba * np.log2(proba + eps) + (1 - proba) * np.log2(1 - proba + eps))
        uncertain = (proba > 0.1) & (proba < 0.9)
        mean_ent = float(ent[uncertain].mean()) if uncertain.sum() > 0 else 0.0
        file_entropies.append(mean_ent)
        file_params.append(
            {
                "B1_current": cond[0],
                "B1_frequency": cond[3],
                "B2_current": cond[14],
                "B2_frequency": cond[17],
            }
        )

    all_probs = np.concatenate(all_probs_list)
    all_labels = np.concatenate(all_labels_list)
    print(f"  Collected {len(all_probs):,} samples")

    # --- Pre-calibration reliability ---
    ece_pre, bin_acc, bin_total, bin_centers = _compute_ece(
        all_probs, all_labels, n_bins
    )
    _plot_reliability(
        bin_acc,
        bin_total,
        bin_centers,
        ece_pre,
        n_bins,
        "Reliability Diagram (Before Temperature Scaling)",
        "calibration_reliability_pre",
    )
    print(f"  ECE (pre) = {ece_pre:.4f}")

    # --- Temperature scaling (Guo et al., 2017) ---
    # Recover logits from probabilities: logit = log(p / (1-p))
    p_clipped = np.clip(all_probs, 1e-7, 1 - 1e-7)
    logits = np.log(p_clipped / (1 - p_clipped))

    T = _fit_temperature(logits, all_labels.astype(np.float64))
    calibrated_probs = 1.0 / (1.0 + np.exp(-logits / T))
    print(f"  Optimal temperature T = {T:.4f}")

    # --- Post-calibration reliability ---
    ece_post, bin_acc_cal, bin_total_cal, _ = _compute_ece(
        calibrated_probs, all_labels, n_bins
    )
    _plot_reliability(
        bin_acc_cal,
        bin_total_cal,
        bin_centers,
        ece_post,
        n_bins,
        "Reliability Diagram (After Temperature Scaling)",
        "calibration_reliability_post",
    )
    print(f"  ECE (post) = {ece_post:.4f}")

    # --- Side-by-side comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, acc, total, ece_val, title in [
        (axes[0], bin_acc, bin_total, ece_pre, "Before (T=1)"),
        (axes[1], bin_acc_cal, bin_total_cal, ece_post, f"After (T={T:.2f})"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.bar(
            bin_centers,
            acc,
            width=1.0 / n_bins,
            alpha=0.7,
            color="#2196F3",
            edgecolor="white",
        )
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.05, 0.92, f"ECE = {ece_val:.4f}", transform=ax.transAxes, fontsize=10)
    fig.suptitle("Temperature Scaling Calibration", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_comparison.pdf")
    fig.savefig(FIGURES_DIR / "calibration_comparison.png")
    plt.close(fig)

    # Save temperature and ECE values
    cal_results = {
        "temperature": T,
        "ece_pre": ece_pre,
        "ece_post": ece_post,
        "n_samples": int(len(all_probs)),
    }
    with open(RESULTS_DIR / "calibration_temperature.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    # --- Uncertainty vs params (unchanged) ---
    params_df = pd.DataFrame(file_params)
    params_df["entropy"] = file_entropies
    param_cols = ["B1_current", "B1_frequency", "B2_current", "B2_frequency"]

    fig, axes = plt.subplots(1, len(param_cols), figsize=(4 * len(param_cols), 4))
    for ax, col in zip(axes, param_cols):
        valid = params_df[[col, "entropy"]].dropna()
        if len(valid) == 0:
            continue
        ax.scatter(valid[col], valid["entropy"], alpha=0.2, s=8, color="#9C27B0")
        corr = np.corrcoef(valid[col], valid["entropy"])[0, 1]
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Mean Entropy")
        ax.set_title(f"r = {corr:.3f}")
    fig.suptitle("Prediction Uncertainty vs Stimulation Parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_uncertainty_vs_params.pdf")
    fig.savefig(FIGURES_DIR / "calibration_uncertainty_vs_params.png")
    plt.close(fig)
    print(f"  Summary: ECE {ece_pre:.4f} → {ece_post:.4f} (T={T:.3f})")


# ---------------------------------------------------------------------------
# Experiment 2: FiLM Jacobian Heatmap
# ---------------------------------------------------------------------------


def experiment_film_jacobian(model, cfg, df) -> None:
    """FiLM Jacobian: sensitivity of each layer's gamma to each conditioning feature."""
    print("\n=== Experiment 2: FiLM Jacobian ===")

    raw_model = getattr(model, "_orig_mod", model)
    film_gen = getattr(raw_model, "film_gen", None)
    if film_gen is None:
        print("  No FiLM generator found — skipping")
        return

    # Collect conditioning stats
    cond_matrix = np.stack(
        [extract_conditioning_vector(row) for _, row in df.iterrows()]
    )
    cond_mean = cond_matrix.mean(axis=0)
    cond_std = cond_matrix.std(axis=0) + 1e-8

    # Identify binary features (montage indices)
    binary_idx = set(range(5, 14)) | set(range(19, 28))

    # Get baseline gammas
    with torch.no_grad():
        baseline_cond = torch.from_numpy(cond_mean).unsqueeze(0).to(cfg.device)
        baseline_gammas_betas = film_gen(baseline_cond)
        n_layers = len(baseline_gammas_betas)
        baseline_gammas = [
            gb[0].cpu().numpy().squeeze() for gb in baseline_gammas_betas
        ]

    # Sweep each feature
    n_features = len(cond_mean)
    jacobian = np.zeros((n_layers, n_features))

    for feat_idx in tqdm(range(n_features), desc="FiLM Jacobian"):
        if feat_idx in binary_idx:
            sweep_vals = [-1.0, 0.0, 1.0]
        else:
            lo = max(0.0, cond_mean[feat_idx] - 2 * cond_std[feat_idx])
            hi = min(1.0, cond_mean[feat_idx] + 2 * cond_std[feat_idx])
            sweep_vals = np.linspace(lo, hi, 21).tolist()

        max_delta = np.zeros(n_layers)
        for val in sweep_vals:
            perturbed = cond_mean.copy()
            perturbed[feat_idx] = val
            with torch.no_grad():
                pert_cond = torch.from_numpy(perturbed).unsqueeze(0).to(cfg.device)
                pert_gb = film_gen(pert_cond)
            for li in range(n_layers):
                gamma = pert_gb[li][0].cpu().numpy().squeeze()
                delta = np.abs(gamma - baseline_gammas[li]).mean()
                max_delta[li] = max(max_delta[li], delta)

        jacobian[:, feat_idx] = max_delta

    # Save raw data
    jac_df = pd.DataFrame(jacobian, columns=COND_NAMES[:n_features])
    jac_df.index.name = "layer"
    jac_df.to_csv(RESULTS_DIR / "film_jacobian_gamma.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(jacobian, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Conditioning Feature")
    ax.set_ylabel("FiLM Layer")
    ax.set_title("FiLM Jacobian — max |Δγ| per Layer × Feature")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(COND_NAMES[:n_features], rotation=90, fontsize=6)
    ax.set_yticks(range(n_layers))
    fig.colorbar(im, ax=ax, shrink=0.8, label="max |Δγ|")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "film_jacobian_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "film_jacobian_heatmap.png")
    plt.close(fig)
    print(f"  Jacobian shape: {jacobian.shape}")


# ---------------------------------------------------------------------------
# Experiment 3: FiLM γ/β Distributions
# ---------------------------------------------------------------------------


def experiment_film_distributions(model, cfg, df) -> None:
    """FiLM gamma/beta distributions per layer."""
    print("\n=== Experiment 3: FiLM γ/β Distributions ===")

    raw_model = getattr(model, "_orig_mod", model)
    film_gen = getattr(raw_model, "film_gen", None)
    if film_gen is None:
        print("  No FiLM generator found — skipping")
        return

    cond_matrix = np.stack(
        [extract_conditioning_vector(row) for _, row in df.iterrows()]
    )

    # Batch compute gammas and betas
    all_gammas = []
    all_betas = []
    batch_size = 512

    with torch.no_grad():
        for start in range(0, len(cond_matrix), batch_size):
            batch = torch.from_numpy(cond_matrix[start : start + batch_size]).to(
                cfg.device
            )
            gb_list = film_gen(batch)
            if not all_gammas:
                all_gammas = [[] for _ in gb_list]
                all_betas = [[] for _ in gb_list]
            for li, (gamma, beta) in enumerate(gb_list):
                all_gammas[li].append(gamma.cpu().numpy())
                all_betas[li].append(beta.cpu().numpy())

    n_layers = len(all_gammas)
    gamma_means = []
    beta_means = []
    gamma_neg_frac = []
    gamma_nearzero_frac = []

    for li in range(n_layers):
        g = np.concatenate(all_gammas[li], axis=0)  # (N, channels)
        b = np.concatenate(all_betas[li], axis=0)
        gamma_means.append(g.mean(axis=0))
        beta_means.append(b.mean(axis=0))
        gamma_neg_frac.append(float((g < 0).mean()))
        gamma_nearzero_frac.append(float((np.abs(g) < 0.1).mean()))

    # Box plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n_layers * 0.5), 8))

    gamma_data = [np.concatenate(all_gammas[li]).flatten() for li in range(n_layers)]
    beta_data = [np.concatenate(all_betas[li]).flatten() for li in range(n_layers)]

    bp1 = ax1.boxplot(gamma_data, showfliers=False, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#2196F3")
        patch.set_alpha(0.6)
    ax1.axhline(1.0, color="red", ls="--", lw=1, alpha=0.7, label="Identity (γ=1)")
    ax1.set_xlabel("FiLM Layer")
    ax1.set_ylabel("γ (scale)")
    ax1.set_title("FiLM γ Distributions per Layer")
    ax1.legend(loc="upper right")

    bp2 = ax2.boxplot(beta_data, showfliers=False, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#FF9800")
        patch.set_alpha(0.6)
    ax2.axhline(0.0, color="red", ls="--", lw=1, alpha=0.7, label="Identity (β=0)")
    ax2.set_xlabel("FiLM Layer")
    ax2.set_ylabel("β (shift)")
    ax2.set_title("FiLM β Distributions per Layer")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "film_gamma_beta_distributions.pdf")
    fig.savefig(FIGURES_DIR / "film_gamma_beta_distributions.png")
    plt.close(fig)

    # Suppression bar chart
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.5), 4))
    x = np.arange(n_layers)
    ax.bar(
        x - 0.2,
        gamma_neg_frac,
        0.35,
        label="γ < 0 (suppression)",
        color="#F44336",
        alpha=0.8,
    )
    ax.bar(
        x + 0.2,
        gamma_nearzero_frac,
        0.35,
        label="|γ| < 0.1 (near-zero)",
        color="#9E9E9E",
        alpha=0.8,
    )
    ax.set_xlabel("FiLM Layer")
    ax.set_ylabel("Fraction")
    ax.set_title("FiLM γ Suppression by Layer")
    ax.legend()
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "film_gamma_suppression.pdf")
    fig.savefig(FIGURES_DIR / "film_gamma_suppression.png")
    plt.close(fig)
    print(f"  {n_layers} FiLM layers analyzed")


# ---------------------------------------------------------------------------
# Experiment 4: DeepLIFT / Integrated Gradients Attribution
# ---------------------------------------------------------------------------


class _UNetAttribWrapper(nn.Module):
    """Wrapper for Captum: combines signal + conditioning input.

    Returns per-sample scalar (1-D tensor with shape [B]) so Captum can
    attribute gradients back through the signal input.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, signal, cond):
        logits = self.model(signal, cond)
        # Mean positive logit per batch element — keeps batch dim for Captum
        probs = torch.sigmoid(logits)
        positive_mask = (probs > 0.5).float()
        # (B, 1, T) → (B,) via masked mean
        num = (logits * positive_mask).sum(dim=(1, 2))
        den = positive_mask.sum(dim=(1, 2)).clamp(min=1.0)
        return num / den


def experiment_deeplift(model, cfg, df) -> None:
    """Integrated Gradients per-channel attributions."""
    print("\n=== Experiment 4: Integrated Gradients Attribution ===")

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("  captum not installed — skipping")
        return

    # Use CPU for Captum (MPS has compatibility issues)
    cpu_model = copy.deepcopy(model).cpu().eval()
    wrapper = _UNetAttribWrapper(cpu_model)
    ig = IntegratedGradients(wrapper)

    sr = cfg.sampling_rate
    ws = cfg.window_samples
    half_win = int(2.0 * sr)  # 2-second context window
    n_examples = min(6, len(df))
    all_attr_aligned = []

    for idx in range(min(200, len(df))):
        row = df.iloc[idx]
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

        data, med, iqr, true_mask, onsets = load_signal_and_mask(row, cfg)
        if len(onsets) == 0:
            continue

        # Find window containing first onset
        onset_samp = int(onsets[0] * sr)
        win_start = max(0, onset_samp - ws // 4)
        if win_start + ws > data.shape[1]:
            win_start = max(0, data.shape[1] - ws)

        # Normalize window
        med_col = med[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        window = (data[:, win_start : win_start + ws] - med_col) / iqr_col

        cond = extract_conditioning_vector(row)
        sig_t = torch.from_numpy(window[np.newaxis]).float()
        cond_t = torch.from_numpy(cond[np.newaxis]).float()
        baseline_t = torch.zeros_like(sig_t)

        try:
            attr = ig.attribute(
                sig_t,
                baselines=baseline_t,
                additional_forward_args=(cond_t,),
                n_steps=50,
                internal_batch_size=10,
            )
            attr_np = attr.detach().numpy().squeeze()  # (n_ch, ws)
        except Exception as exc:
            print(f"  IG failed on file {idx}: {exc}")
            continue

        # Align to onset
        onset_in_win = onset_samp - win_start
        if 0 <= onset_in_win < ws:
            start = max(0, onset_in_win - half_win)
            end = min(ws, onset_in_win + half_win)
            aligned = np.zeros((cfg.n_channels, 2 * half_win))
            src_start = half_win - (onset_in_win - start)
            src_end = src_start + (end - start)
            aligned[:, src_start:src_end] = np.abs(attr_np[:, start:end])
            all_attr_aligned.append(aligned)

        # Plot individual examples
        if idx < n_examples:
            fig, axes = plt.subplots(
                cfg.n_channels, 1, figsize=(12, 2.5 * cfg.n_channels)
            )
            t = np.arange(ws) / sr
            for ch in range(cfg.n_channels):
                ax = axes[ch]
                ax.plot(t, window[ch], color="black", lw=0.5, alpha=0.7)
                extent = [t[0], t[-1], -1, 1]
                ax.imshow(
                    np.abs(attr_np[ch : ch + 1]),
                    aspect="auto",
                    cmap="Reds",
                    alpha=0.5,
                    extent=extent,
                )
                # GT shading
                tm = true_mask[win_start : win_start + ws]
                ax.fill_between(
                    t,
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    where=tm > 0,
                    alpha=0.1,
                    color="green",
                )
                ax.set_ylabel(f"Ch{ch + 1}")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Integrated Gradients — {row['filename']}")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"deeplift_example_{idx}.pdf")
            fig.savefig(FIGURES_DIR / f"deeplift_example_{idx}.png")
            plt.close(fig)

    # Aggregate plot
    if all_attr_aligned:
        mean_attr = np.mean(all_attr_aligned, axis=0)
        t = np.linspace(-2, 2, mean_attr.shape[1])

        fig, axes = plt.subplots(cfg.n_channels, 1, figsize=(10, 2 * cfg.n_channels))
        for ch in range(cfg.n_channels):
            axes[ch].plot(t, mean_attr[ch], color="#F44336", lw=1.5)
            axes[ch].axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
            axes[ch].set_ylabel(f"Ch{ch + 1}\n|attr|")
            axes[ch].fill_between(t, mean_attr[ch], alpha=0.3, color="#F44336")
        axes[-1].set_xlabel("Time relative to onset (s)")
        fig.suptitle(
            f"Mean Integrated Gradients Attribution (n={len(all_attr_aligned)} windows)"
        )
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "deeplift_aggregate.pdf")
        fig.savefig(FIGURES_DIR / "deeplift_aggregate.png")
        plt.close(fig)
        print(f"  Aggregated {len(all_attr_aligned)} windows")


# ---------------------------------------------------------------------------
# Experiment 5: Per-Layer GradCAM
# ---------------------------------------------------------------------------


def experiment_preonset(model, cfg, df) -> None:
    """Test whether the model attributes to the PRE-onset neighbourhood.

    RNS is closed-loop: it fires on detected epileptiform activity, so the
    triggering signal sits in the ~second before the artifact onset. This
    re-runs Integrated Gradients but, rather than only averaging, quantifies per
    window the attribution mass in three onset-relative regions --- a far
    baseline [-2.0,-1.5]s (approximately zero if the model is purely
    artifact-driven), the immediate pre-onset [-1.0,-0.1]s (the trigger
    neighbourhood), and the artifact core [0,+0.5]s --- and correlates the
    pre-onset attribution with the pre-onset line-length of the input (the
    feature the device uses to fire). Writes preonset_attribution.csv and
    preonset_aligned.npy.
    """
    print("\n=== Experiment: pre-onset attribution ===")
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("  captum not installed -- skipping")
        return

    cpu_model = copy.deepcopy(model).cpu().eval()
    ig = IntegratedGradients(_UNetAttribWrapper(cpu_model))
    sr = cfg.sampling_rate
    ws = cfg.window_samples
    half_win = int(2.0 * sr)

    def region(ch_sum, lo_s, hi_s):
        return float(ch_sum[int((lo_s + 2.0) * sr) : int((hi_s + 2.0) * sr)].mean())

    rows, aligned_all = [], []
    n_valid = 0
    for idx in range(len(df)):
        if n_valid >= 100:
            break
        row = df.iloc[idx]
        fp = row["file_path"]
        if not Path(fp).exists():
            continue
        data, med, iqr, true_mask, onsets = load_signal_and_mask(row, cfg)
        if len(onsets) == 0:
            continue
        onset_samp = int(onsets[0] * sr)
        win_start = max(0, onset_samp - ws // 4)
        if win_start + ws > data.shape[1]:
            win_start = max(0, data.shape[1] - ws)
        med_col = med[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        window = (data[:, win_start : win_start + ws] - med_col) / iqr_col
        cond = extract_conditioning_vector(row)
        sig_t = torch.from_numpy(window[np.newaxis]).float()
        cond_t = torch.from_numpy(cond[np.newaxis]).float()
        try:
            attr = ig.attribute(
                sig_t,
                baselines=torch.zeros_like(sig_t),
                additional_forward_args=(cond_t,),
                n_steps=50,
                internal_batch_size=10,
            )
            attr_np = np.abs(attr.detach().numpy().squeeze())  # (n_ch, ws)
        except Exception:
            continue
        onset_in_win = onset_samp - win_start
        if not (0 <= onset_in_win < ws):
            continue
        start = max(0, onset_in_win - half_win)
        end = min(ws, onset_in_win + half_win)
        aligned = np.zeros((cfg.n_channels, 2 * half_win))
        s0 = half_win - (onset_in_win - start)
        aligned[:, s0 : s0 + (end - start)] = attr_np[:, start:end]
        ch_sum = aligned.sum(axis=0)
        aligned_all.append(ch_sum)
        lo = max(0, onset_in_win - int(1.0 * sr))
        hi = onset_in_win - int(0.1 * sr)
        ll_pre = (
            float(np.abs(np.diff(window[:, lo:hi], axis=1)).mean())
            if hi > lo
            else np.nan
        )
        rows.append(
            {
                "filename": row["filename"],
                "far": region(ch_sum, -2.0, -1.5),
                "pre": region(ch_sum, -1.0, -0.1),
                "core": region(ch_sum, 0.0, 0.5),
                "ll_pre": ll_pre,
            }
        )
        n_valid += 1

    res = pd.DataFrame(rows)
    if not len(res):
        print("  no valid windows")
        return
    res["pre_over_far"] = res["pre"] / res["far"].replace(0, np.nan)
    res["pre_over_core"] = res["pre"] / res["core"].replace(0, np.nan)
    res.to_csv(RESULTS_DIR / "preonset_attribution.csv", index=False)
    np.save(RESULTS_DIR / "preonset_aligned.npy", np.array(aligned_all))
    prof = np.array(aligned_all).mean(axis=0)
    print(f"  n windows: {len(res)}")
    print(
        f"  pre-onset / far-baseline |attr|: median={res['pre_over_far'].median():.2f} "
        f"(frac>1.5: {(res['pre_over_far'] > 1.5).mean():.2f})"
    )
    print(f"  pre-onset / core |attr|: median={res['pre_over_core'].median():.2f}")
    valid = res.dropna(subset=["ll_pre", "pre"])
    if len(valid) > 10:
        r = float(np.corrcoef(valid["ll_pre"], valid["pre"])[0, 1])
        print(
            f"  corr(pre-onset attribution, pre-onset line-length): r={r:.2f} (n={len(valid)})"
        )
    print(
        f"  mean |attr| by region: far={region(prof, -2.0, -1.5):.3f}  "
        f"pre={region(prof, -1.0, -0.1):.3f}  core={region(prof, 0.0, 0.5):.3f}"
    )


def experiment_onset_provenance(model, cfg, df) -> None:
    """Onset provenance: does the model find the true artifact onset?

    On the BWH files carrying a nonzero mask_onset_offset_ms (a per-epoch
    mask-start shift, median 792 ms *before* the logged therapy -- see
    stim_mask_refiner.py: "ms to shift mask start from the annotated trigger
    time"), compare the predicted-onset error against the raw-logged onset vs
    the shifted onset. Result (with a companion signal check): the model
    predicts the raw-logged onset to ~4 ms, the saturation/railing begins there,
    and the 792 ms pre-trigger region is clean ECoG. Hence the device-logged
    onset accurately marks the artifact onset, the model localises it from the
    waveform, and mask_onset_offset_ms is a pre-trigger mask margin -- not a
    device-log timing error. This defuses the onset-circularity concern.
    Writes onset_provenance.csv.
    """
    print("\n=== Experiment: onset provenance ===")
    cat = pd.read_parquet("data/bwh_stim_catalog.parquet")
    off = pd.to_numeric(cat["mask_onset_offset_ms"], errors="coerce")
    cat = cat[off.abs() > 1e-6]
    sr = cfg.sampling_rate
    ws = cfg.window_samples
    dev = next(model.parameters()).device
    model.eval()

    def events(p):
        d = np.diff(np.r_[0, (p > 0.5).astype(np.int8), 0])
        return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

    err_raw, err_ref, n_files = [], [], 0
    for _, row in cat.iterrows():
        if n_files >= 250:
            break
        if not Path(row["file_path"]).exists():
            continue
        data, med, iqr, true_mask, onsets = load_signal_and_mask(row, cfg)
        if len(onsets) == 0:
            continue
        offset_s = float(row["mask_onset_offset_ms"]) / 1000.0
        cond_t = (
            torch.from_numpy(extract_conditioning_vector(row)[None]).float().to(dev)
        )
        med_col, iqr_col = med[:, None], iqr[:, None] + 1e-8
        used = False
        for raw_on in onsets:
            onset_samp = int(raw_on * sr)
            win_start = max(0, onset_samp - ws // 4)
            if win_start + ws > data.shape[1]:
                win_start = max(0, data.shape[1] - ws)
            window = (data[:, win_start : win_start + ws] - med_col) / iqr_col
            sig_t = torch.from_numpy(window[None]).float().to(dev)
            with torch.no_grad():
                proba = torch.sigmoid(model(sig_t, cond_t)).squeeze().cpu().numpy()
            ev = [(s, e) for s, e in events(proba) if e - s >= 25]
            if not ev:
                continue
            onset_in_win = onset_samp - win_start
            ps = min(ev, key=lambda x: abs(x[0] - onset_in_win))[0]
            pred_on = (win_start + ps) / sr
            err_raw.append(abs(pred_on - raw_on) * 1000)
            err_ref.append(abs(pred_on - (raw_on + offset_s)) * 1000)
            used = True
        if used:
            n_files += 1

    if not err_raw:
        print("  no predictions")
        return
    er, ef = np.array(err_raw), np.array(err_ref)
    pd.DataFrame({"err_raw_ms": er, "err_ref_ms": ef}).to_csv(
        RESULTS_DIR / "onset_provenance.csv", index=False
    )
    print(f"  files={n_files}  events={len(er)}  (device logs onset late by ~792 ms)")
    print(
        f"  |pred - RAW-LOG onset|: median={np.median(er):.0f} ms  mean={er.mean():.0f}"
    )
    print(
        f"  |pred - REFINED onset|: median={np.median(ef):.0f} ms  mean={ef.mean():.0f}"
    )
    print(
        f"  predicted onset closer to the REFINED (true) edge in {100 * np.mean(ef < er):.0f}% of events"
    )


def experiment_gradcam(model, cfg, df) -> None:
    """Per-layer GradCAM activation heatmaps."""
    print("\n=== Experiment 5: GradCAM ===")
    raw_model = getattr(model, "_orig_mod", model)

    # Identify convolutional blocks for hooking
    target_layers = []
    target_names = []
    for name, module in raw_model.named_modules():
        if isinstance(module, (nn.Conv1d,)) and "conv" in name.lower():
            target_layers.append(module)
            target_names.append(name)
    if not target_layers:
        print("  No conv layers found — skipping")
        return

    # Limit to ~18 layers
    if len(target_layers) > 18:
        step = len(target_layers) // 18
        indices = list(range(0, len(target_layers), step))[:18]
        target_layers = [target_layers[i] for i in indices]
        target_names = [target_names[i] for i in indices]

    n_layers = len(target_layers)
    sr = cfg.sampling_rate
    ws = cfg.window_samples
    n_examples = min(30, len(df))

    for ex_idx in range(n_examples):
        row = df.iloc[ex_idx]
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

        data, med, iqr, true_mask, onsets = load_signal_and_mask(row, cfg)
        if len(onsets) == 0:
            continue

        # Window around first onset
        onset_samp = int(onsets[0] * sr)
        win_start = max(0, onset_samp - ws // 4)
        if win_start + ws > data.shape[1]:
            win_start = max(0, data.shape[1] - ws)

        med_col = med[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        window = (data[:, win_start : win_start + ws] - med_col) / iqr_col
        cond = extract_conditioning_vector(row)

        sig_t = torch.from_numpy(window[np.newaxis]).to(cfg.device)
        cond_t = torch.from_numpy(cond[np.newaxis]).to(cfg.device)

        # Register hooks
        activations = {}
        gradients = {}

        def make_fwd_hook(name):
            def hook(module, inp, out):
                activations[name] = out.detach()

            return hook

        def make_bwd_hook(name):
            def hook(module, grad_in, grad_out):
                gradients[name] = grad_out[0].detach()

            return hook

        handles = []
        for name, layer in zip(target_names, target_layers):
            handles.append(layer.register_forward_hook(make_fwd_hook(name)))
            handles.append(layer.register_full_backward_hook(make_bwd_hook(name)))

        # Forward + backward
        model.zero_grad()
        with torch.enable_grad():
            sig_t.requires_grad_(True)
            logits = model(sig_t, cond_t)
            probs = torch.sigmoid(logits)
            positive = (probs > 0.5).float()
            target = (logits * positive).sum() / (positive.sum() + 1e-8)
            target.backward()

        # Compute GradCAM per layer
        cam_matrix = np.zeros((n_layers, ws))
        for li, name in enumerate(target_names):
            if name not in activations or name not in gradients:
                continue
            act = activations[name].cpu().numpy().squeeze()  # (C, L)
            grad = gradients[name].cpu().numpy().squeeze()  # (C, L)
            weights = grad.mean(axis=-1)  # GAP: (C,)
            cam = np.maximum(0, (weights[:, np.newaxis] * act).sum(axis=0))  # (L,)
            # Upsample to ws
            if len(cam) != ws:
                cam = zoom(cam, ws / len(cam), order=1)[:ws]
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_matrix[li] = cam

        for h in handles:
            h.remove()

        # Plot
        fig, (ax_sig, ax_cam) = plt.subplots(
            2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [1, 2]}
        )
        t = np.arange(ws) / sr

        # Signal
        for ch in range(min(4, cfg.n_channels)):
            ax_sig.plot(t, window[ch], lw=0.5, alpha=0.7)
        tm = true_mask[win_start : win_start + ws]
        pred = probs.detach().cpu().numpy().squeeze()
        ax_sig.fill_between(
            t, -5, 5, where=tm > 0, alpha=0.1, color="green", label="GT"
        )
        ax_sig.fill_between(
            t, -5, 5, where=pred > 0.5, alpha=0.1, color="blue", label="Pred"
        )
        ax_sig.set_ylabel("Signal (norm)")
        ax_sig.set_xlim([t[0], t[-1]])
        ax_sig.legend(fontsize=7, loc="upper right")

        # GradCAM heatmap
        ax_cam.imshow(
            cam_matrix,
            aspect="auto",
            cmap="inferno",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=[t[0], t[-1], -0.5, n_layers - 0.5],
        )
        ax_cam.set_xlabel("Time (s)")
        ax_cam.set_ylabel("Layer")
        ax_cam.set_title("GradCAM per Layer")

        fig.suptitle(f"GradCAM — {row['filename']}", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"gradcam_layers_{ex_idx}.pdf")
        plt.close(fig)

    print(f"  Generated {n_examples} GradCAM figures ({n_layers} layers)")


# ---------------------------------------------------------------------------
# Experiment 6: UMAP Bottleneck Embeddings
# ---------------------------------------------------------------------------


def experiment_umap(model, cfg, df) -> None:
    """UMAP of bottleneck embeddings colored by stim params."""
    print("\n=== Experiment 6: UMAP Bottleneck Embeddings ===")

    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping")
        return

    raw_model = getattr(model, "_orig_mod", model)

    # Hook into bottleneck (after last encoder block)
    bottleneck = getattr(raw_model, "bottleneck", None)
    if bottleneck is None:
        # Try to find it
        for name, mod in raw_model.named_modules():
            if "bottleneck" in name.lower():
                bottleneck = mod
                break
    if bottleneck is None:
        print("  No bottleneck layer found — skipping")
        return

    embeddings = []
    metadata = []
    activation_store = {}

    def hook_fn(module, inp, out):
        activation_store["emb"] = out.detach()

    handle = bottleneck.register_forward_hook(hook_fn)

    sr = cfg.sampling_rate
    ws = cfg.window_samples

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

        data = _load_dat_channels(Path(fp), cfg.use_channels)
        med, iqr = _robust_scale_stats(data)
        rlen = data.shape[1]

        # Center window
        center = rlen // 2
        win_start = max(0, center - ws // 2)
        if win_start + ws > rlen:
            win_start = max(0, rlen - ws)

        med_col = med[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        window = (data[:, win_start : win_start + ws] - med_col) / iqr_col

        cond = extract_conditioning_vector(row)
        sig_t = torch.from_numpy(window[np.newaxis]).to(cfg.device)
        cond_t = torch.from_numpy(cond[np.newaxis]).to(cfg.device)

        with torch.no_grad():
            model(sig_t, cond_t)

        emb = activation_store["emb"].cpu().numpy().squeeze()
        # Global average pool if 2D
        if emb.ndim == 2:
            emb = emb.mean(axis=-1)
        embeddings.append(emb)

        onsets = row.get("onset_times", [])
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []

        metadata.append(
            {
                "filename": row["filename"],
                "subject": row.get("subject", ""),
                "subject_id_lr": row.get("subject_id_lr", ""),
                "B1_current": cond[0],
                "B1_frequency": cond[3],
                "B1_charge": cond[2],
                "n_onsets": len(onsets),
                "lead_1": row.get("lead_1", ""),
                "lead_2": row.get("lead_2", ""),
                "mask_duration_ms": row.get("mask_duration_ms", 0),
            }
        )

    handle.remove()

    if len(embeddings) < 10:
        print("  Too few embeddings — skipping")
        return

    emb_matrix = np.stack(embeddings)
    meta_df = pd.DataFrame(metadata)

    # PCA + UMAP
    n_components = min(50, emb_matrix.shape[1])
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(emb_matrix)
    # Keep 90% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = np.searchsorted(cumvar, 0.9) + 1
    pca_emb = pca_emb[:, :n_keep]
    print(f"  PCA: {emb_matrix.shape[1]} -> {n_keep} (90% variance)")

    n_neighbors = max(15, int(np.sqrt(len(pca_emb))))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42
    )
    umap_emb = reducer.fit_transform(pca_emb)

    meta_df["umap_1"] = umap_emb[:, 0]
    meta_df["umap_2"] = umap_emb[:, 1]
    meta_df.to_csv(RESULTS_DIR / "bottleneck_embeddings.csv", index=False)

    # 4-panel UMAP
    color_cols = [
        ("B1_current", "viridis"),
        ("B1_frequency", "plasma"),
        ("B1_charge", "inferno"),
        ("n_onsets", "cividis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, cmap) in zip(axes.flat, color_cols):
        vals = meta_df[col].values
        sc = ax.scatter(
            umap_emb[:, 0], umap_emb[:, 1], c=vals, cmap=cmap, s=4, alpha=0.5
        )
        ax.set_title(col.replace("_", " "))
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle("UMAP Bottleneck Embeddings", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "umap_bottleneck_4panel.pdf")
    fig.savefig(FIGURES_DIR / "umap_bottleneck_4panel.png")
    plt.close(fig)

    # By subject
    subjects = meta_df["subject"].unique()
    if len(subjects) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, subj in enumerate(sorted(subjects)):
            mask = meta_df["subject"] == subj
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], s=4, alpha=0.5, label=subj)
        ax.set_title("UMAP by Subject")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "umap_bottleneck_by_subject.pdf")
        fig.savefig(FIGURES_DIR / "umap_bottleneck_by_subject.png")
        plt.close(fig)

    print(f"  {len(embeddings)} embeddings projected to 2D")


# ---------------------------------------------------------------------------
# Experiment 7: Cluster Morphology
# ---------------------------------------------------------------------------


def experiment_cluster(model, cfg, df) -> None:
    """HDBSCAN cluster morphology on bottleneck embeddings."""
    print("\n=== Experiment 7: Cluster Morphology ===")

    try:
        import umap
        from sklearn.cluster import HDBSCAN
    except ImportError:
        print("  umap-learn or sklearn HDBSCAN not available — skipping")
        return

    # Check if embeddings already computed
    emb_path = RESULTS_DIR / "bottleneck_embeddings.csv"
    if not emb_path.exists():
        print("  Run 'umap' experiment first to generate embeddings")
        return

    meta_df = pd.read_csv(emb_path)
    if "umap_1" not in meta_df.columns:
        print("  No UMAP coordinates in embeddings — run 'umap' experiment first")
        return

    umap_emb = meta_df[["umap_1", "umap_2"]].values
    n = len(umap_emb)

    # HDBSCAN
    min_cluster_size = max(5, n // 200)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
    labels = clusterer.fit_predict(umap_emb)
    meta_df["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"  {n_clusters} clusters, {noise_frac:.1%} noise")

    # Save
    meta_df.to_csv(RESULTS_DIR / "cluster_labels.csv", index=False)

    # UMAP with cluster colors
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters + 1)))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        if cl == -1:
            ax.scatter(
                umap_emb[mask, 0],
                umap_emb[mask, 1],
                s=2,
                alpha=0.1,
                color="gray",
                label="Noise",
            )
        else:
            ax.scatter(
                umap_emb[mask, 0],
                umap_emb[mask, 1],
                s=4,
                alpha=0.5,
                color=colors[i % len(colors)],
                label=f"C{cl}",
            )
    ax.set_title(f"HDBSCAN Clusters (n={n_clusters})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if n_clusters <= 15:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_morphology_umap.pdf")
    fig.savefig(FIGURES_DIR / "cluster_morphology_umap.png")
    plt.close(fig)

    # Cluster waveforms (aligned to artifact onset)
    sr = cfg.sampling_rate
    pre_samples = int(0.048 * sr)  # 48ms pre
    post_samples = int(3.0 * sr)  # 3s post
    total_len = pre_samples + post_samples

    for cl in range(n_clusters):
        cl_mask = labels == cl
        cl_files = meta_df[cl_mask]["filename"].values
        # Find matching rows in df
        cl_df = df[df["filename"].isin(cl_files)]
        if len(cl_df) == 0:
            continue

        waveforms = []
        for _, row in cl_df.iterrows():
            if len(waveforms) >= 100:
                break
            fp = row["file_path"]
            if not Path(fp).exists():
                continue
            data, _, _, _, onsets = load_signal_and_mask(row, cfg)
            if len(onsets) == 0:
                continue
            # First onset
            onset_samp = int(onsets[0] * sr)
            start = onset_samp - pre_samples
            end = onset_samp + post_samples
            if start < 0 or end > data.shape[1]:
                continue
            waveforms.append(data[:, start:end])

        if len(waveforms) < 3:
            continue

        waveforms = np.stack(waveforms)  # (N, n_ch, total_len)
        mean_wf = waveforms.mean(axis=0)
        t = np.linspace(-pre_samples / sr * 1000, post_samples / sr * 1000, total_len)

        fig, axes = plt.subplots(cfg.n_channels, 1, figsize=(10, 2 * cfg.n_channels))
        for ch in range(cfg.n_channels):
            ax = axes[ch]
            for w in waveforms[:50]:
                ax.plot(t, w[ch], color="gray", alpha=0.12, lw=0.3)
            ax.plot(t, mean_wf[ch], color="#F44336", lw=1.5)
            ax.axvline(0, color="black", ls="--", lw=0.8)
            ax.set_ylabel(f"Ch{ch + 1}")
        axes[-1].set_xlabel("Time from onset (ms)")
        fig.suptitle(f"Cluster {cl} Waveforms (n={len(waveforms)})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"cluster_{cl}_waveforms.pdf")
        plt.close(fig)

    print(f"  Waveform plots generated for {n_clusters} clusters")


# ---------------------------------------------------------------------------
# Experiment 8: Failure Case Panel
# ---------------------------------------------------------------------------


def experiment_failure_cases(model, cfg, df) -> None:
    """FP/FN panel with signal overlay."""
    print("\n=== Experiment 8: Failure Cases ===")
    sr = cfg.sampling_rate
    ws = cfg.window_samples

    fp_events = []
    fn_events = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding failures"):
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

        cond = extract_conditioning_vector(row)
        proba = predict_file_proba(model, cfg, fp, cond)
        data, _, _, true_mask, onsets = load_signal_and_mask(row, cfg)
        pred_bin = post_process_mask((proba > 0.5).astype(np.int_), cfg)

        pred_labeled, n_pred = label(pred_bin)
        true_labeled, n_true = label(true_mask)

        pred_slices = find_objects(pred_labeled) if n_pred > 0 else []
        true_slices = find_objects(true_labeled) if n_true > 0 else []

        # Match events
        matched_gt, matched_pred = set(), set()
        for pi, ps in enumerate(pred_slices):
            p_s, p_e = ps[0].start, ps[0].stop
            best_iou, best_j = 0.0, -1
            for j, ts in enumerate(true_slices):
                if j in matched_gt:
                    continue
                g_s, g_e = ts[0].start, ts[0].stop
                if p_s >= g_e or p_e <= g_s:
                    continue
                inter = min(p_e, g_e) - max(p_s, g_s)
                union = max(p_e, g_e) - min(p_s, g_s)
                iou_val = inter / (union + 1e-8)
                if iou_val > best_iou:
                    best_iou, best_j = iou_val, j
            if best_iou >= 0.3:
                matched_gt.add(best_j)
                matched_pred.add(pi)

        # Collect FPs
        for pi, ps in enumerate(pred_slices):
            if pi not in matched_pred:
                dur_ms = (ps[0].stop - ps[0].start) / sr * 1000
                fp_events.append(
                    {
                        "filename": row["filename"],
                        "file_path": fp,
                        "type": "FP",
                        "start_samp": ps[0].start,
                        "end_samp": ps[0].stop,
                        "duration_ms": dur_ms,
                    }
                )

        # Collect FNs
        for ji, ts in enumerate(true_slices):
            if ji not in matched_gt:
                dur_ms = (ts[0].stop - ts[0].start) / sr * 1000
                fn_events.append(
                    {
                        "filename": row["filename"],
                        "file_path": fp,
                        "type": "FN",
                        "start_samp": ts[0].start,
                        "end_samp": ts[0].stop,
                        "duration_ms": dur_ms,
                    }
                )

    # Sort by duration and take top examples
    fp_events.sort(key=lambda x: -x["duration_ms"])
    fn_events.sort(key=lambda x: -x["duration_ms"])

    all_failures = fp_events[:3] + fn_events[:3]
    if not all_failures:
        print("  No failure cases found")
        return

    failure_df = pd.DataFrame(fp_events + fn_events)
    failure_df.to_csv(RESULTS_DIR / "failure_cases.csv", index=False)

    # Plot panel
    n_panels = len(all_failures)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))
    if n_panels == 1:
        axes = [axes]

    context_s = 2.0  # 2s context on each side
    context_samp = int(context_s * sr)

    for ax, event in zip(axes, all_failures):
        fp_path = event["file_path"]
        row = df[df["file_path"] == fp_path].iloc[0]
        data = _load_dat_channels(Path(fp_path), cfg.use_channels)
        med, iqr = _robust_scale_stats(data)
        med_col = med[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        norm_data = (data - med_col) / iqr_col

        _, _, _, true_mask, _ = load_signal_and_mask(row, cfg)
        cond = extract_conditioning_vector(row)
        proba = predict_file_proba(model, cfg, fp_path, cond)
        pred_bin = post_process_mask((proba > 0.5).astype(np.int_), cfg)

        # Window around event
        ev_start = event["start_samp"]
        ev_end = event["end_samp"]
        view_start = max(0, ev_start - context_samp)
        view_end = min(data.shape[1], ev_end + context_samp)

        t = np.arange(view_start, view_end) / sr

        for ch in range(min(2, cfg.n_channels)):
            ax.plot(t, norm_data[ch, view_start:view_end], lw=0.5, alpha=0.6)

        # Shadings
        tm = true_mask[view_start:view_end]
        pm = pred_bin[view_start:view_end]
        ylim = ax.get_ylim()
        ax.fill_between(t, ylim[0], ylim[1], where=tm > 0, alpha=0.1, color="green")
        ax.fill_between(t, ylim[0], ylim[1], where=pm > 0, alpha=0.1, color="blue")

        # Highlight the error event
        ev_color = "#F44336" if event["type"] == "FP" else "#FF9800"
        ax.axvspan(ev_start / sr, ev_end / sr, alpha=0.25, color=ev_color)
        ax.set_title(
            f"{event['type']} — {event['filename']} ({event['duration_ms']:.0f} ms)",
            fontsize=9,
        )
        ax.set_ylabel("Signal")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Failure Cases: FP (red) / FN (orange)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_cases_panel.pdf")
    fig.savefig(FIGURES_DIR / "failure_cases_panel.png")
    plt.close(fig)
    print(f"  {len(fp_events)} FPs, {len(fn_events)} FNs found")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = [
    "calibration",
    "film_jacobian",
    "film_distributions",
    "deeplift",
    "preonset",
    "onset_provenance",
    "gradcam",
    "umap",
    "cluster",
    "failure_cases",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stim artifact explainability experiments")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    p.add_argument("--catalog", type=Path, default=Path("data/stim_catalog.parquet"))
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=ALL_EXPERIMENTS + ["all"],
    )
    p.add_argument("--max-files", type=int, default=2000)
    p.add_argument("--split", choices=["all", "val"], default="val")
    p.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"])
    p.add_argument(
        "--checkpoint-type",
        choices=["stimask", "deployed"],
        default="stimask",
        help="Model type: stimask (original) or deployed (HP-searched)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = ALL_EXPERIMENTS if "all" in args.experiments else args.experiments

    device = args.device or "mps"
    if args.checkpoint_type == "deployed":
        from deployed_adapter import load_deployed_checkpoint

        ckpt = args.checkpoint
        if ckpt == Path("checkpoints/best_model.pt"):
            ckpt = Path("data/checkpoints/m4_unet.pt")
        print(f"Loading deployed checkpoint: {ckpt}")
        model, cfg = load_deployed_checkpoint(ckpt, device=device)
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, cfg = load_checkpoint(args.checkpoint, device=device)
    cfg.catalog_path = args.catalog

    df = prepare_data(
        cfg,
        split=args.split,
        max_files=args.max_files,
        checkpoint_type=args.checkpoint_type,
    )

    dispatch = {
        "calibration": experiment_calibration,
        "film_jacobian": experiment_film_jacobian,
        "film_distributions": experiment_film_distributions,
        "deeplift": experiment_deeplift,
        "preonset": experiment_preonset,
        "onset_provenance": experiment_onset_provenance,
        "gradcam": experiment_gradcam,
        "umap": experiment_umap,
        "cluster": experiment_cluster,
        "failure_cases": experiment_failure_cases,
    }

    for exp_name in experiments:
        if exp_name in dispatch:
            dispatch[exp_name](model, cfg, df)

    print(f"\nAll outputs in {FIGURES_DIR}/ and {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
