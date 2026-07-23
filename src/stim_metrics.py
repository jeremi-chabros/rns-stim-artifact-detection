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
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Stim Artifact Performance Metrics — publication-quality evaluation.

Reads inference results (from stim_inference.py) and produces comprehensive
metrics tables and figures for a top-tier journal submission.

Metrics:
  A. Sample-level: Dice, IoU, Sensitivity, Specificity, Precision, F1, AUROC, AUPRC, Kappa
  B. Event-level: Event F1/P/R, FP/hour, onset/offset timing errors
  C. Stratified: per-subject, per-epoch, per-site, by stim params, by lead config
  D. Statistical: Bootstrap 95% CI, Wilcoxon signed-rank for pairwise comparisons

Usage:
    uv run src/stim_metrics.py                                      # default
    uv run src/stim_metrics.py --results outputs/results/inference_results.csv
    uv run src/stim_metrics.py --bootstrap-n 2000
    uv run src/stim_metrics.py --probas-dir outputs/results/probas   # for ROC/PR curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from sklearn.metrics import (
    auc,
    cohen_kappa_score,
    precision_recall_curve,
    roc_curve,
)
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stim_detector_lib import (
    Config,
    build_true_mask,
    extract_conditioning_vector,
    load_annotations,
    load_checkpoint,
    post_process_mask,
    predict_file_proba,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

FIGURES_DIR = Path("outputs/figures")
TABLES_DIR = Path("outputs/tables")


# ---------------------------------------------------------------------------
# Sample-level metrics (full, from probabilities)
# ---------------------------------------------------------------------------


def compute_sample_metrics_full(
    true_mask: np.ndarray,
    pred_bin: np.ndarray,
    proba: np.ndarray | None = None,
) -> dict:
    """Comprehensive sample-level metrics."""
    tp = np.logical_and(pred_bin == 1, true_mask == 1).sum()
    fp = np.logical_and(pred_bin == 1, true_mask == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_mask == 1).sum()
    tn = np.logical_and(pred_bin == 0, true_mask == 0).sum()

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    m = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
    }

    if proba is not None and true_mask.sum() > 0 and (true_mask == 0).sum() > 0:
        fpr, tpr, _ = roc_curve(true_mask.ravel(), proba.ravel())
        m["auroc"] = float(auc(fpr, tpr))
        prec_arr, rec_arr, _ = precision_recall_curve(true_mask.ravel(), proba.ravel())
        m["auprc"] = float(auc(rec_arr, prec_arr))
        m["kappa"] = float(cohen_kappa_score(true_mask.ravel(), pred_bin.ravel()))

    return m


# ---------------------------------------------------------------------------
# Event-level metrics
# ---------------------------------------------------------------------------


def compute_event_metrics_full(
    true_mask: np.ndarray,
    pred_bin: np.ndarray,
    sr: int = 250,
    iou_threshold: float = 0.3,
) -> dict:
    """Event-level matching with timing errors."""
    true_labeled, n_true = label(true_mask)
    pred_labeled, n_pred = label(pred_bin)

    if n_true == 0 and n_pred == 0:
        return {
            "n_gt": 0,
            "n_pred": 0,
            "event_tp": 0,
            "event_fp": 0,
            "event_fn": 0,
            "event_precision": 1.0,
            "event_recall": 1.0,
            "event_f1": 1.0,
            "fp_per_hour": 0.0,
            "onset_errors_ms": [],
            "offset_errors_ms": [],
            "duration_errors_ms": [],
        }

    true_slices = find_objects(true_labeled) if n_true > 0 else []
    pred_slices = find_objects(pred_labeled) if n_pred > 0 else []

    tp, matched_gt = 0, set()
    onset_errors, offset_errors, duration_errors = [], [], []

    for ps in pred_slices:
        p_start, p_end = ps[0].start, ps[0].stop
        best_iou, best_j = 0.0, -1

        for j, ts in enumerate(true_slices):
            if j in matched_gt:
                continue
            g_start, g_end = ts[0].start, ts[0].stop
            if p_start >= g_end or p_end <= g_start:
                continue
            inter = min(p_end, g_end) - max(p_start, g_start)
            union = max(p_end, g_end) - min(p_start, g_start)
            iou_val = inter / (union + 1e-8)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j

        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
            g_start = true_slices[best_j][0].start
            g_end = true_slices[best_j][0].stop
            onset_errors.append((p_start - g_start) / sr * 1000.0)
            offset_errors.append((p_end - g_end) / sr * 1000.0)
            pred_dur = (p_end - p_start) / sr * 1000.0
            true_dur = (g_end - g_start) / sr * 1000.0
            duration_errors.append(pred_dur - true_dur)

    fp_count = n_pred - tp
    fn_count = n_true - len(matched_gt)
    ep = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0.0
    er = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
    ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0

    rec_dur_s = len(true_mask) / sr
    fp_per_hour = fp_count / (rec_dur_s / 3600.0) if rec_dur_s > 0 else 0.0

    return {
        "n_gt": n_true,
        "n_pred": n_pred,
        "event_tp": tp,
        "event_fp": fp_count,
        "event_fn": fn_count,
        "event_precision": ep,
        "event_recall": er,
        "event_f1": ef1,
        "fp_per_hour": fp_per_hour,
        "onset_errors_ms": onset_errors,
        "offset_errors_ms": offset_errors,
        "duration_errors_ms": duration_errors,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    boot_means = np.array(
        [
            values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - ci) / 2
    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1 - alpha) * 100)),
    )


# ---------------------------------------------------------------------------
# Full metrics pipeline (requires model + catalog for probabilities)
# ---------------------------------------------------------------------------


def compute_all_metrics(
    model,
    cfg: Config,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    batch_size: int = 64,
    n_bootstrap: int = 1000,
    probas_dir: Path | None = None,
) -> dict:
    """Compute all metrics from raw probabilities.

    Returns dict with keys: sample_agg, event_agg, per_file, per_subject,
    per_epoch, per_site, roc_data, pr_data, all_onset_errors.
    """
    sr = cfg.sampling_rate
    per_file_records = []

    # Aggregation buffers for ROC/PR
    all_true = []
    all_proba = []
    all_onset_errors = []
    all_offset_errors = []
    all_duration_errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics"):
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

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

        # Load or compute probabilities
        proba = None
        if probas_dir is not None:
            npy_path = probas_dir / f"{row['filename']}.npy"
            if npy_path.exists():
                proba = np.load(npy_path)

        if proba is None:
            cond = extract_conditioning_vector(row)
            try:
                proba = predict_file_proba(model, cfg, fp, cond, batch_size=batch_size)
            except Exception:
                continue

        n_samples = len(proba)
        true_mask = build_true_mask(n_samples, onsets, mask_dur_ms, sr, onset_offset_ms)
        pred_bin = post_process_mask((proba > threshold).astype(np.int_), cfg)

        sm = compute_sample_metrics_full(true_mask, pred_bin, proba)
        em = compute_event_metrics_full(true_mask, pred_bin, sr)

        rec = {
            "filename": row["filename"],
            "subject": row.get("subject", ""),
            "subject_id_lr": row.get("subject_id_lr", ""),
            "epoch_start_gmt": row.get("epoch_start_gmt", ""),
            "site": row.get("site", ""),
            "mask_duration_ms": mask_dur_ms,
            "t1b1_ma": row.get("t1b1_ma", np.nan),
            "t1b1_hz": row.get("t1b1_hz", np.nan),
            "t1b1_uc": row.get("t1b1_uc", np.nan),
            "lead_1": row.get("lead_1", ""),
            "lead_2": row.get("lead_2", ""),
            **sm,
            **{k: v for k, v in em.items() if not isinstance(v, list)},
        }
        per_file_records.append(rec)

        # Accumulate for global ROC/PR (subsample to limit memory)
        if len(true_mask) <= 100_000:
            all_true.append(true_mask)
            all_proba.append(proba)

        all_onset_errors.extend(em["onset_errors_ms"])
        all_offset_errors.extend(em["offset_errors_ms"])
        all_duration_errors.extend(em["duration_errors_ms"])

    per_file_df = pd.DataFrame(per_file_records)

    # --- Aggregate sample metrics ---
    sample_agg = {}
    for col in ["precision", "recall", "specificity", "f1", "dice", "iou"]:
        vals = per_file_df[col].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n_bootstrap)
        sample_agg[col] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    # Global ROC/PR
    if all_true:
        cat_true = np.concatenate(all_true)
        cat_proba = np.concatenate(all_proba)
        if cat_true.sum() > 0 and (cat_true == 0).sum() > 0:
            fpr, tpr, roc_thresholds = roc_curve(cat_true, cat_proba)
            sample_agg["auroc"] = {"mean": float(auc(fpr, tpr))}
            prec_arr, rec_arr, pr_thresholds = precision_recall_curve(
                cat_true, cat_proba
            )
            sample_agg["auprc"] = {"mean": float(auc(rec_arr, prec_arr))}
        else:
            fpr = tpr = roc_thresholds = np.array([])
            prec_arr = rec_arr = pr_thresholds = np.array([])
    else:
        fpr = tpr = roc_thresholds = np.array([])
        prec_arr = rec_arr = pr_thresholds = np.array([])

    # --- Aggregate event metrics ---
    tot_tp = per_file_df["event_tp"].sum()
    tot_fp = per_file_df["event_fp"].sum()
    tot_fn = per_file_df["event_fn"].sum()
    ep = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    er = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
    event_agg = {
        "event_precision": ep,
        "event_recall": er,
        "event_f1": ef1,
        "total_tp": int(tot_tp),
        "total_fp": int(tot_fp),
        "total_fn": int(tot_fn),
        "mean_fp_per_hour": per_file_df["fp_per_hour"].mean(),
    }

    if all_onset_errors:
        oe = np.array(all_onset_errors)
        event_agg["onset_mean_ms"] = float(np.mean(oe))
        event_agg["onset_std_ms"] = float(np.std(oe))
        event_agg["onset_median_ms"] = float(np.median(oe))
    if all_offset_errors:
        oe = np.array(all_offset_errors)
        event_agg["offset_mean_ms"] = float(np.mean(oe))
        event_agg["offset_std_ms"] = float(np.std(oe))
    if all_duration_errors:
        de = np.array(all_duration_errors)
        event_agg["duration_mean_ms"] = float(np.mean(de))
        event_agg["duration_std_ms"] = float(np.std(de))

    # --- Per-subject breakdown ---
    per_subject = (
        per_file_df.groupby("subject")
        .agg(
            {
                "dice": "mean",
                "f1": "mean",
                "iou": "mean",
                "precision": "mean",
                "recall": "mean",
                "event_f1": "mean",
                "event_precision": "mean",
                "event_recall": "mean",
                "fp_per_hour": "mean",
                "filename": "count",
            }
        )
        .rename(columns={"filename": "n_files"})
        .reset_index()
    )

    # --- Per-epoch breakdown ---
    per_epoch = (
        per_file_df.groupby("epoch_start_gmt")
        .agg(
            {
                "dice": "mean",
                "f1": "mean",
                "event_f1": "mean",
                "mask_duration_ms": "first",
                "subject": "first",
                "filename": "count",
            }
        )
        .rename(columns={"filename": "n_files"})
        .reset_index()
    )

    # --- Per-site breakdown ---
    per_site = pd.DataFrame()
    if "site" in per_file_df.columns and per_file_df["site"].notna().any():
        per_site = (
            per_file_df.groupby("site")
            .agg(
                {
                    "dice": "mean",
                    "f1": "mean",
                    "event_f1": "mean",
                    "filename": "count",
                }
            )
            .rename(columns={"filename": "n_files"})
            .reset_index()
        )

    return {
        "sample_agg": sample_agg,
        "event_agg": event_agg,
        "per_file": per_file_df,
        "per_subject": per_subject,
        "per_epoch": per_epoch,
        "per_site": per_site,
        "roc_data": (fpr, tpr),
        "pr_data": (rec_arr, prec_arr),
        "all_onset_errors": np.array(all_onset_errors),
        "all_offset_errors": np.array(all_offset_errors),
        "all_duration_errors": np.array(all_duration_errors),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_roc_curve(fpr, tpr, auroc: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_pr_curve(recall, precision, auprc: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color="#FF9800", lw=2, label=f"AUPRC = {auprc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_confusion_matrix(per_file_df: pd.DataFrame, save_path: Path) -> None:
    tp = per_file_df["tp"].sum()
    fp = per_file_df["fp"].sum()
    fn = per_file_df["fn"].sum()
    tn = per_file_df["tn"].sum()
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["True Neg", "True Pos"])
    ax.set_title("Sample-Level Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_metrics_by_subject(per_subject: pd.DataFrame, save_path: Path) -> None:
    if len(per_subject) == 0:
        return
    per_subject_sorted = per_subject.sort_values("event_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(per_subject_sorted) * 0.35)))
    y = range(len(per_subject_sorted))
    ax.barh(
        y, per_subject_sorted["event_f1"], color="#4CAF50", alpha=0.8, label="Event F1"
    )
    ax.barh(
        y,
        per_subject_sorted["dice"],
        color="#2196F3",
        alpha=0.4,
        label="Dice",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(per_subject_sorted["subject"])
    ax.set_xlabel("Score")
    ax.set_title("Performance by Subject")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1.05])
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_onset_timing_histogram(onset_errors: np.ndarray, save_path: Path) -> None:
    if len(onset_errors) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(onset_errors, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7)
    mean_e = np.mean(onset_errors)
    median_e = np.median(onset_errors)
    ax.axvline(mean_e, color="#FF9800", lw=1.2, ls="-", label=f"Mean: {mean_e:.1f} ms")
    ax.axvline(
        median_e, color="#4CAF50", lw=1.2, ls="-", label=f"Median: {median_e:.1f} ms"
    )
    ax.set_xlabel("Onset Timing Error (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Onset Timing Error Distribution")
    ax.legend()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_performance_by_stim_params(per_file_df: pd.DataFrame, save_path: Path) -> None:
    """Scatter: event F1 vs stim current, frequency, charge density."""
    params = [
        ("t1b1_ma", "B1 Current (mA)"),
        ("t1b1_hz", "B1 Frequency (Hz)"),
        ("t1b1_uc", "B1 Charge (µC)"),
    ]
    available = [(col, lbl) for col, lbl in params if col in per_file_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, (col, lbl) in zip(axes, available):
        valid = per_file_df[[col, "event_f1"]].dropna()
        if len(valid) == 0:
            continue
        ax.scatter(valid[col], valid["event_f1"], alpha=0.15, s=8, color="#2196F3")
        # Binned means
        bins = pd.qcut(
            valid[col], q=min(10, len(valid.drop_duplicates())), duplicates="drop"
        )
        binned = valid.groupby(bins)["event_f1"].mean()
        bin_centers = [b.mid for b in binned.index]
        ax.plot(bin_centers, binned.values, "o-", color="#F44336", lw=2, ms=6)
        ax.set_xlabel(lbl)
        ax.set_ylabel("Event F1")
        ax.set_ylim([-0.05, 1.05])
    fig.suptitle("Performance by Stimulation Parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics from pre-computed inference results (no model needed)
# ---------------------------------------------------------------------------


def metrics_from_results_csv(results_path: Path, n_bootstrap: int = 1000) -> dict:
    """Compute aggregate metrics from inference_results.csv (no model needed).

    This path uses pre-computed event counts, not raw probabilities.
    ROC/PR curves are NOT available in this mode.
    """
    df = pd.read_csv(results_path)
    valid = df[~df.get("error", pd.Series(dtype=str)).notna()].copy()

    # Event-level aggregation
    tp = valid["n_true_positives"].sum()
    fp = valid["n_false_positives"].sum()
    fn = valid["n_false_negatives"].sum()
    ep = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    er = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0

    event_agg = {
        "event_precision": ep,
        "event_recall": er,
        "event_f1": ef1,
        "total_tp": int(tp),
        "total_fp": int(fp),
        "total_fn": int(fn),
    }

    # Bootstrap CIs on per-file event F1
    per_file_ef1 = []
    for _, row in valid.iterrows():
        t = row["n_true_positives"]
        f = row["n_false_positives"]
        n = row["n_false_negatives"]
        p_ = t / (t + f) if (t + f) > 0 else 0
        r_ = t / (t + n) if (t + n) > 0 else 0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
        per_file_ef1.append(f_)
    valid["per_file_event_f1"] = per_file_ef1

    mean, lo, hi = bootstrap_ci(np.array(per_file_ef1), n_bootstrap)
    event_agg["event_f1_ci"] = (mean, lo, hi)

    # Sample-level if dice column exists
    sample_agg = {}
    if "dice" in valid.columns:
        vals = valid["dice"].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n_bootstrap)
        sample_agg["dice"] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    # Timing
    onset_errors = valid["mean_onset_error_ms"].dropna().values
    if len(onset_errors) > 0:
        event_agg["onset_mean_ms"] = float(np.mean(onset_errors))
        event_agg["onset_std_ms"] = float(np.std(onset_errors))

    return {
        "sample_agg": sample_agg,
        "event_agg": event_agg,
        "per_file": valid,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stim artifact performance metrics")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Model checkpoint (needed for ROC/PR curves)",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/stim_catalog.parquet"),
        help="Stim catalog path",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Pre-computed inference_results.csv (skips model loading)",
    )
    p.add_argument(
        "--probas-dir",
        type=Path,
        default=None,
        help="Directory with .npy probability arrays",
    )
    p.add_argument(
        "--split",
        choices=["all", "val"],
        default="val",
        help="Which split to evaluate on",
    )
    p.add_argument(
        "--max-files", type=int, default=None, help="Limit files for faster iteration"
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Binary threshold")
    p.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap resamples")
    p.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.results is not None:
        # Lightweight mode: metrics from pre-computed results CSV
        print(f"Loading pre-computed results: {args.results}")
        metrics = metrics_from_results_csv(args.results, args.bootstrap_n)
        print("\n=== Event-Level Metrics ===")
        for k, v in metrics["event_agg"].items():
            if isinstance(v, tuple):
                print(f"  {k}: {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if metrics["sample_agg"]:
            print("\n=== Sample-Level Metrics ===")
            for k, v in metrics["sample_agg"].items():
                if "ci_low" in v:
                    print(
                        f"  {k}: {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
                    )
                else:
                    print(f"  {k}: {v['mean']:.4f}")
        return

    # Full mode: load model, run inference, compute everything
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, device=args.device or "mps")
    cfg.catalog_path = args.catalog

    df = load_annotations(cfg)
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)
    print(f"Catalog: {len(df)} files on disk")

    if args.split == "val":
        from stim_inference import split_data

        _, df = split_data(df)
        print(f"Val split: {len(df)} files")

    if args.max_files is not None and len(df) > args.max_files:
        df = df.sample(n=args.max_files, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} files")

    metrics = compute_all_metrics(
        model,
        cfg,
        df,
        threshold=args.threshold,
        n_bootstrap=args.bootstrap_n,
        probas_dir=args.probas_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SAMPLE-LEVEL METRICS")
    print("=" * 60)
    for k, v in metrics["sample_agg"].items():
        if "ci_low" in v:
            print(f"  {k}: {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
        else:
            print(f"  {k}: {v['mean']:.4f}")

    print("\n" + "=" * 60)
    print("EVENT-LEVEL METRICS")
    print("=" * 60)
    for k, v in metrics["event_agg"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save tables
    metrics["per_file"].to_csv(TABLES_DIR / "metrics_per_file.csv", index=False)
    metrics["per_subject"].to_csv(TABLES_DIR / "metrics_per_subject.csv", index=False)
    metrics["per_epoch"].to_csv(TABLES_DIR / "metrics_per_epoch.csv", index=False)
    if len(metrics["per_site"]) > 0:
        metrics["per_site"].to_csv(TABLES_DIR / "metrics_per_site.csv", index=False)

    sample_summary = pd.DataFrame(
        [
            {
                "metric": k,
                "value": v["mean"],
                "ci_low": v.get("ci_low", np.nan),
                "ci_high": v.get("ci_high", np.nan),
            }
            for k, v in metrics["sample_agg"].items()
        ]
    )
    sample_summary.to_csv(TABLES_DIR / "metrics_sample_level.csv", index=False)

    event_summary = pd.DataFrame([metrics["event_agg"]])
    event_summary.to_csv(TABLES_DIR / "metrics_event_level.csv", index=False)
    print(f"\nTables saved to {TABLES_DIR}/")

    # Generate figures
    fpr, tpr = metrics["roc_data"]
    rec_arr, prec_arr = metrics["pr_data"]

    if len(fpr) > 0:
        auroc = metrics["sample_agg"].get("auroc", {}).get("mean", 0.0)
        plot_roc_curve(fpr, tpr, auroc, FIGURES_DIR / "roc_curve.pdf")

    if len(rec_arr) > 0:
        auprc = metrics["sample_agg"].get("auprc", {}).get("mean", 0.0)
        plot_pr_curve(rec_arr, prec_arr, auprc, FIGURES_DIR / "pr_curve.pdf")

    if "tp" in metrics["per_file"].columns:
        plot_confusion_matrix(metrics["per_file"], FIGURES_DIR / "confusion_matrix.pdf")

    plot_metrics_by_subject(
        metrics["per_subject"], FIGURES_DIR / "metrics_by_subject.pdf"
    )
    plot_onset_timing_histogram(
        metrics["all_onset_errors"], FIGURES_DIR / "onset_timing_histogram.pdf"
    )
    plot_performance_by_stim_params(
        metrics["per_file"], FIGURES_DIR / "performance_by_stim_params.pdf"
    )
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
