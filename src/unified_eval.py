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
#   "tqdm>=4.60",
#   "xgboost>=2.0",
#   "kymatio>=0.3",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Unified evaluation of M0-M4 on the same val split.

Runs all methods on held-out subjects, computes sample-level and event-level
metrics, onset/offset errors, statistical tests (Friedman, Wilcoxon, bootstrap
CIs), and saves per-file results + aggregate summary.

Usage:
    uv run src/unified_eval.py                        # all methods, full val split
    uv run src/unified_eval.py --methods m4            # single method
    uv run src/unified_eval.py --methods m0 m3 m4      # subset
    uv run src/unified_eval.py --max-files 200         # quick test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from scipy.stats import friedmanchisquare, wilcoxon
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    _load_dat_channels,
    build_true_mask,
    extract_conditioning_vector,
    load_annotations,
    post_process_mask,
    predict_file_proba,
)

RESULTS_DIR = Path("outputs/results")
VAL_SUBJECTS = ["300-002", "301-003", "303-004", "305-001"]
ALL_METHODS = ["m0", "m1", "m2", "m3", "m4"]


# ---------------------------------------------------------------------------
# Event extraction and matching (local, no deployed prepare.py dependency)
# ---------------------------------------------------------------------------


def extract_events(binary_mask: np.ndarray) -> list[tuple[int, int]]:
    """Extract contiguous events from binary mask as (start, end) tuples."""
    labeled_arr, n_feat = label(binary_mask > 0.5)
    events = []
    for obj_slice in find_objects(labeled_arr):
        if obj_slice is None:
            continue
        events.append((obj_slice[0].start, obj_slice[0].stop))
    return events


def match_events(
    pred_events: list[tuple[int, int]],
    true_events: list[tuple[int, int]],
    iou_threshold: float = 0.3,
) -> tuple[int, int, int, list[tuple]]:
    """Match predicted to true events by IoU.

    Returns (TP, FP, FN, matched_pairs) where matched_pairs is a list of
    (pred_idx, true_idx, iou, onset_err, offset_err) tuples.
    """
    if not pred_events and not true_events:
        return 0, 0, 0, []
    if not pred_events:
        return 0, 0, len(true_events), []
    if not true_events:
        return 0, len(pred_events), 0, []

    matched_true: set[int] = set()
    matched_pairs: list[tuple] = []
    tp = 0

    for pi, (ps, pe) in enumerate(pred_events):
        best_iou = 0.0
        best_ti = -1
        for ti, (ts, te) in enumerate(true_events):
            if ti in matched_true:
                continue
            inter = max(0, min(pe, te) - max(ps, ts))
            union = (pe - ps) + (te - ts) - inter
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_iou >= iou_threshold and best_ti >= 0:
            tp += 1
            matched_true.add(best_ti)
            ts, te = true_events[best_ti]
            onset_err = ps - ts  # positive = predicted too late
            offset_err = pe - te  # positive = predicted too late
            matched_pairs.append((pi, best_ti, best_iou, onset_err, offset_err))

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    return tp, fp, fn, matched_pairs


# ---------------------------------------------------------------------------
# Per-file metrics computation
# ---------------------------------------------------------------------------


def compute_file_metrics(
    pred_proba: np.ndarray,
    true_mask: np.ndarray,
    cfg: Config,
    threshold: float = 0.5,
    sr: int = 250,
) -> dict:
    """Compute all metrics for a single file.

    Returns dict with sample-level, event-level, and timing metrics.
    """
    pred_bin = post_process_mask((pred_proba > threshold).astype(np.int8), cfg)

    # Zero boundary margins
    margin = int(cfg.boundary_margin_s * sr)
    n = len(pred_bin)
    pred_bin[:margin] = 0
    pred_bin[max(margin, n - margin) :] = 0
    true_local = true_mask.copy()
    true_local[:margin] = 0
    true_local[max(margin, n - margin) :] = 0

    # Sample-level metrics
    p = pred_bin > 0.5
    t = true_local > 0.5
    tp_s = int((p & t).sum())
    fp_s = int((p & ~t).sum())
    fn_s = int((~p & t).sum())
    sample_p = tp_s / max(tp_s + fp_s, 1)
    sample_r = tp_s / max(tp_s + fn_s, 1)
    sample_f1 = 2 * sample_p * sample_r / max(sample_p + sample_r, 1e-12)

    # Event-level metrics at multiple IoU thresholds
    pred_events = extract_events(pred_bin)
    true_events = extract_events(true_local)

    result = {
        "sample_f1": sample_f1,
        "sample_precision": sample_p,
        "sample_recall": sample_r,
        "n_pred_events": len(pred_events),
        "n_true_events": len(true_events),
    }

    for iou_t in [0.1, 0.3, 0.5, 0.7]:
        tp_e, fp_e, fn_e, pairs = match_events(pred_events, true_events, iou_t)
        denom = 2 * tp_e + fp_e + fn_e
        ef1 = (2 * tp_e) / denom if denom > 0 else (1.0 if not true_events else 0.0)
        ep = tp_e / max(tp_e + fp_e, 1)
        er = tp_e / max(tp_e + fn_e, 1)
        suffix = f"_iou{int(iou_t * 10):02d}"
        result[f"event_f1{suffix}"] = ef1
        result[f"event_precision{suffix}"] = ep
        result[f"event_recall{suffix}"] = er
        result[f"event_tp{suffix}"] = tp_e
        result[f"event_fp{suffix}"] = fp_e
        result[f"event_fn{suffix}"] = fn_e

    # Onset/offset errors on IoU >= 0.1 matches
    _, _, _, pairs_01 = match_events(pred_events, true_events, 0.1)
    if pairs_01:
        onset_errs = np.array([p[3] for p in pairs_01]) / sr * 1000  # ms
        offset_errs = np.array([p[4] for p in pairs_01]) / sr * 1000
        abs_onset = np.abs(onset_errs)
        abs_offset = np.abs(offset_errs)
        result["onset_mae_ms"] = float(abs_onset.mean())
        result["onset_median_ms"] = float(np.median(abs_onset))
        result["onset_p90_ms"] = float(np.percentile(abs_onset, 90))
        result["offset_mae_ms"] = float(abs_offset.mean())
        result["offset_median_ms"] = float(np.median(abs_offset))
        result["offset_p90_ms"] = float(np.percentile(abs_offset, 90))
    else:
        for k in [
            "onset_mae_ms",
            "onset_median_ms",
            "onset_p90_ms",
            "offset_mae_ms",
            "offset_median_ms",
            "offset_p90_ms",
        ]:
            result[k] = np.nan

    return result


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_m4(
    df: pd.DataFrame, max_files: int | None = None, device: str = "mps"
) -> pd.DataFrame:
    """Run M4 (deployed U-Net) on val files."""
    from deployed_adapter import load_deployed_checkpoint

    model, cfg = load_deployed_checkpoint(device=device)
    rows_to_eval = df.head(max_files) if max_files else df

    results = []
    t0 = time.time()
    for _, row in tqdm(rows_to_eval.iterrows(), total=len(rows_to_eval), desc="M4"):
        cond = extract_conditioning_vector(row)
        proba = predict_file_proba(model, cfg, row["file_path"], cond, batch_size=256)
        n_samples = len(proba)
        onset_times = row["onset_times"]
        if isinstance(onset_times, str):
            onset_times = json.loads(onset_times)
        true_mask = build_true_mask(
            n_samples, onset_times, row["mask_duration_ms"], cfg.sampling_rate
        )
        metrics = compute_file_metrics(proba, true_mask, cfg, sr=cfg.sampling_rate)
        metrics["filename"] = row.get("filename", Path(row["file_path"]).stem)
        metrics["subject"] = row["subject"]
        metrics["method"] = "m4"
        results.append(metrics)

    elapsed = time.time() - t0
    n = len(results)
    print(f"M4: {n} files in {elapsed:.1f}s ({n / elapsed:.1f} files/s)")
    return pd.DataFrame(results)


def run_baseline(
    method_name: str,
    df: pd.DataFrame,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Run a baseline method (M0-M3) on val files."""
    from comparison_methods import (
        AmplitudeThreshold,
        ScatteringXGBoost,
        SpectralNoiseSub,
        VAEReconstruction,
    )

    method_map = {
        "m0": ("m0", AmplitudeThreshold),
        "m1": ("m1", SpectralNoiseSub),
        "m2": ("m2", VAEReconstruction),
        "m3": ("m3", ScatteringXGBoost),
    }

    baseline_dir = Path("data/baselines")
    name, cls = method_map[method_name]
    method = cls.load(baseline_dir / name)
    print(f"Loaded {name} from {baseline_dir / name}")

    # Create a Config for post-processing parameters
    cfg = Config(
        min_artifact_samples=75,
        merge_gap_ms=300.0,
        boundary_margin_s=4.0,
    )

    rows_to_eval = df.head(max_files) if max_files else df

    results = []
    t0 = time.time()
    for _, row in tqdm(
        rows_to_eval.iterrows(), total=len(rows_to_eval), desc=method_name.upper()
    ):
        fp = row["file_path"]
        subj = row["subject"]
        proba = method.predict_file(fp, subject=subj)

        n_samples = len(proba)
        onset_times = row["onset_times"]
        if isinstance(onset_times, str):
            onset_times = json.loads(onset_times)
        true_mask = build_true_mask(
            n_samples, onset_times, row["mask_duration_ms"], cfg.sampling_rate
        )
        metrics = compute_file_metrics(proba, true_mask, cfg, sr=cfg.sampling_rate)
        metrics["filename"] = row.get("filename", Path(fp).stem)
        metrics["subject"] = subj
        metrics["method"] = method_name
        results.append(metrics)

    elapsed = time.time() - t0
    n = len(results)
    print(
        f"{method_name.upper()}: {n} files in {elapsed:.1f}s ({n / elapsed:.1f} files/s)"
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap BCa 95% CI for the mean.

    Returns (mean, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    mean_val = float(np.mean(values))

    boot_means = np.array(
        [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return mean_val, lo, hi


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (non-parametric)."""
    n_x, n_y = len(x), len(y)
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (n_x * n_y))


def compute_statistics(all_results: pd.DataFrame, methods: list[str]) -> dict:
    """Compute aggregate statistics with CIs and pairwise tests."""
    stats = {}

    # Aggregate metrics per method
    for method in methods:
        mdf = all_results[all_results["method"] == method]
        method_stats = {}
        for col in [
            "sample_f1",
            "event_f1_iou03",
            "event_f1_iou05",
            "onset_mae_ms",
            "offset_mae_ms",
        ]:
            vals = mdf[col].dropna().values
            if len(vals) > 0:
                mean, lo, hi = bootstrap_ci(vals)
                method_stats[col] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        stats[method] = method_stats

    # Per-subject breakdown
    per_subject = {}
    for method in methods:
        mdf = all_results[all_results["method"] == method]
        per_subject[method] = mdf.groupby("subject")["sample_f1"].mean().to_dict()
    stats["per_subject"] = per_subject

    # Friedman test (requires all methods to have values for same files)
    if len(methods) >= 3:
        pivot = all_results.pivot_table(
            values="sample_f1", index="filename", columns="method"
        ).dropna()
        if len(pivot) > 10 and all(m in pivot.columns for m in methods):
            arrays = [pivot[m].values for m in methods]
            stat, p = friedmanchisquare(*arrays)
            stats["friedman"] = {"statistic": float(stat), "p_value": float(p)}

    # Pairwise Wilcoxon: M4 vs each baseline
    if "m4" in methods:
        m4_df = all_results[all_results["method"] == "m4"].set_index("filename")
        pairwise = {}
        p_values = []
        for other in [m for m in methods if m != "m4"]:
            other_df = all_results[all_results["method"] == other].set_index("filename")
            common = m4_df.index.intersection(other_df.index)
            if len(common) > 10:
                m4_vals = m4_df.loc[common, "sample_f1"].values
                other_vals = other_df.loc[common, "sample_f1"].values
                stat, p = wilcoxon(m4_vals, other_vals)
                delta = cliffs_delta(m4_vals, other_vals)
                pairwise[f"m4_vs_{other}"] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "cliffs_delta": delta,
                    "n_pairs": len(common),
                }
                p_values.append(p)

        # Holm correction
        if p_values:
            sorted_idx = np.argsort(p_values)
            n_tests = len(p_values)
            keys = [f"m4_vs_{m}" for m in methods if m != "m4"]
            for rank, idx in enumerate(sorted_idx):
                corrected_p = min(p_values[idx] * (n_tests - rank), 1.0)
                pairwise[keys[idx]]["p_value_holm"] = float(corrected_p)

        stats["pairwise_wilcoxon"] = pairwise

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified M0-M4 evaluation")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit val files")
    parser.add_argument("--device", default="mps", help="Device for M4")
    parser.add_argument(
        "--output", default=str(RESULTS_DIR / "unified_eval.csv"), help="Output CSV"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load catalog and filter to val subjects
    cfg = Config()
    df = load_annotations(cfg)
    val_df = df[df["subject"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    print(
        f"Val set: {len(val_df)} files, "
        f"subjects={sorted(val_df['subject'].unique())}"
    )

    all_results = []
    for method in args.methods:
        if method == "m4":
            result_df = run_m4(val_df, args.max_files, args.device)
        else:
            result_df = run_baseline(method, val_df, args.max_files)
        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\nPer-file results saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("UNIFIED EVALUATION SUMMARY")
    print("=" * 80)
    for method in args.methods:
        mdf = combined[combined["method"] == method]
        sf1 = mdf["sample_f1"]
        ef1 = mdf["event_f1_iou03"]
        print(
            f"  {method.upper():4s}: sample_F1={sf1.mean():.4f} +/- {sf1.std():.4f}"
            f"  event_F1(IoU0.3)={ef1.mean():.4f}"
            f"  onset_MAE={mdf['onset_mae_ms'].mean():.1f}ms"
        )

    # Compute and save statistics
    stats = compute_statistics(combined, args.methods)
    stats_path = RESULTS_DIR / "unified_summary.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nAggregate statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
