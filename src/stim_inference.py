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
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Stim Artifact Inference — standalone prediction pipeline.

Loads trained weights and runs inference on .dat files from the stim catalog
(or a single file), outputting predicted stim masks and onset/offset times.

Usage:
    uv run src/stim_inference.py                           # all catalog files
    uv run src/stim_inference.py --file /path/to.dat       # single file
    uv run src/stim_inference.py --max-files 500           # subset
    uv run src/stim_inference.py --threshold 0.45          # custom threshold
    uv run src/stim_inference.py --save-probas             # save .npy arrays
    uv run src/stim_inference.py --output results.csv      # custom output path
    uv run src/stim_inference.py --split val               # val split only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    build_true_mask,
    extract_conditioning_vector,
    load_annotations,
    load_checkpoint,
    post_process_mask,
    predict_file_proba,
)

# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def extract_events(
    proba: np.ndarray,
    cfg: Config,
    threshold: float = 0.5,
) -> list[dict]:
    """Extract discrete stim events from probability array.

    Returns list of dicts with keys:
        onset_s, offset_s, duration_ms, peak_prob, mean_prob
    """
    pred_bin = post_process_mask((proba > threshold).astype(np.int_), cfg)
    labeled, n_events = label(pred_bin)
    if n_events == 0:
        return []

    sr = cfg.sampling_rate
    events = []
    for slc in find_objects(labeled):
        s, e = slc[0].start, slc[0].stop
        events.append(
            {
                "onset_s": s / sr,
                "offset_s": e / sr,
                "duration_ms": (e - s) / sr * 1000.0,
                "peak_prob": float(proba[s:e].max()),
                "mean_prob": float(proba[s:e].mean()),
            }
        )
    return events


def match_events(
    pred_events: list[dict],
    true_mask: np.ndarray,
    sr: int = 250,
    iou_threshold: float = 0.3,
) -> dict:
    """Match predicted events to ground-truth mask, return counts + timing errors."""
    true_labeled, n_true = label(true_mask)
    n_pred = len(pred_events)

    if n_true == 0 and n_pred == 0:
        return {
            "n_gt": 0,
            "n_pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "timing_errors_ms": [],
        }

    true_slices = find_objects(true_labeled) if n_true > 0 else []

    tp, matched_gt = 0, set()
    timing_errors = []

    for pe in pred_events:
        ps = int(pe["onset_s"] * sr)
        pe_end = int(pe["offset_s"] * sr)
        best_iou, best_j = 0.0, -1

        for j, ts in enumerate(true_slices):
            if j in matched_gt:
                continue
            gs, ge = ts[0].start, ts[0].stop
            if ps >= ge or pe_end <= gs:
                continue
            inter = min(pe_end, ge) - max(ps, gs)
            union = max(pe_end, ge) - min(ps, gs)
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)
            gt_onset = true_slices[best_j][0].start
            timing_errors.append((ps - gt_onset) / sr * 1000.0)

    return {
        "n_gt": n_true,
        "n_pred": n_pred,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_true - len(matched_gt),
        "timing_errors_ms": timing_errors,
    }


# ---------------------------------------------------------------------------
# Catalog inference
# ---------------------------------------------------------------------------


def predict_catalog(
    model,
    cfg: Config,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    batch_size: int = 64,
    tta: bool = False,
    save_probas: bool = False,
    probas_dir: Path | None = None,
) -> pd.DataFrame:
    """Run inference on all files in a DataFrame.

    Returns DataFrame with per-file prediction results + optional metrics
    when ground-truth onset_times are available.
    """
    sr = cfg.sampling_rate
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        fp = row["file_path"]
        if not Path(fp).exists():
            rows.append(
                {
                    "filename": row["filename"],
                    "file_path": fp,
                    "error": "file_not_found",
                }
            )
            continue

        cond = extract_conditioning_vector(row)

        try:
            proba = predict_file_proba(
                model, cfg, fp, cond, batch_size=batch_size, tta=tta
            )
        except Exception as exc:
            rows.append(
                {
                    "filename": row["filename"],
                    "file_path": fp,
                    "error": str(exc),
                }
            )
            continue

        events = extract_events(proba, cfg, threshold=threshold)

        result = {
            "filename": row["filename"],
            "file_path": fp,
            "subject": row.get("subject", ""),
            "subject_id_lr": row.get("subject_id_lr", ""),
            "epoch_start_gmt": row.get("epoch_start_gmt", ""),
            "n_pred_events": len(events),
            "pred_onset_times": json.dumps([e["onset_s"] for e in events]),
            "pred_offset_times": json.dumps([e["offset_s"] for e in events]),
            "pred_durations_ms": json.dumps([e["duration_ms"] for e in events]),
            "pred_peak_probs": json.dumps([round(e["peak_prob"], 4) for e in events]),
            "pred_mean_probs": json.dumps([round(e["mean_prob"], 4) for e in events]),
            "recording_length_s": len(proba) / sr,
        }

        # Ground-truth metrics if onset_times are available
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

        result["n_gt_events"] = len(onsets)
        result["mask_duration_ms"] = mask_dur_ms

        if len(onsets) > 0 or len(events) > 0:
            true_mask = build_true_mask(
                len(proba), onsets, mask_dur_ms, sr, onset_offset_ms
            )
            match = match_events(events, true_mask, sr)
            result["n_true_positives"] = match["tp"]
            result["n_false_positives"] = match["fp"]
            result["n_false_negatives"] = match["fn"]

            if match["timing_errors_ms"]:
                errs = np.array(match["timing_errors_ms"])
                result["mean_onset_error_ms"] = float(np.mean(errs))
                result["std_onset_error_ms"] = float(np.std(errs))
                result["median_onset_error_ms"] = float(np.median(errs))

            # Sample-level dice
            pred_bin = post_process_mask((proba > threshold).astype(np.int_), cfg)
            inter = np.logical_and(pred_bin, true_mask).sum()
            union = pred_bin.sum() + true_mask.sum()
            result["dice"] = float(2 * inter / (union + 1e-8))

        if save_probas and probas_dir is not None:
            np.save(probas_dir / f"{row['filename']}.npy", proba)

        rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split helper (mirrors stim_detector.py)
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-split by biological subject (no bilateral leakage)."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = df["subject"].values
    unique_groups = np.unique(groups)
    if len(unique_groups) == 1:
        groups = df["filename"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stim artifact inference pipeline")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/stim_catalog.parquet"),
        help="Path to stim_catalog.parquet",
    )
    p.add_argument("--file", type=Path, default=None, help="Single .dat file")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/inference_results.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Binary threshold (default: 0.5)"
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=None,
        help="Override merge_gap_ms in post-processing",
    )
    p.add_argument(
        "--min-dur",
        type=float,
        default=None,
        help="Override min_artifact_samples (as ms)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process",
    )
    p.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Which data split to run on",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Windows per inference batch"
    )
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    p.add_argument(
        "--save-probas", action="store_true", help="Save .npy probability arrays"
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Override device",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    assert args.checkpoint.exists(), f"Checkpoint not found: {args.checkpoint}"

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, device=args.device or "mps")
    cfg.catalog_path = args.catalog

    # Override post-processing if requested
    if args.merge_gap is not None:
        cfg.merge_gap_ms = args.merge_gap
    if args.min_dur is not None:
        cfg.min_artifact_samples = max(
            1, int(args.min_dur / 1000.0 * cfg.sampling_rate)
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters on {cfg.device}")
    print(f"Threshold: {args.threshold}, merge_gap: {cfg.merge_gap_ms}ms")

    # Load data
    if args.file is not None:
        # Single file mode — create a minimal DataFrame row
        assert args.file.exists(), f"File not found: {args.file}"
        df = pd.DataFrame(
            [
                {
                    "filename": args.file.stem,
                    "file_path": str(args.file),
                    "onset_times": "[]",
                    "mask_duration_ms": 1000.0,
                    "mask_onset_offset_ms": 0.0,
                    "subject": "unknown",
                    "subject_id_lr": "unknown",
                    "epoch_start_gmt": "",
                }
            ]
        )
        print(f"Single file mode: {args.file}")
    else:
        df = load_annotations(cfg)
        exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
        n_total = len(df)
        df = df[exists_mask].reset_index(drop=True)
        print(f"Catalog: {len(df)}/{n_total} files on disk")

        if args.split in ("train", "val"):
            train_df, val_df = split_data(df)
            df = val_df if args.split == "val" else train_df
            print(f"Split '{args.split}': {len(df)} files")

        if args.max_files is not None and len(df) > args.max_files:
            df = df.sample(n=args.max_files, random_state=42).reset_index(drop=True)
            print(f"Subsampled to {len(df)} files")

    # Setup probas directory
    probas_dir = None
    if args.save_probas:
        probas_dir = args.output.parent / "probas"
        probas_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df = predict_catalog(
        model,
        cfg,
        df,
        threshold=args.threshold,
        batch_size=args.batch_size,
        tta=args.tta,
        save_probas=args.save_probas,
        probas_dir=probas_dir,
    )

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(results_df)} rows -> {args.output}")

    # Summary statistics
    errors = (
        results_df["error"].dropna() if "error" in results_df.columns else pd.Series()
    )
    valid = results_df[~results_df.get("error", pd.Series(dtype=str)).notna()]
    if len(errors) > 0:
        print(f"  Errors: {len(errors)}")

    if "n_true_positives" in valid.columns:
        tp = valid["n_true_positives"].sum()
        fp = valid["n_false_positives"].sum()
        fn = valid["n_false_negatives"].sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"\n  Event-level:  TP={tp}  FP={fp}  FN={fn}")
        print(f"  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

        if "mean_onset_error_ms" in valid.columns:
            merr = valid["mean_onset_error_ms"].dropna().mean()
            mbias = valid["median_onset_error_ms"].dropna().mean()
            print(f"  Onset error: mean={merr:.2f}ms, median_bias={mbias:.2f}ms")

        if "dice" in valid.columns:
            mean_dice = valid["dice"].dropna().mean()
            print(f"  Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
