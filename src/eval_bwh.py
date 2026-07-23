#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "scipy>=1.10",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""
Evaluate the training pipeline U-Net on BWH data for external validation.

Uses the model architecture from the training pipeline (which diverged
from stim_detector_lib.py) with the best checkpoint, and the BWH catalog
built by build_bwh_catalog.py.

Usage:
    uv run src/eval_bwh.py                            # 500 files
    uv run src/eval_bwh.py --max-files 10             # quick test
    uv run src/eval_bwh.py --max-files 1000 --tta     # larger eval + TTA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from scipy.ndimage import find_objects, label
from tqdm.auto import tqdm

# Import the training pipeline model + helpers
DEPLOYED_DIR = Path("/path/to/Research/training-pipeline")
sys.path.insert(0, str(DEPLOYED_DIR))
from prepare import (
    MontageParser,
    LeadParser,
    PARAM_RANGES,
    STIM_PARAM_SUFFIXES,
    extract_conditioning_vector,
    parse_onset_times,
    extract_events,
    N_COND_FEATURES,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
    STRIDE_SAMPLES,
    BOUNDARY_MARGIN_S,
)
from train_5b0d152 import StimArtifactUNet

from lgs_db import read_dat, to_microvolts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = DEPLOYED_DIR / "best_model.pt"
BWH_CATALOG_PATH = Path("data/bwh_stim_catalog.parquet")

# Postprocessing constants — lowered from deployed defaults to handle
# short-duration BWH therapies (200ms = 50 samples at 250 Hz).
# Original: MIN_ARTIFACT_SAMPLES=75, MERGE_GAP_MS=300
MIN_ARTIFACT_SAMPLES = 25  # 100ms — allows detection of 200ms artifacts
MERGE_GAP_MS = 200.0  # tighter merge gap for short events
EVENT_IOU_THRESHOLD = 0.3
ONSET_TOLERANCE_SAMPLES = 75  # 300ms onset tolerance for alternative matching


# ---------------------------------------------------------------------------
# Postprocessing (local — uses lowered MIN_ARTIFACT_SAMPLES)
# ---------------------------------------------------------------------------


def postprocess_predictions(pred_mask: np.ndarray) -> np.ndarray:
    """Morphological closing + small-component removal.

    Uses lowered MIN_ARTIFACT_SAMPLES (25 vs deployed default 75) to retain
    short-duration predictions that correspond to BWH 200ms therapies.
    """
    close_samples = int(MERGE_GAP_MS / 1000 * SAMPLING_RATE)
    struct = np.ones(max(close_samples, 1))
    closed = scipy.ndimage.binary_closing(pred_mask.astype(bool), structure=struct)
    labeled_arr, n_components = label(closed)
    for obj_slice in find_objects(labeled_arr):
        if obj_slice is None:
            continue
        component_len = obj_slice[0].stop - obj_slice[0].start
        if component_len < MIN_ARTIFACT_SAMPLES:
            closed[obj_slice] = False
    return closed.astype(np.float32)


def calibrate_event_widths(
    pred_bin: np.ndarray,
    mask_dur_ms: float,
    sr: int = SAMPLING_RATE,
) -> np.ndarray:
    """Resize predicted events to match the known therapy duration.

    Keeps each event's onset but sets its width to ``mask_dur_ms``.
    This corrects the width mismatch between model predictions (trained on
    longer LGS artifacts) and shorter BWH programmed therapy durations.
    """
    target_samp = max(int(mask_dur_ms / 1000.0 * sr), 1)
    events = extract_events(pred_bin)
    if not events:
        return pred_bin
    out = np.zeros_like(pred_bin)
    n = len(out)
    for start, _end in events:
        out[start : min(n, start + target_samp)] = 1.0
    return out


def match_events_iou(
    pred_events: list[tuple[int, int]],
    true_events: list[tuple[int, int]],
    iou_threshold: float = EVENT_IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Match predicted events to true events by IoU. Returns (TP, FP, FN)."""
    if not pred_events and not true_events:
        return (0, 0, 0)
    if not pred_events:
        return (0, 0, len(true_events))
    if not true_events:
        return (0, len(pred_events), 0)

    tp = 0
    matched: set[int] = set()

    for ps, pe in pred_events:
        best_iou = 0.0
        best_j = -1
        for j, (ts, te) in enumerate(true_events):
            if j in matched:
                continue
            if ps >= te or pe <= ts:
                continue
            inter = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_j)

    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched)
    return (tp, fp, fn)


def match_events_onset(
    pred_events: list[tuple[int, int]],
    true_events: list[tuple[int, int]],
    tolerance: int = ONSET_TOLERANCE_SAMPLES,
) -> tuple[int, int, int]:
    """Match events by onset proximity: pred onset within ±tolerance of GT onset.

    More forgiving for short-duration events where IoU penalizes any width
    mismatch. A predicted event that starts near the true onset is a TP
    regardless of duration mismatch.
    """
    if not pred_events and not true_events:
        return (0, 0, 0)
    if not pred_events:
        return (0, 0, len(true_events))
    if not true_events:
        return (0, len(pred_events), 0)

    tp = 0
    matched: set[int] = set()

    for ps, _pe in pred_events:
        best_dist = float("inf")
        best_j = -1
        for j, (ts, _te) in enumerate(true_events):
            if j in matched:
                continue
            dist = abs(ps - ts)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist <= tolerance:
            tp += 1
            matched.add(best_j)

    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched)
    return (tp, fp, fn)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_deployed_model(
    device: str = "mps",
) -> StimArtifactUNet:
    """Load the training pipeline best model checkpoint."""
    sd = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    # Strip _orig_mod. prefix from torch.compile
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = StimArtifactUNet()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params on {device}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _robust_scale_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and IQR for robust z-normalization."""
    median = np.median(data, axis=1)
    q25 = np.percentile(data, 25, axis=1)
    q75 = np.percentile(data, 75, axis=1)
    return median, q75 - q25


@torch.no_grad()
def predict_file(
    model: StimArtifactUNet,
    dat_path: str | Path,
    cond: np.ndarray,
    *,
    device: str = "mps",
    batch_size: int = 64,
    tta: bool = False,
) -> np.ndarray:
    """Run sliding-window inference on a single .dat file.

    Returns per-sample probability array of shape (n_samples,).
    """
    raw = read_dat(str(dat_path))
    data = to_microvolts(raw).astype(np.float32)
    median, iqr = _robust_scale_stats(data)
    med_col = median[:, np.newaxis]
    iqr_col = iqr[:, np.newaxis] + 1e-8

    n_samples = data.shape[1]
    ws = WINDOW_SAMPLES
    stride = STRIDE_SAMPLES

    starts = list(range(0, n_samples - ws + 1, stride))
    if not starts:
        return np.zeros(n_samples, dtype=np.float32)

    cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)

    scales = [1.0, 0.9, 1.1] if tta else [1.0]

    pred_acc = np.zeros(n_samples, dtype=np.float32)
    count = np.zeros(n_samples, dtype=np.float32)

    for scale in scales:
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            windows = np.stack(
                [
                    (data[:, s : s + ws] - med_col) / iqr_col * scale
                    for s in batch_starts
                ]
            )
            sig_t = torch.from_numpy(windows).to(device)
            cond_batch = cond_t.expand(len(batch_starts), -1)

            logits, _aux = model(sig_t, cond_batch)
            probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)

            for i, s in enumerate(batch_starts):
                pred_acc[s : s + ws] += probs[i]
                count[s : s + ws] += 1.0

    return np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def build_true_mask(
    n_samples: int,
    onset_times: list[float],
    mask_dur_ms: float,
    sr: int = SAMPLING_RATE,
    onset_offset_ms: float = 0.0,
) -> np.ndarray:
    """Build binary ground-truth mask from onset times."""
    mask = np.zeros(n_samples, dtype=np.int8)
    dur_samp = max(int(mask_dur_ms / 1000.0 * sr), 1)
    offset_samp = int(onset_offset_ms / 1000.0 * sr)
    for t in onset_times:
        s = int(t * sr) + offset_samp
        mask[max(0, s) : min(n_samples, s + dur_samp)] = 1
    return mask


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_bwh(
    model: StimArtifactUNet,
    catalog_path: Path,
    *,
    max_files: int | None = None,
    device: str = "mps",
    batch_size: int = 64,
    threshold: float = 0.5,
    tta: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate model on BWH stim catalog files.

    Returns DataFrame with per-file metrics.
    """
    df = pd.read_parquet(catalog_path)

    # Drop files without paths on disk
    df = df[df["file_path"].notna()].copy()
    # Verify files exist
    df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy()

    if max_files and len(df) > max_files:
        # Stratified sample across subjects
        df = df.sample(n=max_files, random_state=seed).reset_index(drop=True)

    print(f"Evaluating {len(df):,} files from {df['subject'].nunique()} subjects")

    margin = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BWH eval"):
        fp = row["file_path"]
        filename = row["filename"]
        subject = row["subject"]

        # Conditioning vector
        cond = extract_conditioning_vector(row)

        # Predict
        try:
            proba = predict_file(
                model, fp, cond, device=device, batch_size=batch_size, tta=tta
            )
        except Exception as exc:
            rows.append({"filename": filename, "subject": subject, "error": str(exc)})
            continue

        n_samples = len(proba)

        # Ground truth
        onsets = row.get("onset_times", "[]")
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []

        mask_dur_ms = row.get("mask_duration_ms", 1024.0)
        if pd.isna(mask_dur_ms):
            mask_dur_ms = 1024.0
        onset_offset_ms = row.get("mask_onset_offset_ms", 0.0)
        if pd.isna(onset_offset_ms):
            onset_offset_ms = 0.0

        true_mask = build_true_mask(n_samples, onsets, mask_dur_ms)

        # Postprocess predictions
        pred_raw = postprocess_predictions((proba > threshold).astype(np.float32))

        # Calibrate predicted event widths to match known therapy duration
        pred_cal = calibrate_event_widths(pred_raw, mask_dur_ms)

        # Zero out boundary margins
        for arr in (pred_raw, pred_cal):
            arr[:margin] = 0.0
            arr[max(margin, n_samples - margin) :] = 0.0
        true_trimmed = true_mask.copy()
        true_trimmed[:margin] = 0.0
        true_trimmed[max(margin, n_samples - margin) :] = 0.0

        # Sample-level metrics (on raw predictions, not calibrated)
        tp_s = int(((pred_raw > 0.5) & (true_trimmed > 0)).sum())
        fp_s = int(((pred_raw > 0.5) & (true_trimmed == 0)).sum())
        fn_s = int(((pred_raw <= 0.5) & (true_trimmed > 0)).sum())
        tn_s = int(((pred_raw <= 0.5) & (true_trimmed == 0)).sum())

        p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) > 0 else 0.0

        # Sample-level metrics on calibrated predictions
        tp_sc = int(((pred_cal > 0.5) & (true_trimmed > 0)).sum())
        fp_sc = int(((pred_cal > 0.5) & (true_trimmed == 0)).sum())
        fn_sc = int(((pred_cal <= 0.5) & (true_trimmed > 0)).sum())

        p_sc = tp_sc / (tp_sc + fp_sc) if (tp_sc + fp_sc) > 0 else 0.0
        r_sc = tp_sc / (tp_sc + fn_sc) if (tp_sc + fn_sc) > 0 else 0.0
        f1_sc = 2 * p_sc * r_sc / (p_sc + r_sc) if (p_sc + r_sc) > 0 else 0.0

        # Event-level metrics (IoU matching on calibrated predictions)
        pred_events = extract_events(pred_cal)
        true_events = extract_events(true_trimmed > 0.5)

        # True negatives: no GT and no predictions → skip F1 (not a failure)
        is_true_neg = len(pred_events) == 0 and len(true_events) == 0

        tp_e, fp_e, fn_e = match_events_iou(pred_events, true_events)

        p_e = (
            tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else (1.0 if is_true_neg else 0.0)
        )
        r_e = (
            tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else (1.0 if is_true_neg else 0.0)
        )
        f1_e = (
            2 * p_e * r_e / (p_e + r_e)
            if (p_e + r_e) > 0
            else (1.0 if is_true_neg else 0.0)
        )

        # Onset-tolerance matching
        tp_o, fp_o, fn_o = match_events_onset(pred_events, true_events)
        p_o = (
            tp_o / (tp_o + fp_o) if (tp_o + fp_o) > 0 else (1.0 if is_true_neg else 0.0)
        )
        r_o = (
            tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else (1.0 if is_true_neg else 0.0)
        )
        f1_o = (
            2 * p_o * r_o / (p_o + r_o)
            if (p_o + r_o) > 0
            else (1.0 if is_true_neg else 0.0)
        )

        # Event-level on raw (uncalibrated) for comparison
        pred_events_raw = extract_events(pred_raw)
        tp_er, fp_er, fn_er = match_events_iou(pred_events_raw, true_events)
        p_er = (
            tp_er / (tp_er + fp_er)
            if (tp_er + fp_er) > 0
            else (1.0 if is_true_neg else 0.0)
        )
        r_er = (
            tp_er / (tp_er + fn_er)
            if (tp_er + fn_er) > 0
            else (1.0 if is_true_neg else 0.0)
        )
        f1_er = (
            2 * p_er * r_er / (p_er + r_er)
            if (p_er + r_er) > 0
            else (1.0 if is_true_neg else 0.0)
        )

        rows.append(
            {
                "filename": filename,
                "subject": subject,
                "mask_duration_ms": mask_dur_ms,
                "n_gt_events": len(true_events),
                "n_pred_events": len(pred_events),
                "sample_precision": round(p_s, 4),
                "sample_recall": round(r_s, 4),
                "sample_f1": round(f1_s, 4),
                "cal_sample_precision": round(p_sc, 4),
                "cal_sample_recall": round(r_sc, 4),
                "cal_sample_f1": round(f1_sc, 4),
                "event_precision": round(p_e, 4),
                "event_recall": round(r_e, 4),
                "event_f1": round(f1_e, 4),
                "event_tp": tp_e,
                "event_fp": fp_e,
                "event_fn": fn_e,
                "raw_event_f1": round(f1_er, 4),
                "onset_precision": round(p_o, 4),
                "onset_recall": round(r_o, 4),
                "onset_f1": round(f1_o, 4),
                "onset_tp": tp_o,
                "onset_fp": fp_o,
                "onset_fn": fn_o,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deployed U-Net on BWH stim catalog"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=BWH_CATALOG_PATH,
        help="Path to bwh_stim_catalog.parquet",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/results/bwh_unet_eval.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-files", type=int, default=500, help="Max files to evaluate"
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default="mps",
        help="Compute device",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    model = load_deployed_model(device=args.device)

    results = evaluate_bwh(
        model,
        args.catalog,
        max_files=args.max_files,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        tta=args.tta,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")

    # Summary
    valid = results[results["error"].isna()] if "error" in results.columns else results
    print(f"\n{'='*60}")
    print(f"  Files: {len(valid):,} evaluated, {len(results) - len(valid)} errors")
    print(f"  Subjects: {valid['subject'].nunique()}")
    print(
        f"  MIN_ARTIFACT_SAMPLES: {MIN_ARTIFACT_SAMPLES} ({MIN_ARTIFACT_SAMPLES/SAMPLING_RATE*1000:.0f}ms)"
    )
    print(f"  MERGE_GAP_MS: {MERGE_GAP_MS}")
    for m in [
        "sample_f1",
        "sample_precision",
        "sample_recall",
        "raw_event_f1",
        "event_f1",
        "onset_f1",
    ]:
        if m in valid.columns:
            label = m
            if m == "raw_event_f1":
                label = "event_f1 (raw)"
            elif m == "event_f1":
                label = "event_f1 (calibrated)"
            print(f"  {label}: {valid[m].mean():.3f} ± {valid[m].std():.3f}")

    print(f"\n  Per-subject event F1 (raw → calibrated → onset):")
    subj_summary = valid.groupby("subject").agg(
        raw_f1=("raw_event_f1", "mean"),
        cal_f1=("event_f1", "mean"),
        onset_f1=("onset_f1", "mean"),
        mask_ms=("mask_duration_ms", "first"),
        n=("filename", "count"),
    )
    subj_summary = subj_summary.sort_values("cal_f1")
    print(subj_summary.to_string())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
