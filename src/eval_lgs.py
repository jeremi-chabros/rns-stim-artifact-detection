#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0","numpy>=2.0","pandas>=2.0","pyarrow>=14.0","scipy>=1.10","tqdm>=4.60","lgs-db"]
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Evaluate the SAME deployed checkpoint on held-out LGS test subjects, using
the SAME scoring as eval_bwh.py, to produce a cross-cohort-comparable CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, "src")
from eval_bwh import (  # noqa: E402
    load_deployed_model,
    predict_file,
    postprocess_predictions,
    calibrate_event_widths,
    match_events_iou,
    match_events_onset,
    build_true_mask,
    SAMPLING_RATE,
    BOUNDARY_MARGIN_S,
)

sys.path.insert(0, "/path/to/Research/training-pipeline")
from prepare import extract_conditioning_vector, extract_events  # noqa: E402

VAL_SUBJECTS = ["300-002", "301-003", "303-004", "305-001"]


def evaluate_lgs(
    model,
    catalog_path: Path,
    *,
    device: str = "mps",
    threshold: float = 0.5,
    max_files: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate deployed U-Net on held-out LGS test subjects.

    Args:
        model: Loaded StimArtifactUNet in eval mode.
        catalog_path: Path to stim_catalog.parquet.
        device: Compute device (mps/cuda/cpu).
        threshold: Binary threshold for predictions.
        max_files: If set, randomly sample this many files (for smoke tests).
        seed: Random seed for sampling.

    Returns:
        DataFrame with per-file metrics in BWH-compatible schema.
    """
    df = pd.read_parquet(catalog_path)
    df = df[df["subject"].isin(VAL_SUBJECTS)].copy()
    df = df[
        df["file_path"].notna() & df["file_path"].apply(lambda p: Path(p).exists())
    ].copy()
    if max_files and len(df) > max_files:
        df = df.sample(n=max_files, random_state=seed).reset_index(drop=True)
    print(f"LGS held-out eval: {len(df)} files / {df['subject'].nunique()} subjects")
    margin = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LGS eval"):
        cond = extract_conditioning_vector(row)
        try:
            proba = predict_file(model, row["file_path"], cond, device=device)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "filename": row["filename"],
                    "subject": row["subject"],
                    "error": str(exc),
                }
            )
            continue
        n = len(proba)
        onsets = row.get("onset_times", "[]")
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []
        elif not isinstance(onsets, list):
            onsets = []
        md = row.get("mask_duration_ms", 1024.0)
        md = 1024.0 if pd.isna(md) else float(md)
        true_mask = build_true_mask(n, onsets, md)
        pred_raw = postprocess_predictions((proba > threshold).astype(np.float32))
        pred_cal = calibrate_event_widths(pred_raw, md)
        # Zero out boundary margins (in-place on copies to avoid aliasing)
        pred_raw = pred_raw.copy()
        pred_cal = pred_cal.copy()
        for arr in (pred_raw, pred_cal):
            arr[:margin] = 0.0
            arr[max(margin, n - margin) :] = 0.0
        true_mask[:margin] = 0
        true_mask[max(margin, n - margin) :] = 0
        true_ev = extract_events(true_mask > 0.5)
        pred_ev = extract_events(pred_cal)
        pred_ev_raw = extract_events(pred_raw)
        is_tn = len(pred_ev) == 0 and len(true_ev) == 0

        def f1(tp: int, fp: int, fn: int) -> float:
            d = 2 * tp + fp + fn
            return (2 * tp) / d if d else (1.0 if is_tn else 0.0)

        tp_e, fp_e, fn_e = match_events_iou(pred_ev, true_ev)
        tp_o, fp_o, fn_o = match_events_onset(pred_ev, true_ev)
        tp_r, fp_r, fn_r = match_events_iou(pred_ev_raw, true_ev)
        rows.append(
            {
                "filename": row["filename"],
                "subject": row["subject"],
                "mask_duration_ms": md,
                "n_gt_events": len(true_ev),
                "n_pred_events": len(pred_ev),
                "event_f1": round(f1(tp_e, fp_e, fn_e), 4),
                "event_tp": tp_e,
                "event_fp": fp_e,
                "event_fn": fn_e,
                "raw_event_f1": round(f1(tp_r, fp_r, fn_r), 4),
                "onset_f1": round(f1(tp_o, fp_o, fn_o), 4),
                "onset_tp": tp_o,
                "onset_fp": fp_o,
                "onset_fn": fn_o,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Evaluate deployed U-Net on held-out LGS test subjects"
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/stim_catalog.parquet"),
        help="Path to stim_catalog.parquet",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/results/lgs_unet_eval_test.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--max-files", type=int, default=None, help="Max files (smoke test)"
    )
    ap.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default="mps",
        help="Compute device",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = load_deployed_model(device=args.device)
    res = evaluate_lgs(
        model,
        args.catalog,
        device=args.device,
        threshold=args.threshold,
        max_files=args.max_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    valid = res[res["error"].isna()] if "error" in res.columns else res
    print(f"saved {len(valid)} rows -> {args.output}")
    for m in ["onset_f1", "event_f1", "raw_event_f1"]:
        if m in valid.columns:
            print(f"  {m}: {valid[m].mean():.4f} ± {valid[m].std():.4f}")


if __name__ == "__main__":
    main()
