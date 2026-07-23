#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0","numpy>=2.0","pandas>=2.0","pyarrow>=14.0","scipy>=1.10","tqdm>=4.60","lgs-db"]
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Cache per-file predicted event onsets + ground truth for the BWH cohort.

One inference pass so the duration-perturbation (A2) and refined-subset (A3)
sensitivity analyses become cheap post-processing. Reuses the exact inference +
postprocessing functions that produced the published BWH results.
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
    SAMPLING_RATE,
    BOUNDARY_MARGIN_S,
)

sys.path.insert(0, "/path/to/Research/training-pipeline")
from prepare import extract_conditioning_vector, extract_events  # noqa: E402


def cache_predicted_events(
    catalog_path: Path,
    out_path: Path,
    *,
    model,
    device: str = "mps",
    threshold: float = 0.5,
    max_files: int | None = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run inference and store predicted event onsets + GT per file.

    Output columns: filename, subject, mask_duration_ms, manually_refined,
    n_samples, pred_onsets (json list of int sample indices),
    onset_times (json list of float seconds).

    Args:
        catalog_path: Path to bwh_stim_catalog.parquet.
        out_path: Destination parquet path.
        model: Loaded StimArtifactUNet model.
        device: Torch device string.
        threshold: Binary classification threshold applied to U-Net probabilities.
        max_files: Random subsample cap; None to use full catalog.
        seed: RNG seed for reproducible sampling.

    Returns:
        DataFrame with one row per file.
    """
    df = pd.read_parquet(catalog_path)
    df = df[df["file_path"].notna()].copy()
    df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy()
    if max_files and len(df) > max_files:
        df = df.sample(n=max_files, random_state=seed).reset_index(drop=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="cache BWH"):
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

        pred_bin = postprocess_predictions((proba > threshold).astype(np.float32))
        m = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
        pred_bin[:m] = 0.0
        pred_bin[max(m, len(pred_bin) - m) :] = 0.0
        pred_events = extract_events(pred_bin)

        onsets = row.get("onset_times", "[]")
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []
        elif not isinstance(onsets, list):
            onsets = []

        md = row.get("mask_duration_ms", 1024.0)
        md = 1024.0 if pd.isna(md) else float(md)

        rows.append(
            {
                "filename": row["filename"],
                "subject": row["subject"],
                "mask_duration_ms": md,
                "manually_refined": bool(row.get("manually_refined", False)),
                "n_samples": int(len(proba)),
                "pred_onsets": json.dumps([int(s) for s, _e in pred_events]),
                "onset_times": json.dumps([float(t) for t in onsets]),
            }
        )

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"cached {len(out)} rows -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache U-Net predicted event onsets for the BWH cohort."
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/bwh_stim_catalog.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/bwh_pred_events.parquet"),
    )
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = load_deployed_model(device=args.device)
    cache_predicted_events(
        args.catalog,
        args.output,
        model=model,
        device=args.device,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
