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
"""Seed-level external headline (reviewer Major 4).

Runs each of the five LGS-trained FiLM-last seed checkpoints through the
*published* BWH evaluation pipeline (``eval_bwh.evaluate_bwh`` -- calibrated
event F1, sample F1, onset recall, with the real conditioning vector) on a fixed
stratified subsample, so the central generalization claim carries seed-level
uncertainty rather than resting on a single best-validation checkpoint.

Usage:
    uv run src/eval_seed_ensemble.py --max-files 4000

CAVEAT (reviewer Major 4, 2026-06-18): the phase-2 seed checkpoints in
``data/checkpoints/phase2/ckpts`` are a DIFFERENT architecture than the deployed
model (depthwise kernel 7 + explicit-qkv attention bottleneck, i.e. the
``train.py`` definition) and do NOT load into the deployed eval model
(``train_5b0d152``: kernel 3, scaled-dot-product attention). A clean
deployed-config seed CI therefore requires retraining five seeds of the deployed
architecture in the separate training repository; this harness is ready for those.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import eval_bwh as E  # noqa: E402  (sets deployed path, imports model + pipeline)

CKPT_DIR = SRC.parent / "data" / "checkpoints" / "phase2" / "ckpts"
BWH_CATALOG = SRC.parent / "data" / "bwh_stim_catalog.parquet"
OUT = SRC.parent / "outputs" / "results" / "seed_ensemble_bwh.csv"
METRICS = ["event_f1", "sample_f1", "onset_recall"]


def load_checkpoint(path: Path, device: str):
    """Load a seed checkpoint into a fresh StimArtifactUNet (strip compile prefix)."""
    sd = torch.load(str(path), map_location=device, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = E.StimArtifactUNet()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--seed", type=int, default=42, help="subsample seed (fixed across ckpts)"
    )
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    ckpts = sorted(CKPT_DIR.glob("best_model_lgs_last_seed*.pt"))
    assert ckpts, f"no lgs_last seed checkpoints in {CKPT_DIR}"
    print(
        f"Found {len(ckpts)} seed checkpoints; {args.max_files} files each on {device}"
    )

    rows = []
    for ck in ckpts:
        model = load_checkpoint(ck, device)
        df = E.evaluate_bwh(
            model, BWH_CATALOG, max_files=args.max_files, device=device, seed=args.seed
        )
        df = df[df.get("error").isna()] if "error" in df.columns else df
        rec = {"checkpoint": ck.stem, "n_files": len(df)}
        for m in METRICS:
            rec[m] = float(df[m].mean()) if m in df.columns else float("nan")
        rows.append(rec)
        print(f"  {ck.stem}: " + "  ".join(f"{m}={rec[m]:.4f}" for m in METRICS))

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print("\n=== SEED ENSEMBLE (mean +/- sd across 5 seeds) ===")
    for m in METRICS:
        v = R[m].to_numpy(dtype=float)
        print(
            f"  {m:13s} mean={v.mean():.4f}  sd={v.std(ddof=1):.4f}  "
            f"range=[{v.min():.4f}, {v.max():.4f}]"
        )


if __name__ == "__main__":
    main()
