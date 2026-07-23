#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scikit-learn",
#     "scipy",
#     "torch",
#     "kymatio",
#     "xgboost",
#     "joblib",
#     "matplotlib",
#     "tqdm",
#     "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Retrain M3 ScatteringXGBoost with artifact_overlap_thresh=0.0 (presence-based).

Reuses the same training catalog and identical hyperparameters as the baseline;
only label-generation changes. Uses the pinned stim catalog snapshot
(data/stim_catalog_m3eval.parquet) so A1 differs from B0 only in the
label threshold, not in training data composition.

Output: data/baselines/m3_lowthresh/

**Idempotency:** re-running overwrites the output directory. Existing files in
`data/baselines/m3_lowthresh/` are overwritten by `ScatteringXGBoost.save()`.
"""

from __future__ import annotations

import os

# OpenMP conflict between torch/kymatio and XGBoost segfaults clf.fit() on
# macOS + Python 3.14 unless OMP is single-threaded. Must be set before any
# numeric library imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison_methods import ScatteringXGBoost


def main() -> None:
    """Retrain M3 with threshold=0.0 and save to data/baselines/m3_lowthresh/."""
    stim_catalog = PROJECT_ROOT / "data" / "stim_catalog_m3eval.parquet"
    neg_catalog = PROJECT_ROOT / "data" / "neg_catalog.parquet"
    assert stim_catalog.exists(), (
        f"Missing pinned stim catalog: {stim_catalog}. "
        "This snapshot reproduces B0's training conditions — regenerate from git commit 67f54a1 if lost."
    )
    assert neg_catalog.exists(), (
        f"Missing neg catalog: {neg_catalog}. "
        "Expected to exist; this is a checked-in project data file."
    )

    model = ScatteringXGBoost(
        window=256,
        J=5,
        Q=8,
        n_estimators=300,
        max_depth=6,
        sr=250,
        artifact_overlap_thresh=0.0,  # presence-based (the only change)
    )
    model.fit(
        neg_catalog_path=str(neg_catalog),
        stim_catalog_path=str(stim_catalog),
        max_stim_files=500,
        max_neg_files=500,
        windows_per_file=8,
        seed=42,
    )
    out = PROJECT_ROOT / "data" / "baselines" / "m3_lowthresh"
    model.save(out)
    print(f"[A1] Retrained model saved to {out}")


if __name__ == "__main__":
    main()
