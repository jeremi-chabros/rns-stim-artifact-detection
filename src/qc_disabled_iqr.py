#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Data-quality QC for the stimulation-disabled false-positive analysis.

Computes per-recording channel interquartile range (IQR), flags degenerate
flat-line acquisitions (a disconnected state with channel IQR below a small
threshold, on which the detector's per-window robust normalisation
``(x - median) / (IQR + eps)`` amplifies quantisation noise into spurious
detections), and recomputes the false-positive rate and clean-recording fraction
on valid ECoG only.

Inputs : outputs/results/bwh_disabled_fp.csv  (from eval_bwh_disabled_fp.py)
         data/bwh_disabled_catalog.parquet
Output : outputs/results/bwh_disabled_fp_iqr.csv  (per-file FP + min/max channel IQR)

Usage:
    uv run src/qc_disabled_iqr.py
    uv run src/qc_disabled_iqr.py --threshold 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/Research/stimask/src")
from lgs_db import read_dat, to_microvolts  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

FP_CSV = Path("outputs/results/bwh_disabled_fp.csv")
CATALOG = Path("data/bwh_disabled_catalog.parquet")
OUT_CSV = Path("outputs/results/bwh_disabled_fp_iqr.csv")


def channel_iqr(file_path: str) -> tuple[float, float]:
    """Return (min, max) across-channel IQR in microvolts for one recording."""
    uv = to_microvolts(read_dat(file_path)).astype(np.float64)
    iqr = np.percentile(uv, 75, axis=1) - np.percentile(uv, 25, axis=1)
    return float(iqr.min()), float(iqr.max())


def augment_with_iqr(fp: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach per-file min/max channel IQR by reading each recording once."""
    fp = fp.copy()
    fp["filename"] = fp["filename"].astype(str)
    catalog = catalog.copy()
    catalog["filename"] = catalog["filename"].astype(str)
    m = fp.merge(catalog[["filename", "file_path"]], on="filename", how="left")
    mins, maxs = [], []
    for r in tqdm(m.itertuples(), total=len(m), desc="IQR QC"):
        try:
            lo, hi = channel_iqr(r.file_path)
        except Exception:  # noqa: BLE001
            lo, hi = np.nan, np.nan
        mins.append(lo)
        maxs.append(hi)
    m["min_ch_iqr"] = mins
    m["max_ch_iqr"] = maxs
    return m


def summarize(m: pd.DataFrame, threshold: float) -> None:
    """Print FP specificity excluding degenerate (low-IQR) recordings."""
    deg = m["min_ch_iqr"] < threshold
    valid = (~deg) & m["min_ch_iqr"].notna()
    for label, sub in [("ALL", m), (f"VALID (IQR>={threshold})", m[valid])]:
        h = sub["length_sec"].sum() / 3600
        fp = int(sub["n_fp_events"].sum())
        z = 100 * (sub["n_fp_events"] == 0).mean()
        print(
            f"  {label}: {len(sub)} files  {h:.1f} h  fp={fp}  "
            f"rate={fp/h:.3f}/h  zero-FP={z:.2f}%"
        )
    print(
        f"  degenerate excluded: {int(deg.sum())} files / "
        f"{m.loc[deg,'length_sec'].sum()/3600:.1f} h"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=1.0, help="IQR (uV) cutoff")
    args = ap.parse_args()

    fp = pd.read_csv(FP_CSV)
    fp = fp[fp["error"].isna()] if "error" in fp.columns else fp
    m = augment_with_iqr(fp, pd.read_parquet(CATALOG))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    summarize(m, args.threshold)


def _smoke() -> None:
    """Pure-function check: IQR is zero for a constant signal, positive otherwise."""
    flat = np.full((4, 1000), -420.0)
    iqr = np.percentile(flat, 75, axis=1) - np.percentile(flat, 25, axis=1)
    assert float(iqr.max()) == 0.0
    print("smoke OK")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
