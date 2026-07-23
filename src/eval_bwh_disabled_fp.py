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
#   "bwh-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# bwh-db = { path = "/path/to/Research/bwh-db" }
# ///
"""Specificity / false-positive test on stimulation-DISABLED BWH recordings.

During ``rx_enabled='Disabled'`` epochs the device delivered no therapy, so the
recordings contain no stimulation artifact: every detected event is a false
positive.  These recordings are fully held out (no BWH data was used in
training), making this a clean specificity benchmark for the deployed detector.
The conditioning vector is zeros (no therapy), matching how negatives were
presented during training.

Usage:
    uv run src/eval_bwh_disabled_fp.py --max-files 1500       # subset run
    uv run src/eval_bwh_disabled_fp.py --smoke                # 5-file check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

STIMASK_SRC = Path("/path/to/Research/stimask/src")
sys.path.insert(0, str(STIMASK_SRC))

# eval_bwh sets up the deployed sys.path and re-exports the prepare helpers.
from eval_bwh import (  # noqa: E402
    BOUNDARY_MARGIN_S,
    SAMPLING_RATE,
    extract_events,
    load_deployed_model,
    postprocess_predictions,
    predict_file,
)
from prepare import N_COND_FEATURES  # noqa: E402
from bwh_db import query_df  # noqa: E402

DB_PATH = Path("/path/to/Research/bwh-db/data/bwh.db")
BWH_DATA_ROOT = Path(
    "/path/to/data/RNS/"
    "bwh"
)
STIM_CATALOG = Path("/path/to/Research/stimask/data/bwh_stim_catalog.parquet")
DISABLED_CATALOG = Path("/path/to/Research/stimask/data/bwh_disabled_catalog.parquet")


def build_disabled_catalog(rebuild: bool = False) -> pd.DataFrame:
    """Catalog of BWH ECoG files recorded during stim-disabled epochs.

    Joins ``ecog_catalog`` to ``programming_epochs`` on timestamp overlap, keeps
    only ``rx_enabled='Disabled'`` rows, excludes any file that also carries a
    stim annotation, and resolves the on-disk ``.dat`` path.
    """
    if DISABLED_CATALOG.exists() and not rebuild:
        return pd.read_parquet(DISABLED_CATALOG)

    sql = """
    SELECT REPLACE(ec.filename, '.lay', '') AS filename,
           ec.patient_id, pc.subject, ec.length_sec, ec.timestamp_utc
    FROM ecog_catalog ec
    JOIN patient_crosswalk pc ON ec.patient_id = pc.patient_id
    JOIN programming_epochs pe
      ON ec.patient_id = pe.patient_id
     AND ec.timestamp_utc >= pe.epoch_start_gmt
     AND ec.timestamp_utc <  pe.epoch_end_gmt
    WHERE pe.rx_enabled = 'Disabled'
    """
    df = query_df(sql, db=DB_PATH).drop_duplicates("filename")

    stim_fns = set(pd.read_parquet(STIM_CATALOG)["filename"].astype(str))
    df = df[~df["filename"].astype(str).isin(stim_fns)].copy()

    print(f"Scanning {BWH_DATA_ROOT} for .dat files ...")
    idx = {p.stem: str(p) for p in BWH_DATA_ROOT.rglob("*.dat")}
    df["file_path"] = df["filename"].astype(str).map(idx)
    df = df[df["file_path"].notna()].copy()
    df = df[df["length_sec"].notna() & (df["length_sec"] > 0)].copy()
    df = df.reset_index(drop=True)

    DISABLED_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DISABLED_CATALOG, index=False)
    return df


def count_false_positives(
    model, file_path: str, cond: np.ndarray, *, device: str
) -> tuple[int, int]:
    """Run the deployed detection pipeline; return (n_fp_events, n_fp_samples).

    Any predicted event on a stim-disabled recording is a false positive.
    """
    proba = predict_file(model, file_path, cond, device=device)
    n = len(proba)
    pred = postprocess_predictions((proba > 0.5).astype(np.float32))
    margin = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
    pred[:margin] = 0.0
    pred[max(margin, n - margin) :] = 0.0
    return len(extract_events(pred)), int((pred > 0.5).sum())


def summarize(res: pd.DataFrame) -> None:
    """Print the specificity / false-positive summary."""
    valid = res[res["error"].isna()] if "error" in res.columns else res
    hours = valid["length_sec"].sum() / 3600
    tot_fp = int(valid["n_fp_events"].sum())
    clean = float((valid["n_fp_events"] == 0).mean() * 100)
    print(f"\n{'='*60}\n  SPECIFICITY / FALSE-POSITIVE (stim-disabled BWH)")
    print(
        f"  files={len(valid)}  subjects={valid['subject'].nunique()}  hours={hours:.1f}"
    )
    print(f"  total false-positive events = {tot_fp}")
    print(f"  FP rate = {tot_fp/hours:.3f} events/hour")
    print(f"  recordings with ZERO false detections = {clean:.1f}%")
    print(f"  mean FP events/file = {valid['n_fp_events'].mean():.4f}")
    print(f"{'='*60}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files", type=int, default=1500)
    ap.add_argument("--per-subject-cap", type=int, default=50)
    ap.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-catalog", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="5-file integrity check")
    ap.add_argument(
        "-o",
        "--output",
        default="/path/to/Research/stimask/outputs/results/bwh_disabled_fp.csv",
    )
    args = ap.parse_args()

    cat = build_disabled_catalog(rebuild=args.rebuild_catalog)
    print(
        f"disabled catalog: {len(cat)} files, {cat['subject'].nunique()} subjects, "
        f"{cat['length_sec'].sum()/3600:.1f} h"
    )

    if args.smoke:
        cat = cat.groupby("subject", group_keys=False).head(1).head(5)
    else:
        # Per-subject cap via explicit concat (pandas 3 excludes the grouping
        # column inside groupby.apply, which dropped 'subject').
        parts = [
            g.sample(min(len(g), args.per_subject_cap), random_state=args.seed)
            for _, g in cat.groupby("subject", sort=False)
        ]
        cat = pd.concat(parts, ignore_index=True)
        if len(cat) > args.max_files:
            cat = cat.sample(args.max_files, random_state=args.seed)
    cat = cat.reset_index(drop=True)
    print(
        f"evaluating {len(cat)} files, {cat['subject'].nunique()} subjects, "
        f"{cat['length_sec'].sum()/3600:.1f} h"
    )

    model = load_deployed_model(device=args.device)
    cond = np.zeros(N_COND_FEATURES, dtype=np.float32)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, r in enumerate(tqdm(cat.itertuples(), total=len(cat), desc="disabled FP")):
        try:
            n_fp, n_fp_s = count_false_positives(
                model, r.file_path, cond, device=args.device
            )
            rows.append(
                {
                    "filename": r.filename,
                    "subject": r.subject,
                    "length_sec": r.length_sec,
                    "n_fp_events": n_fp,
                    "fp_samples": n_fp_s,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "filename": r.filename,
                    "subject": r.subject,
                    "length_sec": getattr(r, "length_sec", np.nan),
                    "n_fp_events": np.nan,
                    "fp_samples": np.nan,
                    "error": str(exc),
                }
            )
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    summarize(res)


if __name__ == "__main__":
    main()
