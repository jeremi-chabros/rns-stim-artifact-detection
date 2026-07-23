#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""
Build stim catalog from lgs.db.

Replaces the old 4-script pipeline (parse_stimdata.jl → build_master_csv.py →
add_durations.py → convert_h5.py) with a single DB query + aggregation.

Output: data/stim_catalog.parquet — one row per ECoG file containing stim events.

Columns:
  filename, file_path, patient_id, subject, subject_id_lr, side, site,
  trigger, length_sec, sampling_rate, n_stim_events, onset_times,
  therapy_counts, epoch_start_gmt, epoch_end_gmt, rx_enabled,
  lead_1, lead_2, t1b1_{ma,us,uc,hz,ms}, t1b2_{ma,us,uc,hz,ms},
  mask_duration_ms, mask_onset_offset_ms, manually_refined
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from lgs_db import query_df

DB_PATH = Path("/path/to/Research/lgs-db/data/lgs.db")

STIM_EVENTS = (
    "IPROG_DATA_RESP_THERAPY",
    "IPROG_DATA_PC_THERAPY",
)

# Therapy parameter columns to carry from programming_epochs (T1 only —
# T1 fires first and is the most common; higher tiers have the same or
# similar params within an epoch).
THERAPY_PARAM_COLS = [
    "t1b1_ma",
    "t1b1_us",
    "t1b1_uc",
    "t1b1_hz",
    "t1b1_ms",
    "t1b1_path",
    "t1b2_ma",
    "t1b2_us",
    "t1b2_uc",
    "t1b2_hz",
    "t1b2_ms",
    "t1b2_path",
]


def fetch_stim_annotations() -> pd.DataFrame:
    """Fetch all stim annotations joined with catalog, crosswalk, and epoch data."""
    sql = """
    SELECT
        REPLACE(c.filename, '.dat', '') AS filename,
        c.file_path,
        c.patient_id,
        pc.subject,
        a.subject_id_lr,
        pc.side,
        pc.site,
        c.trigger,
        c.length_sec,
        c.sampling_rate,
        a.start_at,
        a.therapy_count,
        a.epoch_start_gmt,
        a.epoch_end_gmt,
        pe.rx_enabled,
        pe.lead_1,
        pe.lead_2,
        pe.t1b1_ma, pe.t1b1_us, pe.t1b1_uc, pe.t1b1_hz, pe.t1b1_ms, pe.t1b1_path,
        pe.t1b2_ma, pe.t1b2_us, pe.t1b2_uc, pe.t1b2_hz, pe.t1b2_ms, pe.t1b2_path
    FROM ecog_annotations a
    JOIN ecog_catalog c
        ON REPLACE(a.lay_filename, '.lay', '.dat') = c.filename
    JOIN patient_crosswalk pc
        ON c.patient_id = pc.patient_id
    JOIN programming_epochs pe
        ON a.subject_id_lr = pe.subject_id_lr
        AND a.epoch_start_gmt = pe.epoch_start_gmt
    WHERE a.event_type IN ('IPROG_DATA_RESP_THERAPY', 'IPROG_DATA_PC_THERAPY')
    ORDER BY c.patient_id, a.start_at
    """
    return query_df(sql, db=DB_PATH)


def compute_mask_duration_ms(row: pd.Series) -> float:
    """Compute stim mask duration from T1 B1+B2 durations.

    Matches legacy add_durations.py logic:
    - If B2 disabled (NaN) or B1+B2 < 500 ms → use A2_window_total fallback
      (legacy value = 1024 ms across all epochs)
    - Otherwise → B1 + B2 duration
    """
    A2_WINDOW_FALLBACK = 1024.0  # legacy A2_window_total, constant across epochs

    b1 = row.get("t1b1_ms")
    b2 = row.get("t1b2_ms")

    if pd.isna(b1) or b1 is None:
        return A2_WINDOW_FALLBACK

    b1 = float(b1)
    if pd.isna(b2) or b2 is None:
        return A2_WINDOW_FALLBACK

    b2 = float(b2)
    if b1 + b2 < 500:
        return A2_WINDOW_FALLBACK
    return b1 + b2


def aggregate_per_file(ann: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event annotations to one row per file."""
    # Sort events within each file by start_at
    ann = ann.sort_values(["filename", "start_at"])

    # Columns that are constant per file (take first)
    first_cols = [
        "file_path",
        "patient_id",
        "subject",
        "subject_id_lr",
        "side",
        "site",
        "trigger",
        "length_sec",
        "sampling_rate",
        "epoch_start_gmt",
        "epoch_end_gmt",
        "rx_enabled",
        "lead_1",
        "lead_2",
    ] + THERAPY_PARAM_COLS

    def agg_file(g: pd.DataFrame) -> dict:
        first = g.iloc[0]
        row = {col: first[col] for col in first_cols}
        row["n_stim_events"] = len(g)
        row["onset_times"] = g["start_at"].tolist()
        row["therapy_counts"] = g["therapy_count"].astype(int).tolist()
        row["mask_duration_ms"] = compute_mask_duration_ms(first)
        row["mask_onset_offset_ms"] = 0.0
        row["manually_refined"] = False
        return row

    records = []
    for filename, g in ann.groupby("filename", sort=False):
        rec = agg_file(g)
        rec["filename"] = filename
        records.append(rec)

    return pd.DataFrame(records)


def validate_catalog(cat: pd.DataFrame, ann: pd.DataFrame) -> None:
    """Run sanity checks on the catalog."""
    n_files = len(cat)
    n_events_cat = cat["n_stim_events"].sum()
    n_events_ann = len(ann)

    print(f"\n{'='*60}")
    print(f"  Catalog: {n_files:,} files, {n_events_cat:,} stim events")
    print(f"  Raw annotations: {n_events_ann:,} events")
    assert (
        n_events_cat == n_events_ann
    ), f"Event count mismatch: catalog={n_events_cat}, annotations={n_events_ann}"

    # Check file paths exist
    missing = []
    for fp in cat["file_path"]:
        if not Path(fp).exists():
            missing.append(fp)
    n_missing = len(missing)
    pct_avail = (n_files - n_missing) / n_files * 100
    print(f"  Files on disk: {n_files - n_missing:,}/{n_files:,} ({pct_avail:.1f}%)")
    if n_missing > 0:
        print(f"  WARNING: {n_missing:,} files not found on disk")
        for fp in missing[:5]:
            print(f"    {fp}")
        if n_missing > 5:
            print(f"    ... and {n_missing - 5} more")

    # Per-subject summary
    subj = cat.groupby("subject").agg(
        n_files=("filename", "count"),
        n_events=("n_stim_events", "sum"),
        n_epochs=("epoch_start_gmt", "nunique"),
    )
    print(f"\n  Per-subject summary ({len(subj)} subjects):")
    print(subj.to_string(max_rows=25))

    # Mask duration distribution
    print("\n  mask_duration_ms distribution:")
    print(cat["mask_duration_ms"].describe().to_string())
    print(f"  Unique values: {sorted(cat['mask_duration_ms'].unique())}")
    print(f"{'='*60}")


def merge_refinements(cat: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Preserve manual refinements from existing catalog on rebuild.

    If an existing catalog exists and has rows with manually_refined=True,
    merge those mask_duration_ms, mask_onset_offset_ms, and manually_refined
    values back into the freshly built catalog (keyed by filename).
    """
    if not output_path.exists():
        return cat

    old = pd.read_parquet(output_path)
    if "manually_refined" not in old.columns:
        return cat

    refined = old[old["manually_refined"] == True][
        ["filename", "mask_duration_ms", "mask_onset_offset_ms", "manually_refined"]
    ]
    if refined.empty:
        print("  No manual refinements to preserve.")
        return cat

    n_before = len(refined)
    cat = cat.set_index("filename")
    refined = refined.set_index("filename")

    # Only update rows that exist in the new catalog
    common = cat.index.intersection(refined.index)
    cat.loc[common, "mask_duration_ms"] = refined.loc[common, "mask_duration_ms"]
    cat.loc[common, "mask_onset_offset_ms"] = refined.loc[
        common, "mask_onset_offset_ms"
    ]
    cat.loc[common, "manually_refined"] = True
    cat = cat.reset_index()

    n_after = int(cat["manually_refined"].sum())
    print(f"  Preserved {n_after}/{n_before} manual refinements.")
    return cat


def build_catalog(
    output: Path | None = None,
    *,
    raw: bool = False,
    validate: bool = True,
    backup: bool = True,
) -> pd.DataFrame:
    """Build stim catalog parquet from lgs.db.

    Parameters
    ----------
    output : Path or None
        Where to write the parquet.  None = data/stim_catalog.parquet.
    raw : bool
        If True, skip merging refinements from any existing catalog —
        produces a clean catalog with mask_onset_offset_ms=0 and
        manually_refined=False for every row.
    validate : bool
        Print validation summary.
    backup : bool
        Backup existing file before overwriting.

    Returns
    -------
    pd.DataFrame with onset_times/therapy_counts as Python lists (not JSON).
    """
    if output is None:
        output = Path(__file__).parent.parent / "data" / "stim_catalog.parquet"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if backup and output.exists():
        backup_dir = output.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"stim_catalog_{ts}_pre_rebuild.parquet"
        shutil.copy2(output, backup_path)
        print(f"  Backed up existing catalog → {backup_path.name}")

    print("Fetching stim annotations from lgs.db ...")
    ann = fetch_stim_annotations()
    print(f"  {len(ann):,} annotation rows")

    print("Aggregating per file ...")
    cat = aggregate_per_file(ann)
    print(f"  {len(cat):,} files with stim events")

    if not raw:
        cat = merge_refinements(cat, output)
    else:
        print("  Skipping refinement merge (raw mode).")

    if validate:
        validate_catalog(cat, ann)

    # Save — onset_times and therapy_counts stored as JSON strings for parquet
    cat_out = cat.copy()
    cat_out["onset_times"] = cat_out["onset_times"].apply(json.dumps)
    cat_out["therapy_counts"] = cat_out["therapy_counts"].apply(json.dumps)

    cat_out.to_parquet(output, index=False, engine="pyarrow")
    print(f"\nSaved → {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")

    return cat


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build stim catalog from lgs.db")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output parquet path (default: data/stim_catalog.parquet)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip merging refinements — produce clean raw labels",
    )
    args = parser.parse_args()

    build_catalog(output=args.output, raw=args.raw)


if __name__ == "__main__":
    main()
