#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "bwh-db",
# ]
#
# [tool.uv.sources]
# bwh-db = { path = "/path/to/Research/bwh-db" }
# ///
"""
Build stim catalog from bwh.db.

Mirrors build_stim_catalog.py (LGS) but adapted for BWH's schema:
- Stim events: 'Therapy Delivered', 'Programmer Command Stimulation Delivered'
- Epoch linkage via timestamp overlap (epoch_start_gmt/epoch_end_gmt are NULL
  in ecog_annotations, so we join programming_epochs on timestamp range)
- Therapy params from programming_therapies table (not flattened on epochs)
- File paths resolved from disk scan (ecog_catalog.file_path is NULL)
- Pin-based electrode montage converted to LGS format: (xxxx)(xxxx)(x)

Output: data/bwh_stim_catalog.parquet — one row per ECoG file containing stim events.
Schema matches LGS stim_catalog.parquet exactly for downstream compatibility.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from bwh_db import query_df

DB_PATH = Path("/path/to/Research/bwh-db/data/bwh.db")

BWH_DATA_ROOT = Path(
    "/path/to/data/RNS/"
    "bwh"
)

STIM_EVENTS = (
    "Therapy Delivered",
    "Programmer Command Stimulation Delivered",
)

# epoch_id in programming_therapies = programming_epochs.rowid + EPOCH_ID_OFFSET
EPOCH_ID_OFFSET = 4059

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


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def parse_ipi_frequency(ipi_raw: str | float | None) -> float | None:
    """Parse IPI field to frequency in Hz.

    Handles the mixed formats found in programming_therapies.ipi_fix:
      "200.0 Hz  (5 ms)" -> 200.0
      "200"              -> 200.0
      142.9              -> 142.9
      "Off" / None       -> None
    """
    if ipi_raw is None or (isinstance(ipi_raw, float) and pd.isna(ipi_raw)):
        return None
    if isinstance(ipi_raw, (int, float)):
        return float(ipi_raw)
    s = str(ipi_raw).strip()
    if not s or s.lower() == "off":
        return None
    # Try "NNN.N Hz ..." format first
    m = re.match(r"([\d.]+)\s*Hz", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fall back to raw numeric
    try:
        return float(s)
    except ValueError:
        return None


def pins_to_montage_path(
    pin_0: str,
    pin_1: str,
    pin_2: str,
    pin_3: str,
    pin_4: str,
    pin_5: str,
    pin_6: str,
    pin_7: str,
    pin_8: str,
) -> str:
    """Convert 9 BWH pin values to LGS montage format.

    BWH stores per-pin polarity as '+', '-', or '0'.
    LGS format groups pins 0-3, 4-7, 8 into "(xxxx)(xxxx)(x)".
    """
    pins = [pin_0, pin_1, pin_2, pin_3, pin_4, pin_5, pin_6, pin_7, pin_8]
    chars = [str(p) if p is not None else "0" for p in pins]
    return f"({''.join(chars[:4])})({''.join(chars[4:8])})({''.join(chars[8:])})"


def build_file_index(root: Path) -> dict[str, str]:
    """Scan directory tree for .dat files, return {stem: absolute_path}.

    BWH .dat files live under:
      root/BWH_XX_PID .../BWH_XX_PID Data .../STEM.dat
    """
    index: dict[str, str] = {}
    for p in root.rglob("*.dat"):
        index[p.stem] = str(p)
    return index


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------


def fetch_stim_annotations() -> pd.DataFrame:
    """Fetch stim annotations joined with catalog, crosswalk, and epochs via timestamp overlap."""
    sql = """
    SELECT
        REPLACE(ec.filename, '.lay', '') AS filename,
        ec.patient_id,
        pc.subject,
        pc.subject_id_lr,
        pc.side,
        pc.site,
        ec.trigger,
        ec.length_sec,
        ec.sampling_rate,
        ec.timestamp_utc,
        ea.start_at,
        ea.therapy_count,
        pe.epoch_start_gmt,
        pe.epoch_end_gmt,
        pe.rx_enabled,
        pe.lead_1,
        pe.lead_2,
        pe.rowid AS epoch_rowid
    FROM ecog_annotations ea
    JOIN ecog_catalog ec
        ON ec.filename = ea.lay_filename
    JOIN patient_crosswalk pc
        ON ec.patient_id = pc.patient_id
    JOIN programming_epochs pe
        ON ec.patient_id = pe.patient_id
        AND ec.timestamp_utc >= pe.epoch_start_gmt
        AND ec.timestamp_utc < pe.epoch_end_gmt
    WHERE ea.event_type IN ('Therapy Delivered',
                            'Programmer Command Stimulation Delivered')
    ORDER BY ec.patient_id, ea.start_at
    """
    return query_df(sql, db=DB_PATH)


def fetch_therapy_params() -> pd.DataFrame:
    """Fetch T1 B1+B2 therapy parameters from programming_therapies.

    Returns a DataFrame with epoch_rowid + burst columns pivoted into
    t1b1_* and t1b2_* columns matching LGS schema.
    """
    sql = """
    SELECT
        epoch_id,
        burst_num,
        amp,
        pulse_width,
        charge_density,
        duration,
        ipi_fix,
        pin_0, pin_1, pin_2, pin_3, pin_4, pin_5, pin_6, pin_7, pin_8
    FROM programming_therapies
    WHERE therapy_num = 1 AND burst_num IN (1, 2)
    """
    raw = query_df(sql, db=DB_PATH)

    # Convert epoch_id to epoch_rowid matching programming_epochs.rowid.
    # Validate the offset: first therapy epoch_id should map to a valid epoch.
    min_eid = int(raw["epoch_id"].min())
    check = query_df(
        f"SELECT COUNT(*) AS n FROM programming_epochs WHERE rowid = {min_eid - EPOCH_ID_OFFSET}",
        db=DB_PATH,
    )
    assert check["n"].iloc[0] > 0, (
        f"EPOCH_ID_OFFSET {EPOCH_ID_OFFSET} invalid: epoch_id {min_eid} "
        f"maps to rowid {min_eid - EPOCH_ID_OFFSET} which doesn't exist"
    )
    raw["epoch_rowid"] = raw["epoch_id"] - EPOCH_ID_OFFSET

    records: list[dict] = []
    for epoch_rowid, grp in raw.groupby("epoch_rowid"):
        rec: dict = {"epoch_rowid": int(epoch_rowid)}
        for _, row in grp.iterrows():
            b = int(row["burst_num"])
            prefix = f"t1b{b}"
            rec[f"{prefix}_ma"] = row["amp"]
            rec[f"{prefix}_us"] = row["pulse_width"]
            rec[f"{prefix}_uc"] = row["charge_density"]
            rec[f"{prefix}_hz"] = parse_ipi_frequency(row["ipi_fix"])
            rec[f"{prefix}_ms"] = row["duration"]
            rec[f"{prefix}_path"] = pins_to_montage_path(
                row["pin_0"],
                row["pin_1"],
                row["pin_2"],
                row["pin_3"],
                row["pin_4"],
                row["pin_5"],
                row["pin_6"],
                row["pin_7"],
                row["pin_8"],
            )
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation (duplicated from build_stim_catalog.py — uv scripts have
# isolated venvs so we cannot import across scripts)
# ---------------------------------------------------------------------------


def compute_mask_duration_ms(row: pd.Series) -> float:
    """Compute stim mask duration from T1 B1+B2 durations.

    Uses actual B1+B2 sum directly. BWH therapy durations of 200-400ms
    are real — the LGS legacy fallback (1024ms when B1+B2 < 500ms) was a
    heuristic that inflates GT masks ~4.5x beyond the actual artifact,
    destroying eval metrics (IoU < 0.3 on correctly-predicted events).

    Falls back to B1-only (doubled) if B2 is missing, or 1024ms only
    when no therapy duration data is available at all.
    """
    FALLBACK = 1024.0

    b1 = row.get("t1b1_ms")
    b2 = row.get("t1b2_ms")

    if pd.isna(b1) or b1 is None:
        return FALLBACK

    b1 = float(b1)
    if pd.isna(b2) or b2 is None:
        # B2 disabled — use B1 alone (single burst)
        return b1 if b1 > 0 else FALLBACK

    b2 = float(b2)
    total = b1 + b2
    return total if total > 0 else FALLBACK


def aggregate_per_file(ann: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event annotations to one row per file."""
    ann = ann.sort_values(["filename", "start_at"])

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
        row = {col: first.get(col) for col in first_cols}
        row["n_stim_events"] = len(g)
        row["onset_times"] = g["start_at"].tolist()
        row["therapy_counts"] = g["therapy_count"].fillna(0).astype(int).tolist()
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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

    # Check file paths resolved
    has_path = cat["file_path"].notna().sum()
    no_path = cat["file_path"].isna().sum()
    pct = has_path / n_files * 100 if n_files else 0
    print(f"  File paths resolved: {has_path:,}/{n_files:,} ({pct:.1f}%)")
    if no_path > 0:
        print(f"  WARNING: {no_path:,} files not found on disk")

    # Check file existence for resolved paths
    if has_path > 0:
        missing = sum(1 for fp in cat["file_path"].dropna() if not Path(fp).exists())
        if missing > 0:
            print(f"  WARNING: {missing:,} resolved paths don't exist on disk")

    # Per-subject summary
    subj = cat.groupby("subject").agg(
        n_files=("filename", "count"),
        n_events=("n_stim_events", "sum"),
        n_epochs=("epoch_start_gmt", "nunique"),
    )
    print(f"\n  Per-subject summary ({len(subj)} subjects):")
    print(subj.to_string(max_rows=30))

    # Mask duration distribution
    print(f"\n  mask_duration_ms distribution:")
    print(cat["mask_duration_ms"].describe().to_string())
    print(f"  Unique values: {sorted(cat['mask_duration_ms'].unique())}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_catalog(
    output: Path | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Build BWH stim catalog parquet from bwh.db.

    Parameters
    ----------
    output : Path or None
        Where to write the parquet. None = data/bwh_stim_catalog.parquet.
    validate : bool
        Print validation summary.

    Returns
    -------
    pd.DataFrame with onset_times/therapy_counts as Python lists (not JSON).
    """
    if output is None:
        output = Path(__file__).parent.parent / "data" / "bwh_stim_catalog.parquet"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Build file index from disk
    print(f"Scanning {BWH_DATA_ROOT} for .dat files ...")
    file_index = build_file_index(BWH_DATA_ROOT)
    print(f"  {len(file_index):,} .dat files indexed")

    # Step 2: Fetch stim annotations with epoch linkage
    print("Fetching stim annotations from bwh.db ...")
    ann = fetch_stim_annotations()
    print(f"  {len(ann):,} annotation rows")

    # Step 3: Fetch therapy params and merge onto annotations
    print("Fetching therapy parameters ...")
    therapy = fetch_therapy_params()
    print(f"  {len(therapy):,} epoch therapy records")

    ann = ann.merge(therapy, on="epoch_rowid", how="left")
    # Drop the helper column
    ann = ann.drop(columns=["epoch_rowid"])

    # Step 4: Resolve file paths from disk index
    print("Resolving file paths ...")
    ann["file_path"] = ann["filename"].map(file_index)
    n_resolved = ann["file_path"].notna().sum()
    n_unique_files = ann["filename"].nunique()
    n_resolved_files = ann.loc[ann["file_path"].notna(), "filename"].nunique()
    print(
        f"  {n_resolved_files:,}/{n_unique_files:,} unique files resolved "
        f"({n_resolved:,}/{len(ann):,} annotation rows)"
    )

    # Step 5: Aggregate to one row per file
    print("Aggregating per file ...")
    cat = aggregate_per_file(ann)
    print(f"  {len(cat):,} files with stim events")

    if validate:
        validate_catalog(cat, ann)

    # Step 6: Save -- onset_times and therapy_counts stored as JSON strings
    cat_out = cat.copy()
    cat_out["onset_times"] = cat_out["onset_times"].apply(json.dumps)
    cat_out["therapy_counts"] = cat_out["therapy_counts"].apply(json.dumps)

    cat_out.to_parquet(output, index=False, engine="pyarrow")
    print(f"\nSaved -> {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")

    return cat


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def smoke_test() -> None:
    """Run pure-function smoke tests (no DB or disk access)."""
    print("Running smoke tests ...")

    # parse_ipi_frequency
    assert parse_ipi_frequency("200.0 Hz  (5 ms)") == 200.0
    assert parse_ipi_frequency("200") == 200.0
    assert parse_ipi_frequency("142.9") == 142.9
    assert parse_ipi_frequency(142.9) == 142.9
    assert parse_ipi_frequency("5") == 5.0
    assert parse_ipi_frequency("Off") is None
    assert parse_ipi_frequency(None) is None
    assert parse_ipi_frequency("") is None
    assert parse_ipi_frequency(float("nan")) is None
    print("  parse_ipi_frequency: OK")

    # pins_to_montage_path
    # pins 0-3 -> group1, pins 4-7 -> group2, pin 8 -> group3
    result = pins_to_montage_path("+", "-", "-", "-", "-", "0", "0", "0", "0")
    assert result == "(+---)(-000)(0)", f"Got: {result}"

    result = pins_to_montage_path("0", "+", "+", "+", "+", "-", "-", "-", "-")
    assert result == "(0+++)(+---)(-)", f"Got: {result}"

    result = pins_to_montage_path("0", "0", "0", "0", "0", "0", "0", "0", "0")
    assert result == "(0000)(0000)(0)", f"Got: {result}"

    result = pins_to_montage_path("+", "0", "0", "-", "-", "0", "0", "+", "-")
    assert result == "(+00-)(-00+)(-)", f"Got: {result}"
    print("  pins_to_montage_path: OK")

    # compute_mask_duration_ms — uses actual B1+B2, no 500ms threshold
    row_both = pd.Series({"t1b1_ms": 300, "t1b2_ms": 300})
    assert compute_mask_duration_ms(row_both) == 600.0

    row_small = pd.Series({"t1b1_ms": 100, "t1b2_ms": 100})
    assert compute_mask_duration_ms(row_small) == 200.0  # actual, not fallback

    row_nan = pd.Series({"t1b1_ms": None, "t1b2_ms": 300})
    assert compute_mask_duration_ms(row_nan) == 1024.0  # no B1 → fallback

    row_b2nan = pd.Series({"t1b1_ms": 300, "t1b2_ms": None})
    assert compute_mask_duration_ms(row_b2nan) == 300.0  # B1-only

    row_empty = pd.Series(dtype=float)
    assert compute_mask_duration_ms(row_empty) == 1024.0
    print("  compute_mask_duration_ms: OK")

    print("All smoke tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build BWH stim catalog from bwh.db")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output parquet path (default: data/bwh_stim_catalog.parquet)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run pure-function smoke tests only (no DB access)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    build_catalog(output=args.output)


if __name__ == "__main__":
    main()
