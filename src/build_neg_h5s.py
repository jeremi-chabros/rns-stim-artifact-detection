#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "h5py>=3.8.0",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Build H5 files for negative (non-stim) ECoG recordings.

Identifies files from disabled-therapy epochs and enabled-but-no-fire epochs,
converts .dat → H5 (matching existing format), and outputs neg_catalog.parquet.

Strategy:
  Tier 1: ALL files from disabled/no-epoch periods (guaranteed artifact-free)
  Tier 2: Per-subject sample from enabled epochs where stim didn't fire (hard negatives)

Usage:
    uv run src/build_neg_h5s.py                    # default: all disabled + 30% enabled
    uv run src/build_neg_h5s.py --enabled-frac 0.5 # 50% of enabled-epoch negatives
    uv run src/build_neg_h5s.py --dry-run           # just report counts
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd
from lgs_db import query_df, read_dat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
LAY_FILE = Path("/path/to/data/RNS/LGS/test.lay")
H5_OUTPUT_DIR = Path("/path/to/data/RNS/LGS/h5s_neg")
DB_PATH = Path("/path/to/Research/lgs-db/data/lgs.db")
CATALOG_PATH = Path("data/stim_catalog.parquet")
NEG_CATALOG_PATH = Path("data/neg_catalog.parquet")


# ── Lay file parsing (from convert_h5.py) ────────────────────────────────


@dataclass
class ECoGHeader:
    sampling_rate: int
    waveform_count: int
    enabled_channels: list[int]


def parse_lay_file(lay_path: Path) -> ECoGHeader:
    with open(lay_path, encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]

    def read_hdr(name: str, numeric: bool = False):
        for line in lines:
            if line.startswith(f"{name}="):
                v = line.split("=", 1)[1]
                return int(v) if numeric else v
        return 0 if numeric else ""

    return ECoGHeader(
        sampling_rate=read_hdr("SamplingRate", True),
        waveform_count=read_hdr("WaveformCount", True),
        enabled_channels=[1, 1, 1, 1],  # all channels enabled for our data
    )


# ── Conversion ────────────────────────────────────────────────────────────


def convert_one(
    dat_path: Path,
    output_dir: Path,
    header: ECoGHeader,
    patient_id: str,
) -> bool:
    """Convert a single .dat to H5 (matching existing stim H5 format)."""
    try:
        out_path = output_dir / f"{dat_path.stem}.h5"
        if out_path.exists():
            return True

        # Read raw int16 via lgs_db (de-interleaved, offset-subtracted)
        raw = read_dat(str(dat_path))  # (4, N) int16

        with h5py.File(str(out_path), "w") as f:
            f.attrs["sampling_rate"] = header.sampling_rate
            f.attrs["waveform_count"] = header.waveform_count
            f.attrs["patient_id"] = patient_id
            f.attrs["source_file"] = dat_path.name
            f.attrs["negative"] = True

            for ch in range(4):
                dset = f.create_dataset(
                    f"channel_{ch + 1}",
                    data=raw[ch],
                    compression="gzip",
                    compression_opts=4,
                )
                dset.attrs["channel_index"] = ch + 1
        return True
    except Exception as e:
        log.error(f"Failed {dat_path.name}: {e}")
        return False


def _worker(args: tuple) -> bool:
    dat_path, output_dir, header, patient_id = args
    return convert_one(Path(dat_path), Path(output_dir), header, patient_id)


# ── DB queries ────────────────────────────────────────────────────────────


def get_negative_files(
    stim_filenames: set[str], enabled_frac: float = 0.3
) -> pd.DataFrame:
    """Query DB for negative files, stratified by epoch type.

    Returns DataFrame with columns:
      filename, file_path, patient_id, subject, subject_id_lr, side, site,
      length_sec, sampling_rate, epoch_type ('disabled', 'enabled_no_fire', 'no_epoch')
    """
    log.info("Querying all ECoG files with epoch info...")

    # Get all files with their epoch rx_enabled status
    df = query_df(
        """
        SELECT
            e.filename,
            e.file_path,
            e.patient_id,
            p.subject,
            p.subject_id_lr,
            p.side,
            p.site,
            e.length_sec,
            e.sampling_rate,
            pe.rx_enabled
        FROM ecog_catalog e
        JOIN patient_crosswalk p ON e.patient_id = p.patient_id
        LEFT JOIN programming_epochs pe
            ON p.subject_id_lr = pe.subject_id_lr
            AND e.timestamp_utc >= pe.epoch_start_gmt
            AND e.timestamp_utc < pe.epoch_end_gmt
    """,
        db=DB_PATH,
    )

    log.info(f"  Total files in DB: {len(df)}")

    # Classify epoch type per file
    # A file might match multiple epochs; take the "most active" classification
    def classify_epoch(group):
        rx_vals = set(group["rx_enabled"].dropna())
        if "Enabled" in rx_vals:
            return "enabled"
        elif "Disabled" in rx_vals:
            return "disabled"
        else:
            return "no_epoch"

    file_epochs = (
        df.groupby("filename").apply(classify_epoch, include_groups=False).reset_index()
    )
    file_epochs.columns = ["filename", "epoch_type"]

    # Deduplicate file info (take first row per filename)
    file_info = df.drop_duplicates(subset="filename").drop(columns="rx_enabled")
    file_info = file_info.merge(file_epochs, on="filename")

    # Filter to non-stim files only
    neg = file_info[~file_info["filename"].isin(stim_filenames)].copy()
    log.info(f"  Non-stim files: {len(neg)}")

    # Stratify
    disabled = neg[neg["epoch_type"].isin(["disabled", "no_epoch"])]
    enabled = neg[neg["epoch_type"] == "enabled"]

    log.info(f"  Disabled/no-epoch (Tier 1): {len(disabled)}")
    log.info(f"  Enabled no-fire (Tier 2): {len(enabled)}")

    # Tier 1: ALL disabled/no-epoch files
    result_parts = [disabled]

    # Tier 2: per-subject sample of enabled-epoch negatives
    if enabled_frac > 0 and len(enabled) > 0:
        sampled = enabled.groupby("subject", group_keys=False).apply(
            lambda g: g.sample(frac=enabled_frac, random_state=42),
            include_groups=False,
        )
        # Re-attach the subject column that gets lost with include_groups=False
        sampled = enabled.loc[sampled.index]
        sampled = sampled.copy()
        sampled["epoch_type"] = "enabled_no_fire"
        result_parts.append(sampled)
        log.info(f"  Sampled enabled negatives: {len(sampled)} ({enabled_frac:.0%})")

    result = pd.concat(result_parts, ignore_index=True)

    # Per-subject summary
    summary = result.groupby(["subject", "epoch_type"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    log.info(f"\nPer-subject negative files:\n{summary.to_string()}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert negative ECoG files to H5")
    parser.add_argument(
        "--enabled-frac",
        type=float,
        default=0.3,
        help="Fraction of enabled-epoch negatives to include per subject (default: 0.3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts only")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: ncores-2)",
    )
    args = parser.parse_args()

    # Load stim catalog to identify which files already have stim
    stim_df = pd.read_parquet(CATALOG_PATH)
    stim_filenames = set(stim_df["filename"].unique())
    log.info(f"Stim catalog: {len(stim_filenames)} files")

    # Get negative files
    neg_df = get_negative_files(stim_filenames, enabled_frac=args.enabled_frac)
    log.info(f"\nTotal negative files to convert: {len(neg_df)}")

    if args.dry_run:
        log.info("Dry run — no conversion performed")
        return

    # Parse lay file
    header = parse_lay_file(LAY_FILE)
    log.info(f"Lay header: SR={header.sampling_rate}, WF={header.waveform_count}")

    # Create output dirs per patient
    H5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pid in neg_df["patient_id"].unique():
        (H5_OUTPUT_DIR / str(pid)).mkdir(exist_ok=True)

    # Use file_path from DB directly — no directory scanning needed
    tasks = []
    missing = 0
    already_done = 0
    for _, row in neg_df.iterrows():
        pid = int(row["patient_id"])
        fn = row["filename"]
        out_dir = H5_OUTPUT_DIR / str(pid)

        # Check if already converted
        if (out_dir / f"{fn}.h5").exists():
            already_done += 1
            continue

        # Use file_path from DB (full absolute path to .dat)
        dat_path = Path(row["file_path"])
        if dat_path.exists():
            tasks.append((str(dat_path), str(out_dir), header, str(pid)))
        else:
            missing += 1

    log.info(f"Already converted: {already_done}")
    log.info(f"To convert: {len(tasks)}")
    log.info(f"Missing .dat: {missing}")

    if not tasks:
        log.info("Nothing to convert")
    else:
        n_workers = args.workers or max(1, mp.cpu_count() - 2)
        log.info(f"Converting with {n_workers} workers...")

        with mp.Pool(n_workers) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=100))

        ok = sum(results)
        log.info(f"Converted: {ok}/{len(tasks)} successful")

    # Save negative catalog
    neg_df.to_parquet(NEG_CATALOG_PATH, index=False)
    log.info(f"\nSaved negative catalog: {NEG_CATALOG_PATH} ({len(neg_df)} rows)")

    # Summary stats
    log.info("\n=== SUMMARY ===")
    log.info(f"Stim files:     {len(stim_filenames)}")
    log.info(f"Negative files: {len(neg_df)}")
    log.info(
        f"  Disabled:     {(neg_df['epoch_type'].isin(['disabled', 'no_epoch'])).sum()}"
    )
    log.info(f"  Enabled:      {(neg_df['epoch_type'] == 'enabled_no_fire').sum()}")
    log.info(f"Ratio neg/stim: {len(neg_df) / len(stim_filenames):.2f}")
    log.info(f"H5 output:      {H5_OUTPUT_DIR}")
    log.info(f"Catalog:        {NEG_CATALOG_PATH}")


if __name__ == "__main__":
    main()
