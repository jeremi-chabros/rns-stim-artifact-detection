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
"""Build H5 files for stim ECoG recordings from stim_catalog.parquet.

Reads the stim catalog, converts each unique .dat → H5 (per-channel int16,
gzip compressed), organized by patient_id subdirectories.

Usage:
    uv run src/build_stim_h5s.py                  # full conversion
    uv run src/build_stim_h5s.py --dry-run         # report counts only
    uv run src/build_stim_h5s.py --clean            # remove old H5s not in catalog first
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd
from lgs_db import read_dat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
LAY_FILE = Path("/path/to/data/RNS/LGS/test.lay")
H5_OUTPUT_DIR = Path("/path/to/data/RNS/LGS/h5s")
CATALOG_PATH = Path("data/stim_catalog.parquet")


# ── Lay file parsing (shared with build_neg_h5s.py) ──────────────────────


@dataclass
class ECoGHeader:
    sampling_rate: int
    waveform_count: int


def parse_lay_file(lay_path: Path) -> ECoGHeader:
    with open(lay_path, encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]

    def read_hdr(name: str) -> int:
        for line in lines:
            if line.startswith(f"{name}="):
                return int(line.split("=", 1)[1])
        return 0

    return ECoGHeader(
        sampling_rate=read_hdr("SamplingRate"),
        waveform_count=read_hdr("WaveformCount"),
    )


# ── Conversion ────────────────────────────────────────────────────────────


def convert_one(
    dat_path: Path,
    output_dir: Path,
    header: ECoGHeader,
    patient_id: str,
) -> bool:
    """Convert a single .dat to H5 (per-channel int16, gzip)."""
    try:
        out_path = output_dir / f"{dat_path.stem}.h5"
        if out_path.exists():
            return True

        raw = read_dat(str(dat_path))  # (4, N) int16

        with h5py.File(str(out_path), "w") as f:
            f.attrs["sampling_rate"] = header.sampling_rate
            f.attrs["waveform_count"] = header.waveform_count
            f.attrs["patient_id"] = patient_id
            f.attrs["source_file"] = dat_path.name

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


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert stim ECoG files to H5")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old H5s not in current catalog before converting",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: ncores-2)",
    )
    args = parser.parse_args()

    # Load stim catalog
    df = pd.read_parquet(CATALOG_PATH)
    log.info(f"Stim catalog: {len(df)} files, {df['subject'].nunique()} subjects")

    # Deduplicate: one row per (filename, patient_id, file_path)
    files = df[["filename", "patient_id", "file_path"]].drop_duplicates(
        subset="filename"
    )
    files["patient_id"] = files["patient_id"].astype(str)
    log.info(f"Unique files to convert: {len(files)}")

    # Per-patient summary
    summary = files.groupby("patient_id").size()
    log.info(f"Patient dirs needed: {len(summary)}")

    if args.clean and not args.dry_run:
        catalog_stems = set(files["filename"].values)
        removed = 0
        for pid_dir in H5_OUTPUT_DIR.iterdir():
            if not pid_dir.is_dir():
                continue
            for h5f in pid_dir.glob("*.h5"):
                if h5f.stem not in catalog_stems:
                    h5f.unlink()
                    removed += 1
        log.info(f"Cleaned {removed} stale H5 files")

    if args.dry_run:
        log.info("Dry run — no conversion performed")
        for pid, count in summary.items():
            log.info(f"  {pid}: {count} files")
        return

    # Parse lay file for header info
    header = parse_lay_file(LAY_FILE)
    log.info(f"Lay header: SR={header.sampling_rate}, WF={header.waveform_count}")

    # Create output dirs
    H5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pid in files["patient_id"].unique():
        (H5_OUTPUT_DIR / pid).mkdir(exist_ok=True)

    # Build task list
    tasks = []
    missing = 0
    already_done = 0
    for _, row in files.iterrows():
        pid = row["patient_id"]
        fn = row["filename"]
        out_dir = H5_OUTPUT_DIR / pid

        if (out_dir / f"{fn}.h5").exists():
            already_done += 1
            continue

        dat_path = Path(row["file_path"])
        if dat_path.exists():
            tasks.append((str(dat_path), str(out_dir), header, pid))
        else:
            missing += 1

    log.info(f"Already converted: {already_done}")
    log.info(f"To convert: {len(tasks)}")
    log.info(f"Missing .dat: {missing}")

    if not tasks:
        log.info("Nothing to convert — all files up to date")
    else:
        n_workers = args.workers or max(1, mp.cpu_count() - 2)
        log.info(f"Converting with {n_workers} workers...")

        with mp.Pool(n_workers) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=100))

        ok = sum(results)
        fail = len(results) - ok
        log.info(f"Converted: {ok}/{len(tasks)} ({fail} failures)")

    # Final validation
    total_h5 = sum(1 for _ in H5_OUTPUT_DIR.rglob("*.h5"))
    log.info("\n=== SUMMARY ===")
    log.info(f"Catalog files:  {len(files)}")
    log.info(f"H5 files:       {total_h5}")
    log.info(f"Missing .dat:   {missing}")
    log.info(f"Patient dirs:   {len(list(H5_OUTPUT_DIR.iterdir()))}")
    log.info(f"Output:         {H5_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
