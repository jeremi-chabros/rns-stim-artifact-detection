#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "scipy>=1.10",
# ]
# ///
"""In-distribution FiLM-placement ablation from the Phase-2 training logs.

The Phase-2 matrix trained the same capacity-controlled architecture with
FiLM applied at no layer (``none``), the final decoder level (``last``), or
every level (``every``), 5 seeds each. Each run's validation set is the
held-out subject split used throughout the paper (4 LGS / 4 BWH subjects), so
the logged ``val_event_f1`` / ``val_sample_f1`` are test-set numbers with the
same event-matching as the main results.

This script parses those logs and reports per-configuration 5-seed mean ± 95%
(t) CI, plus the FiLM effect (placement minus ``none``). It is the controlled,
same-architecture evidence that FiLM is not a performance driver (roadmap S3):
``last - none`` is within seed noise and ``every`` is worse.

Usage:
    uv run src/ablation_film.py
    uv run src/ablation_film.py --smoke
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_DIR / "data" / "checkpoints" / "phase2"
OUT_TABLES = REPO_DIR / "outputs" / "tables"

COHORTS = ("lgs", "bwh")
PLACEMENTS = ("none", "last", "every")
SEEDS = range(1, 6)
_PATTERNS = {
    "event_f1": re.compile(r"val_event_f1:\s*([\d.]+)"),
    "sample_f1": re.compile(r"val_sample_f1:\s*([\d.]+)"),
}


def best_val(cohort: str, placement: str, seed: int) -> dict[str, float] | None:
    """Best (over training) validation metrics for one run, or None if absent.

    Args:
        cohort: ``"lgs"`` or ``"bwh"``.
        placement: FiLM placement (``none``/``last``/``every``).
        seed: Training seed.

    Returns:
        ``{"event_f1", "sample_f1"}`` at the best epoch, or None if the log is
        missing (e.g. the untrained ``bwh/none/seed5`` or ``bwh/every``).
    """
    log = LOG_DIR / f"run_{cohort}_{placement}_{seed}.log"
    if not log.exists():
        return None
    text = log.read_text()
    out = {}
    for key, pat in _PATTERNS.items():
        vals = [float(x) for x in pat.findall(text)]
        if not vals:
            return None
        out[key] = max(vals)
    return out


def _ci(values: list[float]) -> tuple[int, float, float, float]:
    """Count, mean, and 95% t-CI bounds (interval collapses for n < 2)."""
    arr = np.asarray(values, float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return n, mean, mean, mean
    half = float(stats.t.ppf(0.975, n - 1)) * float(arr.std(ddof=1) / np.sqrt(n))
    return n, mean, mean - half, mean + half


def collect() -> dict[tuple[str, str], dict]:
    """Gather per-(cohort, placement) seed metrics and their CIs."""
    table: dict[tuple[str, str], dict] = {}
    for cohort in COHORTS:
        for placement in PLACEMENTS:
            runs = [best_val(cohort, placement, s) for s in SEEDS]
            runs = [r for r in runs if r is not None]
            if not runs:
                continue
            table[(cohort, placement)] = {
                metric: _ci([r[metric] for r in runs])
                for metric in ("event_f1", "sample_f1")
            }
    return table


def write_table(table: dict[tuple[str, str], dict]) -> Path:
    """Write the ablation table to ``outputs/tables`` as CSV."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    csv = OUT_TABLES / "ablation_film_placement.csv"
    lines = ["cohort,placement,n_seeds,metric,mean,ci_lo,ci_hi"]
    for (cohort, placement), metrics in table.items():
        for metric, (n, mean, lo, hi) in metrics.items():
            lines.append(
                f"{cohort},{placement},{n},{metric},{mean:.4f},{lo:.4f},{hi:.4f}"
            )
    csv.write_text("\n".join(lines) + "\n")
    return csv


def report(table: dict[tuple[str, str], dict]) -> None:
    """Print the ablation table and the FiLM effect vs the metadata-free model."""
    for cohort in COHORTS:
        present = [p for p in PLACEMENTS if (cohort, p) in table]
        if not present:
            continue
        print(f"\n=== {cohort.upper()} in-distribution (held-out subjects) ===")
        print(f"{'placement':<10}{'n':<3}{'event F1 (mean [95% CI])':<32}{'sample F1'}")
        for p in present:
            n, me, le, he = table[(cohort, p)]["event_f1"]
            _, ms, ls, hs = table[(cohort, p)]["sample_f1"]
            print(f"{p:<10}{n:<3}{me:.4f} [{le:.4f}, {he:.4f}]      {ms:.4f}")
        if (cohort, "none") in table:
            base_e = table[(cohort, "none")]["event_f1"][1]
            base_s = table[(cohort, "none")]["sample_f1"][1]
            for p in present:
                if p == "none":
                    continue
                de = table[(cohort, p)]["event_f1"][1] - base_e
                ds = table[(cohort, p)]["sample_f1"][1] - base_s
                print(f"  {p} - none: event F1 {de:+.4f} | sample F1 {ds:+.4f}")


def main() -> None:
    """Collect, report, and persist the FiLM-placement ablation."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="parse-only integrity check")
    args = ap.parse_args()

    table = collect()
    assert ("lgs", "none") in table and ("lgs", "last") in table, "LGS logs missing"

    if args.smoke:
        n = table[("lgs", "last")]["event_f1"][0]
        print(f"smoke OK: lgs/last parsed {n} seed(s)")
        return

    report(table)
    print(f"\nSaved -> {write_table(table)}")


if __name__ == "__main__":
    main()
