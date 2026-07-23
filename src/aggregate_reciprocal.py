#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "scipy>=1.10",
#   "matplotlib>=3.8",
# ]
# ///
"""Aggregate the metadata-free reciprocal cross-cohort evaluation.

Consumes the per-seed reciprocal eval CSVs in ``data/checkpoints/phase2/recip/``
produced by evaluating the ``film=none`` (metadata-free) Phase-2 checkpoints
across cohorts, and produces the headline numbers + the reciprocal forest for
the "metadata-free workhorse" reframe (roadmap S1; validity gate S5).

Two transfer directions:
  * ``LGS-trained -> BWH``  (``lgs_none_seed*_on_bwh.csv``)
  * ``BWH-trained -> LGS``  (``bwh_none_seed*_on_lgs.csv``)

What uses metadata, and what does not, matters for the claim:
  * ``onset_f1`` / ``raw_event_f1`` -- genuinely metadata-free (the model is
    ``film=none``; onset matching and the uncalibrated mask use no therapy log).
  * ``event_f1`` -- *calibrated*: still uses the device-logged duration
    (``mask_duration_ms``) in the deterministic width step, so it is reported
    separately as "with the optional calibration step".

Outputs:
  * ``outputs/tables/reciprocal_summary.csv`` (+ ``.md`` companion)
  * ``outputs/figures/fig_reciprocal_forest.{png,svg}`` (draft for roadmap F2)

Usage:
    uv run src/aggregate_reciprocal.py
    uv run src/aggregate_reciprocal.py --smoke
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from meta_analysis import dersimonian_laird, subject_onset_recall_effects  # noqa: E402

RECIP_DIR = REPO_DIR / "data" / "checkpoints" / "phase2" / "recip"
OUT_TABLES = REPO_DIR / "outputs" / "tables"
OUT_FIGS = REPO_DIR / "outputs" / "figures"

# (label, glob pattern, plot colour)
DIRECTIONS: tuple[tuple[str, str, str], ...] = (
    ("LGS-trained → BWH", "lgs_none_seed*_on_bwh.csv", "#264653"),
    ("BWH-trained → LGS", "bwh_none_seed*_on_lgs.csv", "#e76f51"),
)

_REQUIRED = [
    "subject",
    "event_f1",
    "raw_event_f1",
    "onset_f1",
    "event_tp",
    "event_fp",
    "event_fn",
    "onset_tp",
    "onset_fp",
    "onset_fn",
]


def _pooled_f1(tp: float, fp: float, fn: float) -> float:
    """Micro-averaged F1 from summed counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _seed_of(path: Path) -> int:
    """Extract the integer seed from a recip CSV filename."""
    m = re.search(r"seed(\d+)", path.name)
    if m is None:
        raise ValueError(f"no seed in {path.name}")
    return int(m.group(1))


def load_direction(pattern: str) -> list[tuple[int, pd.DataFrame]]:
    """Load every per-seed CSV for one direction, sorted by seed.

    Args:
        pattern: Glob pattern relative to the recip directory.

    Returns:
        List of ``(seed, dataframe)`` with data-integrity checks enforced.
    """
    out: list[tuple[int, pd.DataFrame]] = []
    for p in sorted(RECIP_DIR.glob(pattern), key=_seed_of):
        df = pd.read_csv(p)
        missing = [c for c in _REQUIRED if c not in df.columns]
        assert not missing, f"{p.name}: missing columns {missing}"
        assert len(df) > 1000, f"{p.name}: only {len(df)} rows (expected ~4000)"
        assert df["onset_f1"].between(0, 1).all(), f"{p.name}: onset_f1 out of [0,1]"
        out.append((_seed_of(p), df))
    return out


def run_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Per-run scalar metrics (per-file means and micro-pooled counts)."""
    return {
        "onset_f1_pooled": _pooled_f1(
            df["onset_tp"].sum(), df["onset_fp"].sum(), df["onset_fn"].sum()
        ),
        "onset_recall_pooled": df["onset_tp"].sum()
        / max(df["onset_tp"].sum() + df["onset_fn"].sum(), 1.0),
        "onset_f1_mean": float(df["onset_f1"].mean()),
        "event_f1_pooled": _pooled_f1(
            df["event_tp"].sum(), df["event_fp"].sum(), df["event_fn"].sum()
        ),
        "event_f1_mean": float(df["event_f1"].mean()),
        "raw_event_f1_mean": float(df["raw_event_f1"].mean()),
    }


@dataclass
class SeedCI:
    """Mean and 95% (t-based) confidence interval across seeds."""

    n: int
    mean: float
    lo: float
    hi: float

    def fmt(self, nd: int = 4) -> str:
        """Format as ``mean [lo, hi]``."""
        return f"{self.mean:.{nd}f} [{self.lo:.{nd}f}, {self.hi:.{nd}f}]"


def seed_ci(values: list[float]) -> SeedCI:
    """95% t-CI of a metric across training seeds.

    With a single seed the interval collapses to the point estimate.
    """
    arr = np.asarray(values, float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return SeedCI(n, mean, mean, mean)
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * sem
    return SeedCI(n, mean, mean - half, mean + half)


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate both directions across seeds + per-subject forest effects.

    Returns:
        ``(summary, effects, pooled_overall)`` where ``summary`` has one row per
        (direction, metric) with the seed CI, ``effects`` holds per-subject
        onset-recall logit effects labelled by direction, and ``pooled_overall``
        is the DerSimonian-Laird pool across all subjects in both directions.
    """
    summary_rows = []
    effect_frames = []
    metrics = [
        ("onset_f1_pooled", "onset F1 (pooled, metadata-free)"),
        ("onset_f1_mean", "onset F1 (per-file mean)"),
        ("raw_event_f1_mean", "raw event F1 (uncalibrated, metadata-free)"),
        ("event_f1_pooled", "event F1 (pooled, +width-calibration)"),
        ("event_f1_mean", "event F1 (per-file mean, +width-calibration)"),
    ]

    for label, pattern, _color in DIRECTIONS:
        runs = load_direction(pattern)
        assert runs, f"no CSVs for {label} ({pattern})"
        seeds = [s for s, _ in runs]
        per_seed = [run_metrics(df) for _, df in runs]
        n_files = int(np.mean([len(df) for _, df in runs]))
        n_subj = int(runs[0][1]["subject"].nunique())

        for key, pretty in metrics:
            ci = seed_ci([m[key] for m in per_seed])
            summary_rows.append(
                {
                    "direction": label,
                    "metric": pretty,
                    "n_seeds": ci.n,
                    "seeds": ",".join(map(str, seeds)),
                    "mean": round(ci.mean, 4),
                    "ci_lo": round(ci.lo, 4),
                    "ci_hi": round(ci.hi, 4),
                    "n_files": n_files,
                    "n_subjects": n_subj,
                }
            )

        # Per-subject onset-recall effects, pooling all seeds (sum tp/fn).
        pooled_df = pd.concat([df for _, df in runs], ignore_index=True)
        eff = subject_onset_recall_effects(pooled_df).assign(cohort=label)
        effect_frames.append(eff)

    effects = pd.concat(effect_frames, ignore_index=True)
    pooled_overall = dersimonian_laird(
        effects["yi"].to_numpy(), effects["vi"].to_numpy()
    )
    return pd.DataFrame(summary_rows), effects, pooled_overall


def reciprocal_forest(
    effects: pd.DataFrame,
    pooled: dict,
    stem: str = "fig_reciprocal_forest",
) -> Path:
    """Per-subject onset-recall forest for the two transfer directions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {label: color for label, _pat, color in DIRECTIONS}
    e = effects.sort_values(["cohort", "yi"]).reset_index(drop=True)
    p = expit(e["yi"].to_numpy())
    lo = expit(e["yi"].to_numpy() - 1.96 * np.sqrt(e["vi"].to_numpy()))
    hi = expit(e["yi"].to_numpy() + 1.96 * np.sqrt(e["vi"].to_numpy()))
    y = np.arange(len(e))

    fig, ax = plt.subplots(figsize=(6.2, min(7.5, max(3.6, 0.16 * len(e)))))
    for label, _pat, color in DIRECTIONS:
        m = (e["cohort"] == label).to_numpy()
        d = dersimonian_laird(e.loc[m, "yi"].to_numpy(), e.loc[m, "vi"].to_numpy())
        ax.hlines(y[m], lo[m], hi[m], color=color, lw=1, alpha=0.55)
        ax.plot(
            p[m],
            y[m],
            "o",
            ms=3.5,
            color=color,
            label=f"{label} (k={d['k']}, pooled {expit(d['mu']):.3f})",
        )
    mu = expit(pooled["mu"])
    ax.axvspan(
        expit(pooled["pi_lo"]),
        expit(pooled["pi_hi"]),
        color="#e9c46a",
        alpha=0.2,
        label=f"overall 95% PI [{expit(pooled['pi_lo']):.3f}, {expit(pooled['pi_hi']):.3f}]",
    )
    ax.axvline(mu, color="#000000", lw=1.4, label=f"overall pooled {mu:.3f}")
    ax.set_xlabel("per-subject onset recall")
    ax.set_ylabel("subject (sorted within direction)")
    ax.set_title("Metadata-free reciprocal generalization (onset recall)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    png = OUT_FIGS / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIGS / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def _print_report(summary: pd.DataFrame, pooled: dict) -> bool:
    """Print the headline table + validity gate; return True if gate passes."""
    print("\n=== Metadata-free reciprocal generalization (film=none) ===")
    for direction in summary["direction"].unique():
        sub = summary[summary["direction"] == direction]
        n_seeds = int(sub["n_seeds"].iloc[0])
        n_files = int(sub["n_files"].iloc[0])
        n_subj = int(sub["n_subjects"].iloc[0])
        print(f"\n{direction}  ({n_seeds} seeds, {n_files} files, {n_subj} subjects)")
        for _, r in sub.iterrows():
            print(
                f"  {r['metric']:<46} {r['mean']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            )

    print(
        f"\nOverall metadata-free onset recall (DL pooled, k={pooled['k']}): "
        f"{expit(pooled['mu']):.4f} "
        f"[{expit(pooled['ci_lo']):.4f}, {expit(pooled['ci_hi']):.4f}], "
        f"I²={pooled['I2']:.0f}%, 95% PI "
        f"[{expit(pooled['pi_lo']):.4f}, {expit(pooled['pi_hi']):.4f}]"
    )

    onset = summary[summary["metric"].str.startswith("onset F1 (pooled")]
    gate = bool((onset["mean"] > 0.99).all())
    worst = onset.loc[onset["mean"].idxmin()]
    print(
        f"\nValidity gate (S5) -- metadata-free onset F1 > 0.99 both directions: "
        f"{'PASS' if gate else 'FAIL'} "
        f"(min = {worst['mean']:.4f}, {worst['direction']})"
    )
    return gate


def write_tables(summary: pd.DataFrame, pooled: dict) -> Path:
    """Write the summary CSV + a Markdown companion."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    csv = OUT_TABLES / "reciprocal_summary.csv"
    summary.to_csv(csv, index=False)
    md = OUT_TABLES / "reciprocal_summary.md"
    lines = [
        "# Metadata-free reciprocal generalization (film=none)",
        "",
        "Per-seed reciprocal eval aggregated to mean ± 95% (t) seed CI. "
        "`onset_f1` / `raw_event_f1` are metadata-free; the calibrated "
        "`event_f1` additionally uses the device-logged duration.",
        "",
        "| Direction | Metric | n_seeds | Mean | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['direction']} | {r['metric']} | {r['n_seeds']} | "
            f"{r['mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |"
        )
    lines += [
        "",
        f"Overall DL-pooled onset recall (k={pooled['k']}): "
        f"{expit(pooled['mu']):.4f} "
        f"[{expit(pooled['ci_lo']):.4f}, {expit(pooled['ci_hi']):.4f}], "
        f"I²={pooled['I2']:.0f}%, 95% PI "
        f"[{expit(pooled['pi_lo']):.4f}, {expit(pooled['pi_hi']):.4f}].",
        "",
    ]
    md.write_text("\n".join(lines))
    return csv


def main() -> None:
    """Aggregate, report, and write tables + forest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="load-only integrity check")
    args = ap.parse_args()

    if args.smoke:
        for _label, pattern, _c in DIRECTIONS:
            runs = load_direction(pattern)
            assert runs, f"no CSVs for {pattern}"
            print(f"smoke OK: {pattern} -> {len(runs)} seed(s), {len(runs[0][1])} rows")
        return

    summary, effects, pooled = aggregate()
    csv = write_tables(summary, pooled)
    fig = reciprocal_forest(effects, pooled)
    gate = _print_report(summary, pooled)
    print(f"\nSaved -> {csv}\nSaved -> {fig}")
    assert gate, "validity gate FAILED: metadata-free onset F1 below 0.99"


if __name__ == "__main__":
    main()
