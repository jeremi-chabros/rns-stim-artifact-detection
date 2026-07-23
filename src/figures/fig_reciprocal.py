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
"""F2: metadata is not the source of performance (two panels).

(a) FiLM-placement ablation (LGS held-out, 5 seeds): event F1 for ``none`` /
    ``last`` / ``every`` -- placement sits within seed noise and every-layer
    degrades, so FiLM is not a performance lever (data: ``ablation_film``).

(b) Raw-detection parity on the external BWH cohort: the deployed FiLM model
    versus the metadata-free model in both transfer directions. Removing the
    therapy metadata leaves raw event F1 essentially unchanged (0.93), and the
    onset F1 gap is small -- the headline comes from width-calibration, not the
    conditioning pathway (data: ``aggregate_reciprocal`` + the published BWH
    eval).

Outputs ``manuscript/figures/fig_reciprocal.{pdf,png}``.

Usage:
    uv run src/figures/fig_reciprocal.py
    uv run src/figures/fig_reciprocal.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from ablation_film import collect as collect_ablation  # noqa: E402
from aggregate_reciprocal import aggregate  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DEPLOYED_CSV = REPO_DIR / "outputs" / "results" / "bwh_unet_eval_full_refined.csv"
OUT_DIR = REPO_DIR / "manuscript" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, PALETTE  # noqa: E402

apply_style()

C_DEPLOYED = "#6a6a6a"  # deployed FiLM (reference)
C_LGS = PALETTE["lgs"]  # metadata-free, LGS-trained -> BWH
C_BWH = PALETTE["bwh"]  # metadata-free, BWH-trained -> LGS
C_NEUTRAL = "#888780"


def deployed_refs() -> dict[str, float]:
    """Mean raw event F1 and onset F1 of the deployed model on full BWH."""
    df = pd.read_csv(DEPLOYED_CSV)
    return {
        "raw_event_f1": float(df["raw_event_f1"].mean()),
        "onset_f1": float(df["onset_f1"].mean()),
    }


def _pick(
    summary: pd.DataFrame, direction: str, prefix: str
) -> tuple[float, float, float]:
    """Pull (mean, ci_lo, ci_hi) for one (direction, metric-prefix) row."""
    row = summary[
        (summary["direction"] == direction) & (summary["metric"].str.startswith(prefix))
    ].iloc[0]
    return float(row["mean"]), float(row["ci_lo"]), float(row["ci_hi"])


def panel_ablation(ax, table: dict) -> None:
    """Dot plot of LGS held-out event F1 for none/last/every with 5-seed CIs.

    Dots (not bars) on a zoomed axis: the differences are sub-1% and a bar
    truncated above zero would exaggerate them (proportional-ink violation).
    """
    placements = ["none", "last", "every"]
    labels = ["none\n(metadata-free)", "last\n(deployed)", "every"]
    colors = [C_LGS, "#b03a3a", C_NEUTRAL]
    means = [table[("lgs", p)]["event_f1"][1] for p in placements]
    los = [table[("lgs", p)]["event_f1"][2] for p in placements]
    his = [table[("lgs", p)]["event_f1"][3] for p in placements]
    x = np.arange(len(placements))
    for xi, m, c, lo, hi in zip(x, means, colors, los, his):
        ax.errorbar(
            xi,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            ms=7,
            color=c,
            capsize=4,
            elinewidth=1.2,
            capthick=1.2,
            zorder=3,
        )
        # value label beside the dot, clear of the vertical whisker
        ax.text(xi + 0.14, m, f"{m:.4f}", ha="left", va="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(placements) - 0.05)
    ax.set_ylim(0.975, 1.001)
    ax.set_ylabel("event F$_1$ (LGS held-out)")
    ax.set_title("(a)  FiLM placement (LGS held-out, 5 seeds)", loc="left", fontsize=9)


def panel_parity(ax, summary: pd.DataFrame, ref: dict[str, float]) -> None:
    """Grouped dot plot: deployed vs metadata-free (both directions) on BWH.

    Dots (not bars) on a zoomed axis — same proportional-ink reasoning as (a).
    """
    groups = [
        ("raw event F$_1$", "raw event F1 (uncalibrated", "raw_event_f1"),
        ("onset F$_1$", "onset F1 (pooled", "onset_f1"),
    ]
    series = [
        ("deployed (FiLM)", C_DEPLOYED),
        ("metadata-free, LGS→BWH", C_LGS),
        ("metadata-free, BWH→LGS", C_BWH),
    ]
    width = 0.22
    gx = np.arange(len(groups))
    for si, (name, color) in enumerate(series):
        offs = (si - 1) * width
        vals, los, his = [], [], []
        for _glabel, prefix, refkey in groups:
            if si == 0:  # deployed reference (single model, no seed CI)
                v = ref[refkey]
                vals.append(v)
                los.append(v)
                his.append(v)
            else:
                direction = "LGS-trained → BWH" if si == 1 else "BWH-trained → LGS"
                m, lo, hi = _pick(summary, direction, prefix)
                vals.append(m)
                los.append(lo)
                his.append(hi)
        err = [
            [v - lo for v, lo in zip(vals, los)],
            [hi - v for v, hi in zip(vals, his)],
        ]
        ax.errorbar(
            gx + offs,
            vals,
            yerr=err,
            fmt="o",
            ms=6,
            color=color,
            label=name,
            capsize=3,
            elinewidth=1.1,
            capthick=1.1,
            zorder=3,
        )
    ax.set_xticks(gx)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("F$_1$ (external BWH)")
    ax.set_title(
        "(b)  External BWH: deployed vs. metadata-free", loc="left", fontsize=9
    )


def build_figure():
    """Assemble the two-panel reciprocal/ablation figure."""
    table = collect_ablation()
    summary, _effects, _pooled = aggregate()
    ref = deployed_refs()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.6), width_ratios=[0.85, 1.15])
    panel_ablation(axa, table)
    panel_parity(axb, summary, ref)
    handles, labels = axb.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def save(fig, name: str = "fig_reciprocal") -> Path:
    """Write the figure to ``manuscript/figures`` as PDF + PNG."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    return png


def main() -> None:
    """Build and save the figure (or run the smoke check)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="data-load check, no render")
    args = ap.parse_args()

    if args.smoke:
        table = collect_ablation()
        ref = deployed_refs()
        assert ("lgs", "none") in table and "raw_event_f1" in ref
        print(
            f"smoke OK: ablation configs={len(table)}, deployed raw F1={ref['raw_event_f1']:.4f}"
        )
        return

    out = save(build_figure())
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
