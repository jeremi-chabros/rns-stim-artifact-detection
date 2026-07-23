#!/usr/bin/env python3
"""Conceptual figure explaining the evaluation metrics.

Illustrative (synthetic) schematic that *teaches* how the three F1 variants are
computed — it is not a result, so synthetic toy data is correct here:

  (a) Sample-level F1   — per-sample TP/FP/FN over the mask timeline.
  (b) Event-level F1    — events matched by intersection-over-union (IoU >= 0.3).
  (c) Onset F1          — onset matched if within a timing tolerance.

Renders ``manuscript/figures/fig_metrics.{pdf,png}``.

Usage:
    uv run python src/figures/fig_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style  # noqa: E402

apply_style()

ROOT = Path("/path/to/Research/stimask")
OUT = ROOT / "manuscript" / "figures"

C_GT = "#6a6a6a"  # ground truth
C_PRED = "#5a7a9a"  # prediction
C_TP = "#2a9d8f"  # true positive  (teal)
C_FP = "#e76f51"  # false positive (orange)
C_FN = "#8a6bbf"  # false negative (purple)


def _cellrow(ax, y, members, total, color, h=0.74):
    """Draw a row of ``total`` unit cells; fill those in ``members``."""
    for i in range(total):
        ax.add_patch(
            Rectangle(
                (i, y),
                0.92,
                h,
                facecolor=color if i in members else "white",
                edgecolor="#c4c4c4",
                lw=0.5,
            )
        )


def panel_sample(ax) -> None:
    """(a) Sample level: per-sample TP / FP / FN -> precision, recall, F1."""
    n = 10
    gt = set(range(2, 8))  # samples 2..7
    pred = set(range(3, 9))  # samples 3..8
    tp = gt & pred  # 3..7  -> 5
    fp = pred - gt  # 8     -> 1
    fn = gt - pred  # 2     -> 1

    ax.set_xlim(-2.6, 15.2)
    ax.set_ylim(0.2, 4.3)
    ax.axis("off")

    _cellrow(ax, 3.2, gt, n, C_GT)
    _cellrow(ax, 2.2, pred, n, C_PRED)
    # classification row
    for i in range(n):
        c = C_TP if i in tp else C_FP if i in fp else C_FN if i in fn else "white"
        ax.add_patch(
            Rectangle((i, 1.0), 0.92, 0.74, facecolor=c, edgecolor="#c4c4c4", lw=0.5)
        )

    for y, lab in [(3.57, "ground truth"), (2.57, "prediction"), (1.37, "per-sample")]:
        ax.text(-0.3, y, lab, ha="right", va="center", fontsize=8)

    # legend chips for TP/FP/FN
    for i, (c, lab) in enumerate([(C_TP, "TP"), (C_FP, "FP"), (C_FN, "FN")]):
        ax.add_patch(Rectangle((0.2 + i * 1.5, 0.32), 0.3, 0.3, facecolor=c))
        ax.text(0.6 + i * 1.5, 0.47, lab, va="center", fontsize=7.5)

    # formulas on the right
    fx = 10.6
    ax.text(
        fx,
        3.55,
        r"$\mathrm{P}=\dfrac{TP}{TP+FP}=\dfrac{5}{6}$",
        va="center",
        fontsize=9,
    )
    ax.text(
        fx,
        2.55,
        r"$\mathrm{R}=\dfrac{TP}{TP+FN}=\dfrac{5}{6}$",
        va="center",
        fontsize=9,
    )
    ax.text(
        fx,
        1.4,
        r"$\mathrm{F_1}=\dfrac{2PR}{P+R}=0.83$",
        va="center",
        fontsize=9,
    )
    ax.text(-2.5, 4.12, "(a)  Sample level", fontsize=10, fontweight="bold")


def panel_event(ax) -> None:
    """(b) Event level: events matched by IoU >= 0.3."""
    ax.set_xlim(-0.7, 15.2)
    ax.set_ylim(-0.2, 4.3)
    ax.axis("off")

    # --- matched example (left) ---
    gt0, gt1 = 1.0, 6.0
    pr0, pr1 = 2.2, 7.2
    inter0, inter1 = max(gt0, pr0), min(gt1, pr1)
    uni0, uni1 = min(gt0, pr0), max(gt1, pr1)
    yb = 2.4
    ax.add_patch(
        Rectangle(
            (gt0, yb + 0.55), gt1 - gt0, 0.5, facecolor="none", edgecolor=C_GT, lw=1.4
        )
    )
    ax.add_patch(
        Rectangle(
            (pr0, yb), pr1 - pr0, 0.5, facecolor=C_PRED, alpha=0.55, edgecolor=C_PRED
        )
    )
    # intersection band
    ax.add_patch(
        Rectangle(
            (inter0, yb),
            inter1 - inter0,
            1.05,
            facecolor=C_TP,
            alpha=0.30,
            edgecolor="none",
        )
    )
    ax.text(7.4, yb + 0.8, "GT event", va="center", fontsize=8, color=C_GT)
    ax.text(7.4, yb + 0.25, "pred event", va="center", fontsize=8, color=C_PRED)
    ax.text(
        (inter0 + inter1) / 2,
        yb + 1.25,
        "intersection",
        ha="center",
        fontsize=7.5,
        color="#1f6f63",
    )
    # union bracket
    ax.plot([uni0, uni1], [yb - 0.28, yb - 0.28], color="#888", lw=0.9)
    ax.plot([uni0, uni0], [yb - 0.18, yb - 0.38], color="#888", lw=0.9)
    ax.plot([uni1, uni1], [yb - 0.18, yb - 0.38], color="#888", lw=0.9)
    ax.text(
        (uni0 + uni1) / 2, yb - 0.62, "union", ha="center", fontsize=7.5, color="#666"
    )
    ax.text(
        10.0,
        yb + 0.78,
        r"$\mathrm{IoU}=\dfrac{|\cap|}{|\cup|}=\dfrac{3.8}{6.2}=0.61$",
        va="center",
        fontsize=9,
    )
    ax.text(
        10.0,
        yb - 0.05,
        r"$0.61 \geq 0.3 \Rightarrow$ matched (TP)",
        va="center",
        fontsize=8.5,
        color="#1f6f63",
    )

    # --- miss example (lower) ---
    ym = 0.5
    ax.add_patch(
        Rectangle((1.0, ym + 0.5), 1.6, 0.45, facecolor="none", edgecolor=C_GT, lw=1.4)
    )
    ax.add_patch(
        Rectangle((3.4, ym), 1.5, 0.45, facecolor=C_PRED, alpha=0.55, edgecolor=C_PRED)
    )
    ax.text(
        10.0,
        ym + 0.22,
        r"$\mathrm{IoU}=0 < 0.3 \Rightarrow$ miss (FN $+$ FP)",
        va="center",
        fontsize=8.5,
        color=C_FP,
    )
    ax.text(
        -0.6, 4.12, "(b)  Event level (IoU matching)", fontsize=10, fontweight="bold"
    )


def panel_onset(ax) -> None:
    """(c) Onset timing: onset matched if |Delta t| <= tolerance."""
    ax.set_xlim(-0.7, 15.2)
    ax.set_ylim(-0.4, 3.0)
    ax.axis("off")

    base = 1.0
    gt_on = 5.0
    pr_on = 6.1
    # ground-truth and predicted event extents (as light bands)
    ax.add_patch(
        Rectangle(
            (gt_on, base + 0.55), 2.3, 0.42, facecolor="none", edgecolor=C_GT, lw=1.4
        )
    )
    ax.add_patch(
        Rectangle(
            (pr_on, base), 2.3, 0.42, facecolor=C_PRED, alpha=0.5, edgecolor=C_PRED
        )
    )
    # onset markers
    ax.plot([gt_on, gt_on], [base - 0.2, base + 1.45], color=C_GT, lw=1.6)
    ax.plot([pr_on, pr_on], [base - 0.2, base + 1.45], color=C_PRED, lw=1.6, ls="--")
    ax.text(gt_on - 0.12, base + 1.58, "GT onset", ha="right", fontsize=8, color=C_GT)
    ax.text(
        pr_on + 0.12, base + 1.58, "pred onset", ha="left", fontsize=8, color=C_PRED
    )
    # delta-t arrow
    ax.add_patch(
        FancyArrowPatch(
            (gt_on, base - 0.35),
            (pr_on, base - 0.35),
            arrowstyle="<->",
            mutation_scale=10,
            color="#333",
            lw=1.0,
        )
    )
    ax.text((gt_on + pr_on) / 2, base - 0.62, r"$\Delta t$", ha="center", fontsize=9)
    ax.text(
        9.2,
        base + 0.55,
        r"onset matched if $|\Delta t| \leq$ tolerance",
        va="center",
        fontsize=9,
    )
    ax.text(
        9.2,
        base - 0.15,
        r"onset recall $=\dfrac{\#\,\mathrm{matched\ onsets}}{\#\,\mathrm{GT\ onsets}}$",
        va="center",
        fontsize=9,
    )
    ax.text(-0.6, 2.78, "(c)  Onset timing", fontsize=10, fontweight="bold")


def main() -> None:
    """Render the 3-panel metrics-explainer figure."""
    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 6.6), gridspec_kw={"height_ratios": [1.15, 1.1, 0.85]}
    )
    panel_sample(axes[0])
    panel_event(axes[1])
    panel_onset(axes[2])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03, hspace=0.32)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_metrics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_metrics.pdf'}")


if __name__ == "__main__":
    main()
    assert (OUT / "fig_metrics.pdf").exists(), "fig_metrics.pdf not written"
