#!/usr/bin/env python3
"""Shared publication style for all TBME manuscript figures.

Call :func:`apply_style` at the top of every figure generator so the whole
figure set shares one font family, palette, and sizing. Helpers:
:func:`panel_label` for ``(a)``/``(b)``/``(c)`` tags and :func:`forest_xzoom`
for the forest plots.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Canonical palette — IEEE print-safe, colorblind-aware, grayscale-separable.
# Semantic mapping is STABLE across every figure (do not remap per figure).
PALETTE: dict[str, str] = {
    "m0": "#7c8c9a",
    "m1": "#a0724a",
    "m2": "#4d7d5a",
    "m3": "#b89e3f",
    "m4": "#b03a3a",  # baseline models M0..M4
    "lgs": "#264653",
    "bwh": "#e76f51",  # cohorts (colorblind-safe teal/orange; fig4/8/10/recip)
    "raw": "#5a7a9a",
    "cal": "#b03a3a",
    "gt": "#4a4a4a",
    "signal": "#222222",
    "accent": "#b03a3a",  # signal roles
}
PAL = PALETTE  # backwards-compat alias for existing scripts


def apply_style() -> None:
    """Set global rcParams for consistent, publication-quality figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def panel_label(
    ax: plt.Axes,
    letter: str,
    dx: float = -0.12,
    dy: float = 1.02,
    fontsize: int = 11,
) -> None:
    """Bold ``(a)``/``(b)``/``(c)`` panel tag in axes-fraction coords."""
    ax.text(
        dx,
        dy,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def forest_xzoom(
    ax: plt.Axes,
    lo: float = 0.85,
    hi: float = 1.005,
    clip_annots: list[tuple[float, str]] | None = None,
) -> None:
    """Zoom a forest-plot x-axis to ``[lo, hi]`` so data fills the panel.

    CIs extending below ``lo`` are clipped by the limit; pass ``(y, text)``
    pairs in ``clip_annots`` to mark each with a left-pointing arrow + label at
    the axis edge. Do not use a broken axis on the bounded [0, 1] metric.
    """
    ax.set_xlim(lo, hi)
    if clip_annots:
        for y, text in clip_annots:
            ax.annotate(
                text,
                xy=(lo, y),
                xytext=(lo + 0.02, y),
                fontsize=7,
                va="center",
                ha="left",
                color="#555555",
                arrowprops=dict(arrowstyle="<-", lw=0.7, color="#888888"),
            )


if __name__ == "__main__":  # smoke test
    apply_style()
    assert PALETTE["m4"] == "#b03a3a"
    fig, ax = plt.subplots()
    forest_xzoom(ax, 0.85, 1.0, clip_annots=[(0.5, "CI → 0.41")])
    panel_label(ax, "a")
    assert ax.get_xlim() == (0.85, 1.0)
    print("style.py smoke test OK")
