#!/usr/bin/env python3
"""Tile all 11 final manuscript figures into one contact sheet for review.

Renders each figure's vector PDF to a thumbnail (via ``pdftoppm``) and lays
them out in a grid so the whole set can be eyeballed for consistency (font,
palette, panel-label style, sizing) at a glance.

Usage:
    uv run python src/figures/make_contact_sheet.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/path/to/Research/stimask")
FIGDIR = ROOT / "manuscript" / "figures"
OUT = ROOT / "outputs" / "diagnostics" / "figure_contact_sheet.png"

FIGS = [
    ("fig1_overview", "Fig 1 · overview"),
    ("fig2_architecture", "Fig 2 · architecture"),
    ("fig3_baselines", "Fig 3 · baselines"),
    ("fig4_bwh_hero", "Fig 4 · BWH hero"),
    ("fig5_ablations_xai", "Fig 5 · ablations / XAI"),
    ("fig6_qualitative", "Fig 6 · qualitative spectrum"),
    ("fig7_onset_calibration", "Fig 7 · onset calibration"),
    ("fig8_bwh_forest", "Fig 8 · BWH forest"),
    ("fig9_duration_sensitivity", "Fig 9 · duration sensitivity"),
    ("fig10_cross_cohort_forest", "Fig 10 · cross-cohort forest"),
    ("fig_reciprocal", "Fig R · reciprocal / metadata-free"),
    ("fig_metrics", "Fig M · metrics explainer"),
    ("fig_umap", "Fig U · bottleneck UMAP"),
]


def _pdf_to_png(pdf: Path, out_stem: Path, dpi: int = 80) -> Path:
    """Rasterize a single-page PDF to PNG with poppler's pdftoppm."""
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(out_stem)],
        check=True,
    )
    return Path(str(out_stem) + ".png")


def main() -> None:
    """Build and save the contact sheet."""
    tmp = Path(tempfile.mkdtemp())
    ncol = 3
    nrow = (len(FIGS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.2))
    axes = axes.flatten()
    for ax, (stem, title) in zip(axes, FIGS):
        pdf = FIGDIR / f"{stem}.pdf"
        png = _pdf_to_png(pdf, tmp / stem)
        ax.imshow(mpimg.imread(png))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(FIGS) :]:
        ax.axis("off")
    fig.suptitle(
        "TBME figures — final (publication pass)",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
