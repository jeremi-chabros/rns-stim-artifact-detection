#!/usr/bin/env python3
"""Multi-panel UMAP of the trained model's bottleneck embeddings.

One 2-D embedding of the **held-out (test) patients'** bottleneck activations,
colored four ways: two therapy parameters, lead type, and patient identity.
The patient panel (d) is a confound check — if the latent separated by patient
rather than by stimulation parameters, the "organizes by stim" reading would be
identity structure instead of the interesting signal.

Reuses ``outputs/results/cluster_labels.csv`` (embedding already computed by
``src/stim_explainability.py``); no model re-run. Renders
``manuscript/figures/fig_umap.{pdf,png}``.

Usage:
    uv run python src/figures/fig_umap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style  # noqa: E402

apply_style()

ROOT = Path("/path/to/Research/stimask")
OUT = ROOT / "manuscript" / "figures"
SRC = ROOT / "outputs" / "results" / "cluster_labels.csv"

# qualitative, colorblind-aware palette for categorical panels
QUAL = ["#4c6c8a", "#b77938", "#3b7d5a", "#b03a3a", "#7c5aa0", "#c2a33a", "#5a8fa0"]


def _scatter_cont(ax, df, col, label, cmap):
    """Scatter the shared embedding, colored by a continuous variable."""
    sc = ax.scatter(
        df["umap_1"],
        df["umap_2"],
        c=df[col],
        s=5,
        cmap=cmap,
        alpha=0.7,
        edgecolors="none",
    )
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _scatter_cat(ax, df, col, title):
    """Scatter the shared embedding, colored by a categorical variable."""
    cats = sorted(df[col].astype(str).unique())
    for i, cat in enumerate(cats):
        m = df[col].astype(str) == cat
        ax.scatter(
            df.loc[m, "umap_1"],
            df.loc[m, "umap_2"],
            s=5,
            color=QUAL[i % len(QUAL)],
            alpha=0.7,
            edgecolors="none",
            label=cat,
        )
    ax.legend(
        fontsize=6.5,
        markerscale=2.2,
        loc="best",
        frameon=True,
        framealpha=0.92,
        handletextpad=0.2,
        borderpad=0.3,
        labelspacing=0.25,
    )


def main() -> None:
    """Render the 2x2 multi-panel UMAP."""
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))

    _scatter_cont(axes[0, 0], df, "B1_current", "B1 current (mA)", "viridis")
    axes[0, 0].set_title("(a) B1 stim current", fontsize=9, loc="left")

    _scatter_cont(axes[0, 1], df, "B1_frequency", "B1 frequency (norm.)", "plasma")
    axes[0, 1].set_title("(b) B1 frequency", fontsize=9, loc="left")

    _scatter_cont(
        axes[1, 0], df, "mask_duration_ms", "therapy duration (ms)", "cividis"
    )
    axes[1, 0].set_title("(c) therapy duration", fontsize=9, loc="left")

    _scatter_cat(axes[1, 1], df, "subject", "(d) patient")
    axes[1, 1].set_title("(d) patient identity (held-out)", fontsize=9, loc="left")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)

    fig.suptitle(
        f"Bottleneck UMAP — held-out patients "
        f"(n={len(df)} files, {df['subject'].nunique()} patients)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_umap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_umap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_umap.pdf'}")


if __name__ == "__main__":
    main()
    assert (OUT / "fig_umap.pdf").exists(), "fig_umap.pdf not written"
