"""Validity decomposition for therapy-informed width calibration.

Answers: is the calibrated event F1 underwritten by genuine (metadata-free)
onset localization, or fabricated by the duration patch? Operates on the
per-file BWH evaluation CSV produced by ``src/eval_bwh.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
import matplotlib  # headless backend before pyplot is imported (incl. via style)

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from style import apply_style  # noqa: E402

apply_style()

DEFAULT_EVAL = Path("outputs/results/bwh_unet_eval_full_refined.csv")
FIG_DIR_MANUSCRIPT = Path("manuscript/figures")
FIG_DIR_OUTPUTS = Path("outputs/figures")


def load_eval(path: Path = DEFAULT_EVAL) -> pd.DataFrame:
    """Load the per-file BWH evaluation CSV.

    Applies the external-validation inclusion criterion (drop single-recording
    subjects), matching ``meta_analysis.apply_inclusion_criterion``, so the
    decomposition is computed on the same 46-subject cohort as the rest of the
    external-validation results.
    """
    df = pd.read_csv(path)
    return df[df.groupby("subject")["filename"].transform("size") >= 2].copy()


def decompose_raw_failures(
    df: pd.DataFrame, raw_thresh: float = 0.3, ok_thresh: float = 0.9
) -> dict[str, float | int]:
    """Classify files where the raw detector fails the IoU/event metric.

    A 'raw failure' is a file with ``raw_event_f1 < raw_thresh``. Among those:
    legit width-rescue = calibrated and onset both > ``ok_thresh`` (correct
    localization, only width was wrong); fabricated = calibrated > ``ok_thresh``
    but onset <= ``ok_thresh`` (score conjured without localization); residual
    fail = neither rescued.
    """
    raw_fail = df["raw_event_f1"] < raw_thresh
    calib_ok = df["event_f1"] > ok_thresh
    onset_ok = df["onset_f1"] > ok_thresh
    n = int(raw_fail.sum())
    n_legit = int((raw_fail & calib_ok & onset_ok).sum())
    n_fab = int((raw_fail & calib_ok & ~onset_ok).sum())
    n_resid = int((raw_fail & ~calib_ok).sum())
    return {
        "n_files": int(len(df)),
        "n_raw_fail": n,
        "n_legit": n_legit,
        "n_fabricated": n_fab,
        "n_residual_fail": n_resid,
        "frac_legit": n_legit / n if n else 0.0,
        "frac_fabricated": n_fab / n if n else 0.0,
        "frac_residual_fail": n_resid / n if n else 0.0,
        "pct_cohort_raw_fail": 100.0 * n / max(len(df), 1),
        "pct_cohort_residual": 100.0 * n_resid / max(len(df), 1),
    }


def localization_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarise that the calibrated event F1 tracks onset localization."""
    return {
        "mean_onset_f1": float(df["onset_f1"].mean()),
        "mean_event_f1": float(df["event_f1"].mean()),
        "mean_raw_event_f1": float(df["raw_event_f1"].mean()),
        "corr_onset_calib": float(df["onset_f1"].corr(df["event_f1"])),
        "corr_onset_raw": float(df["onset_f1"].corr(df["raw_event_f1"])),
    }


def plot_decomposition(
    df: pd.DataFrame,
    raw_thresh: float = 0.3,
    ok_thresh: float = 0.9,
    stem: str = "fig7_onset_calibration",
) -> Path:
    """Two-panel figure: (a) onset vs raw/calibrated event F1; (b) raw-fail breakdown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = decompose_raw_failures(df, raw_thresh, ok_thresh)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax[0].scatter(df["onset_f1"], df["raw_event_f1"], s=3, alpha=0.15, label="raw")
    ax[0].scatter(df["onset_f1"], df["event_f1"], s=3, alpha=0.15, label="calibrated")
    ax[0].set_xlabel("onset F$_1$ (metadata-free localization)")
    ax[0].set_ylabel("event F$_1$")
    ax[0].set_title("(a) calibrated event F$_1$ tracks onset")
    ax[0].set_xlim(-0.02, 1.02)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(markerscale=4, framealpha=0.9, loc="upper left")

    cats = ["legit\nwidth-rescue", "fabricated", "residual\nfail"]
    vals = [d["n_legit"], d["n_fabricated"], d["n_residual_fail"]]
    ax[1].bar(cats, vals, color=["#2a9d8f", "#e76f51", "#888888"])
    ax[1].set_ylabel(f"files with raw event F$_1$ < {raw_thresh}")
    ax[1].set_title(f"(b) of {d['n_raw_fail']} raw failures")
    for i, v in enumerate(vals):
        ax[1].text(i, v, str(v), ha="center", va="bottom")

    fig.tight_layout()
    FIG_DIR_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    FIG_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR_MANUSCRIPT / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR_MANUSCRIPT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR_OUTPUTS / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def smoke_test() -> None:
    """Run the decomposition on the real CSV if present and print the verdict."""
    if not DEFAULT_EVAL.exists():
        print(f"[smoke] {DEFAULT_EVAL} not found; skipping real-data smoke test.")
        return
    df = load_eval()
    d = decompose_raw_failures(df)
    s = localization_summary(df)
    assert d["n_fabricated"] == 0, "fabrication detected — calibration may be circular"
    print("[smoke] decomposition:", d)
    print("[smoke] localization:", s)
    print("[smoke] figure:", plot_decomposition(df))


if __name__ == "__main__":
    smoke_test()
