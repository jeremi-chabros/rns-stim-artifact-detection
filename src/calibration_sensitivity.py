"""Sensitivity of therapy-informed width calibration to device-log error.

A2: recompute calibrated event F1 as the calibration width is perturbed away
from the true therapy duration. A3: on the manually-refined subset, compare
calibrating to the raw device-logged duration vs the clinician-refined truth.
Operates on the cache from src/cache_bwh_events.py — no model inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import matplotlib  # headless backend before pyplot is imported (incl. via style)

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from style import apply_style  # noqa: E402

apply_style()

SR = 250
IOU_THRESHOLD = 0.3
BOUNDARY_MARGIN_S = 4.0
DEFAULT_CACHE = Path("outputs/results/bwh_pred_events.parquet")
FIG_DIR_MANUSCRIPT = Path("manuscript/figures")
FIG_DIR_OUTPUTS = Path("outputs/figures")


def _events_from_onsets(
    onsets: list[int], width_samp: int, n: int
) -> list[tuple[int, int]]:
    """Build (start, end) events of fixed width from onset sample indices."""
    return [(s, min(n, s + width_samp)) for s in onsets]


def _true_events(
    onset_times_s: list[float], dur_ms: float, n: int
) -> list[tuple[int, int]]:
    w = max(int(dur_ms / 1000.0 * SR), 1)
    return [(int(t * SR), min(n, int(t * SR) + w)) for t in onset_times_s]


def _zero_margin(events: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    m = int(BOUNDARY_MARGIN_S * SR)
    lo, hi = m, max(m, n - m)
    return [(s, e) for (s, e) in events if s >= lo and e <= hi]


def _match_iou(
    pred: list[tuple[int, int]], true: list[tuple[int, int]], thr: float = IOU_THRESHOLD
) -> tuple[int, int, int]:
    if not pred and not true:
        return (0, 0, 0)
    if not pred:
        return (0, 0, len(true))
    if not true:
        return (0, len(pred), 0)
    tp = 0
    matched: set[int] = set()
    for ps, pe in pred:
        best, bj = 0.0, -1
        for j, (ts, te) in enumerate(true):
            if j in matched or ps >= te or pe <= ts:
                continue
            inter = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou = inter / (union + 1e-8)
            if iou > best:
                best, bj = iou, j
        if best >= thr:
            tp += 1
            matched.add(bj)
    return (tp, len(pred) - tp, len(true) - len(matched))


def calibrated_event_f1_at_scale(cache: pd.DataFrame, delta: float) -> float:
    """Pooled calibrated event F1 across files when calibration width is
    set to mask_duration_ms*(1+delta) (truth stays at mask_duration_ms)."""
    tp = fp = fn = 0
    for _, r in cache.iterrows():
        n = int(r["n_samples"])
        onsets = json.loads(r["pred_onsets"])
        true_t = json.loads(r["onset_times"])
        dur = float(r["mask_duration_ms"])
        w = max(int(dur * (1.0 + delta) / 1000.0 * SR), 1)
        pred = _zero_margin(_events_from_onsets(onsets, w, n), n)
        true = _zero_margin(_true_events(true_t, dur, n), n)
        a, b, c = _match_iou(pred, true)
        tp, fp, fn = tp + a, fp + b, fn + c
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 1.0


def duration_sensitivity_curve(
    cache: pd.DataFrame,
    deltas: tuple[float, ...] = (
        -0.5,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.5,
        1.0,
        2.0,
    ),
) -> pd.DataFrame:
    """Pooled calibrated event F1 at each width-perturbation delta."""
    return pd.DataFrame(
        {
            "delta": list(deltas),
            "event_f1": [calibrated_event_f1_at_scale(cache, d) for d in deltas],
        }
    )


def plot_sensitivity(
    curve: pd.DataFrame, stem: str = "fig9_duration_sensitivity"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(curve["delta"]) * 100, curve["event_f1"], "o-", color="#264653")
    ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("device-logged duration error (%)")
    ax.set_ylabel("calibrated event F$_1$ (pooled)")
    ax.set_title("Sensitivity of width calibration to duration error")
    fig.tight_layout()
    FIG_DIR_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    FIG_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR_MANUSCRIPT / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR_OUTPUTS / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def refined_subset_summary(cache: pd.DataFrame) -> dict:
    """Quantify clinician refinement on the manually-refined subset.

    The catalog's ``manually_refined`` flag is set globally, so the
    discriminating signal for an actual clinician correction is a nonzero
    ``mask_onset_offset_ms``. Reports how far refined labels moved from the
    device log (onset-offset magnitude), bounding the device-log timing error
    the width calibration inherits.
    """
    off_all = pd.to_numeric(
        cache.get("mask_onset_offset_ms", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    offs = off_all[off_all != 0].abs()
    return {
        "n_refined": int((off_all != 0).sum()),
        "median_abs_onset_offset_ms": float(offs.median()) if len(offs) else 0.0,
        "p90_abs_onset_offset_ms": float(offs.quantile(0.9)) if len(offs) else 0.0,
        "max_abs_onset_offset_ms": float(offs.max()) if len(offs) else 0.0,
    }


def smoke_test() -> None:
    if not DEFAULT_CACHE.exists():
        print(f"[smoke] {DEFAULT_CACHE} not found; run src/cache_bwh_events.py first.")
        return
    cache = pd.read_parquet(DEFAULT_CACHE)
    cache = cache[cache.get("error").isna()] if "error" in cache.columns else cache
    curve = duration_sensitivity_curve(cache)
    print("[smoke] duration sensitivity:\n", curve.to_string(index=False))
    print(
        f"[smoke] F1 at delta=0 (should ~match published calibrated F1): "
        f"{calibrated_event_f1_at_scale(cache, 0.0):.4f}"
    )
    print("[smoke] figure:", plot_sensitivity(curve))


if __name__ == "__main__":
    smoke_test()
