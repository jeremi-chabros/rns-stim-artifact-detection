#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "scipy>=1.10",
#   "matplotlib>=3.8",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Qualitative behaviour figure: works -> borderline -> fails.

Renders real ECoG traces with the detector's predicted mask versus ground
truth across three regimes, using the *same* model with no retraining:

  Band 1 (routine wins)   pred == ground truth, internal LGS + external BWH.
  Band 2 (benign limit)   over-segmentation: onset exact, raw width over-wide,
                          width-calibration trims it back (the visual twin of
                          the onset-underwrites-calibration argument).
  Band 3 (honest limit)   a genuinely dropped onset on an otherwise-clean file
                          (onset_fn > 0) -- the dominant, rare failure mode.

The detection pipeline (model load, inference, postprocessing, width
calibration, ground-truth mask) is imported verbatim from ``src/eval_bwh.py``
so this figure scores identically to the published evaluation. Exemplars were
selected from the held-out eval CSVs (see the repo's exemplar shortlist);
every panel is a real file, no synthetic data.

Replaces the previous ``fig6_failures``.

Usage:
    uv run src/figures/fig_qualitative_spectrum.py            # render PDF+PNG
    uv run src/figures/fig_qualitative_spectrum.py --smoke    # 1-file integrity check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo paths (CWD-independent) -------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]  # .../stimask/src
REPO_DIR = SRC_DIR.parent  # .../stimask
sys.path.insert(0, str(SRC_DIR))

# Canonical detection pipeline (model + inference + masking). Importing
# eval_bwh also puts the deployed model dir on sys.path and re-exports the
# prepare.py helpers, so this is the single source of truth.
from eval_bwh import (  # noqa: E402
    BOUNDARY_MARGIN_S,
    SAMPLING_RATE,
    build_true_mask,
    calibrate_event_widths,
    extract_conditioning_vector,
    extract_events,
    load_deployed_model,
    postprocess_predictions,
    predict_file,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Catalogs (repo-root copies).
LGS_CATALOG = REPO_DIR / "data" / "stim_catalog.parquet"
BWH_CATALOG = REPO_DIR / "data" / "bwh_stim_catalog.parquet"
OUT_DIR = REPO_DIR / "manuscript" / "figures"

# ---------------------------------------------------------------------------
# House style (mirrors src/figures/make_main_figures.py for visual consistency)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style  # noqa: E402

apply_style()
# waveform panels carry no y-axis, so hide the left spine for this figure:
mpl.rcParams["axes.spines.left"] = False

C_SIGNAL = "#222222"
C_GT = "#b4b4b4"
C_GT_EDGE = "#6a6a6a"
C_PRED_WIN = "#3f7d57"  # green  -- correct detection
C_RAW = "#5a7a9a"  # blue   -- raw (over-wide) prediction
C_CAL = "#b03a3a"  # red    -- calibrated prediction
C_MISS = "#b03a3a"  # red    -- the dropped event
C_BAND = {"win": "#2f6b49", "border": "#b07d1a", "miss": "#a82f2f"}


# ---------------------------------------------------------------------------
# Exemplar specification (verified real files; see exemplar shortlist)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exemplar:
    """One panel: a real catalog file plus its display label.

    Attributes:
        filename: Event-id timestamp (matches the catalog ``filename`` column).
        cohort: ``"lgs"`` (internal) or ``"bwh"`` (external).
        tag: Short label drawn on the panel (e.g. cohort + event count).
    """

    filename: str
    cohort: str
    tag: str


WINS: tuple[Exemplar, ...] = (
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · 17/17 events"),
    Exemplar("REDACTED_ID", "lgs", "LGS int. · 8/8 events"),
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · short artifacts"),
)
BORDERLINE: tuple[Exemplar, ...] = (
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · raw F1 0.75 → 1.00"),
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · raw F1 0.83 → 1.00"),
)
MISSES: tuple[Exemplar, ...] = (
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · caught 2 of 3"),
    Exemplar("REDACTED_ID", "bwh", "BWH ext. · caught 5 of 6"),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PanelData:
    """Everything one panel needs to draw, already windowed to the events."""

    t: np.ndarray  # time axis (s), windowed
    sig: np.ndarray  # chosen-channel signal (uV), windowed
    gt: np.ndarray  # ground-truth mask (bool), windowed
    raw: np.ndarray  # raw predicted mask (bool), windowed
    cal: np.ndarray  # calibrated predicted mask (bool), windowed
    onsets: np.ndarray  # ground-truth onset times (s), within window
    tag: str
    cohort: str


_CATALOG_CACHE: dict[Path, pd.DataFrame] = {}


def _load_catalog(path: Path) -> pd.DataFrame:
    """Load and cache a stim catalog with ``filename`` cast to ``str``."""
    if path not in _CATALOG_CACHE:
        cat = pd.read_parquet(path)
        cat["filename"] = cat["filename"].astype(str)
        _CATALOG_CACHE[path] = cat
    return _CATALOG_CACHE[path]


def fetch_row(cohort: str, filename: str) -> tuple[pd.Series, list[float]]:
    """Fetch a catalog row and its decoded onset-time list.

    Args:
        cohort: ``"lgs"`` or ``"bwh"`` -- selects the catalog.
        filename: Event-id timestamp string.

    Returns:
        The catalog row and the list of onset times in seconds.
    """
    path = LGS_CATALOG if cohort == "lgs" else BWH_CATALOG
    cat = _load_catalog(path)
    hits = cat[cat["filename"] == str(filename)]
    if len(hits) == 0:
        raise KeyError(f"{filename} not in {path.name}")
    row = hits.iloc[0]
    ot = row["onset_times"]
    onsets = json.loads(ot) if isinstance(ot, str) else list(ot)
    return row, [float(x) for x in onsets]


def _pick_channel(uv: np.ndarray, gt: np.ndarray) -> int:
    """Return the channel index with the most signal energy under the GT mask."""
    if gt.any():
        energy = np.array([float(np.std(uv[c, gt])) for c in range(uv.shape[0])])
    else:
        energy = np.std(uv, axis=1)
    return int(np.argmax(energy))


def infer_panel(
    model,
    ex: Exemplar,
    *,
    device: str,
    span_s: float | None = None,
    focus: str = "first",
    pad_s: float = 1.0,
) -> PanelData:
    """Run the full detection pipeline on one exemplar and window it.

    Args:
        model: Loaded ``StimArtifactUNet``.
        ex: The exemplar specification.
        device: Torch device string.
        span_s: Width of the display window in seconds. ``None`` shows the full
            active span; a value zooms to that many seconds.
        focus: ``"first"`` anchors the window on the earliest events;
            ``"miss"`` centres it on the dropped (unmatched) ground-truth event.
        pad_s: Seconds of context before the first event (``focus="first"``).

    Returns:
        A :class:`PanelData` windowed for display, with data-integrity
        assertions enforced.
    """
    from lgs_db import read_dat, to_microvolts

    row, onsets = fetch_row(ex.cohort, ex.filename)
    cond = extract_conditioning_vector(row)
    fp = row["file_path"]

    uv = to_microvolts(read_dat(str(fp))).astype(np.float32)
    proba = predict_file(model, fp, cond, device=device)
    n = len(proba)

    mask_ms = float(row["mask_duration_ms"])
    gt = build_true_mask(n, onsets, mask_ms).astype(bool)
    raw = postprocess_predictions((proba > 0.5).astype(np.float32)).astype(bool)
    cal = calibrate_event_widths(raw.astype(np.float32), mask_ms).astype(bool)

    # Match the published pipeline: zero the 4 s boundary margins.
    margin = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
    for arr in (gt, raw, cal):
        arr[:margin] = False
        arr[max(margin, n - margin) :] = False

    # --- data-integrity assertions -----------------------------------------
    assert uv.shape[0] == 4, f"{ex.filename}: expected 4 channels"
    assert len(gt) == len(raw) == len(cal) == n, "mask/probability length mismatch"
    assert gt.any(), f"{ex.filename}: no ground-truth events after margin trim"

    onset_arr = np.array(onsets, dtype=float)
    in_win = onset_arr[
        (onset_arr * SAMPLING_RATE > margin) & (onset_arr * SAMPLING_RATE < n - margin)
    ]
    sr_max = n / SAMPLING_RATE

    if focus == "miss":
        # Centre on the genuinely dropped ground-truth event.
        gt_ev = extract_events(gt.astype(np.float32))
        miss = next((e for e in gt_ev if not cal[e[0] : e[1]].any()), None)
        assert miss is not None, f"{ex.filename}: expected an unmatched GT event"
        centre = 0.5 * (miss[0] + miss[1]) / SAMPLING_RATE
        width = span_s if span_s is not None else 13.0
        lo = max(0.0, centre - width * 0.45)
        hi = min(sr_max, centre + width * 0.55)
    else:
        first = float(in_win.min()) if in_win.size else margin / SAMPLING_RATE
        lo = max(0.0, first - pad_s)
        if span_s is not None:
            hi = min(sr_max, lo + span_s)
        else:
            last = (
                float(in_win.max()) if in_win.size else sr_max - margin / SAMPLING_RATE
            )
            hi = min(sr_max, last + mask_ms / 1000.0 + pad_s)

    s0, s1 = int(lo * SAMPLING_RATE), int(hi * SAMPLING_RATE)
    ch = _pick_channel(uv, gt)
    t = np.arange(s0, s1) / SAMPLING_RATE
    sig = uv[ch, s0:s1] - float(np.median(uv[ch, s0:s1]))
    onsets_win = in_win[(in_win >= lo) & (in_win <= hi)]

    return PanelData(
        t=t,
        sig=sig,
        gt=gt[s0:s1],
        raw=raw[s0:s1],
        cal=cal[s0:s1],
        onsets=onsets_win,
        tag=ex.tag,
        cohort=ex.cohort,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _bar(ax, t, mask, y0, h, color, alpha, edge=None):
    """Draw a horizontal mask ribbon via ``fill_between``."""
    ax.fill_between(
        t,
        y0,
        y0 + h,
        where=mask,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        step="mid",
    )
    if edge is not None:
        ax.fill_between(
            t,
            y0,
            y0 + h,
            where=mask,
            facecolor="none",
            edgecolor=edge,
            linewidth=0.4,
            step="mid",
        )


def _style_axes(ax, pd_: PanelData, ymax: float, ylo: float) -> None:
    """Common per-panel axis cosmetics."""
    ax.set_xlim(pd_.t[0], pd_.t[-1])
    ax.set_ylim(ylo, ymax * 1.18)
    ax.set_yticks([])
    ax.set_xticks([pd_.t[0], pd_.t[-1]])
    ax.set_xticklabels([f"{pd_.t[0]:.0f}", f"{pd_.t[-1]:.0f} s"])
    ax.tick_params(length=2, pad=1)
    ax.text(
        0.015,
        0.98,
        pd_.tag,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#333333",
    )


def _signal_scale(sig: np.ndarray) -> float:
    """Robust amplitude for y-limits."""
    return float(max(np.quantile(np.abs(sig), 0.995), 1e-3))


def _panel_letter(ax, letter: str) -> None:
    """Draw a bold panel letter at the top-left of an axes."""
    ax.text(
        -0.055,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_win(
    ax, pd_: PanelData, *, letter: str | None = None, show_labels: bool = True
) -> None:
    """Routine win: trace with overlapping GT and predicted masks (they match)."""
    ymax = _signal_scale(pd_.sig)
    ax.plot(pd_.t, pd_.sig, color=C_SIGNAL, linewidth=0.45)
    h = ymax * 0.14
    y_gt = -ymax * 1.02
    y_pr = y_gt - h * 1.55
    _bar(ax, pd_.t, pd_.gt, y_gt, h, C_GT, 0.9, edge=C_GT_EDGE)
    _bar(ax, pd_.t, pd_.cal, y_pr, h, C_PRED_WIN, 0.85)
    if show_labels:
        ax.text(
            pd_.t[0],
            y_gt + h * 0.5,
            "truth ",
            ha="right",
            va="center",
            fontsize=8,
            color=C_GT_EDGE,
        )
        ax.text(
            pd_.t[0],
            y_pr + h * 0.5,
            "pred ",
            ha="right",
            va="center",
            fontsize=8,
            color=C_PRED_WIN,
        )
    if letter:
        _panel_letter(ax, letter)
    _style_axes(ax, pd_, ymax, y_pr - h * 0.8)


def draw_border(
    ax, pd_: PanelData, *, letter: str | None = None, show_labels: bool = True
) -> None:
    """Benign limit: GT, raw (over-wide), and calibrated (trimmed) masks."""
    ymax = _signal_scale(pd_.sig)
    ax.plot(pd_.t, pd_.sig, color=C_SIGNAL, linewidth=0.45)
    h = ymax * 0.13
    y_gt = -ymax * 1.02
    y_raw = y_gt - h * 1.5
    y_cal = y_raw - h * 1.5
    _bar(ax, pd_.t, pd_.gt, y_gt, h, C_GT, 0.9, edge=C_GT_EDGE)
    _bar(ax, pd_.t, pd_.raw, y_raw, h, C_RAW, 0.7)
    _bar(ax, pd_.t, pd_.cal, y_cal, h, C_CAL, 0.8)
    if show_labels:
        for y, lab, col in [
            (y_gt, "truth ", C_GT_EDGE),
            (y_raw, "raw ", C_RAW),
            (y_cal, "calib. ", C_CAL),
        ]:
            ax.text(
                pd_.t[0],
                y + h * 0.5,
                lab,
                ha="right",
                va="center",
                fontsize=8,
                color=col,
            )
    ax.annotate(
        "calibration trims width",
        xy=(0.5, 0.04),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_CAL,
    )
    if letter:
        _panel_letter(ax, letter)
    _style_axes(ax, pd_, ymax, y_cal - h * 0.8)


def draw_miss(
    ax, pd_: PanelData, *, letter: str | None = None, show_labels: bool = True
) -> None:
    """Honest limit: a GT event with no predicted mask beneath it (the miss)."""
    ymax = _signal_scale(pd_.sig)
    ax.plot(pd_.t, pd_.sig, color=C_SIGNAL, linewidth=0.45)
    h = ymax * 0.14
    y_gt = -ymax * 1.02
    y_pr = y_gt - h * 1.55
    _bar(ax, pd_.t, pd_.gt, y_gt, h, C_GT, 0.9, edge=C_GT_EDGE)
    _bar(ax, pd_.t, pd_.cal, y_pr, h, C_PRED_WIN, 0.85)
    if show_labels:
        ax.text(
            pd_.t[0],
            y_gt + h * 0.5,
            "truth ",
            ha="right",
            va="center",
            fontsize=8,
            color=C_GT_EDGE,
        )
        ax.text(
            pd_.t[0],
            y_pr + h * 0.5,
            "pred ",
            ha="right",
            va="center",
            fontsize=8,
            color=C_PRED_WIN,
        )

    # Find a GT event with no overlapping prediction -> the dropped onset.
    gt_ev = extract_events(pd_.gt.astype(np.float32))
    miss = next((e for e in gt_ev if not pd_.cal[e[0] : e[1]].any()), None)
    assert miss is not None, f"{pd_.tag}: expected an unmatched GT event"
    m0 = miss[0] / SAMPLING_RATE + pd_.t[0]
    m1 = miss[1] / SAMPLING_RATE + pd_.t[0]
    mc = 0.5 * (m0 + m1)
    ax.axvspan(m0, m1, color=C_MISS, alpha=0.08, zorder=0)
    ax.annotate(
        "missed",
        xy=(mc, y_gt + h * 0.5),
        xytext=(mc, ymax * 0.72),
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_MISS,
        arrowprops=dict(
            arrowstyle="-|>", color=C_MISS, linewidth=0.8, shrinkA=1, shrinkB=2
        ),
    )
    if letter:
        _panel_letter(ax, letter)
    _style_axes(ax, pd_, ymax, y_pr - h * 0.8)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

_BANDS = (
    (
        "win",
        "Routine wins  ·  prediction matches ground truth",
        WINS,
        draw_win,
        11.0,
        "first",
    ),
    (
        "border",
        "Benign limit  ·  onset exact, width over-wide → calibration trims it",
        BORDERLINE,
        draw_border,
        9.0,
        "first",
    ),
    (
        "miss",
        "Honest limit  ·  the rare dropped onset (onset recall 0.997)",
        MISSES,
        draw_miss,
        13.0,
        "miss",
    ),
)


def build_figure(model, *, device: str):
    """Render the full three-band figure.

    Args:
        model: Loaded detector.
        device: Torch device string.

    Returns:
        The assembled matplotlib ``Figure``.
    """
    fig = plt.figure(figsize=(7.2, 8.0))
    subfigs = fig.subfigures(3, 1, height_ratios=[1.04, 0.74, 0.74], hspace=0.03)

    letters = iter("abcdefg")
    for sub, (key, title, exemplars, draw, span_s, focus) in zip(subfigs, _BANDS):
        axes = np.atleast_1d(sub.subplots(1, 3 if key == "win" else 2))
        for i, (ax, ex) in enumerate(zip(axes, exemplars)):
            draw(
                ax,
                infer_panel(model, ex, device=device, span_s=span_s, focus=focus),
                letter=next(letters),
                show_labels=(i == 0),
            )
        sub.suptitle(
            title, x=0.012, ha="left", fontsize=9.5, color=C_BAND[key], weight="bold"
        )
        sub.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.16, wspace=0.18)

    fig.text(
        0.5,
        0.012,
        "Grey = ground truth · coloured bar = predicted mask · same model, no retraining "
        "(internal LGS + external BWH).",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    return fig


def save(fig, name: str = "fig6_qualitative") -> Path:
    """Write the figure to ``outputs/figures`` as both PNG and PDF."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    return png


def _device() -> str:
    """Pick MPS if available, else CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _smoke() -> None:
    """Single-file integrity check (fast): load model, infer one win, assert."""
    dev = _device()
    model = load_deployed_model(device=dev)
    pd_ = infer_panel(model, WINS[0], device=dev)
    assert pd_.sig.size > 0 and pd_.t.size == pd_.sig.size
    assert pd_.gt.any() and pd_.cal.any()
    print(
        f"smoke OK: {WINS[0].filename} window={pd_.t.size} samples, gt={int(pd_.gt.sum())}"
    )


def main() -> None:
    """Render the figure (or run the smoke check with ``--smoke``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke", action="store_true", help="one-file integrity check, no render"
    )
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (default: auto)")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    dev = args.device or _device()
    model = load_deployed_model(device=dev)
    fig = build_figure(model, device=dev)
    out = save(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
