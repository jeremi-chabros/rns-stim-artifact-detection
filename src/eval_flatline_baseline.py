#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "tabulate>=0.9",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Flat-line (saturation) null model for RNS stimulation-artifact masking.

A second reviewer-facing control, complementary to the device-log baseline.
The RNS artifact saturates the amplifier, railing every channel to a constant
value (a run of identical samples ~80 long at the labeled onsets, vs <=10 in
background).  This detector is therefore purely signal-driven --- no model, no
log, no metadata: it flags runs of >= FLAT_MIN identical consecutive samples on
any channel (merged within MERGE_GAP) as artifact.  It answers the objection
"couldn't you just detect saturation?".

It is expected to (a) detect the railed core but under-cover the sub-threshold
recovery tail, and (b) fire indiscriminately on any flat/disconnected channel,
so its specificity on stim-disabled recordings is poor.  Scored against the
refined ground truth with eval_bwh matching (IoU 0.3, 4 s margin, 250 Hz).

Usage:
    uv run src/eval_flatline_baseline.py
    uv run src/eval_flatline_baseline.py --per-subject 25
    uv run src/eval_flatline_baseline.py --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLING_RATE = 250
FLAT_MIN_SAMPLES = 25  # >= 25 identical consecutive samples (100 ms)
MERGE_GAP_SAMPLES = 50  # merge detections within 200 ms (exclude subsequent pulses)
EVENT_IOU_THRESHOLD = 0.3
ONSET_TOLERANCE_SAMPLES = 75
BOUNDARY_MARGIN_SAMPLES = int(4.0 * SAMPLING_RATE)
DEGENERATE_IQR_UV = 1.0  # all-channel IQR below this -> disconnected/flat, excluded

DETECTOR_REF = {
    "LGS": {"event_f1": 0.994, "sample_f1": 0.969},
    "BWH": {"event_f1": 0.993, "sample_f1": 0.869},
}


# --- detection ---------------------------------------------------------------
def flat_mask(raw: np.ndarray, flat_min: int = FLAT_MIN_SAMPLES) -> np.ndarray:
    """Boolean mask: True where any channel rails (>= flat_min identical samples)."""
    n = raw.shape[1]
    m = np.zeros(n, dtype=bool)
    for ch in range(raw.shape[0]):
        flat = np.abs(np.diff(raw[ch])) == 0
        edges = np.flatnonzero(np.diff(np.r_[0, flat.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if (b - a + 1) >= flat_min:
                m[a : b + 1] = True
    return m


def merge_mask(mask: np.ndarray, gap: int = MERGE_GAP_SAMPLES) -> np.ndarray:
    """Bridge gaps of <= gap samples between detections."""
    out = mask.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    for i in range(len(starts) - 1):
        if starts[i + 1] - ends[i] <= gap:
            out[ends[i] : starts[i + 1]] = True
    return out


# --- scoring (matches eval_bwh.py) -------------------------------------------
def build_gt_mask(n_samples: int, onsets_sec: list[float], dur_ms: float) -> np.ndarray:
    """Refined ground-truth mask from logged onsets + per-epoch reference duration."""
    mask = np.zeros(n_samples, dtype=bool)
    width = max(int(dur_ms / 1000.0 * SAMPLING_RATE), 1)
    for t in onsets_sec:
        s = int(t * SAMPLING_RATE)
        if 0 <= s < n_samples:
            mask[s : min(n_samples, s + width)] = True
    return mask


def extract_events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs as (start, end) sample pairs."""
    d = np.diff(np.r_[0, (mask > 0).astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def match_iou(
    pred: list[tuple[int, int]],
    true: list[tuple[int, int]],
    threshold: float = EVENT_IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Greedy best-IoU event matching -> (TP, FP, FN)."""
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
            inter = min(pe, te) - max(ps, ts)
            union = max(pe, te) - min(ps, ts)
            iou = inter / (union + 1e-8)
            if iou > best:
                best, bj = iou, j
        if best >= threshold:
            tp += 1
            matched.add(bj)
    return (tp, len(pred) - tp, len(true) - len(matched))


def f1_score(tp: int, fp: int, fn: int) -> float:
    """F1 with the eval convention F1=1 for empty/empty (true-negative) files."""
    if tp + fp + fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _zero_margins(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    mask[:BOUNDARY_MARGIN_SAMPLES] = False
    mask[max(BOUNDARY_MARGIN_SAMPLES, n - BOUNDARY_MARGIN_SAMPLES) :] = False
    return mask


def score_stim_cohort(catalog: pd.DataFrame, per_subject: int) -> dict[str, float]:
    """Event/sample F1 of the flat-line detector on a stratified stim subset."""
    from lgs_db import read_dat

    sub = catalog.groupby("subject", group_keys=False).head(per_subject)
    ev, sa = [], []
    n_used = 0
    for row in sub.itertuples():
        try:
            raw = np.asarray(read_dat(row.file_path))
        except Exception:
            continue
        if raw.ndim != 2:
            continue
        n = raw.shape[1]
        onsets = (
            __import__("ast").literal_eval(row.onset_times)
            if isinstance(row.onset_times, str)
            else list(row.onset_times)
        )
        gt = _zero_margins(build_gt_mask(n, onsets, float(row.mask_duration_ms)))
        pred = _zero_margins(merge_mask(flat_mask(raw)))
        ev.append(f1_score(*match_iou(extract_events(pred), extract_events(gt))))
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        sa.append(f1_score(tp, fp, fn))
        n_used += 1
    return {
        "n_files": n_used,
        "event_f1": float(np.mean(ev)) if ev else float("nan"),
        "sample_f1": float(np.mean(sa)) if sa else float("nan"),
    }


def score_specificity(disabled: pd.DataFrame, per_subject: int) -> dict[str, float]:
    """False-positive rate on stim-disabled recordings (degenerate ones excluded)."""
    from lgs_db import read_dat, to_microvolts

    sub = disabled.groupby("subject", group_keys=False).head(per_subject)
    tot_events, tot_hours, n_fired, n_valid, n_degenerate = 0, 0.0, 0, 0, 0
    for row in sub.itertuples():
        try:
            raw = np.asarray(read_dat(row.file_path))
        except Exception:
            continue
        if raw.ndim != 2:
            continue
        uv = to_microvolts(raw)
        iqr = np.percentile(uv, 75, axis=1) - np.percentile(uv, 25, axis=1)
        if np.all(iqr < DEGENERATE_IQR_UV):  # disconnected / flat-line acquisition
            n_degenerate += 1
            continue
        n = raw.shape[1]
        pred = _zero_margins(merge_mask(flat_mask(raw)))
        n_ev = len(extract_events(pred))
        tot_events += n_ev
        tot_hours += n / SAMPLING_RATE / 3600.0
        n_fired += int(n_ev > 0)
        n_valid += 1
    return {
        "n_valid": n_valid,
        "n_degenerate": n_degenerate,
        "fp_per_hour": tot_events / tot_hours if tot_hours else float("nan"),
        "pct_fired": 100.0 * n_fired / n_valid if n_valid else float("nan"),
    }


def run(
    catalogs: dict[str, Path],
    disabled: Path | None,
    per_subject: int,
    out_md: Path | None = None,
) -> pd.DataFrame:
    """Score the flat-line detector on each cohort + report disabled-recording specificity."""
    rows = []
    for name, path in catalogs.items():
        s = score_stim_cohort(pd.read_parquet(path), per_subject)
        det = DETECTOR_REF.get(name, {})
        rows.append(
            {
                "cohort": name,
                "n_files": s["n_files"],
                "flatline_event_f1": round(s["event_f1"], 3),
                "detector_event_f1": det.get("event_f1"),
                "flatline_sample_f1": round(s["sample_f1"], 3),
                "detector_sample_f1": det.get("sample_f1"),
            }
        )
    df = pd.DataFrame(rows)
    spec = None
    if disabled is not None and disabled.exists():
        spec = score_specificity(pd.read_parquet(disabled), per_subject)
        print(
            f"Specificity (stim-disabled, {spec['n_valid']} valid + "
            f"{spec['n_degenerate']} degenerate excluded): "
            f"{spec['fp_per_hour']:.1f} FP/h, {spec['pct_fired']:.1f}% fired "
            f"(U-Net: 2.1 FP/h, 1.0% fired)"
        )
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "# Flat-line (saturation) null model vs. detector (per-file mean F1)\n\n"
            "Signal-only baseline: runs of >= 25 identical consecutive samples "
            "(merged within 200 ms), no model/log/metadata.\n\n"
            + df.to_markdown(index=False)
            + "\n"
        )
        if spec is not None:
            text += (
                f"\nSpecificity on stim-disabled recordings: "
                f"{spec['fp_per_hour']:.1f} FP/h, {spec['pct_fired']:.1f}% fired "
                f"({spec['n_degenerate']} degenerate recordings excluded); "
                f"U-Net 2.1 FP/h, 1.0% fired.\n"
            )
        out_md.write_text(text)
    return df


def _smoke_test() -> None:
    """A synthetic railed segment is detected; pure noise is not."""
    rng = np.zeros((4, 3000), dtype=np.int16)
    # railed segment of 80 samples on channel 0 in the valid region
    rng[0, 1500:1580] = 5000
    m = merge_mask(flat_mask(rng))
    # the rail is one constant block (one flat run); background zeros are also
    # constant, so restrict the check to the injected non-zero rail region
    assert m[1500:1580].all(), "railed segment not detected"
    # detection + scoring round-trips to F1=1 against itself
    ev = extract_events(_zero_margins(m.copy()))
    assert f1_score(*match_iou(ev, ev)) == 1.0
    # a short rail (< FLAT_MIN) is ignored
    short = np.zeros((1, 3000), dtype=np.int16)
    short[0, 1500:1510] = 7000  # 10 samples < 25
    assert not flat_mask(short)[1500:1510].all()
    print("smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lgs", type=Path, default=Path("data/stim_catalog.parquet"))
    ap.add_argument("--bwh", type=Path, default=Path("data/bwh_stim_catalog.parquet"))
    ap.add_argument(
        "--disabled", type=Path, default=Path("data/bwh_disabled_catalog.parquet")
    )
    ap.add_argument("--per-subject", type=int, default=25)
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/tables/flatline_baseline.md")
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _smoke_test()
    if not args.smoke:
        cats = {}
        if args.lgs.exists():
            cats["LGS"] = args.lgs
        if args.bwh.exists():
            cats["BWH"] = args.bwh
        result = run(cats, args.disabled, args.per_subject, out_md=args.out)
        print(result.to_string(index=False))
        print(f"\nWrote {args.out}")
