#!/usr/bin/env -S uv run --no-project --with pandas --with pyarrow --with numpy --with tabulate
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy", "tabulate"]
# ///
"""Device-log-only baseline for RNS stimulation-artifact masking.

Reviewer-requested control: how well does the *device log alone* recover the
artifact, with no model? The device log records only the therapy *onset*
(the ecog_annotations ``duration`` field is 0 for every therapy event in both
cohorts, and no inter-burst-interval is stored), so the therapy duration can
only be derived from the programmed burst parameters: B1+B2. The baseline mask
is the raw logged onset (``onset_times``) plus this B1+B2 duration, scored
against the same refined ground truth with the same matching as ``eval_bwh.py``
(IoU 0.3 event matching, 75-sample onset tolerance, 4 s boundary margin,
250 Hz). This isolates what the model adds *over* the log.

Key result (2026-06-17): at the event level a log-only mask nearly matches the
detector (~0.97 vs 0.99), but at the *sample* level the detector is clearly
better (BWH 0.869 vs 0.702), because the logged duration is a crude per-epoch
constant while the model tracks true extent per event from the waveform alone.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLING_RATE = 250
EVENT_IOU_THRESHOLD = 0.3
ONSET_TOLERANCE_SAMPLES = 75  # 300 ms
BOUNDARY_MARGIN_SAMPLES = int(4.0 * SAMPLING_RATE)  # 1000

# Detector reference numbers (per-file mean) from the U-Net evaluation, for
# side-by-side reporting (see eval_bwh.py / manuscript Table III & external).
DETECTOR_REF = {
    "LGS": {
        "event_f1": 0.994,
        "sample_f1": 0.969,
        "onset_f1": 0.994,
        "raw_event_f1": None,
    },
    "BWH": {
        "event_f1": 0.993,
        "sample_f1": 0.869,
        "onset_f1": 0.993,
        "raw_event_f1": 0.927,
    },
}


def device_duration_ms(t1b1_ms: float, t1b2_ms: float) -> float:
    """Device-derived therapy duration = B1 + B2 programmed burst durations.

    The device log itself records no therapy duration (annotation duration = 0
    for therapy events) and no inter-burst interval, so B1+B2 is the device's
    best duration estimate. Single-burst therapies have B2 NaN/0.
    """
    b1 = 0.0 if pd.isna(t1b1_ms) else float(t1b1_ms)
    b2 = 0.0 if pd.isna(t1b2_ms) else float(t1b2_ms)
    return max(b1 + b2, 1.0)


def build_mask(n_samples: int, onsets_sec: list[float], dur_ms: float) -> np.ndarray:
    """Binary mask: a ``dur_ms`` block starting at each onset (samples)."""
    mask = np.zeros(n_samples, dtype=np.int8)
    width = max(int(dur_ms / 1000.0 * SAMPLING_RATE), 1)
    for t in onsets_sec:
        s = int(t * SAMPLING_RATE)
        if 0 <= s < n_samples:
            mask[s : min(n_samples, s + width)] = 1
    return mask


def extract_events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1 as (start, end) sample pairs."""
    d = np.diff(np.concatenate([[0], (mask > 0).astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def match_iou(
    pred: list[tuple[int, int]],
    true: list[tuple[int, int]],
    threshold: float = EVENT_IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Greedy best-IoU event matching. Returns (TP, FP, FN)."""
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


def match_onset(
    pred: list[tuple[int, int]],
    true: list[tuple[int, int]],
    tol: int = ONSET_TOLERANCE_SAMPLES,
) -> tuple[int, int, int]:
    """Onset-proximity matching (pred onset within +/-tol of a GT onset)."""
    if not pred and not true:
        return (0, 0, 0)
    if not pred:
        return (0, 0, len(true))
    if not true:
        return (0, len(pred), 0)
    tp = 0
    matched: set[int] = set()
    for ps, _pe in pred:
        bj, bd = -1, tol + 1
        for j, (ts, _te) in enumerate(true):
            if j in matched:
                continue
            dd = abs(ps - ts)
            if dd <= tol and dd < bd:
                bd, bj = dd, j
        if bj >= 0:
            tp += 1
            matched.add(bj)
    return (tp, len(pred) - tp, len(true) - len(matched))


def f1_score(tp: int, fp: int, fn: int) -> float:
    """F1 with the eval convention F1=1 for true-negative (empty/empty) files."""
    if tp + fp + fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_cohort(catalog: pd.DataFrame) -> dict[str, float]:
    """Per-file mean event/onset/sample F1 for the device-log-only baseline."""
    ev, on, sa = [], [], []
    margin = BOUNDARY_MARGIN_SAMPLES
    for row in catalog.itertuples():
        n = int(round(row.length_sec * SAMPLING_RATE))
        onsets = (
            ast.literal_eval(row.onset_times)
            if isinstance(row.onset_times, str)
            else list(row.onset_times)
        )
        gt = build_mask(n, onsets, float(row.mask_duration_ms))
        log = build_mask(n, onsets, device_duration_ms(row.t1b1_ms, row.t1b2_ms))
        for arr in (gt, log):
            arr[:margin] = 0
            arr[max(margin, n - margin) :] = 0
        gte, loge = extract_events(gt), extract_events(log)
        ev.append(f1_score(*match_iou(loge, gte)))
        on.append(f1_score(*match_onset(loge, gte)))
        tp = int(((log > 0) & (gt > 0)).sum())
        fp = int(((log > 0) & (gt == 0)).sum())
        fn = int(((log == 0) & (gt > 0)).sum())
        sa.append(f1_score(tp, fp, fn))
    return {
        "n_files": len(catalog),
        "event_f1": float(np.mean(ev)),
        "onset_f1": float(np.mean(on)),
        "sample_f1": float(np.mean(sa)),
    }


def run(catalogs: dict[str, Path], out_md: Path | None = None) -> pd.DataFrame:
    """Score each cohort and emit a comparison table vs. the detector."""
    rows = []
    for name, path in catalogs.items():
        base = score_cohort(pd.read_parquet(path))
        det = DETECTOR_REF.get(name, {})
        rows.append(
            {
                "cohort": name,
                "n_files": base["n_files"],
                "logbaseline_event_f1": round(base["event_f1"], 3),
                "detector_event_f1": det.get("event_f1"),
                "detector_raw_event_f1": det.get("raw_event_f1"),
                "logbaseline_sample_f1": round(base["sample_f1"], 3),
                "detector_sample_f1": det.get("sample_f1"),
                "logbaseline_onset_f1": round(base["onset_f1"], 3),
            }
        )
    df = pd.DataFrame(rows)
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            "# Device-log-only baseline vs. detector (per-file mean F1)\n\n"
            "Baseline = raw logged onset + device-programmed duration, no model, "
            "scored against the refined ground truth with eval_bwh matching.\n\n"
            + df.to_markdown(index=False)
            + "\n"
        )
    return df


def _smoke_test() -> None:
    """Self-check on masks whose onsets sit inside the valid (post-margin) region."""
    n = 10000  # 40 s; onsets at 10 s & 20 s are within [margin, n-margin]
    gt = build_mask(n, [10.0, 20.0], 1000.0)  # two 250-sample events
    gt[:BOUNDARY_MARGIN_SAMPLES] = 0
    gt[max(BOUNDARY_MARGIN_SAMPLES, n - BOUNDARY_MARGIN_SAMPLES) :] = 0
    ge = extract_events(gt)
    assert len(ge) == 2, ge
    assert f1_score(*match_iou(ge, ge)) == 1.0
    assert f1_score(*match_onset(ge, ge)) == 1.0
    # Half-width (IoU 0.5 >= 0.3) -> both TP; quarter-width (IoU 0.25 < 0.3) -> none
    assert match_iou(extract_events(build_mask(n, [10.0, 20.0], 500.0)), ge)[0] == 2
    assert match_iou(extract_events(build_mask(n, [10.0, 20.0], 250.0)), ge)[0] == 0
    # Device duration = B1 + B2 (single-burst -> B2 NaN treated as 0)
    assert device_duration_ms(300, 300) == 600.0
    assert device_duration_ms(160, 160) == 320.0
    assert device_duration_ms(200, float("nan")) == 200.0
    print("smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lgs", type=Path, default=Path("data/stim_catalog.parquet"))
    ap.add_argument("--bwh", type=Path, default=Path("data/bwh_stim_catalog.parquet"))
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/tables/device_log_baseline.md")
    )
    ap.add_argument("--smoke", action="store_true", help="run smoke test only")
    args = ap.parse_args()
    if args.smoke:
        _smoke_test()
    else:
        _smoke_test()
        cats = {}
        if args.lgs.exists():
            cats["LGS"] = args.lgs
        if args.bwh.exists():
            cats["BWH"] = args.bwh
        result = run(cats, out_md=args.out)
        print(result.to_string(index=False))
        print(f"\nWrote {args.out}")
