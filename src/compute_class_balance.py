#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
# ]
# ///
"""Class-balance and trivial-baseline statistics for the artifact-detection task.

Reproduces the numbers in the "not trivially solvable by class imbalance"
paragraph: per-cohort artifact prevalence at the sample and analysis-window
level (ground-truth mask built from logged onset times), the trivial all-positive
sample-F1, and a random-onset chance event-F1 floor (events placed uniformly at
random, matched by IoU>=0.3). Excludes single-recording BWH subjects to match
the 46-subject external cohort used elsewhere.

Usage:
    uv run src/compute_class_balance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SR, WIN, STRIDE, MARGIN = 250, 2048, 1024, int(4.0 * 250)
BWH_CAT = Path("data/bwh_stim_catalog.parquet")
LGS_CAT = Path("data/stim_catalog.parquet")
BWH_EVAL = Path("outputs/results/bwh_unet_eval_full_refined.csv")


def _onsets(x) -> list[float]:
    if isinstance(x, str):
        return [float(t) for t in (json.loads(x) if x.strip() else [])]
    return [float(t) for t in (x if x is not None else [])]


def prevalence(cat: pd.DataFrame) -> dict[str, float]:
    """Sample- and window-level artifact prevalence from the GT mask."""
    cat = cat[cat["length_sec"].notna() & (cat["length_sec"] > 0)]
    tot = art = tot_win = pos_win = nev = 0
    for r in cat.itertuples():
        n = int(round(r.length_sec * SR))
        if n < 1:
            continue
        dur = max(int((r.mask_duration_ms or 0) / 1000 * SR), 1)
        onsets = _onsets(r.onset_times)
        m = np.zeros(n, bool)
        for t in onsets:
            s = int(t * SR)
            m[max(0, s) : min(n, s + dur)] = True
        art += int(m.sum())
        tot += n
        nev += len(onsets)
        nw = max(0, (n - WIN) // STRIDE + 1)
        tot_win += nw
        if onsets and nw > 0:
            st = np.arange(nw) * STRIDE
            wp = np.zeros(nw, bool)
            for t in onsets:
                s = int(t * SR)
                wp |= (st < s + dur) & (st + WIN > s)
            pos_win += int(wp.sum())
    p = art / tot
    return {
        "sample_pct": 100 * p,
        "window_pct": 100 * pos_win / tot_win,
        "events_per_file": nev / len(cat),
        "all_positive_sample_f1": 2 * p / (1 + p),
    }


def _iou_match(pred, gt, thr=0.3) -> tuple[int, int, int]:
    if not pred:
        return (0, 0, len(gt))
    if not gt:
        return (0, len(pred), 0)
    tp, matched = 0, set()
    for ps, pe in pred:
        best, bj = 0.0, -1
        for j, (ts, te) in enumerate(gt):
            if j in matched or ps >= te or pe <= ts:
                continue
            iou = (min(pe, te) - max(ps, ts)) / (max(pe, te) - min(ps, ts) + 1e-8)
            if iou > best:
                best, bj = iou, j
        if best >= thr:
            tp += 1
            matched.add(bj)
    return (tp, len(pred) - tp, len(gt) - len(matched))


def chance_event_f1(cat: pd.DataFrame, reps: int = 5, seed: int = 0) -> float:
    """Macro event-F1 of a predictor placing n_gt events at uniformly random times."""
    cat = cat[cat["length_sec"].notna() & (cat["length_sec"] > 0)]
    if len(cat) > 6000:
        cat = cat.sample(6000, random_state=seed)
    rng = np.random.default_rng(seed)
    macro = []
    for _ in range(reps):
        per = []
        for r in cat.itertuples():
            n = int(round(r.length_sec * SR))
            dur = max(int((r.mask_duration_ms or 0) / 1000 * SR), 1)
            gt = []
            for t in _onsets(r.onset_times):
                s = int(t * SR)
                if s < n - MARGIN and s + dur > MARGIN:
                    gt.append((max(MARGIN, s), min(n - MARGIN, s + dur)))
            lo, hi = MARGIN, max(MARGIN + 1, n - MARGIN - dur)
            starts = rng.integers(lo, hi, size=len(gt)) if (gt and hi > lo) else []
            pred = [(int(s), int(s) + dur) for s in starts]
            tp, fp, fn = _iou_match(pred, gt)
            p = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per.append(2 * p * rec / (p + rec) if p + rec else 0.0)
        macro.append(float(np.mean(per)))
    return float(np.mean(macro))


def main() -> None:
    ev = pd.read_csv(BWH_EVAL)
    keep = set(
        ev.loc[
            ev.groupby("subject")["filename"].transform("size") >= 2, "subject"
        ].astype(str)
    )
    bwh = pd.read_parquet(BWH_CAT)
    bwh = bwh[bwh["subject"].astype(str).isin(keep)]
    lgs = pd.read_parquet(LGS_CAT)
    for name, cat in [("BWH (46-subj external)", bwh), ("LGS (internal)", lgs)]:
        pv = prevalence(cat)
        ch = chance_event_f1(cat)
        print(f"\n[{name}]  n_files={len(cat)}")
        print(f"  sample +prevalence   = {pv['sample_pct']:.1f}%")
        print(f"  window +prevalence   = {pv['window_pct']:.1f}%")
        print(f"  events / recording   = {pv['events_per_file']:.1f}")
        print(
            f"  all-positive sample F1 = {pv['all_positive_sample_f1']:.3f}  (all-negative = 0)"
        )
        print(f"  random-onset chance event F1 = {ch:.3f}")


if __name__ == "__main__":
    main()
