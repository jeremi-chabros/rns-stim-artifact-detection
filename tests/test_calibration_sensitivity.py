"""Tests for src/calibration_sensitivity.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.calibration_sensitivity import (
    calibrated_event_f1_at_scale,
    duration_sensitivity_curve,
)

SR = 250


def _toy_cache() -> pd.DataFrame:
    # one file: a true event at t=5.0s, dur 200ms; predicted onset exactly right.
    # n_samples=5000 @ 250 Hz = 20s file; BOUNDARY_MARGIN_S=4s → lo=1000, hi=4000.
    # onset_samp=1250 is inside the valid window [1000, 4000).
    onset_samp = int(5.0 * SR)
    return pd.DataFrame(
        [
            {
                "filename": "f1",
                "subject": 1,
                "mask_duration_ms": 200.0,
                "manually_refined": False,
                "n_samples": 5000,
                "pred_onsets": json.dumps([onset_samp]),
                "onset_times": json.dumps([5.0]),
            }
        ]
    )


def test_scale_zero_perfect_when_onset_correct():
    f1 = calibrated_event_f1_at_scale(_toy_cache(), delta=0.0)
    assert f1 == pytest.approx(1.0, abs=1e-9)


def test_large_overestimate_degrades_f1():
    # delta=5 → pred width = 200ms*6 = 1200ms; IoU = 200/1200 ≈ 0.167 < IOU_THRESHOLD=0.3
    f1 = calibrated_event_f1_at_scale(_toy_cache(), delta=5.0)
    assert f1 < 1.0


def test_curve_has_all_deltas_and_peaks_at_zero():
    curve = duration_sensitivity_curve(_toy_cache(), deltas=(-0.5, 0.0, 0.5, 2.0))
    assert list(curve["delta"]) == [-0.5, 0.0, 0.5, 2.0]
    assert curve.loc[curve.delta == 0.0, "event_f1"].iloc[0] == pytest.approx(
        1.0, abs=1e-9
    )


def test_refined_subset_summary_keys():
    from src.calibration_sensitivity import refined_subset_summary

    cache = pd.DataFrame(
        [
            {
                "manually_refined": True,
                "mask_onset_offset_ms": 40.0,
                "mask_duration_ms": 200.0,
            },
            {
                "manually_refined": False,
                "mask_onset_offset_ms": 0.0,
                "mask_duration_ms": 200.0,
            },
        ]
    )
    out = refined_subset_summary(cache)
    assert out["n_refined"] == 1
    assert "median_abs_onset_offset_ms" in out
