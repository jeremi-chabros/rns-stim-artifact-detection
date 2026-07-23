"""Tests for src/calibration_validity.py."""

from pathlib import Path

import pandas as pd
import pytest

from src.calibration_validity import (
    decompose_raw_failures,
    localization_summary,
    DEFAULT_EVAL,
)


def _toy() -> pd.DataFrame:
    # 4 files: legit width-rescue, fabricated, clean, genuine-fail
    return pd.DataFrame(
        {
            "raw_event_f1": [0.05, 0.05, 0.99, 0.10],
            "event_f1": [0.99, 0.99, 0.99, 0.20],
            "onset_f1": [0.99, 0.10, 0.99, 0.15],
        }
    )


def test_decompose_toy_counts():
    out = decompose_raw_failures(_toy(), raw_thresh=0.3, ok_thresh=0.9)
    assert out["n_raw_fail"] == 3  # rows 0,1,3
    assert out["n_legit"] == 1  # row 0
    assert out["n_fabricated"] == 1  # row 1
    assert out["n_residual_fail"] == 1  # row 3


def test_decompose_fractions_sum():
    out = decompose_raw_failures(_toy())
    assert 0.0 <= out["frac_legit"] <= 1.0
    assert out["frac_fabricated"] >= 0.0


def test_localization_summary_keys():
    s = localization_summary(_toy())
    for k in ("mean_onset_f1", "mean_event_f1", "corr_onset_calib", "corr_onset_raw"):
        assert k in s


def test_localization_summary_values():
    s = localization_summary(_toy())
    expected = _toy()["onset_f1"].corr(_toy()["event_f1"])
    assert s["corr_onset_calib"] == pytest.approx(expected, abs=1e-9)
    assert -1.0 <= s["corr_onset_raw"] <= 1.0
    assert s["mean_onset_f1"] == pytest.approx(_toy()["onset_f1"].mean(), abs=1e-9)


@pytest.mark.skipif(not Path(DEFAULT_EVAL).exists(), reason="BWH eval CSV not present")
def test_real_bwh_anchors():
    import pandas as pd

    df = pd.read_csv(DEFAULT_EVAL)
    out = decompose_raw_failures(df)
    assert out["n_raw_fail"] == 4154
    assert out["n_legit"] == 3997
    assert out["n_fabricated"] == 0
    s = localization_summary(df)
    assert s["mean_onset_f1"] == pytest.approx(0.9934, abs=1e-3)
