"""Tests for src/meta_analysis.py."""

import numpy as np
import pytest
from scipy.special import expit, logit

from src.meta_analysis import (
    dersimonian_laird,
    subject_onset_recall_effects,
    meta_regression,
    subgroup_q_between,
    pool_cohorts,
)


def test_dl_homogeneous_zero_tau():
    # identical effects, equal variance -> no heterogeneity
    yi = np.array([0.5, 0.5, 0.5, 0.5])
    vi = np.array([0.04, 0.04, 0.04, 0.04])
    out = dersimonian_laird(yi, vi)
    assert out["tau2"] == pytest.approx(0.0, abs=1e-9)
    assert out["I2"] == pytest.approx(0.0, abs=1e-6)
    assert out["mu"] == pytest.approx(0.5, abs=1e-9)


def test_dl_heterogeneous_positive_tau():
    yi = np.array([0.1, 0.9, 0.2, 0.8])
    vi = np.array([0.01, 0.01, 0.01, 0.01])
    out = dersimonian_laird(yi, vi)
    assert out["tau2"] > 0.0
    assert out["I2"] > 50.0
    # pooled estimate lies within the data range
    assert yi.min() <= out["mu"] <= yi.max()
    # prediction interval is wider than the confidence interval
    assert (out["pi_hi"] - out["pi_lo"]) > (out["ci_hi"] - out["ci_lo"])


def test_logit_roundtrip():
    p = np.array([0.01, 0.3, 0.99])
    assert np.allclose(expit(logit(p)), p, atol=1e-9)


def test_subject_recall_effects_shapes():
    import pandas as pd

    df = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2],
            "onset_tp": [5, 4, 9, 1],
            "onset_fn": [0, 1, 1, 0],
        }
    )
    eff = subject_onset_recall_effects(df)
    assert set(eff["subject"]) == {1, 2}
    assert {"yi", "vi", "recall"}.issubset(eff.columns)
    # subject 1: recall = 9/10
    s1 = eff.loc[eff.subject == 1, "recall"].iloc[0]
    assert s1 == pytest.approx(0.9, abs=1e-9)


def test_meta_regression_linear_relationship():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 10)
    yi = 0.3 + 0.5 * x + rng.normal(0, 0.01, 10)
    vi = np.full(10, 0.01)
    mr = meta_regression(yi, vi, x)
    assert mr["slope"] == pytest.approx(0.5, abs=0.05)
    assert mr["slope_p"] < 0.05


def test_meta_regression_constant_moderator_raises():
    yi = np.array([0.1, 0.2, 0.3])
    vi = np.array([0.01, 0.01, 0.01])
    x = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        meta_regression(yi, vi, x)


def test_subgroup_q_between_detects_difference():
    """Two clearly separated groups -> significant; identical values -> Q=0."""
    yi = np.array([0.0, 0.05, 2.0, 2.05])
    vi = np.full(4, 0.01)
    groups = np.array(["a", "a", "b", "b"])

    out = subgroup_q_between(yi, vi, groups)
    assert out["p_value"] < 0.05
    assert out["Q_between"] > 0
    assert out["n_levels"] == 2
    assert out["n_qualifying"] == 2
    assert out["singleton_levels"] == []

    # Identical values -> no between-group variance
    yi2 = np.array([1.0, 1.0, 1.0, 1.0])
    out2 = subgroup_q_between(yi2, vi, groups)
    assert out2["Q_between"] == pytest.approx(0.0, abs=1e-9)


def test_subgroup_q_between_singleton_excluded():
    """Singleton level should be listed but not contribute to Q_within."""
    yi = np.array([0.0, 0.05, 2.0, 2.05, 1.0])
    vi = np.full(5, 0.01)
    groups = np.array(["a", "a", "b", "b", "c"])  # 'c' is a singleton

    out = subgroup_q_between(yi, vi, groups)
    assert "c" in out["singleton_levels"]
    assert out["n_qualifying"] == 2
    assert out["n_levels"] == 3
    # df = qualifying levels - 1 = 1
    assert out["df"] == 1


def test_pool_cohorts_adds_cohort_column_and_between_test():
    import pandas as pd
    from src.meta_analysis import pool_cohorts

    lgs = pd.DataFrame({"subject": [10, 11], "onset_tp": [100, 90], "onset_fn": [1, 2]})
    bwh = pd.DataFrame(
        {"subject": [20, 21], "onset_tp": [200, 150], "onset_fn": [3, 1]}
    )
    out = pool_cohorts(lgs, bwh)
    assert set(out["pooled"].keys()) >= {"mu", "ci_lo", "ci_hi", "I2", "pi_lo", "pi_hi"}
    assert set(out["by_cohort"].keys()) == {"LGS", "BWH"}
    assert "p_value" in out["between"]
