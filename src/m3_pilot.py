#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scikit-learn",
#     "scipy",
#     "torch",
#     "kymatio",
#     "xgboost",
#     "joblib",
#     "matplotlib",
#     "tqdm",
#     "lgs-db",
#     "tabulate",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""M3 ablation pilot: label-threshold / per-subject-scaling / temporal-smoothing.

Design: docs/superpowers/specs/2026-04-19-m3-ablation-pilot-design.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_EVAL_CSV = PROJECT_ROOT / "outputs" / "results" / "m3_eval.csv"
ROSTER_CSV = PROJECT_ROOT / "data" / "m3_pilot_roster.csv"

# Hoisted so `from comparison_methods import ...` works when running this script
# directly (e.g. `uv run src/m3_pilot.py ...`) without PYTHONPATH gymnastics.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FN_CLUSTER = ["300-002", "301-001", "301-003", "300-006"]
FP_CLUSTER = ["302-002", "302-003"]
CONTROL_CLUSTER = ["300-005", "303-001"]

FN_PER_SUBJECT = 8
FP_302_002 = 6
FP_302_003 = 5  # take all available
CONTROL_300_005 = 4
CONTROL_303_001 = 3


def build_roster(
    baseline_csv: Path = BASELINE_EVAL_CSV,
    output: Path = ROSTER_CSV,
) -> pd.DataFrame:
    """Build 50-file pilot roster from the 500-file baseline eval CSV.

    Args:
        baseline_csv: Path to m3_eval.csv with per-file metrics.
        output: Where to write the roster CSV.

    Returns:
        The roster DataFrame with columns
        (filename, subject, cluster, baseline_sample_f1).
    """
    df = pd.read_csv(baseline_csv)
    rows = []

    # FN cluster: lowest sample_F1 per subject
    for subj in FN_CLUSTER:
        sub = df[df["subject"] == subj].nsmallest(FN_PER_SUBJECT, "sample_f1")
        for _, r in sub.iterrows():
            rows.append((r["filename"], subj, "FN", r["sample_f1"]))

    # FP cluster: highest FP count
    sub = df[df["subject"] == "302-002"].nlargest(FP_302_002, "sample_fp")
    for _, r in sub.iterrows():
        rows.append((r["filename"], "302-002", "FP", r["sample_f1"]))

    sub = df[df["subject"] == "302-003"].nlargest(FP_302_003, "sample_fp")
    for _, r in sub.iterrows():
        rows.append((r["filename"], "302-003", "FP", r["sample_f1"]))

    # Control: median F1
    for subj, n in [("300-005", CONTROL_300_005), ("303-001", CONTROL_303_001)]:
        sub = df[df["subject"] == subj].copy()
        sub["_dist"] = (sub["sample_f1"] - sub["sample_f1"].median()).abs()
        sub = sub.nsmallest(n, "_dist")
        for _, r in sub.iterrows():
            rows.append((r["filename"], subj, "control", r["sample_f1"]))

    roster = pd.DataFrame(
        rows, columns=["filename", "subject", "cluster", "baseline_sample_f1"]
    )
    assert len(roster) == 50, (
        f"Expected 50-file roster but got {len(roster)}. "
        f"Cluster counts: {roster.groupby('cluster').size().to_dict()}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    roster.to_csv(output, index=False)
    return roster


def _cli_build_roster(args: argparse.Namespace) -> None:
    roster = build_roster()
    print(f"[roster] {len(roster)} files written to {ROSTER_CSV}")
    print(roster.groupby("cluster").size().to_string())


def _smoke_task1_roster() -> None:
    """Task 1 smoke test: build_roster() produces the expected 50-file roster."""
    roster = build_roster()
    assert len(roster) == 50, f"Expected 50 rows, got {len(roster)}"
    counts = roster.groupby("cluster").size().to_dict()
    expected = {"FN": 32, "FP": 11, "control": 7}
    assert counts == expected, f"Expected cluster counts {expected}, got {counts}"
    required_cols = {"filename", "subject", "cluster", "baseline_sample_f1"}
    missing = required_cols - set(roster.columns)
    assert not missing, f"Missing required columns: {missing}"


# Registry of smoke tests — later tasks append their own callables here.
_smoke_tests: list = [_smoke_task1_roster]


def _cli_smoke_test(args: argparse.Namespace) -> None:
    total = len(_smoke_tests)
    for i, fn in enumerate(_smoke_tests, start=1):
        name = fn.__name__
        try:
            fn()
            print(f"[smoke {i}/{total}] {name} OK")
        except Exception as e:
            print(f"[smoke {i}/{total}] {name} FAILED: {e}")
            raise
    print(f"All {total} smoke tests passed.")


# ---------------------------------------------------------------------------
# Task 2 — feature cache
# ---------------------------------------------------------------------------

FEATURE_CACHE = PROJECT_ROOT / "data" / "m3_pilot_features.npz"
# Pinned snapshot of the refined-labels catalog at the commit that produced
# outputs/results/m3_eval.csv (commit 67f54a1). The live stim_catalog.parquet
# has been re-refined since (durations shortened per 2026-03-08 session notes),
# so using it here would break reproducibility against the baseline. See
# docs/superpowers/specs/2026-04-19-m3-ablation-pilot-design.md for rationale.
STIM_CATALOG = PROJECT_ROOT / "data" / "stim_catalog_m3eval.parquet"


def _cache_key(roster: pd.DataFrame, window: int, J: int, Q: int) -> str:
    """Deterministic hash of (file list, feature params) for cache invalidation.

    Args:
        roster: Roster DataFrame; only the ``filename`` column participates.
        window: Window size in samples.
        J: Scattering J parameter.
        Q: Scattering Q parameter.

    Returns:
        16-char hex prefix of the SHA-256 of the sorted filenames + params.
    """
    files_sorted = sorted(roster["filename"].astype(str).tolist())
    payload = json.dumps(
        {"files": files_sorted, "window": window, "J": J, "Q": Q},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_features(
    roster: pd.DataFrame,
    window: int = 256,
    J: int = 5,
    Q: int = 8,
    stride: int | None = None,
    output: Path = FEATURE_CACHE,
    catalog_path: Path = STIM_CATALOG,
) -> dict:
    """Extract scattering features for all pilot files and cache to .npz.

    Reuses ``ScatteringXGBoost._extract_features`` for per-window features and
    the same overlapping-windows layout as inference. Labels and file paths
    are joined from ``stim_catalog.parquet`` (the refined-labels catalog used
    by the original M3 training/eval).

    Args:
        roster: Roster DataFrame from ``build_roster``.
        window: Window size in samples.
        J: Scattering J parameter.
        Q: Scattering Q parameter.
        stride: Window stride (default ``window // 2``).
        output: Cache file path.
        catalog_path: Parquet catalog providing ``file_path`` and mask params.

    Returns:
        Dict mirroring the ``.npz`` contents.
    """
    from comparison_methods import (
        ScatteringXGBoost,
        _load_dat_channels,
        build_true_mask,
    )
    from stim_detector_lib import parse_onset_times

    stride = stride or window // 2
    extractor = ScatteringXGBoost(window=window, J=J, Q=Q)
    key = _cache_key(roster, window, J, Q)

    if output.exists():
        cached = np.load(output, allow_pickle=True)
        cached_key = str(cached["cache_key"]) if "cache_key" in cached.files else ""
        if cached_key == key:
            print(f"[cache] HIT ({output}, key={key})")
            return {k: cached[k] for k in cached.files}
        print(f"[cache] STALE (expected {key}, found {cached_key})")

    # Join roster → catalog on filename for labels and file paths.
    catalog = pd.read_parquet(catalog_path)
    catalog["filename"] = catalog["filename"].astype(str)
    roster = roster.copy()
    roster["filename"] = roster["filename"].astype(str)
    meta_cols = [
        "filename",
        "file_path",
        "onset_times",
        "mask_duration_ms",
        "mask_onset_offset_ms",
        "sampling_rate",
    ]
    joined = roster.merge(catalog[meta_cols], on="filename", how="left")

    feats_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    file_ids: list[int] = []
    subjects_win: list[str] = []
    window_starts: list[int] = []
    file_nsamples: list[int] = []
    file_names: list[str] = []

    miss_count = 0
    for file_id, row in enumerate(joined.itertuples(index=False)):
        if row.file_path is None or (
            isinstance(row.file_path, float) and np.isnan(row.file_path)
        ):
            print(f"[cache] MISS filename={row.filename} subject={row.subject}")
            miss_count += 1
            continue

        fpath = Path(row.file_path)
        if not fpath.exists():
            print(f"[cache] missing file: {fpath}")
            miss_count += 1
            continue

        data = _load_dat_channels(fpath)
        n_samples = data.shape[1]
        if n_samples < window:
            print(f"[cache] SHORT filename={row.filename} n={n_samples} < {window}")
            miss_count += 1
            continue

        sr = (
            int(row.sampling_rate)
            if row.sampling_rate is not None
            and not (
                isinstance(row.sampling_rate, float) and np.isnan(row.sampling_rate)
            )
            and row.sampling_rate
            else 250
        )
        onset_offset_ms = (
            float(row.mask_onset_offset_ms)
            if row.mask_onset_offset_ms is not None
            and not (
                isinstance(row.mask_onset_offset_ms, float)
                and np.isnan(row.mask_onset_offset_ms)
            )
            else 0.0
        )
        mask_duration_ms = (
            float(row.mask_duration_ms)
            if row.mask_duration_ms is not None
            and not (
                isinstance(row.mask_duration_ms, float)
                and np.isnan(row.mask_duration_ms)
            )
            else 0.0
        )
        # onset_times is stored as a JSON string in the parquet catalog.
        onsets = parse_onset_times(row.onset_times)
        true_mask = build_true_mask(
            n_samples,
            onsets,
            mask_duration_ms,
            sr=sr,
            onset_offset_ms=onset_offset_ms,
        )

        # Overlapping windows matching ScatteringXGBoost.predict layout.
        starts = list(range(0, n_samples - window + 1, stride))
        if starts and starts[-1] + window < n_samples:
            starts.append(n_samples - window)

        for s in starts:
            win = data[:, s : s + window]
            feats_all.append(extractor._extract_features(win))
            window_starts.append(s)
            file_ids.append(file_id)
            subjects_win.append(row.subject)

        mask_all.append(true_mask)
        file_nsamples.append(n_samples)
        file_names.append(str(row.filename))

    X = np.array(feats_all, dtype=np.float32)
    payload = dict(
        cache_key=key,
        X=X,
        file_ids=np.asarray(file_ids, dtype=np.int32),
        subjects=np.asarray(subjects_win),
        window_starts=np.asarray(window_starts, dtype=np.int64),
        file_nsamples=np.asarray(file_nsamples, dtype=np.int64),
        file_names=np.asarray(file_names),
        window=np.int64(window),
        stride=np.int64(stride),
        # Masks are variable-length; store as object array.
        true_masks=np.array(mask_all, dtype=object),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(
        f"[cache] WROTE {output} "
        f"(key={key}, X={X.shape}, files={len(file_names)}, miss={miss_count})"
    )
    return payload


def _cli_cache_features(args: argparse.Namespace) -> None:
    roster = pd.read_csv(ROSTER_CSV)
    cache_features(roster)


# ---------------------------------------------------------------------------
# Ablation predict paths — each takes the feature cache + a fitted
# ScatteringXGBoost instance and returns a sample-level probability timeline
# for every file in the cache.
# ---------------------------------------------------------------------------

BASELINE_MODEL_DIR = PROJECT_ROOT / "data" / "baselines" / "m3"
LOWTHRESH_MODEL_DIR = PROJECT_ROOT / "data" / "baselines" / "m3_lowthresh"


def _window_probs(
    model,
    X: np.ndarray,
    scaler=None,
) -> np.ndarray:
    """Run one forward pass through scaler + XGBoost.

    Args:
        model: Fitted ScatteringXGBoost with .clf.
        X: (n_windows, n_features) float32.
        scaler: Override scaler (else use model.scaler).

    Returns:
        (n_windows,) float32 probabilities in [0, 1].
    """
    s = scaler if scaler is not None else model.scaler
    if s is not None:
        X = s.transform(X)
    return model.clf.predict_proba(X)[:, 1].astype(np.float32)


def _overlap_average(
    win_probs: np.ndarray,
    cache: dict,
) -> list[np.ndarray]:
    """Map per-window probabilities back to per-sample timelines.

    Returns:
        List of (n_samples,) arrays, one per file, same order as file_names.
    """
    window = int(cache["window"])
    file_ids = cache["file_ids"]
    window_starts = cache["window_starts"]
    file_nsamples = cache["file_nsamples"]
    n_files = len(file_nsamples)

    timelines: list[np.ndarray] = []
    for f in range(n_files):
        mask = file_ids == f
        probs_f = win_probs[mask]
        starts_f = window_starts[mask]
        n = int(file_nsamples[f])
        prob_sum = np.zeros(n, dtype=np.float64)
        prob_count = np.zeros(n, dtype=np.float64)
        for p, s in zip(probs_f, starts_f):
            end = min(s + window, n)
            prob_sum[s:end] += p
            prob_count[s:end] += 1.0
        prob_count = np.maximum(prob_count, 1.0)
        timelines.append((prob_sum / prob_count).astype(np.float32))
    return timelines


def _load_baseline_model(model_dir: Path):
    """Load a fitted ``ScatteringXGBoost`` from disk.

    Imports ``xgboost`` and ``joblib`` BEFORE ``comparison_methods`` to avoid
    a macOS libomp double-load segfault: xgboost bundles its own libomp, and
    on macOS loading it after torch's libomp crashes ``load_model``. Preloading
    these two dependencies first gets everything onto the same OMP runtime.

    Args:
        model_dir: Directory containing m3_xgb.json / m3_scaler.pkl / m3_meta.json.

    Returns:
        A loaded ``ScatteringXGBoost`` instance (model.clf + model.scaler set).
    """
    # TODO: move libomp preload into comparison_methods.py top of file
    # so all M3 consumers inherit the fix transparently.
    import xgboost  # noqa: F401  -- preload to avoid torch/xgb libomp conflict
    import joblib  # noqa: F401

    from comparison_methods import ScatteringXGBoost

    return ScatteringXGBoost.load(model_dir)


def predict_B0(cache: dict) -> list[np.ndarray]:
    """Baseline: existing model, global scaler, no smoothing."""
    model = _load_baseline_model(BASELINE_MODEL_DIR)
    win_probs = _window_probs(model, cache["X"])
    return _overlap_average(win_probs, cache)


def predict_A1(cache: dict) -> list[np.ndarray]:
    """A1: presence-label-retrained model, global scaler, no smoothing."""
    model = _load_baseline_model(LOWTHRESH_MODEL_DIR)  # reuse libomp preload
    win_probs = _window_probs(model, cache["X"])
    return _overlap_average(win_probs, cache)


A2_MIN_CLEAN_WINDOWS = 20


def _fit_subject_scalers(
    cache: dict,
    min_windows: int = A2_MIN_CLEAN_WINDOWS,
) -> tuple[dict, list[str]]:
    """Fit a StandardScaler on each subject's clean scattering windows.

    "Clean" = window where ground-truth mask has zero artifact samples.
    Primary source: disabled recordings via lgs_db (if available).
    Fallback: windows from the subject's own pilot files where the mask is
    entirely zero.

    Returns:
        (scalers_by_subject, fallback_subjects) — the second list names
        subjects that did not meet min_windows and will reuse the global scaler.
    """
    from sklearn.preprocessing import StandardScaler

    X = cache["X"]
    window = int(cache["window"])
    file_ids = cache["file_ids"]
    window_starts = cache["window_starts"]
    subjects_win = cache["subjects"]
    true_masks = cache["true_masks"]

    scalers: dict[str, StandardScaler] = {}
    fallback: list[str] = []

    subj_set = sorted(set(subjects_win.tolist()))
    for subj in subj_set:
        # Clean pilot windows for this subject: mask=0 across the whole window
        clean_rows = []
        subj_mask = subjects_win == subj
        for idx in np.where(subj_mask)[0]:
            f = int(file_ids[idx])
            s = int(window_starts[idx])
            m = true_masks[f][s : s + window]
            if m.size == window and not np.any(m):
                clean_rows.append(idx)
        clean_X = (
            X[np.asarray(clean_rows, dtype=np.int64)]
            if clean_rows
            else np.empty((0, X.shape[1]))
        )
        # Note: disabled-recording augmentation via lgs_db could be added here
        # for subjects that fall below min_windows. For the pilot, we log
        # any fallback subjects so we can assess whether augmentation is
        # needed before the 500-file confirmation run.
        if len(clean_X) < min_windows:
            fallback.append(subj)
            continue
        s = StandardScaler().fit(clean_X)
        scalers[subj] = s
    return scalers, fallback


def predict_A2(cache: dict) -> list[np.ndarray]:
    """A2: per-subject scaler override, existing model, no smoothing."""
    model = _load_baseline_model(BASELINE_MODEL_DIR)
    scalers, fallback = _fit_subject_scalers(cache)
    if fallback:
        print(f"[A2] fallback to global scaler for subjects: {fallback}")

    X = cache["X"]
    subjects_win = cache["subjects"]
    X_scaled = np.empty_like(X)

    for subj in np.unique(subjects_win):
        mask = subjects_win == subj
        scaler = scalers.get(subj, model.scaler)
        X_scaled[mask] = scaler.transform(X[mask])

    win_probs = model.clf.predict_proba(X_scaled)[:, 1].astype(np.float32)
    return _overlap_average(win_probs, cache)


A3_MEDIAN_KERNEL = 125  # samples = 0.5 s @ 250 Hz
A3_CLOSING_KERNEL = 50  # samples = 0.2 s @ 250 Hz


def _smooth_and_close(
    probs: np.ndarray,
    median_k: int = A3_MEDIAN_KERNEL,
    closing_k: int = A3_CLOSING_KERNEL,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply A3's two-stage smoothing.

    1. 1-D median filter on probability timeline (pre-threshold).
    2. Binary threshold.
    3. Morphological closing (post-threshold, fills short gaps).

    Returns:
        Float32 timeline in [0, 1] where the values are either 0.0 or 1.0
        (post-threshold). Consumers treat any value >= threshold as positive
        in downstream metric code; emitting 0/1 keeps that invariant.
    """
    from scipy.ndimage import binary_closing, median_filter

    smoothed = median_filter(probs.astype(np.float64), size=median_k, mode="nearest")
    binary = smoothed >= threshold
    if binary.any():
        binary = binary_closing(binary, structure=np.ones(closing_k, dtype=bool))
    return binary.astype(np.float32)


def predict_A3(cache: dict) -> list[np.ndarray]:
    """A3: baseline model, global scaler, post-hoc median + closing."""
    model = _load_baseline_model(BASELINE_MODEL_DIR)
    win_probs = _window_probs(model, cache["X"])
    raw_timelines = _overlap_average(win_probs, cache)
    return [_smooth_and_close(tl) for tl in raw_timelines]


# ---------------------------------------------------------------------------
# Task 3 — regression check: B0 on pilot files must reproduce m3_eval.csv.
# ---------------------------------------------------------------------------


def _sample_f1_from_probs(
    probs: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
    post_process: bool = True,
) -> float:
    """Compute sample-level F1 from a probability timeline vs a boolean mask.

    Mirrors ``evaluate_method``: binarize with ``proba > threshold`` (strict),
    then run ``post_process_mask`` (morphological closing + min-artifact-size).

    Args:
        probs: (n_samples,) probabilities in [0, 1].
        true_mask: (n_samples,) ground-truth mask.
        threshold: Decision threshold applied to ``probs``.
        post_process: Apply ``post_process_mask`` to match ``evaluate_method``.

    Returns:
        F1 in [0, 1]. Convention: zero-GT + zero-pred = 1.0 (true negative).
    """
    pred = (probs > threshold).astype(np.int_)
    if post_process:
        from stim_detector_lib import Config, post_process_mask

        pred = post_process_mask(pred, Config())
    pred = pred.astype(np.int8)
    gt = (true_mask > 0).astype(np.int8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # zero-GT + zero-pred = TN
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def regression_check_B0(cache: dict, tol: float = 1e-3) -> None:
    """Assert B0 sample F1 on pilot files matches m3_eval.csv within tol.

    Uses 1e-3 (not 1e-6) because overlap-window rounding and XGBoost stochastic
    determinism across machine arches can shift F1 at the 4th decimal.
    """
    baseline = pd.read_csv(BASELINE_EVAL_CSV, dtype={"filename": str}).set_index(
        "filename"
    )
    timelines = predict_B0(cache)
    names = cache["file_names"]
    masks = cache["true_masks"]
    worst = (0.0, None, None, None)
    for tl, fname, mask in zip(timelines, names, masks):
        key = str(fname)
        if key not in baseline.index:
            print(f"[regression] skip (not in baseline): {key}")
            continue
        ref = baseline.loc[key]
        ours = _sample_f1_from_probs(tl, mask)
        diff = abs(ours - float(ref["sample_f1"]))
        if diff > worst[0]:
            worst = (diff, key, ours, float(ref["sample_f1"]))
    print(f"[regression] worst |Δ sample_f1| = {worst[0]:.5f} on {worst[1]}")
    assert worst[0] <= tol, (
        f"B0 regression: |Δ F1| = {worst[0]:.5f} > {tol} "
        f"(file={worst[1]} ours={worst[2]} ref={worst[3]})"
    )
    print(f"[regression] OK (N={len(timelines)}, tol={tol})")


def _cli_regression_check(args: argparse.Namespace) -> None:
    cache = dict(np.load(FEATURE_CACHE, allow_pickle=True))
    regression_check_B0(cache)


# ---------------------------------------------------------------------------
# Task 8 — unified eval loop + per-file CSV writer.
# ---------------------------------------------------------------------------

RESULTS_CSV = PROJECT_ROOT / "outputs" / "results" / "m3_pilot_ablation.csv"

ABLATIONS = {
    "B0": predict_B0,
    "A1": predict_A1,
    "A2": predict_A2,
    "A3": predict_A3,
}


def _post_process_binary(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold + post_process_mask, returns binary int8 mask."""
    from comparison_methods import post_process_mask, Config

    pred = (probs > threshold).astype(np.int_)
    return post_process_mask(pred, Config()).astype(np.int8)


def _event_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    min_overlap: float = 0.5,
) -> dict:
    """Event-level TP/FP/FN using contiguous-run matching.

    An event = contiguous run of 1s. A predicted event matches a GT event
    if their overlap is >= min_overlap of either event's length.
    """
    from scipy.ndimage import label as _label

    gt_runs, n_gt = _label(gt_mask.astype(np.int8))
    pred_runs, n_pred = _label(pred_mask.astype(np.int8))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for p in range(1, n_pred + 1):
        p_where = pred_runs == p
        p_len = int(p_where.sum())
        if p_len == 0:
            continue
        for g in range(1, n_gt + 1):
            if g in matched_gt:
                continue
            g_where = gt_runs == g
            overlap = int((p_where & g_where).sum())
            g_len = int(g_where.sum())
            if g_len == 0:
                continue
            if overlap >= min_overlap * p_len or overlap >= min_overlap * g_len:
                matched_pred.add(p)
                matched_gt.add(g)
                break

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - len(matched_gt)
    if tp + fp == 0:
        prec = 1.0 if (tp == 0 and fn == 0) else 0.0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = 1.0 if (tp == 0 and fp == 0) else 0.0
    else:
        rec = tp / (tp + fn)
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 1.0
    elif prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "event_n_gt": int(n_gt),
        "event_n_pred": int(n_pred),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_precision": float(prec),
        "event_recall": float(rec),
        "event_f1": float(f1),
    }


def _per_file_metrics(
    probs: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute sample + event metrics for one (probs, gt_mask) pair.

    Applies the same threshold + post_process_mask pipeline that
    evaluate_method uses, so numbers are directly comparable to m3_eval.csv.
    """
    pred = _post_process_binary(probs, threshold=threshold)
    gt = (gt_mask > 0).astype(np.int8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 1.0
        iou = 1.0
    elif prec + rec == 0:
        f1 = 0.0
        iou = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
        iou = tp / max(tp + fp + fn, 1)
    out = {
        "sample_tp": tp,
        "sample_fp": fp,
        "sample_fn": fn,
        "sample_tn": tn,
        "sample_precision": prec,
        "sample_recall": rec,
        "sample_f1": f1,
        "sample_iou": iou,
    }
    out.update(_event_metrics(pred, gt))
    return out


def run_pilot(
    cache: dict, roster: pd.DataFrame, output: Path = RESULTS_CSV
) -> pd.DataFrame:
    """Run all four ablations on cached features and write long-format CSV."""
    roster_by_name = {
        str(r.filename): (r.subject, r.cluster) for r in roster.itertuples(index=False)
    }
    file_names = [str(f) for f in cache["file_names"].tolist()]
    masks = cache["true_masks"]

    rows = []
    for ab_name, fn in ABLATIONS.items():
        print(f"[run] {ab_name} ...")
        timelines = fn(cache)
        for tl, name, m in zip(timelines, file_names, masks):
            subj, cluster = roster_by_name.get(name, ("?", "?"))
            metrics = _per_file_metrics(tl, m)
            rows.append(
                {
                    "ablation": ab_name,
                    "filename": name,
                    "subject": subj,
                    "cluster": cluster,
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[run] wrote {len(df)} rows to {output}")
    return df


def _cli_run(args: argparse.Namespace) -> None:
    roster = pd.read_csv(ROSTER_CSV, dtype={"filename": str})
    cache = dict(np.load(FEATURE_CACHE, allow_pickle=True))
    run_pilot(cache, roster)


# ---------------------------------------------------------------------------
# Task 9 — aggregation with bootstrap CIs and Wilcoxon paired tests vs B0.
# ---------------------------------------------------------------------------

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def _bootstrap_ci(
    vals: np.ndarray, n_boot: int = 1000, seed: int = RNG_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon_vs_b0(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Paired Wilcoxon signed-rank on sample_f1 (per cluster, per ablation vs B0)."""
    from scipy.stats import wilcoxon

    out: dict[tuple[str, str], float] = {}
    b0 = df[df["ablation"] == "B0"].set_index(["cluster", "filename"])["sample_f1"]
    for ab in ["A1", "A2", "A3"]:
        ax = df[df["ablation"] == ab].set_index(["cluster", "filename"])["sample_f1"]
        for cluster in sorted(df["cluster"].unique()):
            try:
                diff = (ax[cluster] - b0[cluster]).dropna().values
                if len(diff) == 0 or np.all(diff == 0):
                    out[(ab, cluster)] = float("nan")
                    continue
                stat = wilcoxon(diff, zero_method="wilcox").pvalue
                out[(ab, cluster)] = float(stat)
            except Exception:
                out[(ab, cluster)] = float("nan")
    return out


def aggregate(
    input_csv: Path = RESULTS_CSV,
    stamp: str = "2026-04-19",
) -> pd.DataFrame:
    """Produce per-(ablation, cluster) summary with bootstrap CIs and Wilcoxon p vs B0."""
    df = pd.read_csv(input_csv)
    wpv = _wilcoxon_vs_b0(df)

    rows = []
    for (ab, cluster), g in df.groupby(["ablation", "cluster"]):
        for metric in ["sample_f1", "sample_precision", "sample_recall", "event_f1"]:
            vals = g[metric].to_numpy()
            lo, hi = _bootstrap_ci(vals)
            rows.append(
                {
                    "ablation": ab,
                    "cluster": cluster,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "wilcoxon_p_vs_b0": (
                        wpv.get((ab, cluster), float("nan"))
                        if metric == "sample_f1"
                        else float("nan")
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = TABLES_DIR / f"tab_m3_ablation_{stamp}.csv"
    summary.to_csv(csv_out, index=False)
    print(f"[aggregate] wrote {csv_out}")

    # Markdown companion: pivot on sample_f1 only for readability
    pivot = (
        summary[summary["metric"] == "sample_f1"]
        .assign(
            cell=lambda d: d.apply(
                lambda r: f"{r['mean']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]",
                axis=1,
            )
        )
        .pivot(index="ablation", columns="cluster", values="cell")
    )
    md_out = TABLES_DIR / f"tab_m3_ablation_{stamp}.md"
    md_out.write_text(
        "# M3 Pilot Ablation — Sample F1 by (ablation, cluster)\n\n"
        "Values: mean [95% bootstrap CI, 1000 resamples]. See\n"
        f"`{csv_out.name}` for all metrics including precision, recall, event F1\n"
        "and paired Wilcoxon p-values vs B0.\n\n" + pivot.to_markdown() + "\n"
    )
    print(f"[aggregate] wrote {md_out}")
    return summary


def _cli_aggregate(args: argparse.Namespace) -> None:
    aggregate()


# ---------------------------------------------------------------------------
# Task 10 — figure generator: grouped bar chart with 95% bootstrap CIs.
# ---------------------------------------------------------------------------

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ABLATION_COLORS = {
    "B0": "#888888",
    "A1": "#1f77b4",
    "A2": "#2ca02c",
    "A3": "#d62728",
}


def plot_ablation(stamp: str = "2026-04-19") -> None:
    """Grouped bar chart: 4 ablations x 3 clusters, 95% CI error bars."""
    import matplotlib.pyplot as plt

    csv_path = TABLES_DIR / f"tab_m3_ablation_{stamp}.csv"
    df = pd.read_csv(csv_path)
    df = df[df["metric"] == "sample_f1"]

    clusters = ["FN", "FP", "control"]
    ablations = ["B0", "A1", "A2", "A3"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=300)
    width = 0.2
    xpos = np.arange(len(clusters))

    for i, ab in enumerate(ablations):
        means, errs_lo, errs_hi = [], [], []
        for c in clusters:
            r = df[(df["ablation"] == ab) & (df["cluster"] == c)]
            if r.empty:
                means.append(np.nan)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
                continue
            r = r.iloc[0]
            means.append(r["mean"])
            errs_lo.append(r["mean"] - r["ci95_lo"])
            errs_hi.append(r["ci95_hi"] - r["mean"])
        ax.bar(
            xpos + (i - 1.5) * width,
            means,
            width=width,
            label=ab,
            color=ABLATION_COLORS[ab],
            yerr=[errs_lo, errs_hi],
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels(clusters)
    ax.set_ylabel("Sample F1 (mean, 95% bootstrap CI)")
    ax.set_title(f"M3 ablation pilot — {stamp}")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = FIGURES_DIR / f"fig_m3_ablation_{stamp}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[plot] wrote {out}")
    plt.close(fig)


def _cli_plot(args: argparse.Namespace) -> None:
    plot_ablation()


# ---------------------------------------------------------------------------
# Task 11 — end-to-end smoke tests extending the registry.
# ---------------------------------------------------------------------------


def _smoke_synthetic_burst() -> None:
    """Real-data burst localization: B0 + A3 concentrate mass on stim regions.

    Uses the pre-computed pilot feature cache and checks that across all stim
    files with non-empty GT masks, aggregated probability mass is higher
    inside the ground-truth mask than outside. This is the data-level analog
    of ``does the model localize stim bursts?`` — a pass/fail check on real
    data rather than a shape-only sanity test.

    Pure-synthetic scattering is avoided here because computing torch-based
    scattering and loading xgboost in the same process triggers macOS
    libomp double-load issues; the feature cache was produced by a fresh
    process and is safe to reuse. If the cache doesn't exist, the test SKIPs.
    """
    if not FEATURE_CACHE.exists():
        print("[smoke] SKIP _smoke_synthetic_burst (feature cache not built yet)")
        return

    cache = dict(np.load(FEATURE_CACHE, allow_pickle=True))
    masks = cache["true_masks"]
    stim_indices = [i for i, m in enumerate(masks) if int(np.asarray(m).sum()) > 0]
    assert stim_indices, "no stim files in feature cache"

    for name, fn in [("B0", predict_B0), ("A3", predict_A3)]:
        timelines = fn(cache)
        inside_vals: list[np.ndarray] = []
        outside_vals: list[np.ndarray] = []
        for i in stim_indices:
            tl = timelines[i]
            m = np.asarray(masks[i])
            assert (
                tl.shape == m.shape
            ), f"{name}[{i}] timeline shape {tl.shape} != mask shape {m.shape}"
            inside_vals.append(tl[m > 0])
            outside_vals.append(tl[m == 0])
        inside = np.concatenate(inside_vals)
        outside = np.concatenate(outside_vals)
        assert inside.size > 0 and outside.size > 0, (
            f"{name}: degenerate aggregate "
            f"(inside={inside.size}, outside={outside.size})"
        )
        assert inside.mean() > outside.mean(), (
            f"{name}: aggregate inside-GT mean {inside.mean():.3f} should "
            f"exceed outside-GT mean {outside.mean():.3f} "
            f"(mass not concentrated on burst regions)"
        )


def _smoke_a2_scaler_sanity() -> None:
    """Per-subject scaler: post-transform clean windows should have mean≈0, std≈1."""
    if not FEATURE_CACHE.exists():
        print("[smoke] SKIP _smoke_a2_scaler_sanity (feature cache not built yet)")
        return
    cache = dict(np.load(FEATURE_CACHE, allow_pickle=True))
    scalers, _ = _fit_subject_scalers(cache)
    assert len(scalers) >= 1, "no per-subject scalers fitted"

    X = cache["X"]
    subjects = cache["subjects"]
    file_ids = cache["file_ids"]
    win_starts = cache["window_starts"]
    window = int(cache["window"])
    masks = cache["true_masks"]

    for subj, scaler in list(scalers.items())[:2]:
        clean_idx = []
        for idx in np.where(subjects == subj)[0]:
            f = int(file_ids[idx])
            t = int(win_starts[idx])
            m = masks[f][t : t + window]
            if m.size == window and not m.any():
                clean_idx.append(idx)
        Xt = scaler.transform(X[np.asarray(clean_idx)])
        mean_abs = abs(float(Xt.mean()))
        std = float(Xt.std())
        assert (
            mean_abs < 0.05
        ), f"{subj}: post-scale |mean|={mean_abs:.3f} (expected <0.05)"
        assert abs(std - 1.0) < 0.1, f"{subj}: post-scale std={std:.3f} (expected ~1)"


def _smoke_a3_smoothing() -> None:
    """A3 smoothing: idempotence (kernel=1) + empty-input no-op + gap fill."""
    # kernel=1 with matching threshold should equal naive thresholding
    p = np.array([0.0, 0.9, 0.9, 0.1, 0.8] * 200, dtype=np.float32)
    out = _smooth_and_close(p, median_k=1, closing_k=1, threshold=0.5)
    expected = (p >= 0.5).astype(np.float32)
    assert np.array_equal(out, expected), "kernel=1 should equal naive thresholding"

    # empty input
    p_empty = np.zeros(1000, dtype=np.float32)
    out_empty = _smooth_and_close(p_empty)
    assert not out_empty.any(), "empty input should give empty output"

    # gap fill
    p_burst = np.zeros(1000, dtype=np.float32)
    p_burst[100:200] = 0.9
    p_burst[220:400] = 0.9  # 20-sample gap
    out_burst = _smooth_and_close(p_burst, median_k=1, closing_k=50, threshold=0.5)
    assert out_burst[200:220].mean() > 0.5, "closing should fill 20-sample gap"


def _smoke_b0_regression() -> None:
    """B0 regression vs m3_eval.csv (requires feature cache)."""
    if not FEATURE_CACHE.exists():
        print("[smoke] SKIP _smoke_b0_regression (feature cache not built yet)")
        return
    cache = dict(np.load(FEATURE_CACHE, allow_pickle=True))
    regression_check_B0(cache)


_smoke_tests.extend(
    [
        _smoke_synthetic_burst,
        _smoke_a2_scaler_sanity,
        _smoke_a3_smoothing,
        _smoke_b0_regression,
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-roster")
    sub.add_parser("cache-features")
    sub.add_parser("smoke-test")
    sub.add_parser("regression-check")
    sub.add_parser("run")
    sub.add_parser("aggregate")
    sub.add_parser("plot")
    args = parser.parse_args()
    if args.cmd == "build-roster":
        _cli_build_roster(args)
    elif args.cmd == "cache-features":
        _cli_cache_features(args)
    elif args.cmd == "smoke-test":
        _cli_smoke_test(args)
    elif args.cmd == "regression-check":
        _cli_regression_check(args)
    elif args.cmd == "run":
        _cli_run(args)
    elif args.cmd == "aggregate":
        _cli_aggregate(args)
    elif args.cmd == "plot":
        _cli_plot(args)


if __name__ == "__main__":
    main()
