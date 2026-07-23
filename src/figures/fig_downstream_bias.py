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
"""Downstream-bias injection figure.

Inject a real RNS saturation+recovery artifact into clean stim-disabled BWH
ECoG at known times, clean the contaminated trace four ways
(none / 5 s blank / device-log / raw metadata-free U-Net), and compare the
recovered PSD and inter-channel coherence to the known-clean truth. The U-Net
targets the low-bias / low-data-discarded Pareto corner.

The detection pipeline (model load + sliding-window inference + postprocessing)
is reused from ``src/eval_bwh.py`` so the U-Net mask scores identically to the
published evaluation; ``predict_array`` is the in-memory twin of
``eval_bwh.predict_file`` for injected (not-on-disk) data. The pure signal math
lives in ``downstream_bias_lib`` (unit-tested separately).

Spec: docs/superpowers/specs/2026-06-18-downstream-bias-figure-design.md

Usage:
    uv run src/figures/fig_downstream_bias.py --smoke        # 1-trial integrity
    uv run src/figures/fig_downstream_bias.py --pilot 16     # de-risk gate
    uv run src/figures/fig_downstream_bias.py --n-trials 240 # full run + figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_bwh import (  # noqa: E402  (also puts deployed on sys.path)
    BOUNDARY_MARGIN_S,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
    STRIDE_SAMPLES,
    N_COND_FEATURES,
    load_deployed_model,
    postprocess_predictions,
)
import downstream_bias_lib as L  # noqa: E402

DISABLED_CATALOG = REPO_DIR / "data" / "bwh_disabled_catalog.parquet"
# Templates come from the EXTERNAL (BWH) stim cohort -- the population the
# downstream claim is about -- so the device-afforded width (B1+B2) genuinely
# differs from the true artifact extent (it does not on LGS, where they coincide).
TEMPLATE_CATALOG = REPO_DIR / "data" / "bwh_stim_catalog.parquet"
OUT_FIG_DIR = REPO_DIR / "manuscript" / "figures"
OUT_CSV = REPO_DIR / "outputs" / "downstream_bias_trials.csv"

# Welch + injection parameters (chosen so surviving clean gaps host many windows;
# see spec "Open risks 3"). 512 @ 250 Hz == 2.05 s window, 0.49 Hz resolution.
NPERSEG = 512
NOVERLAP = 256
N_EVENTS = 5
SPACING_S = 16.0
BLANK_S = 5.0
TEMPLATE_CAP_S = 5.0  # capture up to this much artifact+recovery tail per template


def _device() -> str:
    """Pick MPS if available, else CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _parse_onsets(val) -> list[float]:
    """Decode the catalog ``onset_times`` cell (JSON string or list)."""
    seq = json.loads(val) if isinstance(val, str) else list(val)
    return [float(x) for x in seq]


def predict_array(
    model, data: np.ndarray, cond: np.ndarray, *, device: str, batch_size: int = 64
) -> np.ndarray:
    """Sliding-window inference on an in-memory (4, N) µV array.

    Mirror of ``eval_bwh.predict_file`` (robust per-channel z-norm,
    ``WINDOW_SAMPLES``/``STRIDE_SAMPLES``, sigmoid, overlap-averaged) but takes
    an array instead of a ``.dat`` path, so it can score injected data.

    Returns:
        (N,) per-sample artifact probability.
    """
    import torch

    median = np.median(data, axis=1)
    q25 = np.percentile(data, 25, axis=1)
    q75 = np.percentile(data, 75, axis=1)
    med_col = median[:, None]
    iqr_col = (q75 - q25)[:, None] + 1e-8

    n = data.shape[1]
    starts = list(range(0, n - WINDOW_SAMPLES + 1, STRIDE_SAMPLES))
    if not starts:
        return np.zeros(n, dtype=np.float32)

    cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)
    acc = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, len(starts), batch_size):
            b = starts[b0 : b0 + batch_size]
            wins = np.stack(
                [(data[:, s : s + WINDOW_SAMPLES] - med_col) / iqr_col for s in b]
            ).astype(np.float32)
            sig_t = torch.from_numpy(wins).to(device)
            logits, _ = model(sig_t, cond_t.expand(len(b), -1))
            probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
            for i, s in enumerate(b):
                acc[s : s + WINDOW_SAMPLES] += probs[i]
                cnt[s : s + WINDOW_SAMPLES] += 1.0
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def extract_template(
    row: pd.Series, *, cap_s: float = TEMPLATE_CAP_S, guard_s: float = 0.5
) -> tuple[np.ndarray, float, float]:
    """Excise a real 4-channel artifact (saturation + recovery tail) from a stim file.

    Captures from the first onset up to ``cap_s`` seconds, or the gap to the next
    onset (minus ``guard_s``), whichever is shorter -- so the low-amplitude
    recovery tail is represented and a too-short device-log mask visibly
    under-removes it. The template is referenced to the source pre-onset baseline
    (deviation), preserving the DC saturation step.

    Returns:
        ``(template (4, L), b1b2_ms, mask_dur_ms)`` where ``b1b2_ms`` is the
        device-afforded B1+B2 burst duration (used for the device-log mask) and
        ``mask_dur_ms`` is the refined label width (used only for regime
        stratification, never for scoring).
    """
    from lgs_db import read_dat, to_microvolts

    uv = to_microvolts(read_dat(str(row["file_path"]))).astype(np.float32)
    onsets = _parse_onsets(row["onset_times"])
    b1 = float(row.get("t1b1_ms") or 0.0)
    b2 = float(row.get("t1b2_ms") or 0.0)
    b1b2_ms = b1 + b2
    md_ms = float(row["mask_duration_ms"])

    cap = int(cap_s * SAMPLING_RATE)
    o0 = onsets[0]
    s = int(o0 * SAMPLING_RATE)
    nxt = next((x for x in onsets if x > o0), None)
    if nxt is not None:
        gap = int((nxt - o0) * SAMPLING_RATE) - int(guard_s * SAMPLING_RATE)
        win = max(min(cap, gap), int(0.3 * SAMPLING_RATE))
    else:
        win = min(cap, uv.shape[1] - s)
    seg = uv[:, s : s + win]
    if seg.shape[1] < int(0.3 * SAMPLING_RATE):  # ran off the end
        s = max(0, uv.shape[1] - cap)
        seg = uv[:, s : s + cap]
    pre_lo = max(0, s - SAMPLING_RATE)  # 1 s pre-onset baseline
    src_base = (
        np.median(uv[:, pre_lo:s], axis=1, keepdims=True)
        if s > pre_lo
        else np.median(uv, axis=1, keepdims=True)
    )
    seg = seg - src_base  # deviation from source baseline (preserves DC rail)
    return seg.astype(np.float32), b1b2_ms, md_ms


def sample_clean_files(n: int, seed: int) -> pd.DataFrame:
    """Sample clean stim-disabled BWH files >= 80 s, spread across subjects."""
    cat = pd.read_parquet(DISABLED_CATALOG)
    cat = cat[cat["length_sec"].astype(float) >= 80.0]
    return cat.sample(min(n, len(cat)), random_state=seed).reset_index(drop=True)


def sample_templates(n: int, seed: int) -> pd.DataFrame:
    """Sample external (BWH) stim files stratified across therapy-duration regimes.

    Requires a usable device-afforded width (B1+B2 > 0) so the device-log mask is
    well-defined.
    """
    cat = pd.read_parquet(TEMPLATE_CATALOG)
    b1b2 = pd.to_numeric(cat["t1b1_ms"], errors="coerce").fillna(0) + pd.to_numeric(
        cat["t1b2_ms"], errors="coerce"
    ).fillna(0)
    cat = cat[b1b2 > 0]
    md = cat["mask_duration_ms"].astype(float)
    regimes = [cat[md < 500], cat[(md >= 500) & (md < 2000)], cat[md >= 2000]]
    per = max(n // 3, 1)
    parts = [r.sample(min(per, len(r)), random_state=seed) for r in regimes if len(r)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def regime_of(dur_ms: float) -> str:
    """Bucket a therapy duration (refined label width, ms) into short/med/long."""
    return "short" if dur_ms < 500 else ("med" if dur_ms < 2000 else "long")


def build_masks(
    n: int,
    onsets: list[int],
    devlog_ms: float,
    model,
    contam: np.ndarray,
    *,
    device: str,
    margin: int,
) -> dict[str, np.ndarray]:
    """Construct the four strategy masks (margins zeroed to match the pipeline).

    ``devlog_ms`` is the device-afforded B1+B2 burst duration -- what a real
    device-log mask can use (the device records no therapy duration). The U-Net
    mask is the RAW postprocessed prediction with zero conditioning: metadata-free
    and NOT width-calibrated.
    """
    proba = predict_array(
        model, contam, np.zeros(N_COND_FEATURES, np.float32), device=device
    )
    unet = postprocess_predictions((proba > 0.5).astype(np.float32)).astype(bool)
    masks = {
        "none": np.zeros(n, dtype=bool),
        "blank5s": L.mask_fixed_blank(n, onsets, BLANK_S, SAMPLING_RATE),
        "devlog": L.mask_device_log(n, onsets, devlog_ms, SAMPLING_RATE),
        "unet": unet,
    }
    for m in masks.values():
        m[:margin] = False
        m[n - margin :] = False
    return masks


def run_trial(
    model,
    clean_uv: np.ndarray,
    template: np.ndarray,
    devlog_ms: float,
    *,
    device: str,
    injection: str = "replace",
    core_frac: float = 0.5,
) -> list[dict] | None:
    """Inject, clean four ways, and score PSD + coherence bias vs clean truth.

    ``devlog_ms`` = device-afforded B1+B2 width for the device-log mask.
    ``injection`` selects the contamination model (review Major 1): ``"replace"``
    overwrites the whole artifact window (saturation destroys signal),
    ``"split"`` overwrites only the saturation core (the leading ``core_frac`` of
    the peak envelope) and ADDS the recovery tail onto intact ECoG, so masking
    the tail now has a real spectral cost. Returns one record per
    (strategy, metric, band), or None if the file is too short to host >= 2
    well-separated events.
    """
    c, n = clean_uv.shape
    margin = int(BOUNDARY_MARGIN_S * SAMPLING_RATE)
    hop = int(SPACING_S * SAMPLING_RATE)
    first = margin + SAMPLING_RATE  # 1 s past the margin
    # each event occupies max(template length, 5 s blank) of the trace
    occ = max(template.shape[1], int(BLANK_S * SAMPLING_RATE))
    onsets = [first + i * hop for i in range(N_EVENTS)]
    onsets = [o for o in onsets if o + occ < n - margin]
    if len(onsets) < 2:
        return None

    if injection == "split":
        core_mask = L.saturation_core_mask(template, core_frac)
        contam = L.inject_artifact_split(clean_uv, template, onsets, core_mask)
    else:
        contam = L.inject_artifact(clean_uv, template, onsets)
    masks = build_masks(
        n, onsets, devlog_ms, model, contam, device=device, margin=margin
    )
    clean_mask = np.zeros(n, dtype=bool)
    pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]

    # Truth band powers (per channel) and band coherence (per pair) from CLEAN.
    truth_bp: dict[tuple[int, str], float] = {}
    for ch in range(c):
        f, p, _ = L.gapped_welch_psd(
            clean_uv[ch], clean_mask, SAMPLING_RATE, NPERSEG, NOVERLAP
        )
        for band, (lo, hi) in L.CANONICAL_BANDS.items():
            truth_bp[(ch, band)] = L.band_power(f, p, lo, hi)
    truth_coh: dict[tuple[int, int, str], float] = {}
    for i, j in pairs:
        f, coh, _ = L.coherence_gapped(
            clean_uv[i], clean_uv[j], clean_mask, SAMPLING_RATE, NPERSEG, NOVERLAP
        )
        for band, (lo, hi) in L.CANONICAL_BANDS.items():
            sel = (f >= lo) & (f < hi)
            truth_coh[(i, j, band)] = (
                float(np.nanmean(coh[sel])) if sel.any() else np.nan
            )

    records: list[dict] = []
    for strat, m in masks.items():
        dd = L.data_discarded_fraction(m)
        # PSD bias: mean over channels of log10(bp_strategy / bp_truth) per band.
        psd_cache = [
            L.gapped_welch_psd(contam[ch], m, SAMPLING_RATE, NPERSEG, NOVERLAP)
            for ch in range(c)
        ]
        for band, (lo, hi) in L.CANONICAL_BANDS.items():
            vals = []
            for ch in range(c):
                f, p, nw = psd_cache[ch]
                if nw == 0:
                    continue
                vals.append(
                    L.psd_log_bias(L.band_power(f, p, lo, hi), truth_bp[(ch, band)])
                )
            records.append(
                dict(
                    strategy=strat,
                    metric="psd",
                    band=band,
                    value=float(np.mean(vals)) if vals else np.nan,
                    data_discarded=dd,
                )
            )
        # Coherence bias: mean over pairs of (coh_strategy - coh_truth) per band.
        coh_cache = {
            (i, j): L.coherence_gapped(
                contam[i], contam[j], m, SAMPLING_RATE, NPERSEG, NOVERLAP
            )
            for (i, j) in pairs
        }
        for band, (lo, hi) in L.CANONICAL_BANDS.items():
            vals = []
            for i, j in pairs:
                f, coh, nw = coh_cache[(i, j)]
                if nw == 0:
                    continue
                sel = (f >= lo) & (f < hi)
                cs = float(np.nanmean(coh[sel])) if sel.any() else np.nan
                vals.append(cs - truth_coh[(i, j, band)])
            records.append(
                dict(
                    strategy=strat,
                    metric="coh",
                    band=band,
                    value=float(np.nanmean(vals)) if vals else np.nan,
                    data_discarded=dd,
                )
            )
    return records


def run_trials(
    model,
    n_trials: int,
    seed: int,
    device: str,
    checkpoint_every: int = 20,
    csv_path: Path | None = None,
    injection: str = "replace",
    core_frac: float = 0.5,
) -> pd.DataFrame:
    """Run ``n_trials`` injection trials (one clean file x one template each).

    Pairs the i-th sampled clean file with the i-th sampled template (templates
    recycled if fewer than n_trials). ``injection``/``core_frac`` select the
    contamination model (see :func:`run_trial`). Writes a partial CSV every
    ``checkpoint_every`` trials (OOM/kill safety per the global policy).
    """
    from lgs_db import read_dat, to_microvolts
    from tqdm.auto import tqdm

    clean = sample_clean_files(n_trials, seed)
    tmpls = sample_templates(max(n_trials, 24), seed + 1)
    rows: list[dict] = []
    done = 0
    for i in tqdm(range(len(clean)), desc=f"trials[{injection}]"):
        crow = clean.iloc[i]
        trow = tmpls.iloc[i % len(tmpls)]
        try:
            cuv = to_microvolts(read_dat(str(crow["file_path"]))).astype(np.float32)
            if cuv.shape[0] != 4:
                continue
            tmpl, b1b2_ms, md_ms = extract_template(trow)
            recs = run_trial(
                model,
                cuv,
                tmpl,
                b1b2_ms,
                device=device,
                injection=injection,
                core_frac=core_frac,
            )
        except Exception as exc:  # skip unreadable files; keep going
            print(f"  skip trial {i}: {exc}")
            continue
        if recs is None:
            continue
        regime = regime_of(md_ms)
        for r in recs:
            r.update(
                trial=i,
                dur_ms=md_ms,
                devlog_ms=b1b2_ms,
                regime=regime,
                injection=injection,
            )
        rows.extend(recs)
        done += 1
        if csv_path is not None and done % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    out = pd.DataFrame(rows)
    if csv_path is not None:
        out.to_csv(csv_path, index=False)
    return out


def _median(
    df: pd.DataFrame, metric: str, band: str, strat: str, absolute: bool
) -> float:
    """Median (optionally of |value|) for one strategy/metric/band slice."""
    sub = df[(df.metric == metric) & (df.band == band) & (df.strategy == strat)]
    v = sub.value.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if absolute:
        v = np.abs(v)
    return float(np.median(v)) if len(v) else float("nan")


def _dd_regime(df: pd.DataFrame, strat: str, regime: str) -> float:
    """Median data-discarded fraction for a strategy within a duration regime."""
    x = df[(df.strategy == strat) & (df.regime == regime)].data_discarded
    return float(x.median()) if len(x) else float("nan")


def _worst_band(df: pd.DataFrame, metric: str, strat: str) -> float:
    """Max over the named bands of the median |bias| (the most-affected band).

    Broadband averages dilute a time-localized artifact, so the physically
    meaningful claim is per-band: the artifact distorts (and the U-Net recovers)
    the worst-hit band. Excludes the 'broadband' aggregate.
    """
    bands = [b for b in L.CANONICAL_BANDS if b != "broadband"]
    return max(_median(df, metric, b, strat, True) for b in bands)


def report_pilot(df: pd.DataFrame) -> bool:
    """Print pilot acceptance metrics (|bias|, stratified); return all-gates-pass.

    Saturation can both inflate (oscillatory artifacts) and deflate (flatline
    rails) band power, so the "is it distorted / recovered" gates use the median
    of |bias|. The data-efficiency win is checked on the short-therapy regime,
    where naive 5 s blanking is most wasteful.
    """
    strategies = ["none", "blank5s", "devlog", "unet"]
    print("\n=== PILOT ACCEPTANCE ===")
    print(
        f"trials={df.trial.nunique()}  "
        f"by_regime={df.groupby('regime').trial.nunique().to_dict()}"
    )
    print("\nBroadband bias (median |.| ; signed):")
    for s in strategies:
        print(
            f"  {s:8s} |psd|={_median(df,'psd','broadband',s,True):.3f} "
            f"({_median(df,'psd','broadband',s,False):+.3f})   "
            f"|coh|={_median(df,'coh','broadband',s,True):.3f} "
            f"({_median(df,'coh','broadband',s,False):+.3f})"
        )
    print("\nWorst-band |bias| (max over delta..gamma):")
    for s in strategies:
        print(
            f"  {s:8s} psd={_worst_band(df, 'psd', s):.3f}   "
            f"coh={_worst_band(df, 'coh', s):.3f}"
        )
    print("\nData discarded (median) overall + per regime:")
    overall = df.groupby("strategy").data_discarded.median().to_dict()
    regimes = [r for r in ["short", "med", "long"] if (df.regime == r).any()]
    for s in strategies:
        per = "  ".join(f"{r}={_dd_regime(df, s, r):.3f}" for r in regimes)
        print(f"  {s:8s} overall={overall.get(s, float('nan')):.3f}   {per}")

    # The honest claim is COMPARATIVE / Pareto, not absolute-zero bias: among
    # low-data strategies the U-Net has the lowest residual (beats the real
    # device-log mask, whose B1+B2 width under-covers the recovery tail), while
    # discarding far less than 5 s blanking. Gates check the worst-affected band.
    eps = 1e-9
    gates = {
        "no-mask distorts PSD (worst |bias|>0.25)": _worst_band(df, "psd", "none")
        > 0.25,
        "U-Net <= device-log PSD residual (worst band)": _worst_band(df, "psd", "unet")
        <= _worst_band(df, "psd", "devlog") + eps,
        "U-Net <= device-log coherence residual": _worst_band(df, "coh", "unet")
        <= _worst_band(df, "coh", "devlog") + eps,
        "U-Net << 5s-blank on data discarded (<0.5x)": overall.get("unet", 1)
        < 0.5 * overall.get("blank5s", 1),
        "no-mask perturbs coherence (worst |bias|>0.05)": _worst_band(df, "coh", "none")
        > 0.05,
    }
    print()
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    return all(gates.values())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Median and 95% bootstrap CI of a 1-D array (NaNs dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array(
        [np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    )
    return (
        float(np.median(v)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (strategy, metric, band, regime + 'all'): median bias, IQR, n."""
    out = []
    for regime in list(df.regime.unique()) + ["all"]:
        sub = df if regime == "all" else df[df.regime == regime]
        for (strat, metric, band), g in sub.groupby(["strategy", "metric", "band"]):
            v = g.value.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            out.append(
                dict(
                    regime=regime,
                    strategy=strat,
                    metric=metric,
                    band=band,
                    median=float(np.median(v)) if len(v) else np.nan,
                    q25=float(np.percentile(v, 25)) if len(v) else np.nan,
                    q75=float(np.percentile(v, 75)) if len(v) else np.nan,
                    data_discarded=float(g.data_discarded.median()),
                    n=len(v),
                )
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STRAT_ORDER = ["none", "blank5s", "devlog", "unet"]
STRAT_LABELS = {
    "none": "No mask",
    "blank5s": "5 s blank",
    "devlog": "Device-log",
    "unet": "U-Net (ours)",
}
STRAT_COLORS = {
    "none": "#b03a3a",
    "blank5s": "#b07d1a",
    "devlog": "#5a7a9a",
    "unet": "#2f6b49",
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def _grouped_band_bars(
    ax, df: pd.DataFrame, metric: str, title: str, ylab: str, legend: bool
) -> None:
    """Grouped bars of signed median bias: x = bands, one bar per strategy."""
    x = np.arange(len(BAND_ORDER))
    w = 0.2
    for k, s in enumerate(STRAT_ORDER):
        meds = []
        for b in BAND_ORDER:
            v = df[
                (df.strategy == s) & (df.metric == metric) & (df.band == b)
            ].value.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            meds.append(float(np.median(v)) if len(v) else np.nan)
        ax.bar(
            x + (k - 1.5) * w,
            meds,
            width=w,
            color=STRAT_COLORS[s],
            label=STRAT_LABELS[s],
        )
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:3] for b in BAND_ORDER])
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold")
    if legend:
        ax.legend(fontsize=6.5, ncol=2, frameon=False)


def build_figure(df: pd.DataFrame, summary: pd.DataFrame):
    """Assemble the 4-panel downstream-bias figure.

    (a) data discarded per strategy; (b) per-band PSD bias; (c) money panel:
    broadband |bias| vs % data discarded with bootstrap CIs; (d) per-band
    coherence bias. Bias panels are signed medians; the money panel uses |bias|.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from style import apply_style

        apply_style()
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    allr = summary[summary.regime == "all"]

    # (a) data discarded
    dd = [
        float(allr[allr.strategy == s].data_discarded.iloc[0]) * 100
        for s in STRAT_ORDER
    ]
    ax_a.bar(
        [STRAT_LABELS[s] for s in STRAT_ORDER],
        dd,
        color=[STRAT_COLORS[s] for s in STRAT_ORDER],
    )
    ax_a.set_ylabel("Data discarded (%)")
    ax_a.set_title("(a) Data discarded", loc="left", fontweight="bold")
    ax_a.tick_params(axis="x", rotation=18)

    # (b) PSD bias per band; (d) coherence bias per band
    _grouped_band_bars(
        ax_b,
        df,
        "psd",
        "(b) Spectral (PSD) bias",
        "Bias (log$_{10}$ ratio)",
        legend=True,
    )
    _grouped_band_bars(
        ax_d, df, "coh", "(d) Coherence bias", "Coherence bias", legend=False
    )

    # (c) money panel: broadband |bias| vs data discarded. Staggered labels so
    # the near-coincident U-Net / Device-log points don't collide.
    label_off = {
        "none": (9, -3),
        "blank5s": (-4, 9),
        "devlog": (4, 11),
        "unet": (4, -16),
    }
    label_ha = {"none": "left", "blank5s": "right", "devlog": "left", "unet": "left"}
    markers = {"none": "o", "blank5s": "o", "devlog": "o", "unet": "*"}
    sizes = {"none": 9, "blank5s": 9, "devlog": 9, "unet": 16}
    for s in STRAT_ORDER:
        v = (
            df[(df.strategy == s) & (df.metric == "psd") & (df.band == "broadband")]
            .value.abs()
            .to_numpy()
        )
        m, lo, hi = bootstrap_ci(v)
        x = float(allr[allr.strategy == s].data_discarded.iloc[0]) * 100
        ax_c.errorbar(
            x,
            m,
            yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
            fmt=markers[s],
            ms=sizes[s],
            color=STRAT_COLORS[s],
            capsize=3,
            zorder=3,
        )
        ax_c.annotate(
            STRAT_LABELS[s],
            (x, m),
            textcoords="offset points",
            xytext=label_off[s],
            fontsize=8,
            color=STRAT_COLORS[s],
            ha=label_ha[s],
            fontweight="bold" if s == "unet" else "normal",
        )
    ax_c.set_xlabel("Data discarded (%)")
    ax_c.set_ylabel("Broadband |bias| (log$_{10}$)")
    ax_c.set_title("(c) Bias vs data discarded", loc="left", fontweight="bold")
    ax_c.annotate(
        "better",
        xy=(0.04, 0.06),
        xytext=(0.30, 0.06),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7,
        color="#666",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    ax_c.margins(x=0.24, y=0.28)

    fig.suptitle(
        "Artifact masking vs downstream spectral fidelity "
        "(real RNS artifact injected into clean ECoG)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_figure(fig, name: str = "fig_downstream_bias") -> Path:
    """Write the figure to manuscript/figures as PDF + PNG + SVG."""
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / f"{name}.svg", bbox_inches="tight")
    return png


def main() -> None:
    """Render the figure, run the pilot, or run a smoke check."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="1-trial integrity check")
    ap.add_argument("--pilot", type=int, default=0, help="run N pilot trials + gate")
    ap.add_argument("--n-trials", type=int, default=0, help="full run trial count")
    ap.add_argument(
        "--injection",
        choices=["replace", "split", "both"],
        default="replace",
        help="contamination model: replace (saturation), split (core + additive "
        "recovery tail; review Major 1), or both for a robustness comparison",
    )
    ap.add_argument(
        "--core-frac",
        type=float,
        default=0.5,
        help="split model: leading peak-envelope fraction treated as the "
        "saturation core (rest of the artifact is added onto intact ECoG)",
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the figure from the cached trial CSV (no model/inference)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device or _device()

    if args.render_only:
        df = pd.read_csv(OUT_CSV)
        summary = summarize(df)
        summary.to_csv(
            REPO_DIR / "outputs" / "downstream_bias_summary.csv", index=False
        )
        out = save_figure(build_figure(df, summary))
        print(f"Re-rendered -> {out}  ({df.trial.nunique()} trials)")
        return

    if args.smoke:
        model = load_deployed_model(device=dev)
        clean = sample_clean_files(1, args.seed)
        tmpl_rows = sample_templates(3, args.seed)
        from lgs_db import read_dat, to_microvolts

        cuv = to_microvolts(read_dat(str(clean.iloc[0]["file_path"]))).astype(
            np.float32
        )
        tmpl, b1b2_ms, md_ms = extract_template(tmpl_rows.iloc[0])
        assert cuv.shape[0] == 4 and tmpl.shape[0] == 4, "expected 4 channels"
        assert tmpl.shape[1] >= 1 and np.isfinite(tmpl).all(), "bad template"
        recs = run_trial(model, cuv, tmpl, b1b2_ms, device=dev)
        assert recs is not None, "smoke file too short for a trial"
        # Also exercise the split (core-replace + additive-tail) model end-to-end.
        core_mask = L.saturation_core_mask(tmpl, 0.5)
        recs_split = run_trial(model, cuv, tmpl, b1b2_ms, device=dev, injection="split")
        assert recs_split is not None, "split smoke too short"
        assert all(np.isfinite(r["value"]) for r in recs_split if r["band"] != "gamma")
        df = pd.DataFrame(recs)
        none_bb = df[
            (df.strategy == "none") & (df.band == "broadband") & (df.metric == "psd")
        ].value.iloc[0]
        unet_bb = df[
            (df.strategy == "unet") & (df.band == "broadband") & (df.metric == "psd")
        ].value.iloc[0]
        dd = df.groupby("strategy").data_discarded.first().to_dict()
        print(
            f"smoke OK: tmpl_len={tmpl.shape[1]} core_n={int(core_mask.sum())} "
            f"b1b2={b1b2_ms:.0f}ms md={md_ms:.0f}ms "
            f"records={len(df)}(+{len(recs_split)} split) psd_bias[none]={none_bb:+.2f} "
            f"psd_bias[unet]={unet_bb:+.2f} "
            f"discarded={ {k: round(v, 3) for k, v in dd.items()} }"
        )
        return

    if args.pilot:
        model = load_deployed_model(device=dev)
        models = ["replace", "split"] if args.injection == "both" else [args.injection]
        results: dict[str, bool] = {}
        for inj in models:
            suffix = "" if inj == "replace" else f"_{inj}"
            df = run_trials(
                model,
                args.pilot,
                args.seed,
                dev,
                csv_path=REPO_DIR / "outputs" / f"downstream_bias_pilot{suffix}.csv",
                injection=inj,
                core_frac=args.core_frac,
            )
            print(
                f"\n########## INJECTION = {inj} "
                f"(core_frac={args.core_frac}) ##########"
            )
            results[inj] = report_pilot(df)
        for inj in models:
            print(
                f"\nGATE[{inj}]:",
                (
                    "PASS — verdict holds"
                    if results[inj]
                    else "FAIL — reframe (see spec)"
                ),
            )
        return

    if args.n_trials:
        model = load_deployed_model(device=dev)
        models = ["replace", "split"] if args.injection == "both" else [args.injection]
        for inj in models:
            suffix = "" if inj == "replace" else f"_{inj}"
            csv_path = (
                OUT_CSV
                if inj == "replace"
                else REPO_DIR / "outputs" / f"downstream_bias_trials{suffix}.csv"
            )
            df = run_trials(
                model,
                args.n_trials,
                args.seed,
                dev,
                csv_path=csv_path,
                injection=inj,
                core_frac=args.core_frac,
            )
            summary = summarize(df)
            summary.to_csv(
                REPO_DIR / "outputs" / f"downstream_bias_summary{suffix}.csv",
                index=False,
            )
            out = save_figure(
                build_figure(df, summary), name=f"fig_downstream_bias{suffix}"
            )
            print(f"\nSaved -> {out}  ({df.trial.nunique()} trials, injection={inj})")
            allr = summary[summary.regime == "all"]
            print(f"HEADLINE (broadband, injection={inj}):")
            for s in STRAT_ORDER:
                r = allr[
                    (allr.strategy == s)
                    & (allr.metric == "psd")
                    & (allr.band == "broadband")
                ]
                print(
                    f"  {s:8s} psd_bias={float(r['median'].iloc[0]):+.3f}  "
                    f"discarded={float(r['data_discarded'].iloc[0]) * 100:.1f}%"
                )
        return

    raise SystemExit("Use --smoke, --pilot N, or --n-trials N")


if __name__ == "__main__":
    main()
