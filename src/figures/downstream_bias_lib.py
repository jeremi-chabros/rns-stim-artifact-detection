"""Pure signal math for the downstream-bias injection figure.

Torch-free (numpy + scipy only) so it imports fast and is unit-testable with
synthetic signals. All spectral estimators are gap-aware: masked samples are
*excised* (never interpolated), and Welch averaging runs only over windows that
lie entirely inside the surviving (unmasked) regions.

See docs/superpowers/specs/2026-06-18-downstream-bias-figure-design.md.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

# numpy >=2.0 renamed trapz -> trapezoid (trapz removed in recent releases).
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Canonical EEG bands (Hz); broadband spans the analysis range. All < Nyquist
# (125 Hz at the 250 Hz RNS sampling rate).
CANONICAL_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "broadband": (1.0, 100.0),
}


def data_discarded_fraction(mask: np.ndarray) -> float:
    """Fraction of samples a strategy removes (mask True == discarded)."""
    return float(np.mean(np.asarray(mask, dtype=bool)))


def psd_log_bias(bp_strategy: float, bp_truth: float, eps: float = 1e-20) -> float:
    """Spectral bias as a base-10 log ratio of band powers (0 == unbiased)."""
    return float(np.log10((bp_strategy + eps) / (bp_truth + eps)))


def inject_artifact(
    clean: np.ndarray,
    template: np.ndarray,
    onset_samples: list[int],
    baseline_pad: int = 250,
) -> np.ndarray:
    """Inject a real artifact by REPLACING each onset window (saturation model).

    Saturation destroys the underlying signal rather than summing with it, so we
    overwrite ``clean[:, onset:onset+L]`` with ``local_baseline + template``,
    where ``template`` is the artifact's deviation from its *source* pre-onset
    baseline and ``local_baseline`` is the clean trace's median over the
    ``baseline_pad`` samples just before the onset (DC-aligning the artifact to
    the host trace, so window edges stay continuous). Trailing recovered ECoG in
    the template (deviation ~0) thus becomes host-baseline-level ECoG -> benign.

    Args:
        clean: (C, N) clean signal (not mutated).
        template: (C, L) artifact deviation from its source pre-onset baseline.
        onset_samples: sample indices at which to place the artifact.
        baseline_pad: samples before each onset used to estimate local baseline.

    Returns:
        (C, N) contaminated copy. The template is truncated past the trace end.
    """
    out = np.array(clean, dtype=np.float32, copy=True)
    n = out.shape[1]
    length = template.shape[1]
    assert template.shape[0] == out.shape[0], "channel-count mismatch"
    global_base = np.median(out, axis=1, keepdims=True)
    for s in onset_samples:
        s = int(s)
        if s >= n:
            continue
        e = min(n, s + length)
        lo = max(0, s - baseline_pad)
        local_base = (
            np.median(out[:, lo:s], axis=1, keepdims=True) if s > lo else global_base
        )
        out[:, s:e] = local_base + template[:, : e - s]
    return out


def saturation_core_mask(template: np.ndarray, core_frac: float = 0.5) -> np.ndarray:
    """Per-sample saturation-core mask for an artifact template.

    A sample is in the rail core (amplifier saturated; no recoverable neural
    signal) where the across-channel envelope ``max_c |template[c]|`` reaches at
    least ``core_frac`` of its peak; the rest is the additive recovery. Defined by
    amplitude rather than position, so it handles both a short leading blank and a
    long pulse train whose rail is not at the onset edge.

    Args:
        template: (C, L) artifact deviation from baseline.
        core_frac: fraction of the peak envelope defining the rail threshold.

    Returns:
        (L,) boolean mask; True == saturation core (replacement region).
    """
    env = np.max(np.abs(np.asarray(template, dtype=float)), axis=0)
    peak = float(env.max()) if env.size else 0.0
    return env >= core_frac * peak


def inject_artifact_split(
    clean: np.ndarray,
    template: np.ndarray,
    onset_samples: list[int],
    core_mask: np.ndarray,
    baseline_pad: int = 250,
) -> np.ndarray:
    """Inject with a saturation CORE (replacement) + recovery (additive).

    The honest stress test for the downstream-bias claim (review Major 1):
    samples where ``core_mask`` is True overwrite the host trace (the saturation
    model of :func:`inject_artifact`), while the remaining recovery samples are
    ADDED onto the intact host signal (a settling transient summed with surviving
    neural activity), so masking the recovery now has a real spectral cost. An
    all-True mask reduces exactly to :func:`inject_artifact`; an all-False mask is
    fully additive.

    Args:
        clean: (C, N) clean signal (not mutated).
        template: (C, L) artifact deviation from its source pre-onset baseline.
        onset_samples: sample indices at which to place the artifact.
        core_mask: (L,) bool; True samples are replaced (rail), False are added.
        baseline_pad: samples before each onset used to estimate local baseline.

    Returns:
        (C, N) contaminated copy. The template is truncated past the trace end.
    """
    out = np.array(clean, dtype=np.float32, copy=True)
    n = out.shape[1]
    length = template.shape[1]
    assert template.shape[0] == out.shape[0], "channel-count mismatch"
    cmask = np.asarray(core_mask, dtype=bool)
    assert cmask.shape[0] == length, "core_mask length must match template"
    global_base = np.median(out, axis=1, keepdims=True)
    for s in onset_samples:
        s = int(s)
        if s >= n:
            continue
        e = min(n, s + length)
        w = e - s
        lo = max(0, s - baseline_pad)
        local_base = (
            np.median(out[:, lo:s], axis=1, keepdims=True) if s > lo else global_base
        )
        idx = np.arange(s, e)
        cw = cmask[:w]
        core_idx, rec_idx = idx[cw], idx[~cw]
        if core_idx.size:  # rail: replacement (destroys underlying signal)
            out[:, core_idx] = local_base + template[:, :w][:, cw]
        if rec_idx.size:  # recovery: additive transient on intact signal
            out[:, rec_idx] += template[:, :w][:, ~cw]
    return out


def _onset_mask(n: int, onset_samples: list[int], width: int) -> np.ndarray:
    """Boolean mask True over [onset, onset+width) for each onset, clipped to n."""
    m = np.zeros(n, dtype=bool)
    for s in onset_samples:
        s = int(s)
        m[max(0, s) : min(n, s + width)] = True
    return m


def mask_fixed_blank(
    n: int, onset_samples: list[int], blank_s: float, sr: int
) -> np.ndarray:
    """Naive fixed-window blanking: blank ``blank_s`` seconds from each onset."""
    return _onset_mask(n, onset_samples, int(round(blank_s * sr)))


def mask_device_log(
    n: int, onset_samples: list[int], dur_ms: float, sr: int
) -> np.ndarray:
    """Device-log mask: logged onset + logged therapy duration (``dur_ms``)."""
    return _onset_mask(n, onset_samples, max(int(round(dur_ms / 1000.0 * sr)), 1))


def surviving_windows(mask: np.ndarray, nperseg: int, noverlap: int) -> list[int]:
    """Start indices of length-``nperseg`` windows lying fully in unmasked data.

    Args:
        mask: (N,) bool; True == excised.
        nperseg: window length in samples.
        noverlap: overlap in samples (hop == nperseg - noverlap).

    Returns:
        List of window start indices ``s`` where ``mask[s:s+nperseg]`` is all
        False.
    """
    n = len(mask)
    hop = max(nperseg - noverlap, 1)
    return [
        s for s in range(0, n - nperseg + 1, hop) if not mask[s : s + nperseg].any()
    ]


def gapped_welch_psd(
    sig: np.ndarray, mask: np.ndarray, sr: int, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Welch PSD (density, Hann, one-sided) over gap-free windows only.

    Args:
        sig: (N,) single-channel signal.
        mask: (N,) bool; True samples are excised (never interpolated).
        sr: sampling rate (Hz).
        nperseg, noverlap: Welch window/overlap in samples.

    Returns:
        ``(freqs, psd, n_windows)``. ``psd`` is all-NaN when ``n_windows == 0``.
    """
    win = get_window("hann", nperseg)
    scale = 1.0 / (sr * float((win**2).sum()))
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
    starts = surviving_windows(mask, nperseg, noverlap)
    if not starts:
        return freqs, np.full(freqs.shape, np.nan), 0
    acc = np.zeros(freqs.shape)
    for s in starts:
        seg = sig[s : s + nperseg].astype(float)
        seg = seg - seg.mean()
        spec = np.fft.rfft(seg * win)
        p = (np.abs(spec) ** 2) * scale
        p[1:-1] *= 2.0  # one-sided density
        acc += p
    return freqs, acc / len(starts), len(starts)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate the PSD over [lo, hi) Hz with the trapezoidal rule."""
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return float("nan")
    return float(_trapz(psd[sel], freqs[sel]))


def coherence_gapped(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, sr: int, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Magnitude-squared coherence over windows where neither channel is masked.

    Args:
        x, y: (N,) single-channel signals.
        mask: (N,) bool; True samples excised from both channels.
        sr, nperseg, noverlap: as in :func:`gapped_welch_psd`.

    Returns:
        ``(freqs, coherence, n_windows)``; coherence is all-NaN if no window.
    """
    win = get_window("hann", nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
    starts = surviving_windows(mask, nperseg, noverlap)
    if not starts:
        return freqs, np.full(freqs.shape, np.nan), 0
    sxx = np.zeros(freqs.shape)
    syy = np.zeros(freqs.shape)
    sxy = np.zeros(freqs.shape, dtype=complex)
    for s in starts:
        xs = x[s : s + nperseg].astype(float)
        ys = y[s : s + nperseg].astype(float)
        xs = (xs - xs.mean()) * win
        ys = (ys - ys.mean()) * win
        fx = np.fft.rfft(xs)
        fy = np.fft.rfft(ys)
        sxx += np.abs(fx) ** 2
        syy += np.abs(fy) ** 2
        sxy += fx * np.conj(fy)
    coh = (np.abs(sxy) ** 2) / (sxx * syy + 1e-20)
    return freqs, coh, len(starts)
