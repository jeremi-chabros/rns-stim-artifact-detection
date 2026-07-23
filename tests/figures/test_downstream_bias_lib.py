"""Unit tests for downstream_bias_lib (torch-free signal math)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "figures"))
import downstream_bias_lib as L  # noqa: E402


def test_data_discarded_fraction():
    m = np.zeros(100, dtype=bool)
    m[:25] = True
    assert L.data_discarded_fraction(m) == 0.25
    assert L.data_discarded_fraction(np.zeros(10, dtype=bool)) == 0.0


def test_psd_log_bias():
    assert L.psd_log_bias(5.0, 5.0) == 0.0
    assert abs(L.psd_log_bias(50.0, 5.0) - 1.0) < 1e-9  # 10x -> +1 decade
    assert abs(L.psd_log_bias(0.5, 5.0) + 1.0) < 1e-9  # 0.1x -> -1 decade


def test_canonical_bands_cover_eeg():
    assert set(L.CANONICAL_BANDS) == {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "broadband",
    }
    for lo, hi in L.CANONICAL_BANDS.values():
        assert 0 < lo < hi <= 125  # below Nyquist at 250 Hz


def test_inject_artifact_replaces_window():
    # Saturation REPLACES the signal: window <- local baseline + deviation.
    clean = np.full((2, 100), 10.0, dtype=np.float32)
    clean[:, 25] = 999.0  # an underlying signal spike inside the window
    tmpl = np.zeros((2, 10), dtype=np.float32)  # zero deviation == flatline rail
    out = L.inject_artifact(clean, tmpl, [20], baseline_pad=10)
    assert out.shape == clean.shape
    assert np.allclose(out[:, 20:30], 10.0)  # window == local baseline (10)
    assert out[0, 25] == 10.0  # underlying spike destroyed (replaced, not summed)
    assert np.allclose(out[:, :20], 10.0) and np.allclose(out[:, 30:], 10.0)
    assert clean[0, 25] == 999.0  # input not mutated


def test_inject_artifact_adds_deviation_on_local_baseline():
    clean = np.full((1, 100), 5.0, dtype=np.float32)
    tmpl = np.full((1, 10), 50.0, dtype=np.float32)  # +50 deviation from baseline
    out = L.inject_artifact(clean, tmpl, [40], baseline_pad=10)
    assert np.allclose(out[:, 40:50], 55.0)  # 5 baseline + 50 deviation


def test_inject_artifact_truncates_at_end():
    clean = np.zeros((2, 100), dtype=np.float32)
    tmpl = np.full((2, 10), 7.0, dtype=np.float32)
    out = L.inject_artifact(clean, tmpl, [95], baseline_pad=10)
    assert np.allclose(out[:, 95:100], 7.0)  # baseline 0 + 7; only 5 samples fit
    assert out[:, :95].sum() == 0.0


def test_mask_fixed_blank():
    m = L.mask_fixed_blank(1000, [100], blank_s=1.0, sr=250)  # 1 s == 250 samples
    assert m.dtype == bool
    assert m[100:350].all() and m.sum() == 250
    assert not m[:100].any() and not m[350:].any()


def test_mask_device_log_clips_at_end():
    m = L.mask_device_log(300, [200], dur_ms=1000.0, sr=250)  # wants 250, clipped
    assert m[200:300].all() and m.sum() == 100


def test_mask_device_log_short_vs_blank():
    # 110 ms therapy masks far less than a 5 s blank
    short = L.mask_device_log(5000, [500], dur_ms=110.0, sr=250)
    blank = L.mask_fixed_blank(5000, [500], blank_s=5.0, sr=250)
    assert short.sum() < blank.sum()


def _sine(freq, n, sr, amp=1.0):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_surviving_windows_skips_masked():
    mask = np.zeros(2000, dtype=bool)
    mask[800:1200] = True  # block the middle
    starts = L.surviving_windows(mask, nperseg=256, noverlap=128)
    assert all((not mask[s : s + 256].any()) for s in starts)
    assert len(starts) > 0


def test_gapped_welch_psd_peaks_at_sine():
    sr, n = 250, 5000
    sig = _sine(20.0, n, sr)
    f, p, nw = L.gapped_welch_psd(sig, np.zeros(n, dtype=bool), sr, 512, 256)
    assert nw > 0 and f.shape == p.shape
    assert abs(f[np.argmax(p)] - 20.0) < 1.0  # peak at the sine frequency


def test_gapped_welch_psd_survives_masking():
    sr, n = 250, 5000
    sig = _sine(20.0, n, sr)
    mask = np.zeros(n, dtype=bool)
    mask[2000:2600] = True  # excise a chunk
    f, p, nw = L.gapped_welch_psd(sig, mask, sr, 512, 256)
    assert nw > 0 and abs(f[np.argmax(p)] - 20.0) < 1.0


def test_gapped_welch_psd_no_window_returns_nan():
    sr, n = 250, 600
    f, p, nw = L.gapped_welch_psd(np.zeros(n), np.ones(n, dtype=bool), sr, 512, 256)
    assert nw == 0 and np.isnan(p).all()


def test_band_power_flat_spectrum():
    freqs = np.linspace(0, 125, 126)  # 1 Hz spacing
    psd = np.ones_like(freqs)
    # Half-open band [8, 13) selects bins 8..12; trapezoid over those 5 points
    # integrates from 8 to 12 Hz == 4.0 (upper edge excluded to avoid
    # double-counting shared band boundaries).
    bp = L.band_power(freqs, psd, 8.0, 13.0)
    assert abs(bp - 4.0) < 0.5


def test_coherence_identical_signals_is_one():
    # Broadband signal so every frequency bin carries power (a sparse-spectrum
    # signal leaves most bins with ~0 power -> undefined coherence there).
    sr, n = 250, 8000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    f, c, nw = L.coherence_gapped(x, x.copy(), np.zeros(n, dtype=bool), sr, 512, 256)
    assert nw > 0 and np.nanmean(c) > 0.99


def test_coherence_independent_noise_is_low():
    sr, n = 250, 40000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    f, c, nw = L.coherence_gapped(x, y, np.zeros(n, dtype=bool), sr, 512, 256)
    assert np.nanmean(c) < 0.3  # many windows -> coherence collapses


def test_coherence_shared_component_high_in_band():
    sr, n = 250, 40000
    rng = np.random.default_rng(1)
    shared = _sine(20.0, n, sr)
    x = shared + rng.standard_normal(n)
    y = shared + rng.standard_normal(n)
    f, c, nw = L.coherence_gapped(x, y, np.zeros(n, dtype=bool), sr, 512, 256)
    band = (f >= 18) & (f < 22)
    # Peak in-band: the shared 20 Hz component (leakage spreads it over a few
    # bins) drives high coherence at its frequency; off-peak bins are noise-only.
    assert np.nanmax(c[band]) > 0.8  # spurious-coherence analogue


# ---------------------------------------------------------------------------
# Additive-tail injection model (review Major 1): high-amplitude artifact samples
# rail the amplifier (no recoverable neural signal -> REPLACEMENT), while
# low-amplitude recovery samples are intact neural signal plus a settling
# transient (-> ADDITIVE). The core is defined by AMPLITUDE, not position, so it
# handles both a short leading blank and a long pulse train (where the rail is
# not at the onset edge). A replacement-only model makes masking the recovery
# spectrally free, favoring wider masks by construction; split is the honest test.
# ---------------------------------------------------------------------------


def test_saturation_core_mask_leading_rail():
    # 40-sample rail at 100, then a 60-sample recovery at 10 (10% of peak).
    tmpl = np.zeros((1, 100), dtype=np.float32)
    tmpl[0, :40] = 100.0
    tmpl[0, 40:] = 10.0
    m = L.saturation_core_mask(tmpl, core_frac=0.5)  # thr=50
    assert m.dtype == bool and m.shape == (100,)
    assert m[:40].all() and not m[40:].any()
    # core_frac=0.05 -> thr=5: the 10-amplitude recovery also rails -> all core.
    assert L.saturation_core_mask(tmpl, core_frac=0.05).all()


def test_saturation_core_mask_constant_all_core():
    tmpl = np.full((1, 50), 3.0, dtype=np.float32)
    assert L.saturation_core_mask(tmpl, core_frac=0.5).all()


def test_saturation_core_mask_uses_max_across_channels():
    # ch0 rails for 5 samples, ch1 for 8 -> envelope (max over channels) rails for 8.
    tmpl = np.zeros((2, 20), dtype=np.float32)
    tmpl[0, :5] = 100.0
    tmpl[1, :8] = 100.0
    m = L.saturation_core_mask(tmpl, core_frac=0.5)
    assert m[:8].all() and not m[8:].any()


def test_saturation_core_mask_peak_not_leading():
    # Regression for the long-therapy case: the rail is in the MIDDLE, not at the
    # onset edge. A leading-run definition would wrongly return an empty core.
    tmpl = np.zeros((1, 60), dtype=np.float32)
    tmpl[0, :20] = 10.0  # pre-rail ramp (recovery-like)
    tmpl[0, 20:40] = 100.0  # rail
    tmpl[0, 40:] = 10.0  # recovery
    m = L.saturation_core_mask(tmpl, core_frac=0.5)
    assert not m[:20].any() and m[20:40].all() and not m[40:].any()


def test_inject_split_core_replaces_recovery_adds():
    clean = np.full((1, 100), 5.0, dtype=np.float32)
    clean[0, 35] = 999.0  # inside the core [30:40) -> destroyed (replacement)
    clean[0, 45] = 100.0  # inside the recovery [40:50) -> preserved (additive)
    tmpl = np.zeros((1, 20), dtype=np.float32)
    tmpl[0, :10] = 50.0  # core deviation
    tmpl[0, 10:] = 3.0  # recovery deviation
    core = L.saturation_core_mask(tmpl, core_frac=0.5)  # first 10 samples
    out = L.inject_artifact_split(clean, tmpl, [30], core, baseline_pad=10)
    assert out.shape == clean.shape
    # core: local baseline (5) + deviation (50), underlying spike obliterated
    assert np.allclose(out[0, 30:40], 55.0)
    assert out[0, 35] == 55.0
    # recovery: clean signal preserved + deviation added
    assert out[0, 40] == 8.0  # 5 + 3
    assert out[0, 45] == 103.0  # 100 + 3 (neural signal survives the recovery)
    assert np.allclose(out[0, :30], 5.0) and np.allclose(out[0, 50:], 5.0)
    assert clean[0, 35] == 999.0  # input not mutated


def test_inject_split_all_core_equals_replacement():
    # An all-True core mask -> identical to the pure-replacement inject_artifact.
    clean = np.full((2, 100), 5.0, dtype=np.float32)
    clean[:, 35] = 999.0
    tmpl = np.full((2, 20), 7.0, dtype=np.float32)
    core = np.ones(20, dtype=bool)
    a = L.inject_artifact_split(clean, tmpl, [30], core, baseline_pad=10)
    b = L.inject_artifact(clean, tmpl, [30], baseline_pad=10)
    assert np.allclose(a, b)


def test_inject_split_no_core_is_fully_additive():
    clean = np.full((1, 100), 5.0, dtype=np.float32)
    clean[0, 35] = 999.0  # fully-additive region -> preserved, not destroyed
    tmpl = np.zeros((1, 20), dtype=np.float32)
    tmpl[0, :10] = 50.0
    core = np.zeros(20, dtype=bool)
    out = L.inject_artifact_split(clean, tmpl, [30], core, baseline_pad=10)
    assert out[0, 30] == 55.0  # 5 + 50
    assert out[0, 35] == 1049.0  # 999 + 50 (additive preserves the underlying sample)


def test_inject_split_truncates_at_end():
    clean = np.zeros((1, 100), dtype=np.float32)
    tmpl = np.full((1, 10), 7.0, dtype=np.float32)
    core = np.ones(10, dtype=bool)
    out = L.inject_artifact_split(clean, tmpl, [95], core, baseline_pad=10)
    assert np.allclose(out[0, 95:100], 7.0)  # 5 core samples fit; rest off-end
