#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "scipy>=1.10",
#   "scikit-learn>=1.3",
#   "tqdm>=4.60",
#   "lgs-db",
#   "PyWavelets>=1.4",
#   "xgboost>=2.0",
#   "kymatio>=0.3",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Comparison baseline methods for stim artifact detection.

Each method exposes a standardized interface:
    .fit(neg_catalog_path)            — learn baseline stats from clean recordings
    .predict_file(dat_path, subject)  — return (n_samples,) float32 proba ∈ [0, 1]
    .save(path) / .load(path)         — persist fitted parameters

Methods:
    M0  AmplitudeThreshold   — z-score of absolute amplitude vs clean baseline
    M1  SpectralNoiseSub     — Audacity-style noise profile → STFT ratio → threshold
    M2  VAEReconstruction    — 1D Conv-VAE reconstruction error on clean data
    M3  ScatteringXGBoost    — kymatio scattering transform + XGBoost (supervised)

Usage:
    # Fit baseline stats
    uv run src/comparison_methods.py fit-m0 --neg-catalog data/neg_catalog.parquet

    # Predict a single file
    uv run src/comparison_methods.py predict-m0 --file /path/to.dat --subject 300-001

    # Evaluate M0 on validation split
    uv run src/comparison_methods.py eval-m0 --catalog data/stim_catalog.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter1d
from scipy.signal import ShortTimeFFT
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    _load_dat_channels,
    build_true_mask,
    load_annotations,
    post_process_mask,
)

BASELINE_DIR = Path("data/baselines")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaselineMethod(ABC):
    """Abstract interface for comparison baseline methods."""

    name: str = "base"

    @abstractmethod
    def fit(self, neg_catalog_path: Path | str, **kwargs) -> None:
        """Learn baseline statistics from clean (stim-disabled) recordings."""

    @abstractmethod
    def predict_file(
        self, dat_path: Path | str, *, subject: str | None = None
    ) -> np.ndarray:
        """Return per-sample artifact probabilities.

        Returns
        -------
        proba : np.ndarray, shape (n_samples,), dtype float32
            Values in [0, 1].  Higher = more likely artifact.
        """

    def save(self, path: Path | str) -> None:
        """Persist fitted parameters to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path | str) -> BaselineMethod:
        """Load a previously fitted instance from disk."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# M0 — Amplitude Threshold
# ---------------------------------------------------------------------------


class AmplitudeThreshold(BaselineMethod):
    """M0: Detect stim artifacts via amplitude z-scoring against clean baseline.

    Algorithm
    ---------
    Fit phase (from stim-disabled recordings per subject):
        1. Load each file → (4, n_samples) µV
        2. Compute median absolute amplitude and MAD per channel
        3. Aggregate across files → per-subject baseline {median, mad}

    Predict phase:
        1. Load test file → (4, n_samples) µV
        2. z = (|x| − median_baseline) / (mad_baseline + ε)  per channel
        3. Apply short-window max filter (catch transient onset/offset)
        4. Aggregate channels: max across 4 channels
        5. Convert to probability: sigmoid(z − z_thresh)

    Multi-channel handling: per-channel z-scores → max (artifact is synchronous
    across all channels; neural signal is not).
    """

    name = "m0_amplitude_threshold"

    def __init__(
        self,
        z_thresh: float = 5.0,
        smooth_ms: float = 80.0,
        sr: int = 250,
    ):
        self.z_thresh = z_thresh
        self.smooth_ms = smooth_ms
        self.smooth_samples = max(1, int(smooth_ms / 1000.0 * sr))
        self.sr = sr
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        self._global_stats: dict[str, np.ndarray] | None = None

    def fit(
        self,
        neg_catalog_path: Path | str,
        *,
        max_files_per_subject: int = 200,
        seed: int = 42,
    ) -> None:
        """Compute baseline amplitude stats from stim-disabled recordings.

        Parameters
        ----------
        neg_catalog_path : path to neg_catalog.parquet
        max_files_per_subject : cap per subject (for speed)
        seed : RNG seed for subsampling
        """
        neg_df = pd.read_parquet(neg_catalog_path)
        disabled = neg_df[neg_df["epoch_type"] == "disabled"]
        rng = np.random.default_rng(seed)

        subjects = sorted(disabled["subject"].unique())
        print(
            f"[M0] Fitting on {len(disabled)} disabled files, {len(subjects)} subjects"
        )

        for subj in tqdm(subjects, desc="[M0] Baseline stats"):
            grp = disabled[disabled["subject"] == subj]
            n = min(len(grp), max_files_per_subject)
            sample = grp.iloc[rng.choice(len(grp), size=n, replace=False)]

            file_medians = []
            file_mads = []
            skipped = 0
            for _, row in sample.iterrows():
                fpath = Path(row["file_path"])
                if not fpath.exists():
                    skipped += 1
                    continue
                data = _load_dat_channels(fpath)  # (4, n_samples)
                abs_data = np.abs(data)
                med = np.median(abs_data, axis=1)  # (4,)
                mad = np.median(np.abs(abs_data - med[:, None]), axis=1)  # (4,)
                file_medians.append(med)
                file_mads.append(mad)

            if not file_medians:
                print(f"  [M0] WARNING: no valid files for {subj}, skipping")
                continue

            if skipped > 0:
                print(
                    f"  [M0] {subj}: {skipped}/{n} files missing, used {len(file_medians)}"
                )

            self.stats[subj] = {
                "median": np.median(file_medians, axis=0).astype(np.float32),
                "mad": np.median(file_mads, axis=0).astype(np.float32),
            }

        # Global fallback for unseen subjects
        all_med = np.array([s["median"] for s in self.stats.values()])
        all_mad = np.array([s["mad"] for s in self.stats.values()])
        self._global_stats = {
            "median": np.median(all_med, axis=0).astype(np.float32),
            "mad": np.median(all_mad, axis=0).astype(np.float32),
        }
        print(
            f"[M0] Fitted {len(self.stats)} subjects. Global MAD: {self._global_stats['mad']}"
        )

    def predict_file(
        self, dat_path: Path | str, *, subject: str | None = None
    ) -> np.ndarray:
        """Predict stim artifact probabilities for a single .dat file.

        Returns (n_samples,) float32 array of probabilities in [0, 1].
        """
        data = _load_dat_channels(Path(dat_path))  # (4, n_samples)

        stats = (
            self.stats.get(subject, self._global_stats)
            if subject
            else self._global_stats
        )
        if stats is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        median = stats["median"][:, None]  # (4, 1)
        mad = stats["mad"][:, None] + 1e-8  # (4, 1)

        # Per-channel z-score of absolute amplitude
        z = (np.abs(data) - median) / mad  # (4, n_samples)

        # Short-window max filter — catches transient onset/offset spikes
        if self.smooth_samples > 1:
            z = maximum_filter1d(z, size=self.smooth_samples, axis=1)

        # Max across channels (artifact is synchronous)
        z_max = np.max(z, axis=0)  # (n_samples,)

        # Sigmoid: z=z_thresh → p=0.5, z=z_thresh+4 → p≈0.98
        prob = 1.0 / (1.0 + np.exp(-(z_max - self.z_thresh)))
        return prob.astype(np.float32)

    def save(self, path: Path | str) -> None:
        """Save fitted statistics to .npz + metadata JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save per-subject stats as .npz
        arrays = {}
        subjects = sorted(self.stats.keys())
        for subj in subjects:
            arrays[f"{subj}_median"] = self.stats[subj]["median"]
            arrays[f"{subj}_mad"] = self.stats[subj]["mad"]
        arrays["global_median"] = self._global_stats["median"]
        arrays["global_mad"] = self._global_stats["mad"]
        np.savez(path / "m0_stats.npz", **arrays)

        # Save hyperparams
        meta = {
            "z_thresh": self.z_thresh,
            "smooth_ms": self.smooth_ms,
            "sr": self.sr,
            "subjects": subjects,
        }
        (path / "m0_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[M0] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> AmplitudeThreshold:
        """Load a previously fitted instance."""
        path = Path(path)
        meta = json.loads((path / "m0_meta.json").read_text())
        data = np.load(path / "m0_stats.npz")

        obj = cls(
            z_thresh=meta["z_thresh"],
            smooth_ms=meta["smooth_ms"],
            sr=meta["sr"],
        )
        for subj in meta["subjects"]:
            obj.stats[subj] = {
                "median": data[f"{subj}_median"],
                "mad": data[f"{subj}_mad"],
            }
        obj._global_stats = {
            "median": data["global_median"],
            "mad": data["global_mad"],
        }
        return obj


# ---------------------------------------------------------------------------
# M1 — Spectral Noise Subtraction
# ---------------------------------------------------------------------------


class SpectralNoiseSub(BaselineMethod):
    """M1: Audacity-style spectral noise subtraction for artifact detection.

    Algorithm
    ---------
    Fit phase (from stim-disabled recordings per subject):
        1. Load each file → (4, n_samples) µV
        2. STFT → |S(t, f)|² → median power across frames → per-file PSD
        3. Median across files → per-subject noise profile N(f)

    Predict phase:
        1. Load test file → (4, n_samples) µV
        2. STFT → |S(t, f)|² per channel (same window/hop as fit)
        3. Log spectral ratio: log₂(|S(t,f)|² / N(f)) per time-frequency bin
        4. Per-frame broadband score: median log-ratio across freq bins
        5. Interpolate frame scores to sample level
        6. Max across channels
        7. Sigmoid(score − threshold) → probability

    Using identical STFT for fit and predict ensures ratio ≈ 1 (log-ratio ≈ 0)
    for clean signal and >> 1 for artifacts. Log-domain makes the threshold
    more interpretable (ratio_thresh=3 → artifact is 8× baseline power).

    Multi-channel handling: per-channel log-ratios → max across channels.
    """

    name = "m1_spectral_noise_sub"

    def __init__(
        self,
        nperseg: int = 64,
        hop: int = 16,
        ratio_thresh: float = 2.0,
        sr: int = 250,
    ):
        self.nperseg = nperseg
        self.hop = hop
        self.ratio_thresh = ratio_thresh
        self.sr = sr
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        self._global_stats: dict[str, np.ndarray] | None = None
        # Build STFT object once
        self._win = np.hanning(nperseg).astype(np.float32)

    def _make_stft(self):
        return ShortTimeFFT(self._win, hop=self.hop, fs=self.sr)

    def fit(
        self,
        neg_catalog_path: Path | str,
        *,
        max_files_per_subject: int = 200,
        seed: int = 42,
    ) -> None:
        """Compute baseline noise PSD from stim-disabled recordings.

        Uses the same STFT as predict_file to ensure consistent normalization.
        Per-file: median STFT power across frames. Per-subject: median across files.
        """
        neg_df = pd.read_parquet(neg_catalog_path)
        disabled = neg_df[neg_df["epoch_type"] == "disabled"]
        rng = np.random.default_rng(seed)
        stft = self._make_stft()

        subjects = sorted(disabled["subject"].unique())
        print(
            f"[M1] Fitting on {len(disabled)} disabled files, {len(subjects)} subjects"
        )

        for subj in tqdm(subjects, desc="[M1] Noise profiles"):
            grp = disabled[disabled["subject"] == subj]
            n = min(len(grp), max_files_per_subject)
            sample = grp.iloc[rng.choice(len(grp), size=n, replace=False)]

            file_psds = []
            skipped = 0
            for _, row in sample.iterrows():
                fpath = Path(row["file_path"])
                if not fpath.exists():
                    skipped += 1
                    continue
                data = _load_dat_channels(fpath)  # (4, n_samples)
                # STFT per channel → median power spectrum across frames
                ch_psds = []
                for ch in range(data.shape[0]):
                    Sx = stft.stft(data[ch])  # (n_freq, n_frames)
                    power = np.abs(Sx) ** 2
                    ch_psds.append(np.median(power, axis=1))  # (n_freq,)
                file_psds.append(np.array(ch_psds))  # (4, n_freq)

            if not file_psds:
                print(f"  [M1] WARNING: no valid files for {subj}, skipping")
                continue

            if skipped > 0:
                print(
                    f"  [M1] {subj}: {skipped}/{n} files missing, used {len(file_psds)}"
                )

            # Median across files → stable noise profile
            median_psd = np.median(file_psds, axis=0).astype(np.float32)
            self.stats[subj] = {"noise_psd": median_psd}

        # Global fallback
        all_psds = np.array([s["noise_psd"] for s in self.stats.values()])
        self._global_stats = {
            "noise_psd": np.median(all_psds, axis=0).astype(np.float32),
        }
        n_freq = self.stats[next(iter(self.stats))]["noise_psd"].shape[1]
        print(f"[M1] Fitted {len(self.stats)} subjects. Freq bins: {n_freq}")

    def predict_file(
        self, dat_path: Path | str, *, subject: str | None = None
    ) -> np.ndarray:
        """Predict stim artifact probabilities via spectral noise subtraction.

        Returns (n_samples,) float32 array of probabilities in [0, 1].
        """
        data = _load_dat_channels(Path(dat_path))  # (4, n_samples)
        n_ch, n_samples = data.shape

        stats = (
            self.stats.get(subject, self._global_stats)
            if subject
            else self._global_stats
        )
        if stats is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        noise_psd = stats["noise_psd"]  # (4, n_freq)
        stft = self._make_stft()

        # Per-channel STFT → log spectral ratio → broadband score
        frame_scores = []
        for ch in range(n_ch):
            Sx = stft.stft(data[ch])  # (n_freq, n_frames) complex
            power = np.abs(Sx) ** 2  # (n_freq, n_frames)
            # Log₂ spectral ratio: how many doublings above baseline
            log_ratio = np.log2(
                power / (noise_psd[ch, :, None] + 1e-10) + 1e-10
            )  # (n_freq, n_frames)
            # Broadband score: median log-ratio across frequencies per frame
            score = np.median(log_ratio, axis=0)  # (n_frames,)
            frame_scores.append(score)

        frame_scores = np.array(frame_scores)  # (4, n_frames)
        # Max across channels
        max_score = np.max(frame_scores, axis=0)  # (n_frames,)

        # Interpolate frame-level scores to sample-level
        n_frames = max_score.shape[0]
        frame_centers = stft.nearest_k_p(0) + np.arange(n_frames) * self.hop
        frame_centers = np.clip(frame_centers, 0, n_samples - 1)
        sample_score = np.interp(np.arange(n_samples), frame_centers, max_score).astype(
            np.float32
        )

        # Sigmoid: score=ratio_thresh → p=0.5
        prob = 1.0 / (1.0 + np.exp(-(sample_score - self.ratio_thresh)))
        return prob.astype(np.float32)

    def save(self, path: Path | str) -> None:
        """Save fitted noise profiles to .npz + metadata JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        arrays = {}
        subjects = sorted(self.stats.keys())
        for subj in subjects:
            arrays[f"{subj}_noise_psd"] = self.stats[subj]["noise_psd"]
        arrays["global_noise_psd"] = self._global_stats["noise_psd"]
        np.savez(path / "m1_stats.npz", **arrays)

        meta = {
            "nperseg": self.nperseg,
            "hop": self.hop,
            "ratio_thresh": self.ratio_thresh,
            "sr": self.sr,
            "subjects": subjects,
        }
        (path / "m1_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[M1] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> SpectralNoiseSub:
        """Load a previously fitted instance."""
        path = Path(path)
        meta = json.loads((path / "m1_meta.json").read_text())
        data = np.load(path / "m1_stats.npz")

        obj = cls(
            nperseg=meta["nperseg"],
            hop=meta["hop"],
            ratio_thresh=meta["ratio_thresh"],
            sr=meta["sr"],
        )
        for subj in meta["subjects"]:
            obj.stats[subj] = {"noise_psd": data[f"{subj}_noise_psd"]}
        obj._global_stats = {"noise_psd": data["global_noise_psd"]}
        return obj


# ---------------------------------------------------------------------------
# M2 — VAE Reconstruction Error
# ---------------------------------------------------------------------------


class ConvVAE1d(nn.Module):
    """1D Convolutional VAE for 4-channel ECoG windows."""

    def __init__(self, n_channels: int = 4, window: int = 256, latent_dim: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.window = window
        self.latent_dim = latent_dim
        self._spatial = window // 8  # 3 stride-2 layers: 256→128→64→32

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_dim = 128 * self._spatial
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, 128, self._spatial)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """MSE reconstruction + beta-weighted KL divergence."""
    mse = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl


class VAEReconstruction(BaselineMethod):
    """M2: Detect stim artifacts via Conv-VAE anomaly score (MSE + KL).

    Algorithm
    ---------
    Fit phase:
        1. Normalization stats from ALL clean recordings (20 subjects)
        2. Train Conv-VAE on disabled-only recordings (5 subjects, best AUROC)
        3. Per-subject calibration on held-out clean data (20 subjects):
           anomaly_score = per-sample MSE + KL_divergence / window
           → store {mean, std} per subject for z-score calibration

    Predict phase:
        1. Z-normalize with per-subject stats (global fallback)
        2. Slide overlapping windows → reconstruct → anomaly_score
        3. Per-subject z-score: z = (score - subj_mean) / subj_std → sigmoid

    Multi-channel handling: joint 4-channel input (implicit cross-channel
    modeling via shared convolutional filters).
    """

    name = "m2_vae_reconstruction"

    def __init__(
        self,
        window: int = 256,
        latent_dim: int = 32,
        beta: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        sr: int = 250,
        cal_percentile: float = 95.0,
    ):
        self.window = window
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sr = sr
        self.cal_percentile = cal_percentile

        self.model: ConvVAE1d | None = None
        self.norm_stats: dict[str, dict[str, np.ndarray]] = {}
        self._global_norm: dict[str, np.ndarray] | None = None
        # Per-subject error calibration (replaces global error_mean/error_std)
        self.calibration: dict[str, dict[str, float]] = {}
        self._global_calibration: dict[str, float] | None = None

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _extract_windows(
        self,
        neg_catalog_path: Path | str,
        *,
        max_files_per_subject: int = 100,
        windows_per_file: int = 10,
        cal_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract windows: disabled for training, ALL clean for calibration.

        Training uses only disabled recordings (cleanest baseline, best AUROC).
        Normalization and calibration use ALL clean data (all subjects covered).

        Returns
        -------
        train_windows : (N_train, 4, window) float32 — z-normalized
        cal_windows   : (N_cal, 4, window) float32 — z-normalized
        cal_subjects  : subject ID per calibration window
        """
        neg_df = pd.read_parquet(neg_catalog_path)
        rng = np.random.default_rng(seed)
        all_subjects = sorted(neg_df["subject"].unique())

        # First pass: per-subject normalization from ALL clean files
        print(
            f"[M2] Computing normalization stats from {len(neg_df)} files, "
            f"{len(all_subjects)} subjects..."
        )
        for subj in all_subjects:
            grp = neg_df[neg_df["subject"] == subj]
            n = min(len(grp), max_files_per_subject)
            sample = grp.iloc[rng.choice(len(grp), size=n, replace=False)]

            ch_sums = np.zeros(4, dtype=np.float64)
            ch_sq_sums = np.zeros(4, dtype=np.float64)
            total_samples = 0

            for _, row in sample.iterrows():
                fpath = Path(row["file_path"])
                if not fpath.exists():
                    continue
                data = _load_dat_channels(fpath)  # (4, n)
                ch_sums += data.sum(axis=1)
                ch_sq_sums += (data**2).sum(axis=1)
                total_samples += data.shape[1]

            if total_samples == 0:
                continue
            mean = (ch_sums / total_samples).astype(np.float32)
            std = np.sqrt(ch_sq_sums / total_samples - mean**2).astype(np.float32)
            std = np.maximum(std, 1e-6)
            self.norm_stats[subj] = {"mean": mean, "std": std}

        # Global fallback
        all_means = np.array([s["mean"] for s in self.norm_stats.values()])
        all_stds = np.array([s["std"] for s in self.norm_stats.values()])
        self._global_norm = {
            "mean": np.median(all_means, axis=0).astype(np.float32),
            "std": np.median(all_stds, axis=0).astype(np.float32),
        }

        # Second pass: extract windows
        disabled = neg_df[neg_df["epoch_type"] == "disabled"]
        disabled_subjects = sorted(disabled["subject"].unique())
        print(
            f"[M2] Training from {len(disabled)} disabled files "
            f"({len(disabled_subjects)} subjects)"
        )
        print(
            f"[M2] Calibrating from {len(neg_df)} files ({len(all_subjects)} subjects)"
        )

        train_windows = []
        cal_windows = []
        cal_subjects = []

        # --- Training windows (disabled only, best discrimination) ---
        for subj in tqdm(disabled_subjects, desc="[M2] Train"):
            grp = disabled[disabled["subject"] == subj]
            n = min(len(grp), max_files_per_subject)
            sample = grp.iloc[rng.choice(len(grp), size=n, replace=False)]

            stats = self.norm_stats.get(subj, self._global_norm)
            mean = stats["mean"][:, None]
            std = stats["std"][:, None]

            for _, row in sample.iterrows():
                fpath = Path(row["file_path"])
                if not fpath.exists():
                    continue
                data = _load_dat_channels(fpath)
                data = (data - mean) / std
                n_samples = data.shape[1]
                if n_samples < self.window:
                    continue
                max_start = n_samples - self.window
                starts = rng.integers(0, max_start + 1, size=windows_per_file)
                for s in starts:
                    train_windows.append(data[:, s : s + self.window])

        # --- Calibration windows (ALL subjects, ALL epoch types) ---
        for subj in tqdm(all_subjects, desc="[M2] Cal"):
            grp = neg_df[neg_df["subject"] == subj]
            n_cal = min(len(grp), max(20, int(max_files_per_subject * cal_fraction)))
            sample = grp.iloc[rng.choice(len(grp), size=n_cal, replace=False)]

            stats = self.norm_stats.get(subj, self._global_norm)
            mean = stats["mean"][:, None]
            std = stats["std"][:, None]

            for _, row in sample.iterrows():
                fpath = Path(row["file_path"])
                if not fpath.exists():
                    continue
                data = _load_dat_channels(fpath)
                data = (data - mean) / std
                n_samples = data.shape[1]
                if n_samples < self.window:
                    continue
                max_start = n_samples - self.window
                starts = rng.integers(0, max_start + 1, size=windows_per_file)
                for s in starts:
                    cal_windows.append(data[:, s : s + self.window])
                    cal_subjects.append(subj)

        train_windows = np.array(train_windows, dtype=np.float32)
        cal_windows = np.array(cal_windows, dtype=np.float32)
        print(
            f"[M2] Extracted {len(train_windows)} train + {len(cal_windows)} cal "
            f"windows from {len(disabled_subjects)}+{len(all_subjects)} subjects"
        )
        return train_windows, cal_windows, cal_subjects

    def fit(
        self,
        neg_catalog_path: Path | str,
        *,
        max_files_per_subject: int = 100,
        windows_per_file: int = 10,
        cal_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Train Conv-VAE on clean recordings + compute per-subject calibration."""
        train_windows, cal_windows, cal_subjects = self._extract_windows(
            neg_catalog_path,
            max_files_per_subject=max_files_per_subject,
            windows_per_file=windows_per_file,
            cal_fraction=cal_fraction,
            seed=seed,
        )

        device = self._get_device()
        self.model = ConvVAE1d(
            n_channels=4, window=self.window, latent_dim=self.latent_dim
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_windows))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        print(
            f"[M2] Training VAE: {len(train_windows)} windows, "
            f"{self.epochs} epochs, device={device}"
        )
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                recon, mu, logvar = self.model(batch)
                loss = _vae_loss(recon, batch, mu, logvar, beta=self.beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs}  "
                    f"loss={epoch_loss / n_batches:.6f}"
                )

        # Per-subject calibration on held-out cal windows
        print("[M2] Computing per-subject error calibration...")
        self.model.eval()

        cal_dataset = torch.utils.data.TensorDataset(torch.from_numpy(cal_windows))
        cal_loader = torch.utils.data.DataLoader(
            cal_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Compute per-sample MSE (matching predict_file granularity)
        all_cal_errors = []
        all_cal_subj_idx = []
        with torch.no_grad():
            for (batch,) in cal_loader:
                batch = batch.to(device)
                recon, mu, logvar = self.model(batch)
                # Per-sample MSE + KL broadcast — same as predict_file
                mse = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
                kl = (
                    0.5
                    * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).cpu().numpy()
                )
                # Broadcast KL per window to each sample
                kl_broadcast = np.repeat(kl[:, None] / self.window, self.window, axis=1)
                all_cal_errors.append((mse + kl_broadcast).ravel())

        all_cal_errors = np.concatenate(all_cal_errors)
        # Expand subject labels: each window contributes `window` samples
        cal_subjects_arr = np.repeat(cal_subjects, self.window)

        for subj in sorted(set(cal_subjects)):
            mask = cal_subjects_arr == subj
            errors = all_cal_errors[mask]
            if len(errors) < 10:
                print(f"  [M2] WARNING: {subj} has only {len(errors)} cal windows")
                continue

            q50, q75, q90, q95, q99 = np.percentile(errors, [50, 75, 90, 95, 99])
            iqr = max(float(q75 - q50), 1e-8)

            self.calibration[subj] = {
                "error_mean": float(np.mean(errors)),
                "error_std": float(np.std(errors)),
                "q50": float(q50),
                "q75": float(q75),
                "q90": float(q90),
                "q95": float(q95),
                "q99": float(q99),
                "iqr": iqr,
                "n_windows": len(errors),
            }

        # Global fallback calibration (median of per-subject stats)
        if self.calibration:
            self._global_calibration = {}
            for key in [
                "error_mean",
                "error_std",
                "q50",
                "q75",
                "q90",
                "q95",
                "q99",
                "iqr",
            ]:
                vals = [c[key] for c in self.calibration.values()]
                self._global_calibration[key] = float(np.median(vals))
            self._global_calibration["n_windows"] = sum(
                c["n_windows"] for c in self.calibration.values()
            )

        print(f"[M2] Calibrated {len(self.calibration)} subjects")
        for subj, cal in sorted(self.calibration.items()):
            print(
                f"  {subj}: q50={cal['q50']:.4f} q95={cal['q95']:.4f} "
                f"iqr={cal['iqr']:.4f} n={cal['n_windows']}"
            )

    def _get_calibration(self, subject: str | None) -> dict[str, float]:
        """Get calibration stats for a subject, falling back to global."""
        if subject and subject in self.calibration:
            return self.calibration[subject]
        if self._global_calibration:
            return self._global_calibration
        raise RuntimeError("No calibration data. Call .fit() first.")

    def predict_file(
        self, dat_path: Path | str, *, subject: str | None = None
    ) -> np.ndarray:
        """Predict stim artifact probabilities via VAE reconstruction error.

        Returns (n_samples,) float32 array of probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        data = _load_dat_channels(Path(dat_path))  # (4, n_samples)
        n_ch, n_samples = data.shape

        # Z-normalize with subject stats
        stats = (
            self.norm_stats.get(subject, self._global_norm)
            if subject
            else self._global_norm
        )
        mean = stats["mean"][:, None]
        std = stats["std"][:, None]
        data = (data - mean) / std

        device = self._get_device()
        self.model.eval()

        stride = self.window // 2
        error_sum = np.zeros(n_samples, dtype=np.float64)
        error_count = np.zeros(n_samples, dtype=np.float64)

        # Pad if file is shorter than window
        if n_samples < self.window:
            pad = self.window - n_samples
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
            n_padded = self.window
        else:
            n_padded = n_samples

        # Collect windows
        starts = list(range(0, n_padded - self.window + 1, stride))
        if starts and starts[-1] + self.window < n_padded:
            starts.append(n_padded - self.window)

        with torch.no_grad():
            # Process in batches
            for i in range(0, len(starts), self.batch_size):
                batch_starts = starts[i : i + self.batch_size]
                batch = np.array(
                    [data[:, s : s + self.window] for s in batch_starts],
                    dtype=np.float32,
                )
                x = torch.from_numpy(batch).to(device)
                recon, mu, logvar = self.model(x)
                # Per-sample MSE across channels: (B, window)
                mse = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
                # KL divergence per window, broadcast to all samples
                kl = (
                    0.5
                    * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).cpu().numpy()
                )

                for j, s in enumerate(batch_starts):
                    end = min(s + self.window, n_samples)
                    valid = end - s
                    kl_per_sample = kl[j] / self.window
                    error_sum[s:end] += mse[j, :valid] + kl_per_sample
                    error_count[s:end] += 1.0

        # Average overlapping predictions
        error_count = np.maximum(error_count, 1.0)
        error = (error_sum / error_count).astype(np.float32)

        # Per-subject z-score calibration → sigmoid
        cal = self._get_calibration(subject)
        z = (error - cal["error_mean"]) / max(cal["error_std"], 1e-8)
        prob = 1.0 / (1.0 + np.exp(-z))
        return prob.astype(np.float32)

    def save(self, path: Path | str) -> None:
        """Save trained VAE model + normalization + calibration stats."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(self.model.state_dict(), path / "m2_model.pt")

        # Normalization stats
        arrays = {}
        subjects = sorted(self.norm_stats.keys())
        for subj in subjects:
            arrays[f"{subj}_mean"] = self.norm_stats[subj]["mean"]
            arrays[f"{subj}_std"] = self.norm_stats[subj]["std"]
        arrays["global_mean"] = self._global_norm["mean"]
        arrays["global_std"] = self._global_norm["std"]
        np.savez(path / "m2_norm.npz", **arrays)

        # Per-subject calibration
        cal_data = {
            "per_subject": self.calibration,
            "global": self._global_calibration,
        }
        (path / "m2_calibration.json").write_text(json.dumps(cal_data, indent=2))

        # Metadata
        meta = {
            "window": self.window,
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "sr": self.sr,
            "cal_percentile": self.cal_percentile,
            "subjects": subjects,
            "calibrated_subjects": sorted(self.calibration.keys()),
        }
        (path / "m2_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[M2] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> VAEReconstruction:
        """Load a previously trained instance."""
        path = Path(path)
        meta = json.loads((path / "m2_meta.json").read_text())
        norm_data = np.load(path / "m2_norm.npz")

        obj = cls(
            window=meta["window"],
            latent_dim=meta["latent_dim"],
            beta=meta["beta"],
            lr=meta["lr"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            sr=meta["sr"],
            cal_percentile=meta.get("cal_percentile", 95.0),
        )

        for subj in meta["subjects"]:
            obj.norm_stats[subj] = {
                "mean": norm_data[f"{subj}_mean"],
                "std": norm_data[f"{subj}_std"],
            }
        obj._global_norm = {
            "mean": norm_data["global_mean"],
            "std": norm_data["global_std"],
        }

        # Load per-subject calibration
        cal_path = path / "m2_calibration.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            obj.calibration = cal_data.get("per_subject", {})
            obj._global_calibration = cal_data.get("global")
        else:
            # Backward compat: synthesize from old error_mean/error_std
            em = meta.get("error_mean", 0.0)
            es = meta.get("error_std", 1.0)
            obj._global_calibration = {
                "error_mean": em,
                "error_std": es,
                "q50": em,
                "q75": em + es * 0.675,
                "q90": em + es * 1.28,
                "q95": em + es * 1.645,
                "q99": em + es * 2.33,
                "iqr": es * 0.675,
                "n_windows": 0,
            }

        obj.model = ConvVAE1d(
            n_channels=4, window=meta["window"], latent_dim=meta["latent_dim"]
        )
        obj.model.load_state_dict(
            torch.load(path / "m2_model.pt", map_location="cpu", weights_only=True)
        )
        obj.model.to(obj._get_device())
        obj.model.eval()
        return obj


# ---------------------------------------------------------------------------
# M3 — Scattering Transform + XGBoost
# ---------------------------------------------------------------------------


def _patch_kymatio_scipy():
    """Shim scipy.special.sph_harm for kymatio 0.3 + scipy >= 2.0.

    Kymatio eagerly imports 3D scattering which uses the removed sph_harm.
    Only 1D scattering is actually needed; the shim avoids the ImportError.
    """
    import scipy.special

    if not hasattr(scipy.special, "sph_harm"):

        def _sph_harm_compat(m, n, theta, phi):
            return scipy.special.sph_harm_y(n, m, theta, phi)

        scipy.special.sph_harm = _sph_harm_compat


class ScatteringXGBoost(BaselineMethod):
    """M3: Kymatio scattering transform + XGBoost classifier.

    Algorithm
    ---------
    Fit phase (supervised — uses labeled stim catalog + clean neg catalog):
        1. From stim files: extract windows, label by overlap with true mask
        2. From neg files: extract windows, all labeled clean
        3. Scattering transform per channel → multi-scale invariant features
        4. Train XGBoost binary classifier

    Predict phase:
        1. Extract overlapping windows from test file
        2. Compute scattering features per window
        3. XGBoost predict_proba per window
        4. Map window probabilities to sample level (average over overlaps)

    Multi-channel handling: explicit — per-channel scattering coefficients
    (S0/S1/S2) with summary stats, plus cross-channel max/mean/std of
    S1 energies, plus per-channel time-domain features.

    The scattering transform (Mallat 2012) provides translation-invariant
    and deformation-stable features — ideal for stim artifacts whose onset
    may shift slightly across recordings.
    """

    name = "m3_scattering_xgboost"

    def __init__(
        self,
        window: int = 256,
        J: int = 5,
        Q: int = 8,
        n_estimators: int = 300,
        max_depth: int = 6,
        sr: int = 250,
        artifact_overlap_thresh: float = 0.3,
    ):
        self.window = window
        self.J = J
        self.Q = Q
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sr = sr
        self.artifact_overlap_thresh = artifact_overlap_thresh

        self.clf = None
        self.scaler = None  # StandardScaler fitted during training
        self._scattering = None  # lazy init

    def _get_scattering(self):
        """Lazily create the Scattering1D object (caches filter bank)."""
        if self._scattering is None:
            _patch_kymatio_scipy()
            from kymatio.torch import Scattering1D

            self._scattering = Scattering1D(J=self.J, shape=(self.window,), Q=self.Q)
            self._scattering_meta = self._scattering.meta()
        return self._scattering

    def _extract_features(self, data_window: np.ndarray) -> np.ndarray:
        """Compute scattering features for a single (4, window) array.

        Per channel:
            - Scattering transform → (n_paths, n_time) coefficients
            - Per path: mean, std, max → 3 × n_paths features
            - Time-domain: RMS, zero-crossing rate, line length, max amplitude
        Cross-channel:
            - S1 path energies: max/mean/std across 4 channels

        Returns a 1D feature vector.
        """
        scattering = self._get_scattering()
        n_ch = data_window.shape[0]

        # Scattering transform: all channels at once (batch=n_ch)
        x = torch.from_numpy(
            np.ascontiguousarray(data_window, dtype=np.float32)
        )  # (4, window)
        with torch.no_grad():
            Sx = scattering(x).numpy()  # (4, n_paths, n_time)

        # Log-scattering (Mallat 2012): compress dynamic range, make features
        # more scale-invariant across subjects with different amplitude scales
        Sx = np.log1p(np.abs(Sx))

        meta = self._scattering_meta
        orders = meta["order"]
        s1_mask = orders == 1

        ch_features = []
        s1_energies = []  # for cross-channel stats

        for ch in range(n_ch):
            s = Sx[ch]  # (n_paths, n_time)
            # Summary stats per path
            path_mean = s.mean(axis=1)
            path_std = s.std(axis=1)
            path_max = s.max(axis=1)

            ch_features.extend(path_mean.tolist())
            ch_features.extend(path_std.tolist())
            ch_features.extend(path_max.tolist())

            # S1 path energies for cross-channel
            s1_energies.append(np.sum(s[s1_mask] ** 2, axis=1))

            # Time-domain features
            sig = data_window[ch].astype(np.float64)
            rms = np.sqrt(np.mean(sig**2))
            zc = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
            line_length = np.sum(np.abs(np.diff(sig))) / len(sig)
            max_amp = np.max(np.abs(sig))
            ch_features.extend([rms, zc, line_length, max_amp])

        # Cross-channel statistics on S1 energies
        s1_energies = np.array(s1_energies)  # (4, n_s1_paths)
        cross_ch = np.concatenate(
            [
                s1_energies.max(axis=0),
                s1_energies.mean(axis=0),
                s1_energies.std(axis=0),
            ]
        )
        features = np.concatenate([ch_features, cross_ch])
        return features.astype(np.float32)

    def _extract_labeled_windows(
        self,
        stim_catalog_path: Path | str,
        neg_catalog_path: Path | str,
        *,
        max_stim_files: int = 500,
        max_neg_files: int = 500,
        windows_per_file: int = 8,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels from stim + neg files.

        Returns
        -------
        X : (N, n_features) float32
        y : (N,) int — 1 = artifact, 0 = clean
        """
        rng = np.random.default_rng(seed)
        cfg = Config()
        features_list = []
        labels_list = []

        # --- Stim files (subject-balanced sampling) ---
        stim_df = load_annotations(Config(catalog_path=Path(stim_catalog_path)))
        if len(stim_df) > max_stim_files:
            # Equal files per subject to prevent majority-subject bias
            subjects = stim_df["subject"].unique()
            per_subj = max(1, max_stim_files // len(subjects))
            balanced = []
            for subj in subjects:
                subj_df = stim_df[stim_df["subject"] == subj]
                n = min(len(subj_df), per_subj)
                idx = rng.choice(len(subj_df), size=n, replace=False)
                balanced.append(subj_df.iloc[idx])
            stim_df = pd.concat(balanced, ignore_index=True)
            # If under budget, fill remaining slots proportionally
            remaining = max_stim_files - len(stim_df)
            if remaining > 0:
                unused = load_annotations(Config(catalog_path=Path(stim_catalog_path)))
                used_files = set(stim_df["file_path"])
                unused = unused[~unused["file_path"].isin(used_files)]
                if len(unused) > 0:
                    extra = min(len(unused), remaining)
                    idx = rng.choice(len(unused), size=extra, replace=False)
                    stim_df = pd.concat([stim_df, unused.iloc[idx]], ignore_index=True)

        print(f"[M3] Extracting features from {len(stim_df)} stim files...")
        for _, row in tqdm(stim_df.iterrows(), total=len(stim_df), desc="[M3] Stim"):
            fpath = Path(row["file_path"])
            if not fpath.exists():
                continue
            data = _load_dat_channels(fpath)  # (4, n_samples)
            n_samples = data.shape[1]
            if n_samples < self.window:
                continue

            true_mask = build_true_mask(
                n_samples,
                row["onset_times"],
                row["mask_duration_ms"],
                sr=int(row.get("sampling_rate", 250)),
                onset_offset_ms=row.get("mask_onset_offset_ms", 0.0),
            )

            # Random windows
            max_start = n_samples - self.window
            starts = rng.integers(0, max_start + 1, size=windows_per_file)
            for s in starts:
                win = data[:, s : s + self.window]
                mask_win = true_mask[s : s + self.window]
                label = int(mask_win.mean() >= self.artifact_overlap_thresh)
                feat = self._extract_features(win)
                features_list.append(feat)
                labels_list.append(label)

        # --- Neg files (subject-balanced sampling) ---
        neg_df = pd.read_parquet(neg_catalog_path)
        disabled = neg_df[neg_df["epoch_type"] == "disabled"]
        if len(disabled) > max_neg_files:
            subjects = disabled["subject"].unique()
            per_subj = max(1, max_neg_files // len(subjects))
            balanced = []
            for subj in subjects:
                subj_df = disabled[disabled["subject"] == subj]
                n = min(len(subj_df), per_subj)
                idx = rng.choice(len(subj_df), size=n, replace=False)
                balanced.append(subj_df.iloc[idx])
            disabled = pd.concat(balanced, ignore_index=True)

        print(f"[M3] Extracting features from {len(disabled)} clean files...")
        for _, row in tqdm(disabled.iterrows(), total=len(disabled), desc="[M3] Clean"):
            fpath = Path(row["file_path"])
            if not fpath.exists():
                continue
            data = _load_dat_channels(fpath)
            n_samples = data.shape[1]
            if n_samples < self.window:
                continue

            max_start = n_samples - self.window
            starts = rng.integers(0, max_start + 1, size=windows_per_file)
            for s in starts:
                win = data[:, s : s + self.window]
                feat = self._extract_features(win)
                features_list.append(feat)
                labels_list.append(0)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)
        print(
            f"[M3] Total: {len(X)} windows, "
            f"{y.sum()} artifact ({y.mean():.1%}), "
            f"{(1-y).sum()} clean ({1-y.mean():.1%})"
        )
        return X, y

    def fit(
        self,
        neg_catalog_path: Path | str,
        *,
        stim_catalog_path: Path | str = "data/stim_catalog.parquet",
        max_stim_files: int = 500,
        max_neg_files: int = 500,
        windows_per_file: int = 8,
        seed: int = 42,
    ) -> None:
        """Train XGBoost on kymatio scattering features from labeled data."""
        import xgboost as xgb

        X, y = self._extract_labeled_windows(
            stim_catalog_path,
            neg_catalog_path,
            max_stim_files=max_stim_files,
            max_neg_files=max_neg_files,
            windows_per_file=windows_per_file,
            seed=seed,
        )

        # Feature normalization — reduces subject-dependent scale effects
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Class weight balancing
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos = n_neg / max(n_pos, 1)

        self.clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=scale_pos,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        print(
            f"[M3] Training XGBoost ({self.n_estimators} trees, scale_pos_weight={scale_pos:.2f})..."
        )
        self.clf.fit(X, y)
        train_proba = self.clf.predict_proba(X)[:, 1]
        from sklearn.metrics import f1_score, roc_auc_score

        train_f1 = f1_score(y, (train_proba > 0.5).astype(int))
        train_auc = roc_auc_score(y, train_proba)
        print(f"[M3] Train F1={train_f1:.4f}, AUC={train_auc:.4f}")

    def predict_file(
        self, dat_path: Path | str, *, subject: str | None = None
    ) -> np.ndarray:
        """Predict stim artifact probabilities via scattering + XGBoost.

        Returns (n_samples,) float32 array of probabilities in [0, 1].
        """
        if self.clf is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        data = _load_dat_channels(Path(dat_path))  # (4, n_samples)
        n_samples = data.shape[1]

        stride = self.window // 2
        prob_sum = np.zeros(n_samples, dtype=np.float64)
        prob_count = np.zeros(n_samples, dtype=np.float64)

        # Pad if shorter than window
        if n_samples < self.window:
            pad = self.window - n_samples
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
            n_padded = self.window
        else:
            n_padded = n_samples

        starts = list(range(0, n_padded - self.window + 1, stride))
        if starts and starts[-1] + self.window < n_padded:
            starts.append(n_padded - self.window)

        # Extract features for all windows
        features = []
        for s in starts:
            win = data[:, s : s + self.window]
            features.append(self._extract_features(win))
        X = np.array(features, dtype=np.float32)

        # Apply scaler if available (trained with normalization)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict probabilities
        proba = self.clf.predict_proba(X)[:, 1]

        # Map window probabilities to samples
        for i, s in enumerate(starts):
            end = min(s + self.window, n_samples)
            prob_sum[s:end] += proba[i]
            prob_count[s:end] += 1.0

        prob_count = np.maximum(prob_count, 1.0)
        return (prob_sum / prob_count).astype(np.float32)

    def save(self, path: Path | str) -> None:
        """Save XGBoost model + scattering parameters."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.clf.save_model(str(path / "m3_xgb.json"))
        meta = {
            "window": self.window,
            "J": self.J,
            "Q": self.Q,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "sr": self.sr,
            "artifact_overlap_thresh": self.artifact_overlap_thresh,
        }
        (path / "m3_meta.json").write_text(json.dumps(meta, indent=2))
        # Save scaler
        if self.scaler is not None:
            import joblib

            joblib.dump(self.scaler, path / "m3_scaler.pkl")
        print(f"[M3] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> ScatteringXGBoost:
        """Load a previously trained instance."""
        import xgboost as xgb

        path = Path(path)
        meta = json.loads((path / "m3_meta.json").read_text())

        obj = cls(
            window=meta["window"],
            J=meta["J"],
            Q=meta["Q"],
            n_estimators=meta["n_estimators"],
            max_depth=meta["max_depth"],
            sr=meta["sr"],
            artifact_overlap_thresh=meta["artifact_overlap_thresh"],
        )
        obj.clf = xgb.XGBClassifier()
        obj.clf.load_model(str(path / "m3_xgb.json"))
        # Load scaler if available
        scaler_path = path / "m3_scaler.pkl"
        if scaler_path.exists():
            import joblib

            obj.scaler = joblib.load(scaler_path)
        return obj


# Backward-compat alias
WaveletXGBoost = ScatteringXGBoost


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _apply_edge_exclusions(
    true_mask: np.ndarray,
    pred_bin: np.ndarray,
    proba: np.ndarray,
    onset_times: np.ndarray,
    mask_dur_ms: float,
    n_samples: int,
    sr: int,
    onset_offset_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Targeted edge exclusion for known label artifacts.

    1. End-truncated artifacts: if last artifact extends past recording end,
       blank from that artifact's onset to end (partial artifact, unclassifiable).
    2. Pre-recording artifact tails: if method predicts artifact at sample 0
       but no ground truth label covers it, blank predictions up to first
       labeled onset (or first 0.5s, whichever is smaller).

    Returns (true_mask, pred_bin, proba, n_end_blanked, n_start_blanked).
    """
    true_mask = true_mask.copy()
    pred_bin = pred_bin.copy()
    proba = proba.copy()
    n_end_blanked = 0
    n_start_blanked = 0

    dur_samp = int(mask_dur_ms / 1000.0 * sr)
    offset_samp = int(onset_offset_ms / 1000.0 * sr)

    # --- End truncation ---
    # Find artifacts whose labeled extent exceeds n_samples
    if len(onset_times) > 0:
        for t in onset_times:
            start_samp = int(t * sr) + offset_samp
            end_samp = start_samp + dur_samp
            if end_samp > n_samples and start_samp < n_samples:
                # This artifact is truncated — blank from onset to end
                blank_from = max(start_samp, 0)
                true_mask[blank_from:] = 0
                pred_bin[blank_from:] = 0
                proba[blank_from:] = 0.0
                n_end_blanked += n_samples - blank_from

    # --- Pre-recording artifact tails ---
    # If predictions fire at sample 0 but no label covers it, these are likely
    # tails of stim that started before the recording.
    if len(onset_times) > 0:
        first_onset_samp = int(min(onset_times) * sr) + offset_samp
        # Cap blanking at 0.5s or first labeled onset, whichever is smaller
        max_blank = min(int(0.5 * sr), max(first_onset_samp, 0))
        if max_blank > 0 and true_mask[:max_blank].sum() == 0 and pred_bin[0] == 1:
            # Predictions at start with no ground truth — likely pre-recording tail
            # Find extent of this leading prediction blob
            end_pred = 0
            while end_pred < max_blank and pred_bin[end_pred] == 1:
                end_pred += 1
            pred_bin[:end_pred] = 0
            proba[:end_pred] = 0.0
            n_start_blanked += end_pred

    return true_mask, pred_bin, proba, n_end_blanked, n_start_blanked


def evaluate_method(
    method: BaselineMethod,
    catalog_path: Path | str,
    *,
    max_files: int | None = None,
    threshold: float = 0.5,
    seed: int = 42,
    edge_blank_s: float = 0.0,
    edge_exclusion: bool = False,
) -> pd.DataFrame:
    """Run a baseline method on the stim catalog and compute per-file metrics.

    Returns a DataFrame with one row per file and columns:
        filename, subject, f1, precision, recall, iou, dice, n_samples,
        artifact_frac, pred_artifact_frac

    Args:
        edge_blank_s: Blank first/last N seconds symmetrically (crude).
        edge_exclusion: Targeted exclusion of end-truncated artifacts and
            pre-recording artifact tails (preferred over edge_blank_s).
    """
    from stim_metrics import compute_event_metrics_full, compute_sample_metrics_full

    cfg = Config(catalog_path=Path(catalog_path))
    df = load_annotations(cfg)

    if max_files is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=min(max_files, len(df)), replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{method.name}] Eval"):
        fpath = Path(row["file_path"])
        if not fpath.exists():
            continue

        # Ground truth
        sr = int(row.get("sampling_rate", 250))
        data = _load_dat_channels(fpath)
        n_samples = data.shape[1]
        true_mask = build_true_mask(
            n_samples,
            row["onset_times"],
            row["mask_duration_ms"],
            sr=sr,
            onset_offset_ms=row.get("mask_onset_offset_ms", 0.0),
        )

        # Prediction
        proba = method.predict_file(fpath, subject=row.get("subject"))
        pred_bin = post_process_mask((proba > threshold).astype(np.int_), cfg)

        # Edge blanking — ignore first/last N seconds (crude)
        if edge_blank_s > 0:
            blank_samp = int(edge_blank_s * sr)
            true_mask[:blank_samp] = 0
            true_mask[-blank_samp:] = 0
            pred_bin[:blank_samp] = 0
            pred_bin[-blank_samp:] = 0
            proba[:blank_samp] = 0.0
            proba[-blank_samp:] = 0.0

        # Targeted edge exclusion
        n_end_blanked = 0
        n_start_blanked = 0
        if edge_exclusion:
            true_mask, pred_bin, proba, n_end_blanked, n_start_blanked = (
                _apply_edge_exclusions(
                    true_mask,
                    pred_bin,
                    proba,
                    row["onset_times"],
                    row["mask_duration_ms"],
                    n_samples,
                    sr,
                    onset_offset_ms=row.get("mask_onset_offset_ms", 0.0),
                )
            )

        # Metrics
        sm = compute_sample_metrics_full(true_mask, pred_bin, proba=proba)
        em = compute_event_metrics_full(true_mask, pred_bin, sr=sr)

        results.append(
            {
                "filename": row["filename"],
                "subject": row.get("subject", ""),
                "n_samples": n_samples,
                "artifact_frac": true_mask.mean(),
                "pred_artifact_frac": pred_bin.mean(),
                "edge_end_blanked": n_end_blanked,
                "edge_start_blanked": n_start_blanked,
                **{
                    f"sample_{k}": v
                    for k, v in sm.items()
                    if not isinstance(v, (list, np.ndarray))
                },
                **{
                    f"event_{k}": v
                    for k, v in em.items()
                    if not isinstance(v, (list, np.ndarray))
                },
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_fit_m0(args):
    m0 = AmplitudeThreshold(
        z_thresh=args.z_thresh,
        smooth_ms=args.smooth_ms,
    )
    m0.fit(args.neg_catalog, max_files_per_subject=args.max_files_per_subject)
    out = Path(args.output)
    m0.save(out)


def _cli_predict_m0(args):
    m0 = AmplitudeThreshold.load(args.model_dir)
    proba = m0.predict_file(args.file, subject=args.subject)
    print(f"Shape: {proba.shape}, min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"Artifact fraction (>0.5): {(proba > 0.5).mean():.4f}")
    if args.output:
        np.save(args.output, proba)
        print(f"Saved to {args.output}")


def _cli_eval_m0(args):
    m0 = AmplitudeThreshold.load(args.model_dir)
    _run_eval(m0, "M0 Amplitude Threshold", args)


def _cli_fit_m1(args):
    m1 = SpectralNoiseSub(
        nperseg=args.nperseg,
        hop=args.hop,
        ratio_thresh=args.ratio_thresh,
    )
    m1.fit(args.neg_catalog, max_files_per_subject=args.max_files_per_subject)
    m1.save(Path(args.output))


def _cli_predict_m1(args):
    m1 = SpectralNoiseSub.load(args.model_dir)
    proba = m1.predict_file(args.file, subject=args.subject)
    print(f"Shape: {proba.shape}, min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"Artifact fraction (>0.5): {(proba > 0.5).mean():.4f}")
    if args.output:
        np.save(args.output, proba)
        print(f"Saved to {args.output}")


def _cli_eval_m1(args):
    m1 = SpectralNoiseSub.load(args.model_dir)
    _run_eval(m1, "M1 Spectral Noise Sub", args)


def _cli_fit_m2(args):
    m2 = VAEReconstruction(
        window=args.window,
        latent_dim=args.latent_dim,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cal_percentile=args.cal_percentile,
    )
    m2.fit(
        args.neg_catalog,
        max_files_per_subject=args.max_files_per_subject,
        windows_per_file=args.windows_per_file,
        cal_fraction=args.cal_fraction,
    )
    m2.save(Path(args.output))


def _cli_predict_m2(args):
    m2 = VAEReconstruction.load(args.model_dir)
    proba = m2.predict_file(args.file, subject=args.subject)
    print(f"Shape: {proba.shape}, min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"Artifact fraction (>0.5): {(proba > 0.5).mean():.4f}")
    if args.output:
        np.save(args.output, proba)
        print(f"Saved to {args.output}")


def _cli_eval_m2(args):
    m2 = VAEReconstruction.load(args.model_dir)
    _run_eval(m2, "M2 VAE Reconstruction", args)


def _cli_fit_m3(args):
    m3 = ScatteringXGBoost(
        window=args.window,
        J=args.J,
        Q=args.Q,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    m3.fit(
        args.neg_catalog,
        stim_catalog_path=args.stim_catalog,
        max_stim_files=args.max_stim_files,
        max_neg_files=args.max_neg_files,
        windows_per_file=args.windows_per_file,
    )
    m3.save(Path(args.output))


def _cli_predict_m3(args):
    m3 = ScatteringXGBoost.load(args.model_dir)
    proba = m3.predict_file(args.file, subject=args.subject)
    print(f"Shape: {proba.shape}, min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"Artifact fraction (>0.5): {(proba > 0.5).mean():.4f}")
    if args.output:
        np.save(args.output, proba)
        print(f"Saved to {args.output}")


def _cli_eval_m3(args):
    m3 = ScatteringXGBoost.load(args.model_dir)
    _run_eval(m3, "M3 Scattering+XGBoost", args)


def _run_eval(method, label, args):
    edge_blank_s = getattr(args, "edge_blank", 0.0)
    edge_exclusion = getattr(args, "edge_exclusion", False)
    results = evaluate_method(
        method,
        args.catalog,
        max_files=args.max_files,
        threshold=args.threshold,
        edge_blank_s=edge_blank_s,
        edge_exclusion=edge_exclusion,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    print(f"\n{'='*60}")
    blank_msg = ""
    if edge_blank_s > 0:
        blank_msg = f" (edge_blank={edge_blank_s}s)"
    elif edge_exclusion:
        blank_msg = " (edge_exclusion=targeted)"
    print(f"{label} — {len(results)} files evaluated{blank_msg}")
    print(f"{'='*60}")
    for col in [
        "sample_f1",
        "sample_precision",
        "sample_recall",
        "sample_iou",
        "sample_dice",
    ]:
        if col in results.columns:
            print(f"  {col:25s}: {results[col].mean():.4f} ± {results[col].std():.4f}")
    for col in ["event_event_f1", "event_event_precision", "event_event_recall"]:
        if col in results.columns:
            print(f"  {col:25s}: {results[col].mean():.4f} ± {results[col].std():.4f}")
    if edge_exclusion and "edge_end_blanked" in results.columns:
        n_end = (results["edge_end_blanked"] > 0).sum()
        n_start = (results["edge_start_blanked"] > 0).sum()
        print(
            f"  Edge exclusion: {n_end} files end-truncated, {n_start} files start-blanked"
        )


def main():
    parser = argparse.ArgumentParser(description="Comparison baseline methods")
    sub = parser.add_subparsers(dest="command")

    # fit-m0
    p_fit = sub.add_parser("fit-m0", help="Fit M0 amplitude threshold baseline stats")
    p_fit.add_argument("--neg-catalog", default="data/neg_catalog.parquet")
    p_fit.add_argument("--output", default="data/baselines/m0")
    p_fit.add_argument("--z-thresh", type=float, default=5.0)
    p_fit.add_argument("--smooth-ms", type=float, default=80.0)
    p_fit.add_argument("--max-files-per-subject", type=int, default=200)

    # predict-m0
    p_pred = sub.add_parser("predict-m0", help="Predict stim artifacts with M0")
    p_pred.add_argument("--file", required=True, help="Path to .dat file")
    p_pred.add_argument(
        "--subject", default=None, help="Subject ID for per-subject stats"
    )
    p_pred.add_argument("--model-dir", default="data/baselines/m0")
    p_pred.add_argument("--output", default=None, help="Save probas to .npy")

    # eval-m0
    p_eval = sub.add_parser("eval-m0", help="Evaluate M0 on stim catalog")
    p_eval.add_argument("--catalog", default="data/stim_catalog.parquet")
    p_eval.add_argument("--model-dir", default="data/baselines/m0")
    p_eval.add_argument("--max-files", type=int, default=None)
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--output", default="outputs/results/m0_eval.csv")
    p_eval.add_argument(
        "--edge-blank",
        type=float,
        default=0.0,
        help="Blank first/last N seconds of each recording for eval (default: 0)",
    )
    p_eval.add_argument(
        "--edge-exclusion",
        action="store_true",
        help="Targeted exclusion of end-truncated and pre-recording artifacts",
    )

    # fit-m1
    p_fit1 = sub.add_parser("fit-m1", help="Fit M1 spectral noise subtraction profiles")
    p_fit1.add_argument("--neg-catalog", default="data/neg_catalog.parquet")
    p_fit1.add_argument("--output", default="data/baselines/m1")
    p_fit1.add_argument("--nperseg", type=int, default=64)
    p_fit1.add_argument("--hop", type=int, default=16)
    p_fit1.add_argument("--ratio-thresh", type=float, default=2.0)
    p_fit1.add_argument("--max-files-per-subject", type=int, default=200)

    # predict-m1
    p_pred1 = sub.add_parser("predict-m1", help="Predict stim artifacts with M1")
    p_pred1.add_argument("--file", required=True, help="Path to .dat file")
    p_pred1.add_argument(
        "--subject", default=None, help="Subject ID for per-subject stats"
    )
    p_pred1.add_argument("--model-dir", default="data/baselines/m1")
    p_pred1.add_argument("--output", default=None, help="Save probas to .npy")

    # eval-m1
    p_eval1 = sub.add_parser("eval-m1", help="Evaluate M1 on stim catalog")
    p_eval1.add_argument("--catalog", default="data/stim_catalog.parquet")
    p_eval1.add_argument("--model-dir", default="data/baselines/m1")
    p_eval1.add_argument("--max-files", type=int, default=None)
    p_eval1.add_argument("--threshold", type=float, default=0.5)
    p_eval1.add_argument("--output", default="outputs/results/m1_eval.csv")
    p_eval1.add_argument(
        "--edge-blank",
        type=float,
        default=0.0,
        help="Blank first/last N seconds of each recording for eval (default: 0)",
    )
    p_eval1.add_argument(
        "--edge-exclusion",
        action="store_true",
        help="Targeted exclusion of end-truncated and pre-recording artifacts",
    )

    # fit-m2
    p_fit2 = sub.add_parser("fit-m2", help="Train M2 Conv-VAE on clean recordings")
    p_fit2.add_argument("--neg-catalog", default="data/neg_catalog.parquet")
    p_fit2.add_argument("--output", default="data/baselines/m2")
    p_fit2.add_argument("--window", type=int, default=256)
    p_fit2.add_argument("--latent-dim", type=int, default=32)
    p_fit2.add_argument("--beta", type=float, default=0.1)
    p_fit2.add_argument("--lr", type=float, default=1e-3)
    p_fit2.add_argument("--epochs", type=int, default=50)
    p_fit2.add_argument("--batch-size", type=int, default=256)
    p_fit2.add_argument("--max-files-per-subject", type=int, default=100)
    p_fit2.add_argument("--windows-per-file", type=int, default=10)
    p_fit2.add_argument("--cal-fraction", type=float, default=0.2)
    p_fit2.add_argument("--cal-percentile", type=float, default=95.0)

    # predict-m2
    p_pred2 = sub.add_parser("predict-m2", help="Predict stim artifacts with M2 VAE")
    p_pred2.add_argument("--file", required=True, help="Path to .dat file")
    p_pred2.add_argument(
        "--subject", default=None, help="Subject ID for per-subject normalization"
    )
    p_pred2.add_argument("--model-dir", default="data/baselines/m2")
    p_pred2.add_argument("--output", default=None, help="Save probas to .npy")

    # eval-m2
    p_eval2 = sub.add_parser("eval-m2", help="Evaluate M2 VAE on stim catalog")
    p_eval2.add_argument("--catalog", default="data/stim_catalog.parquet")
    p_eval2.add_argument("--model-dir", default="data/baselines/m2")
    p_eval2.add_argument("--max-files", type=int, default=None)
    p_eval2.add_argument("--threshold", type=float, default=0.5)
    p_eval2.add_argument("--output", default="outputs/results/m2_eval.csv")
    p_eval2.add_argument(
        "--edge-blank",
        type=float,
        default=0.0,
        help="Blank first/last N seconds of each recording for eval (default: 0)",
    )
    p_eval2.add_argument(
        "--edge-exclusion",
        action="store_true",
        help="Targeted exclusion of end-truncated and pre-recording artifacts",
    )

    # fit-m3
    p_fit3 = sub.add_parser(
        "fit-m3", help="Train M3 scattering+XGBoost on labeled data"
    )
    p_fit3.add_argument("--neg-catalog", default="data/neg_catalog.parquet")
    p_fit3.add_argument("--stim-catalog", default="data/stim_catalog.parquet")
    p_fit3.add_argument("--output", default="data/baselines/m3")
    p_fit3.add_argument("--window", type=int, default=256)
    p_fit3.add_argument("--J", type=int, default=5, help="Scattering max scale (2^J)")
    p_fit3.add_argument("--Q", type=int, default=8, help="Wavelets per octave")
    p_fit3.add_argument("--n-estimators", type=int, default=300)
    p_fit3.add_argument("--max-depth", type=int, default=6)
    p_fit3.add_argument("--max-stim-files", type=int, default=500)
    p_fit3.add_argument("--max-neg-files", type=int, default=500)
    p_fit3.add_argument("--windows-per-file", type=int, default=8)

    # predict-m3
    p_pred3 = sub.add_parser(
        "predict-m3", help="Predict stim artifacts with M3 scattering+XGBoost"
    )
    p_pred3.add_argument("--file", required=True, help="Path to .dat file")
    p_pred3.add_argument("--subject", default=None)
    p_pred3.add_argument("--model-dir", default="data/baselines/m3")
    p_pred3.add_argument("--output", default=None, help="Save probas to .npy")

    # eval-m3
    p_eval3 = sub.add_parser("eval-m3", help="Evaluate M3 on stim catalog")
    p_eval3.add_argument("--catalog", default="data/stim_catalog.parquet")
    p_eval3.add_argument("--model-dir", default="data/baselines/m3")
    p_eval3.add_argument("--max-files", type=int, default=None)
    p_eval3.add_argument("--threshold", type=float, default=0.5)
    p_eval3.add_argument("--output", default="outputs/results/m3_eval.csv")
    p_eval3.add_argument(
        "--edge-blank",
        type=float,
        default=0.0,
        help="Blank first/last N seconds of each recording for eval (default: 0)",
    )
    p_eval3.add_argument(
        "--edge-exclusion",
        action="store_true",
        help="Targeted exclusion of end-truncated and pre-recording artifacts",
    )

    args = parser.parse_args()
    dispatch = {
        "fit-m0": _cli_fit_m0,
        "predict-m0": _cli_predict_m0,
        "eval-m0": _cli_eval_m0,
        "fit-m1": _cli_fit_m1,
        "predict-m1": _cli_predict_m1,
        "eval-m1": _cli_eval_m1,
        "fit-m2": _cli_fit_m2,
        "predict-m2": _cli_predict_m2,
        "eval-m2": _cli_eval_m2,
        "fit-m3": _cli_fit_m3,
        "predict-m3": _cli_predict_m3,
        "eval-m3": _cli_eval_m3,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
