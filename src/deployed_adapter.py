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
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Adapter for the training pipeline trained U-Net model.

Loads the model architecture and checkpoint from the autonomous HP search
(the training pipeline, commit 5b0d152) and exposes it through the same interface
that stim_detector_lib.py uses, so existing ablation/explainability scripts
work without modification.

Usage:
    from deployed_adapter import load_deployed_checkpoint

    model, cfg = load_deployed_checkpoint("data/checkpoints/m4_unet.pt")
    # model is compatible with predict_file_proba(model, cfg, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_stimask_config():
    """Lazy import to avoid lgs_db dependency at module load."""
    from stim_detector_lib import Config

    return Config


# ---------------------------------------------------------------------------
# Frozen hyperparameters from training commit 5b0d152
# ---------------------------------------------------------------------------

_BASE_CHANNELS = 48
_CHANNEL_MULT = (1, 2, 4, 8)
_BOTTLENECK_CHANNELS = 256
_DROPOUT = 0.20
_USE_FILM = True
_FILM_LAST_ONLY = True
_FILM_HIDDEN_DIM = 128
_DEEP_SUPERVISION = False
_N_CHANNELS = 4
_N_COND_FEATURES = 32
_KERNEL_SIZE = 3  # best model used kernel_size=3


# ---------------------------------------------------------------------------
# Model Architecture (copied verbatim from the training pipeline @ 5b0d152)
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = _KERNEL_SIZE,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FiLMGenerator(nn.Module):
    """Generate per-layer (gamma, beta) FiLM parameters from conditioning vector."""

    def __init__(self, n_cond: int, channel_dims: list[int], hidden_dim: int = 128):
        super().__init__()
        self.channel_dims = channel_dims
        total_out = 2 * sum(channel_dims)

        self.mlp = nn.Sequential(
            nn.Linear(n_cond, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, total_out),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        total_gamma = sum(self.channel_dims)
        with torch.no_grad():
            last.bias[:total_gamma] = 1.0
            last.bias[total_gamma:] = 0.0

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = self.mlp(cond)
        total_gamma = sum(self.channel_dims)
        pairs = []
        g_off, b_off = 0, total_gamma
        for dim in self.channel_dims:
            gamma = out[:, g_off : g_off + dim]
            beta = out[:, b_off : b_off + dim]
            pairs.append((gamma, beta))
            g_off += dim
            b_off += dim
        return pairs


class ConditionalBlock(nn.Module):
    """Conv -> GroupNorm -> FiLM -> GELU -> Residual -> Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = _KERNEL_SIZE,
        dilation: int = 1,
        dropout: float = 0.05,
        bn_affine: bool = True,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, dilation)
        num_groups = min(8, out_ch)
        self.bn = nn.GroupNorm(num_groups, out_ch, affine=bn_affine)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()
        self.residual = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        )

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        identity = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = self.act(x)
        x = x + identity
        x = self.drop(x)
        return x


class EncoderLevel(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, dropout: float = 0.05, bn_affine: bool = True
    ):
        super().__init__()
        self.block1 = ConditionalBlock(
            in_ch, out_ch, dropout=dropout, bn_affine=bn_affine
        )
        self.block2 = ConditionalBlock(
            out_ch, out_ch, dropout=dropout, bn_affine=bn_affine
        )
        self.pool = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(
        self, x: torch.Tensor, film1: tuple, film2: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderLevel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        dropout: float = 0.05,
        bn_affine: bool = True,
    ):
        super().__init__()
        self.block1 = ConditionalBlock(
            in_ch + skip_ch, out_ch, dropout=dropout, bn_affine=bn_affine
        )
        self.block2 = ConditionalBlock(
            out_ch, out_ch, dropout=dropout, bn_affine=bn_affine
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, film1: tuple, film2: tuple
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


class SelfAttention1d(nn.Module):
    """Multi-head self-attention via scaled_dot_product_attention."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1, bias=False)
        self.out_proj = nn.Conv1d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, T)
        q, k, v = qkv.unbind(1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, T)
        return x + self.out_proj(out)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.05,
        bn_affine: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.block1 = ConditionalBlock(
            in_ch, out_ch, dilation=1, dropout=dropout, bn_affine=bn_affine
        )
        self.attn = SelfAttention1d(out_ch) if use_attention else nn.Identity()
        self.block2 = ConditionalBlock(
            out_ch, out_ch, dilation=4, dropout=dropout, bn_affine=bn_affine
        )

    def forward(self, x: torch.Tensor, film1: tuple, film2: tuple) -> torch.Tensor:
        x = self.block1(x, *film1)
        x = self.attn(x)
        x = self.block2(x, *film2)
        return x


class DeployedStimArtifactUNet(nn.Module):
    """Deployed 1D U-Net with FiLM conditioning.

    Architecture from training commit 5b0d152 (ta_F1=0.834).
    Key differences from stimask StimArtifactUNet:
    - kernel_size=3 (not 7)
    - GroupNorm (not BatchNorm)
    - GELU (not SiLU)
    - No SE blocks in encoder/decoder
    - No bottleneck attention
    """

    def __init__(
        self,
        n_channels: int = _N_CHANNELS,
        n_cond_features: int = _N_COND_FEATURES,
        base_channels: int = _BASE_CHANNELS,
        channel_mult: tuple = _CHANNEL_MULT,
        bottleneck_channels: int = _BOTTLENECK_CHANNELS,
        dropout: float = _DROPOUT,
        use_film: bool = _USE_FILM,
        film_last_only: bool = _FILM_LAST_ONLY,
        film_hidden_dim: int = _FILM_HIDDEN_DIM,
        deep_supervision: bool = _DEEP_SUPERVISION,
    ):
        super().__init__()
        bn_affine = True

        enc_channels = [base_channels * m for m in channel_mult]
        dec_channels = enc_channels[::-1]

        self.encoders = nn.ModuleList()
        in_ch = n_channels
        for out_ch in enc_channels:
            self.encoders.append(
                EncoderLevel(in_ch, out_ch, dropout, bn_affine=bn_affine)
            )
            in_ch = out_ch

        self.bottleneck = Bottleneck(
            enc_channels[-1],
            bottleneck_channels,
            dropout,
            bn_affine=bn_affine,
        )

        self.decoders = nn.ModuleList()
        in_ch = bottleneck_channels
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i + 1)]
            self.decoders.append(
                DecoderLevel(in_ch, skip_ch, out_ch, dropout, bn_affine=bn_affine)
            )
            in_ch = out_ch

        self.head = nn.Conv1d(dec_channels[-1], 1, kernel_size=1)

        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [nn.Conv1d(ch, 1, kernel_size=1) for ch in dec_channels[:-1]]
            )
        else:
            self.aux_heads = None

        self._film_dims: list[int] = []
        for ch in enc_channels:
            self._film_dims.extend([ch, ch])
        self._film_dims.extend([bottleneck_channels, bottleneck_channels])
        for ch in dec_channels:
            self._film_dims.extend([ch, ch])

        self.film_gen = FiLMGenerator(n_cond_features, self._film_dims, film_hidden_dim)

        n_films = len(self._film_dims)
        if use_film and not film_last_only:
            mask_vals = [1.0] * n_films
        elif use_film and film_last_only:
            mask_vals = [0.0] * (n_films - 2) + [1.0, 1.0]
        else:
            mask_vals = [0.0] * n_films
        self.register_buffer("_film_mask", torch.tensor(mask_vals))

    def set_film_mode(self, use_film: bool, film_last_only: bool = False) -> None:
        """Switch FiLM mode at runtime without recompilation."""
        n = len(self._film_dims)
        if use_film and not film_last_only:
            self._film_mask.fill_(1.0)
        elif use_film and film_last_only:
            self._film_mask[: n - 2] = 0.0
            self._film_mask[n - 2 :] = 1.0
        else:
            self._film_mask.fill_(0.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits only (eval-compatible).

        The original deployed model always returned (logits, aux_list).
        This version returns logits only, matching the stimask convention
        expected by predict_file_proba().
        """
        films = self.film_gen(cond)
        films = [
            (1.0 + m * (g - 1.0), m * b) for (g, b), m in zip(films, self._film_mask)
        ]
        fi = 0

        skips = []
        for enc in self.encoders:
            x, skip = enc(x, films[fi], films[fi + 1])
            skips.append(skip)
            fi += 2

        x = self.bottleneck(x, films[fi], films[fi + 1])
        fi += 2

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)], films[fi], films[fi + 1])
            fi += 2

        return self.head(x)


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------


def load_deployed_checkpoint(
    path: str | Path = "data/checkpoints/m4_unet.pt",
    device: str = "mps",
) -> tuple[nn.Module, "Config"]:
    """Load the deployed U-Net checkpoint.

    Returns (model, cfg) compatible with predict_file_proba() and the
    existing ablation/explainability scripts.

    Args:
        path: Path to the checkpoint file (best_model.pt state dict).
        device: Target device ('mps', 'cuda', 'cpu').

    Returns:
        model: DeployedStimArtifactUNet in eval mode on device.
        cfg: stimask Config instance with matching hyperparameters.
    """
    Config = _get_stimask_config()
    model = DeployedStimArtifactUNet(
        n_channels=_N_CHANNELS,
        n_cond_features=_N_COND_FEATURES,
        base_channels=_BASE_CHANNELS,
        channel_mult=_CHANNEL_MULT,
        bottleneck_channels=_BOTTLENECK_CHANNELS,
        dropout=_DROPOUT,
        use_film=_USE_FILM,
        film_last_only=_FILM_LAST_ONLY,
        film_hidden_dim=_FILM_HIDDEN_DIM,
        deep_supervision=_DEEP_SUPERVISION,
    )

    # The deployed checkpoint is a raw state_dict (not wrapped in a config dict).
    # Keys may have _orig_mod. prefix from torch.compile — strip it.
    raw_state = torch.load(str(path), map_location=device, weights_only=True)
    state_dict = {}
    for k, v in raw_state.items():
        clean_key = k.removeprefix("_orig_mod.")
        state_dict[clean_key] = v
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    cfg = Config(
        catalog_path=Path("data/stim_catalog.parquet"),
        checkpoint_dir=Path("data/checkpoints"),
        sampling_rate=250,
        n_channels=_N_CHANNELS,
        window_samples=2048,
        stride_samples=1024,
        inference_stride_ratio=0.50,
        n_cond_features=_N_COND_FEATURES,
        base_channels=_BASE_CHANNELS,
        channel_mult=_CHANNEL_MULT,
        bottleneck_channels=_BOTTLENECK_CHANNELS,
        use_film=_USE_FILM,
        film_last_only=_FILM_LAST_ONLY,
        film_hidden_dim=_FILM_HIDDEN_DIM,
        deep_supervision=_DEEP_SUPERVISION,
        dropout=_DROPOUT,
        boundary_margin_s=4.0,
        onset_taper_samples=10,
        offset_taper_samples=20,
        min_artifact_samples=75,
        merge_gap_ms=300.0,
        device=device,
        use_ema=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded deployed checkpoint from {path} ({n_params / 1e6:.1f}M params)")
    return model, cfg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from stim_detector_lib import (
        build_true_mask,
        extract_conditioning_vector,
        load_annotations,
        predict_file_proba,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg = load_deployed_checkpoint(device=device)

    df = load_annotations(cfg)
    val_subjects = ["300-002", "301-003", "303-004", "305-001"]
    val_df = df[df["subject"].isin(val_subjects)]
    print(f"Val set: {len(val_df)} files, {val_df['subject'].nunique()} subjects")

    # Predict first val file
    row = val_df.iloc[0]
    cond = extract_conditioning_vector(row)
    proba = predict_file_proba(model, cfg, row["file_path"], cond, batch_size=256)

    n_samples = len(proba)
    onset_times = row["onset_times"]
    if isinstance(onset_times, str):
        import json

        onset_times = json.loads(onset_times)
    true_mask = build_true_mask(
        n_samples, onset_times, row["mask_duration_ms"], cfg.sampling_rate
    )

    pred_bin = (proba > 0.5).astype(np.int8)
    tp = int((pred_bin & true_mask).sum())
    fp = int((pred_bin & ~true_mask.astype(bool)).sum())
    fn = int((~pred_bin.astype(bool) & true_mask).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    print(f"File: {row['file_path']}")
    print(f"  n_samples={n_samples}, n_events={len(onset_times)}")
    print(f"  Sample F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
    print("Smoke test passed." if f1 > 0.5 else "WARNING: low F1, check adapter.")
