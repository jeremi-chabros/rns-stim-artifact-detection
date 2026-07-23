"""Stimulation Artifact Detector — library module.

Multi-Scale Conditional 1D U-Net with FiLM conditioning for
NeuroPace RNS ECoG stim artifact detection.

Contains: conditioning parsers, model architecture, loss functions,
dataset / data loading, post-processing / metrics, and trainer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from lgs_db import read_dat, to_microvolts
from scipy.ndimage import find_objects, label
from scipy.special import ndtr
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Full configuration for model, data, and training."""

    # ---------- Paths ----------
    catalog_path: Path = Path("data/stim_catalog.parquet")
    checkpoint_dir: Path = Path("checkpoints")

    # ---------- Signal ----------
    sampling_rate: int = 250
    n_channels: int = 4
    use_channels: list[int] | None = None
    window_samples: int = 2048
    stride_samples: int = 1024
    inference_stride_ratio: float = 0.50

    # ---------- Conditioning dimensions ----------
    n_cond_features: int = 32

    # ---------- Model architecture ----------
    base_channels: int = 32
    channel_mult: tuple = (1, 2, 4, 8)
    bottleneck_channels: int = 512
    use_film: bool = False
    film_last_only: bool = False  # FiLM at last decoder level only (Dec1)
    film_hidden_dim: int = 128
    deep_supervision: bool = True
    deep_sup_weight: float = 0.3
    dropout: float = 0.10

    # ---------- Training ----------
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-3
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6
    preload_to_ram: bool = True
    val_every_n_epochs: int = 1
    train_file_subsample: float = 1.0  # fraction of files per epoch (1.0 = all)

    # --- LR scheduling ---
    lr_scheduler: str = "plateau"
    warmup_epochs: int = 5
    plateau_patience: int = 4
    plateau_factor: float = 0.2
    plateau_cooldown: int = 0

    # ---------- Boundary exclusion ----------
    boundary_margin_s: float = 4.0

    # ---------- Soft labels (Gaussian CDF boundary taper) ----------
    onset_taper_samples: int = 10  # Gaussian taper width at artifact onset; 0 disables
    offset_taper_samples: int = (
        20  # Gaussian taper width at artifact offset; 0 disables
    )
    checkpoint_metric: str = "iou"

    # ---------- Post-processing ----------
    min_artifact_samples: int = 75
    merge_gap_ms: float = 300.0

    # ---------- Compute options ----------
    num_workers: int = 0  # 0 when preload_to_ram; workers duplicate the cache
    use_compile: bool = False
    amp_dtype: str = "float16"

    # ---------- Device ----------
    device: str = "mps"

    # ---------- EMA ----------
    use_ema: bool = True
    ema_decay: float = 0.9999

    def __post_init__(self):
        if self.use_channels is not None:
            self.n_channels = len(self.use_channels)


# ---------------------------------------------------------------------------
# Conditioning Parsers
# ---------------------------------------------------------------------------


class MontageParser:
    """Parse RNS electrode montage string to numeric vector.

    Montage string formats:
      Legacy CSV:  "0-00|----|+"
      DB/parquet:  "(0-00)(----)(+)"
    Output: length-9 vector [L1_c1..c4, L2_c1..c4, case]
    """

    _MAP = {"0": 0.0, "-": -1.0, "+": 1.0}

    @classmethod
    def parse_single(cls, s: str) -> np.ndarray:
        if pd.isna(s) or s == "" or s == "Disabled":
            return np.zeros(9, dtype=np.float32)
        clean = s.replace("|", "").replace("(", "").replace(")", "")
        try:
            vec = [cls._MAP[c] for c in clean]
        except KeyError:
            return np.zeros(9, dtype=np.float32)
        if len(vec) != 9:
            return np.zeros(9, dtype=np.float32)
        return np.array(vec, dtype=np.float32)

    @classmethod
    def parse_batch(cls, montage_list: list[str]) -> torch.Tensor:
        return torch.tensor(
            np.stack([cls.parse_single(s) for s in montage_list]), dtype=torch.float32
        )


class LeadParser:
    """Parse RNS lead type string to numeric features.

    Output: [is_depth, spacing_mm_normalized]
    """

    _PATTERN = re.compile(r"([DC])(\d+\.?\d*)")

    @classmethod
    def parse_single(cls, s: str) -> np.ndarray:
        if pd.isna(s) or s == "":
            return np.zeros(2, dtype=np.float32)
        m = cls._PATTERN.match(str(s).strip())
        if not m:
            return np.zeros(2, dtype=np.float32)
        is_depth = 1.0 if m.group(1) == "D" else 0.0
        spacing = float(m.group(2)) / 10.0
        return np.array([is_depth, spacing], dtype=np.float32)


PARAM_RANGES = {
    "ma": (0.0, 12.0),
    "us": (40.0, 400.0),
    "uc": (0.0, 50.0),
    "hz": (1.0, 333.0),
    "ms": (10.0, 5000.0),
}

STIM_PARAM_SUFFIXES = ["ma", "us", "uc", "hz", "ms"]


def extract_conditioning_vector(row: pd.Series) -> np.ndarray:
    """Extract full (32,) conditioning vector from a catalog row.

    Layout: [B1_params(5), B1_montage(9), B2_params(5), B2_montage(9),
             lead1(2), lead2(2)] = 32 dims.
    """
    parts = []

    for bank in ["t1b1", "t1b2"]:
        params = []
        for suffix in STIM_PARAM_SUFFIXES:
            val = row.get(f"{bank}_{suffix}", 0.0)
            if pd.isna(val):
                val = 0.0
            lo, hi = PARAM_RANGES[suffix]
            params.append(np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0))
        parts.append(np.array(params, dtype=np.float32))
        montage = row.get(f"{bank}_path", "")
        parts.append(MontageParser.parse_single(montage))

    parts.append(LeadParser.parse_single(row.get("lead_1", "")))
    parts.append(LeadParser.parse_single(row.get("lead_2", "")))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------


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

    def _init_identity(self):
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


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
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


class ConditionalBlock(nn.Module):
    """Conv -> BN -> FiLM -> SiLU -> Residual -> Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.05,
        bn_affine: bool = True,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_ch, affine=bn_affine)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
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
        self.se = SqueezeExcite1d(out_ch)
        self.pool = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor, film1: tuple, film2: tuple):
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        x = self.se(x)
        skip = x
        x = self.pool(x)
        return x, skip


class SqueezeExcite1d(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D signals."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        s = x.mean(dim=-1)  # (B, C) global average pool
        s = self.fc(s).unsqueeze(-1)  # (B, C, 1)
        return x * s


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
        self.se = SqueezeExcite1d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, film1: tuple, film2: tuple):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        x = self.se(x)
        return x


# class Bottleneck(nn.Module):
#     """ASPP-style multi-scale bottleneck with FiLM-conditioned refinement."""

#     def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05,
#                  bn_affine: bool = True):
#         super().__init__()
#         # Parallel dilated branches (FiLM-free multi-scale feature extraction)
#         dilations = [1, 2, 4, 8]
#         branch_ch = out_ch // len(dilations)
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 DepthwiseSeparableConv1d(in_ch, branch_ch, kernel_size=7, dilation=d),
#                 nn.BatchNorm1d(branch_ch),
#                 nn.SiLU(inplace=True),
#             )
#             for d in dilations
#         ])
#         self.fuse = nn.Sequential(
#             nn.Conv1d(branch_ch * len(dilations), out_ch, 1, bias=False),
#             nn.BatchNorm1d(out_ch),
#             nn.SiLU(inplace=True),
#         )
#         self.self_attn = BottleneckAttention(out_ch, num_heads=4)
#         # FiLM-conditioned refinement (preserves 2 FiLM pairs interface)
#         self.block1 = ConditionalBlock(out_ch, out_ch, dilation=1, dropout=dropout,
#                                        bn_affine=bn_affine)
#         self.block2 = ConditionalBlock(out_ch, out_ch, dilation=4, dropout=dropout,
#                                        bn_affine=bn_affine)

#     def forward(self, x: torch.Tensor, film1: tuple, film2: tuple):
#         branches = [b(x) for b in self.branches]
#         x = self.fuse(torch.cat(branches, dim=1))
#         x = self.self_attn(x)
#         x = self.block1(x, *film1)
#         x = self.block2(x, *film2)
#         return x


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05, bn_affine=True):
        super().__init__()
        self.block1 = ConditionalBlock(
            in_ch, out_ch, dilation=1, dropout=dropout, bn_affine=bn_affine
        )
        self.block2 = ConditionalBlock(
            out_ch, out_ch, dilation=4, dropout=dropout, bn_affine=bn_affine
        )

    def forward(self, x, film1, film2):
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


class BottleneckAttention(nn.Module):
    """Lightweight self-attention at bottleneck resolution."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x_t = x.permute(0, 2, 1)
        normed = self.norm(x_t)
        x_t = x_t + self.attn(normed, normed, normed, need_weights=False)[0]
        return x_t.permute(0, 2, 1)


class StimArtifactUNet(nn.Module):
    """Multi-Scale Conditional 1D U-Net with FiLM."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        base = cfg.base_channels
        mult = cfg.channel_mult
        # bn_affine = not cfg.use_film  # FiLM provides affine; avoid redundancy
        bn_affine = True

        enc_channels = [base * m for m in mult]
        dec_channels = enc_channels[::-1]

        self.encoders = nn.ModuleList()
        in_ch = cfg.n_channels
        for out_ch in enc_channels:
            self.encoders.append(
                EncoderLevel(in_ch, out_ch, cfg.dropout, bn_affine=bn_affine)
            )
            in_ch = out_ch

        self.bottleneck = Bottleneck(
            enc_channels[-1],
            cfg.bottleneck_channels,
            cfg.dropout,
            bn_affine=bn_affine,
        )

        self.decoders = nn.ModuleList()
        in_ch = cfg.bottleneck_channels
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i + 1)]
            self.decoders.append(
                DecoderLevel(in_ch, skip_ch, out_ch, cfg.dropout, bn_affine=bn_affine)
            )
            in_ch = out_ch

        self.head = nn.Conv1d(dec_channels[-1], 1, kernel_size=1)

        # Deep supervision: auxiliary heads at intermediate decoder levels
        if cfg.deep_supervision:
            self.aux_heads = nn.ModuleList(
                [nn.Conv1d(ch, 1, kernel_size=1) for ch in dec_channels[:-1]]
            )
        else:
            self.aux_heads = None

        self._film_dims = []
        for ch in enc_channels:
            self._film_dims.extend([ch, ch])
        self._film_dims.extend([cfg.bottleneck_channels, cfg.bottleneck_channels])
        for ch in dec_channels:
            self._film_dims.extend([ch, ch])

        # Always create FiLMGenerator with full dims so the compiled graph is
        # identical regardless of use_film / film_last_only.  Behaviour is
        # controlled by _film_mask (a tensor buffer, not a Python bool) so
        # torch.compile traces a single graph for all FiLM modes.
        self.film_gen = FiLMGenerator(
            cfg.n_cond_features, self._film_dims, cfg.film_hidden_dim
        )
        # _film_mask[i] = 1 → active FiLM, 0 → identity (gamma=1, beta=0).
        n_films = len(self._film_dims)
        if cfg.use_film and not cfg.film_last_only:
            mask_vals = [1.0] * n_films
        elif cfg.use_film and cfg.film_last_only:
            mask_vals = [0.0] * (n_films - 2) + [1.0, 1.0]
        else:
            mask_vals = [0.0] * n_films
        self.register_buffer("_film_mask", torch.tensor(mask_vals))

    def set_film_mode(self, use_film: bool, film_last_only: bool = False):
        """Switch FiLM mode at runtime without recompilation."""
        n = len(self._film_dims)
        if use_film and not film_last_only:
            self._film_mask.fill_(1.0)
        elif use_film and film_last_only:
            self._film_mask[: n - 2] = 0.0
            self._film_mask[n - 2 :] = 1.0
        else:
            self._film_mask.fill_(0.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # Always run FiLM generator — no Python branching.
        # _film_mask blends between identity and learned FiLM per layer:
        #   mask=0 → (1, 0) identity;  mask=1 → (gamma, beta) learned.
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

        aux_outputs = []
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)], films[fi], films[fi + 1])
            fi += 2
            if (
                self.training
                and self.aux_heads is not None
                and i < len(self.decoders) - 1
            ):
                aux_outputs.append(self.aux_heads[i](x))

        main = self.head(x)
        if self.training and aux_outputs:
            return main, aux_outputs
        return main


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, prob, 1 - prob)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        per_sample = alpha_t * (1 - pt) ** self.gamma * bce
        if weight is not None:
            per_sample = per_sample * weight
            denom = weight.sum().clamp(min=1.0)
            return per_sample.sum() / denom
        return per_sample.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        if weight is not None:
            p = (prob * weight).view(-1)
            t = (target * weight).view(-1)
        else:
            p = prob.view(-1)
            t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.dice = DiceLoss()
        self.fw = cfg.focal_weight
        self.dw = cfg.dice_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> dict:
        fl = self.focal(pred, target, weight)
        dl = self.dice(pred, target, weight)
        return {"loss": self.fw * fl + self.dw * dl, "focal": fl, "dice": dl}


# ---------------------------------------------------------------------------
# Dataset & Data Loading
# ---------------------------------------------------------------------------


def parse_onset_times(onset_str: str) -> np.ndarray:
    """Parse onset_times from JSON string (parquet catalog format)."""
    import json

    if pd.isna(onset_str) or str(onset_str).strip() == "":
        return np.array([], dtype=np.float64)
    try:
        return np.array(sorted(json.loads(onset_str)), dtype=np.float64)
    except (json.JSONDecodeError, TypeError):
        return np.array([], dtype=np.float64)


def _robust_scale_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(data, axis=1).astype(np.float32)
    q1 = np.percentile(data, 25, axis=1).astype(np.float32)
    q3 = np.percentile(data, 75, axis=1).astype(np.float32)
    iqr = (q3 - q1).astype(np.float32)
    return median, iqr


def _load_dat_channels(
    dat_path: Path, use_channels: list[int] | None = None
) -> np.ndarray:
    """Load ECoG from .dat file via lgs_db, return float32 µV array."""
    raw = read_dat(str(dat_path))  # (4, n_samples) int16
    uv = to_microvolts(raw)  # (4, n_samples) float64 µV
    if use_channels is not None:
        # use_channels is 1-indexed
        uv = uv[[c - 1 for c in use_channels]]
    return uv.astype(np.float32)


def _soften_mask_gaussian(
    mask: np.ndarray,
    onset_taper: int,
    offset_taper: int,
) -> np.ndarray:
    """Apply Gaussian CDF soft-label taper at artifact onset/offset boundaries.

    Hard 0→1 (onset) and 1→0 (offset) transitions are replaced with
    Gaussian CDF ramps.  Each taper parameter specifies the number of
    samples over which the transition spans (3-sigma convention, so
    sigma = taper / 3).  The taper extends slightly beyond the hard
    boundary, naturally encoding annotation timing uncertainty.
    """
    if (onset_taper <= 0 and offset_taper <= 0) or mask.max() == 0:
        return mask.copy()

    length = len(mask)
    soft = np.zeros(length, dtype=np.float32)
    labeled, n_feat = scipy.ndimage.label(mask > 0.5)
    x = np.arange(length, dtype=np.float64)

    for comp in range(1, n_feat + 1):
        indices = np.where(labeled == comp)[0]
        art_start = float(indices[0])
        art_end = float(indices[-1] + 1)

        if onset_taper > 0:
            sigma_on = onset_taper / 3.0
            rise = ndtr((x - art_start + 0.5) / sigma_on)
        else:
            rise = (x >= art_start).astype(np.float64)

        if offset_taper > 0:
            sigma_off = offset_taper / 3.0
            fall = ndtr((art_end - 0.5 - x) / sigma_off)
        else:
            fall = (x < art_end).astype(np.float64)

        np.maximum(soft, np.minimum(rise, fall).astype(np.float32), out=soft)

    return soft


class StimArtifactDataset(Dataset):
    """Sliding-window dataset for stim artifact detection."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        augment: bool = True,
    ):
        self.cfg = cfg
        self.augment = augment
        self.use_channels = cfg.use_channels
        self.data_cache: dict[str, np.ndarray] = {}
        self.file_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.mask_cache: dict[str, np.ndarray] = {}
        self.weight_cache: dict[str, np.ndarray] = {}
        self.samples: list[dict] = []

        if cfg.preload_to_ram:
            unique_paths = set(df["file_path"].unique())
            print(f"Pre-loading {len(unique_paths)} .dat files into RAM...")
            for p in tqdm(sorted(unique_paths), desc="Caching"):
                if Path(p).exists():
                    self._cache_file(Path(p))
            mem_gb = sum(a.nbytes for a in self.data_cache.values()) / 1e9
            print(f"Cached {len(self.data_cache)} files ({mem_gb:.2f} GB)")

        use_soft = augment and (
            cfg.onset_taper_samples > 0 or cfg.offset_taper_samples > 0
        )
        fs = cfg.sampling_rate
        margin = int(cfg.boundary_margin_s * fs)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building samples"):
            fp = row["file_path"]
            if not Path(fp).exists():
                continue
            onsets = row["onset_times"]
            if len(onsets) == 0:
                continue

            if cfg.preload_to_ram and fp in self.data_cache:
                n_samples = self.data_cache[fp].shape[1]
            else:
                raw = read_dat(fp)
                n_samples = raw.shape[1]

            mask_dur_ms = row.get("mask_duration_ms", 1000.0)
            if pd.isna(mask_dur_ms):
                mask_dur_ms = 1000.0
            mask_dur_samples = max(int(mask_dur_ms / 1000.0 * fs), 1)

            # Onset offset from manual refinement (negative = start before trigger)
            onset_offset_ms = row.get("mask_onset_offset_ms", 0.0)
            if pd.isna(onset_offset_ms):
                onset_offset_ms = 0.0
            onset_offset_samples = int(float(onset_offset_ms) / 1000.0 * fs)

            # Pre-compute per-file mask and weight (once per file)
            if fp not in self.mask_cache:
                hard_mask = np.zeros(n_samples, dtype=np.float32)
                for t_sec in onsets:
                    s_idx = int(t_sec * fs) + onset_offset_samples
                    hard_mask[
                        max(0, s_idx) : min(n_samples, s_idx + mask_dur_samples)
                    ] = 1.0

                if use_soft:
                    self.mask_cache[fp] = _soften_mask_gaussian(
                        hard_mask,
                        cfg.onset_taper_samples,
                        cfg.offset_taper_samples,
                    )
                else:
                    self.mask_cache[fp] = hard_mask

                weight = np.ones(n_samples, dtype=np.float32)
                weight[:margin] = 0.0
                if n_samples > margin:
                    weight[max(margin, n_samples - margin) :] = 0.0
                self.weight_cache[fp] = weight

            cond = extract_conditioning_vector(row)

            for start in range(
                0, max(1, n_samples - cfg.window_samples), cfg.stride_samples
            ):
                end = start + cfg.window_samples
                if end > n_samples:
                    break
                self.samples.append(
                    {
                        "file_path": fp,
                        "file_id": row["filename"],
                        "rec_len": n_samples,
                        "start": start,
                        "end": end,
                        "mask_dur": mask_dur_samples,
                        "cond": cond,
                    }
                )

        print(f"Total samples: {len(self.samples)}")
        mask_mem_gb = (
            sum(a.nbytes for a in self.mask_cache.values())
            + sum(a.nbytes for a in self.weight_cache.values())
        ) / 1e9
        print(f"Mask cache: {len(self.mask_cache)} files ({mask_mem_gb:.2f} GB)")

        if not cfg.preload_to_ram:
            unique_paths = {s["file_path"] for s in self.samples}
            for p in tqdm(unique_paths, desc="File stats"):
                if p not in self.file_stats:
                    data = _load_dat_channels(Path(p), self.use_channels)
                    self.file_stats[p] = _robust_scale_stats(data)

    def _cache_file(self, dat_path: Path) -> None:
        data = _load_dat_channels(dat_path, self.use_channels)
        self.data_cache[str(dat_path)] = data
        self.file_stats[str(dat_path)] = _robust_scale_stats(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        key = s["file_path"]

        if key in self.data_cache:
            sig = self.data_cache[key][:, s["start"] : s["end"]].copy()
        else:
            data = _load_dat_channels(Path(key), self.use_channels)
            sig = data[:, s["start"] : s["end"]].copy()

        median, iqr = self.file_stats[key]
        sig = (sig - median[:, np.newaxis]) / (iqr[:, np.newaxis] + 1e-8)

        mask = self.mask_cache[key][s["start"] : s["end"]].copy()
        weight = self.weight_cache[key][s["start"] : s["end"]].copy()

        if self.augment:
            sig, mask = self._augment(sig, mask)

        return {
            "signal": torch.from_numpy(sig),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "weight": torch.from_numpy(weight).unsqueeze(0),
            "cond": torch.from_numpy(s["cond"]),
            "file_id": s["file_id"],
            "start": torch.tensor(s["start"], dtype=torch.long),
            "end": torch.tensor(s["end"], dtype=torch.long),
            "rec_len": torch.tensor(s["rec_len"], dtype=torch.long),
            "mask_dur": torch.tensor(s["mask_dur"], dtype=torch.long),
        }

    def _augment(self, sig: np.ndarray, mask: np.ndarray):
        # Temporal jitter: shift signal+mask by ±5 samples (~±20ms at 250Hz)
        # to prevent overfitting to exact annotation positions
        if np.random.random() < 0.4:
            shift = np.random.randint(-10, 11)
            if shift != 0:
                sig = np.roll(sig, shift, axis=1)
                mask = np.roll(mask, shift, axis=0)
                # Zero the wrapped-around edges
                if shift > 0:
                    sig[:, :shift] = 0.0
                    mask[:shift] = 0.0
                else:
                    sig[:, shift:] = 0.0
                    mask[shift:] = 0.0
        # Amplitude jitter (cross-patient gain variation)
        if np.random.random() < 0.5:
            sig *= np.random.uniform(0.8, 1.2)
        # Additive Gaussian noise (background ECoG variability)
        if np.random.random() < 0.3:
            sigma = np.random.uniform(0.01, 0.15)
            sig += np.random.normal(0, sigma, sig.shape).astype(np.float32)
        # Channel dropout: zero an entire channel (simulates dead electrode)
        if sig.shape[0] > 1 and np.random.random() < 0.1:
            ch = np.random.randint(0, sig.shape[0])
            sig[ch] = 0.0
        # Channel-wise noise injection (degraded electrode)
        if sig.shape[0] > 1 and np.random.random() < 0.15:
            ch = np.random.randint(0, sig.shape[0])
            sig[ch] += np.random.normal(0, 0.3, sig.shape[1]).astype(np.float32)
        # Per-channel DC offset (residual drift after median/IQR scaling)
        if np.random.random() < 0.3:
            offsets = np.random.uniform(-0.5, 0.5, (sig.shape[0], 1)).astype(np.float32)
            sig += offsets
        # Time masking: zero a random contiguous block (SpecAugment-style)
        if np.random.random() < 0.2:
            max_width = min(50, sig.shape[1] // 10)
            width = np.random.randint(5, max_width + 1)
            start = np.random.randint(0, sig.shape[1] - width)
            sig[:, start : start + width] = 0.0
        if np.random.random() < 0.3:
            sig = -sig
        return sig, mask


def load_annotations(cfg: Config) -> pd.DataFrame:
    """Load stim catalog parquet and parse onset_times."""
    df = pd.read_parquet(cfg.catalog_path)
    df["onset_times"] = df["onset_times"].apply(parse_onset_times)
    df = df[df["onset_times"].apply(len) > 0].reset_index(drop=True)
    return df


class FileSubsetSampler(Sampler):
    """Randomly subsample a fraction of files each epoch, rotating coverage.

    Groups window indices by source file, then each epoch selects a random
    ``subsample`` fraction of files and yields all their window indices.
    """

    def __init__(self, dataset: StimArtifactDataset, subsample: float = 1.0):
        self.subsample = subsample
        # Group sample indices by file
        self.file_indices: dict[str, list[int]] = {}
        for i, s in enumerate(dataset.samples):
            fp = s["file_path"]
            if fp not in self.file_indices:
                self.file_indices[fp] = []
            self.file_indices[fp].append(i)
        self.file_list = list(self.file_indices.keys())
        self.epoch = 0
        self._total = len(dataset)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.epoch)
        n_files = max(1, int(len(self.file_list) * self.subsample))
        selected = rng.choice(self.file_list, size=n_files, replace=False)
        indices = []
        for fp in selected:
            indices.extend(self.file_indices[fp])
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        n_files = max(1, int(len(self.file_list) * self.subsample))
        avg_per_file = self._total / max(len(self.file_list), 1)
        return int(n_files * avg_per_file)


# ---------------------------------------------------------------------------
# Post-Processing & Metrics
# ---------------------------------------------------------------------------


def post_process_mask(pred_mask: np.ndarray, cfg: Config) -> np.ndarray:
    gap_samples = int(cfg.merge_gap_ms / 1000.0 * cfg.sampling_rate)
    struct = np.ones(max(gap_samples, 1))
    closed = scipy.ndimage.binary_closing(pred_mask, structure=struct).astype(int)

    labeled, n_feat = scipy.ndimage.label(closed)
    result = np.zeros_like(closed)
    for i in range(1, n_feat + 1):
        component = labeled == i
        if component.sum() >= cfg.min_artifact_samples:
            result[component] = 1
    return result


def compute_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    weight: torch.Tensor | None = None,
) -> dict[str, float]:
    pred_bin = (pred > threshold).float()
    if weight is not None:
        w = (weight > 0).float()
        pred_bin = pred_bin * w
        target = target * w
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def compute_event_metrics(
    pred_mask: np.ndarray, true_mask: np.ndarray, min_iou: float = 0.3
) -> dict[str, float]:
    pred_lab, n_pred = label(pred_mask)
    true_lab, n_true = label(true_mask)

    if n_pred == 0 and n_true == 0:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0}
    if n_pred == 0:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0}
    if n_true == 0:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0}

    pred_slices = find_objects(pred_lab)
    true_slices = find_objects(true_lab)

    tp = 0
    matched = set()
    for i, ps in enumerate(pred_slices):
        best_iou, best_j = 0.0, -1
        pm = pred_lab == (i + 1)
        for j, ts in enumerate(true_slices):
            if j in matched:
                continue
            if ps[0].start >= ts[0].stop or ps[0].stop <= ts[0].start:
                continue
            tm = true_lab == (j + 1)
            inter = np.logical_and(pm, tm).sum()
            union = np.logical_or(pm, tm).sum()
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= min_iou:
            tp += 1
            matched.add(best_j)

    fp = n_pred - tp
    fn = n_true - len(matched)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"event_precision": prec, "event_recall": rec, "event_f1": f1}


def compute_onset_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    fs: int = 250,
    tolerance_ms: float = 100.0,
) -> dict:
    pred_lab, n_pred = label(pred_mask)
    true_lab, n_true = label(true_mask)

    if n_pred == 0 and n_true == 0:
        return {
            "onset_precision": 1.0,
            "onset_recall": 1.0,
            "mean_error_ms": 0.0,
            "std_error_ms": 0.0,
            "bias_ms": 0.0,
        }
    if n_pred == 0:
        return {
            "onset_precision": 0.0,
            "onset_recall": 0.0,
            "mean_error_ms": 0.0,
            "std_error_ms": 0.0,
            "bias_ms": 0.0,
        }
    if n_true == 0:
        return {
            "onset_precision": 0.0,
            "onset_recall": 0.0,
            "mean_error_ms": 0.0,
            "std_error_ms": 0.0,
            "bias_ms": 0.0,
        }

    pred_slices = find_objects(pred_lab)
    true_slices = find_objects(true_lab)

    pred_onsets = np.array([s[0].start for s in pred_slices])
    true_onsets = np.array([s[0].start for s in true_slices])

    tol_samples = int(tolerance_ms / 1000 * fs)

    tp = 0
    errors = []
    matched_true = set()

    for p_start in pred_onsets:
        if len(true_onsets) == 0:
            break
        dist = np.abs(true_onsets - p_start)
        min_idx = np.argmin(dist)
        min_dist = dist[min_idx]
        if min_dist <= tol_samples and min_idx not in matched_true:
            tp += 1
            matched_true.add(min_idx)
            errors.append(p_start - true_onsets[min_idx])

    errors_ms = np.array(errors) / fs * 1000.0
    fp = n_pred - tp
    fn = n_true - tp

    return {
        "onset_precision": tp / (tp + fp + 1e-8),
        "onset_recall": tp / (tp + fn + 1e-8),
        "mean_error_ms": float(np.mean(errors_ms)) if errors_ms.size > 0 else 0.0,
        "std_error_ms": float(np.std(errors_ms)) if errors_ms.size > 0 else 0.0,
        "bias_ms": float(np.median(errors_ms)) if errors_ms.size > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Training loop with AMP (bfloat16), torch.compile, and early stopping."""

    def __init__(self, model: StimArtifactUNet, cfg: Config):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device

        # AMP setup: MPS supports float16 autocast; CUDA supports bf16
        if cfg.device == "mps":
            self.amp_dtype = torch.float16
            self.amp_device = "mps"
            self.use_grad_scaler = False
            self.scaler = GradScaler(enabled=False)
        elif cfg.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.amp_device = "cuda"
            self.use_grad_scaler = False
            self.scaler = GradScaler("cuda", enabled=False)
        else:
            self.amp_dtype = torch.float16
            self.amp_device = "cuda"
            self.use_grad_scaler = True
            self.scaler = GradScaler("cuda", enabled=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        if cfg.lr_scheduler == "onecycle":
            self.scheduler = None
            self._step_scheduler_per_batch = True
            self._warmup_scheduler = None
            self._plateau_scheduler = None
        elif cfg.lr_scheduler == "plateau":
            self._warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.warmup_epochs,
            )
            self._plateau_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=cfg.plateau_factor,
                patience=cfg.plateau_patience,
                cooldown=cfg.plateau_cooldown,
                min_lr=cfg.min_lr,
            )
            self.scheduler = None
            self._step_scheduler_per_batch = False
        else:
            we = cfg.warmup_epochs
            warmup = LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=we
            )
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs - we, eta_min=cfg.min_lr
            )
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[we]
            )
            self._step_scheduler_per_batch = False
            self._warmup_scheduler = None
            self._plateau_scheduler = None

        self.criterion = CombinedLoss(cfg)
        self.best_f1 = 0.0

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Keep a pre-compile reference so EMA always tracks raw parameter tensors.
        self._raw_model = model

        if cfg.use_compile and hasattr(torch, "compile"):
            compile_mode = "max-autotune" if cfg.device == "cuda" else "default"
            print(f"Compiling model with torch.compile (mode={compile_mode!r})...")
            self.model = torch.compile(self.model, mode=compile_mode)

        # EMA shadow model
        self.ema_model: StimArtifactUNet | None = None
        if cfg.use_ema:
            import copy

            self.ema_model = copy.deepcopy(self._raw_model)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            print(f"EMA enabled (decay={cfg.ema_decay})")
        self._ema_steps = 0

        print(f"AMP dtype: {self.amp_dtype}, GradScaler: {self.use_grad_scaler}")

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        # Accumulate on GPU — avoid per-batch .item() syncs that stall MPS
        total_loss = torch.tensor(0.0, device=self.device)
        running_tp = torch.tensor(0.0, device=self.device)
        running_fp = torch.tensor(0.0, device=self.device)
        running_fn = torch.tensor(0.0, device=self.device)

        pbar = tqdm(loader, desc="  Train")
        for i, batch in enumerate(pbar):
            sig = batch["signal"].to(self.device, non_blocking=True)
            msk = batch["mask"].to(self.device, non_blocking=True)
            wgt = batch["weight"].to(self.device, non_blocking=True)
            cnd = batch["cond"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                output = self.model(sig, cnd)
                if isinstance(output, tuple):
                    logits, aux_outputs = output
                else:
                    logits, aux_outputs = output, []
                losses = self.criterion(logits, msk, wgt)

                # Deep supervision auxiliary loss
                if aux_outputs:
                    aux_loss = 0.0
                    for aux_logits in aux_outputs:
                        aux_msk = F.interpolate(
                            msk, size=aux_logits.shape[-1], mode="nearest"
                        )
                        aux_wgt = F.interpolate(
                            wgt, size=aux_logits.shape[-1], mode="nearest"
                        )
                        aux_loss += self.criterion(aux_logits, aux_msk, aux_wgt)["loss"]
                    losses["loss"] = losses[
                        "loss"
                    ] + self.cfg.deep_sup_weight * aux_loss / len(aux_outputs)

            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self._step_scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            # EMA update: shadow_p = decay * shadow_p + (1 - decay) * model_p
            if self.ema_model is not None:
                with torch.no_grad():
                    decay = min(
                        self.cfg.ema_decay,
                        (1.0 + self._ema_steps) / (10.0 + self._ema_steps),
                    )
                    for ema_p, m_p in zip(
                        self.ema_model.parameters(),
                        self._raw_model.parameters(),
                    ):
                        ema_p.lerp_(m_p, 1.0 - decay)
                    # Copy BatchNorm running stats (buffers are not learnable params).
                    # Guard with is_current_stream_capturing(): torch.compile
                    # (max-autotune) updates running_mean/running_var *inside*
                    # a CUDA graph. Issuing copy_() on those tensors during
                    # graph capture corrupts the tensor weakref lifecycle and
                    # raises AssertionError in cudagraph_trees. Skipping during
                    # capture is correct — we don't want this op baked into the
                    # graph; all non-capture steps copy buffers normally.
                    capturing = (
                        torch.cuda.is_current_stream_capturing()
                        if self.device == "cuda"
                        else False
                    )
                    if not capturing:
                        for ema_b, m_b in zip(
                            self.ema_model.buffers(),
                            self._raw_model.buffers(),
                        ):
                            ema_b.copy_(m_b)
                self._ema_steps += 1

            total_loss.add_(losses["loss"].detach())

            # GPU-side accumulation — no .item() sync
            with torch.no_grad():
                pred_bin = (logits > 0.0).float()
                w = (wgt > 0).float()
                pred_bin = pred_bin * w
                tgt_w = msk * w
                running_tp.add_((pred_bin * tgt_w).sum())
                running_fp.add_((pred_bin * (1 - tgt_w)).sum())
                running_fn.add_(((1 - pred_bin) * tgt_w).sum())

        # Single GPU sync at epoch end
        tp = running_tp.item()
        fp = running_fp.item()
        fn = running_fn.item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        return {
            "loss": total_loss.item() / len(loader),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou": iou,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        # Use uncompiled _raw_model for validation to avoid CUDA graph
        # replay issues with max-autotune (BN/Dropout/return-type mode
        # switches cause stale graph replays → garbage logits).
        eval_model = self.ema_model if self.ema_model is not None else self._raw_model
        eval_model.eval()
        total_loss = 0.0

        file_wins: dict[str, list[dict]] = {}
        file_lens: dict[str, int] = {}

        for batch in tqdm(loader, desc="  Val  "):
            sig = batch["signal"].to(self.device, non_blocking=True)
            msk = batch["mask"].to(self.device, non_blocking=True)
            wgt = batch["weight"].to(self.device, non_blocking=True)
            cnd = batch["cond"].to(self.device, non_blocking=True)

            with autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                logits = eval_model(sig, cnd)
                total_loss += self.criterion(logits, msk, wgt)["loss"].item()

            pred_np = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
            mask_np = msk.float().cpu().numpy().squeeze(1)

            for i in range(len(pred_np)):
                fid = batch["file_id"][i]
                if fid not in file_wins:
                    file_wins[fid] = []
                    file_lens[fid] = batch["rec_len"][i].item()
                file_wins[fid].append(
                    {
                        "pred": pred_np[i],
                        "mask": mask_np[i],
                        "start": batch["start"][i].item(),
                        "end": batch["end"][i].item(),
                    }
                )

        sample_metrics_list = []
        event_metrics_list = []
        onset_metrics_list = []
        fs = self.cfg.sampling_rate
        margin = int(self.cfg.boundary_margin_s * fs)

        for fid, wins in file_wins.items():
            rlen = file_lens[fid]
            pred_acc = np.zeros(rlen, dtype=np.float32)
            mask_acc = np.zeros(rlen, dtype=np.float32)
            wgt_acc = np.zeros(rlen, dtype=np.float32)

            for w in wins:
                s, e = w["start"], w["end"]
                pred_acc[s:e] += w["pred"]
                mask_acc[s:e] = w["mask"]
                wgt_acc[s:e] += 1.0

            pred_avg = np.divide(
                pred_acc, wgt_acc, out=np.zeros_like(pred_acc), where=wgt_acc > 0
            )

            file_weight = np.ones(rlen, dtype=np.float32)
            file_weight[:margin] = 0.0
            file_weight[max(margin, rlen - margin) :] = 0.0

            valid = file_weight > 0
            if valid.sum() == 0:
                continue

            sm = compute_sample_metrics(
                torch.from_numpy(pred_avg[valid]).unsqueeze(0),
                torch.from_numpy(mask_acc[valid]).unsqueeze(0),
            )
            sample_metrics_list.append(sm)

            pred_bin = post_process_mask(
                (pred_avg > 0.5).astype(int),
                self.cfg,
            )
            mask_eval = mask_acc.astype(int).copy()
            pred_bin[:margin] = 0
            pred_bin[rlen - margin :] = 0
            mask_eval[:margin] = 0
            mask_eval[rlen - margin :] = 0

            em = compute_event_metrics(pred_bin, mask_eval)
            event_metrics_list.append(em)

            om = compute_onset_metrics(pred_bin, mask_eval, fs=fs)
            onset_metrics_list.append(om)

        agg = {}
        if sample_metrics_list:
            for key in sample_metrics_list[0]:
                agg[key] = np.mean([m[key] for m in sample_metrics_list])
        if event_metrics_list:
            for key in event_metrics_list[0]:
                agg[key] = np.mean([m[key] for m in event_metrics_list])
        if onset_metrics_list:
            for key in onset_metrics_list[0]:
                agg[key] = np.mean([m[key] for m in onset_metrics_list])
        agg["loss"] = total_loss / max(len(loader), 1)
        return agg

    @torch.no_grad()
    def predict_file(
        self,
        dat_path: Path,
        cond: np.ndarray,
        signal_transform=None,
        batch_size: int = 64,
        tta: bool = False,
    ) -> np.ndarray:
        # Use uncompiled model for inference (same CUDA graph issue as validate)
        eval_model = self.ema_model if self.ema_model is not None else self._raw_model
        eval_model.eval()
        data = _load_dat_channels(dat_path, self.cfg.use_channels)
        if signal_transform is not None:
            data = signal_transform(data)

        median, iqr = _robust_scale_stats(data)
        med_col = median[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        rlen = data.shape[1]
        ws = self.cfg.window_samples
        stride = max(1, int(ws * self.cfg.inference_stride_ratio))

        starts = list(range(0, rlen - ws + 1, stride))
        cond_t = torch.from_numpy(cond).unsqueeze(0).to(self.device)

        # TTA amplitude scales: original + scaled versions
        scales = [1.0]
        if tta:
            scales = [1.0, 0.9, 1.1]

        pred_acc = np.zeros(rlen, dtype=np.float32)
        count = np.zeros(rlen, dtype=np.float32)

        for scale in scales:
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                windows = np.stack(
                    [
                        (data[:, s : s + ws] - med_col) / iqr_col * scale
                        for s in batch_starts
                    ]
                )
                sig_t = torch.from_numpy(windows).to(self.device)
                cond_batch = cond_t.expand(len(batch_starts), -1)

                with autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                    logits = eval_model(sig_t, cond_batch)
                probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)

                for i, s in enumerate(batch_starts):
                    pred_acc[s : s + ws] += probs[i]
                    count[s : s + ws] += 1.0

        return np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Device: {self.device}")
        print(f"Model parameters: {n_params:,}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"LR scheduler: {self.cfg.lr_scheduler}")

        if self.cfg.lr_scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.lr,
                epochs=self.cfg.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )

        patience_ctr = 0
        val_n = self.cfg.val_every_n_epochs
        train_sampler = (
            train_loader.batch_sampler.sampler
            if hasattr(train_loader.batch_sampler, "sampler")
            else None
        )

        for epoch in range(self.cfg.epochs):
            # Rotate file subset each epoch
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.cfg.epochs}  (lr={lr:.2e})")
            print(f"{'='*60}")

            train_m = self.train_epoch(train_loader)
            print(
                f"  Train | loss={train_m['loss']:.4f}  F1={train_m['f1']:.4f}"
                f"  P={train_m['precision']:.4f}  R={train_m['recall']:.4f}  IoU={train_m['iou']:.4f}"
            )

            run_val = (epoch % val_n == 0) or (epoch == self.cfg.epochs - 1)

            if self.cfg.lr_scheduler == "plateau" and epoch < self.cfg.warmup_epochs:
                self._warmup_scheduler.step()
                if not run_val:
                    continue

            if not run_val:
                if not self._step_scheduler_per_batch and self.scheduler is not None:
                    self.scheduler.step()
                continue

            val_m = self.validate(val_loader)
            ckpt_key = self.cfg.checkpoint_metric
            val_score = val_m.get(ckpt_key, val_m.get("f1", 0.0))

            if self.cfg.lr_scheduler == "plateau":
                if epoch >= self.cfg.warmup_epochs:
                    self._plateau_scheduler.step(val_score)
            elif not self._step_scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            print(
                f"  Val   | loss={val_m['loss']:.4f}  F1={val_m['f1']:.4f}"
                f"  P={val_m['precision']:.4f}  R={val_m['recall']:.4f}  IoU={val_m['iou']:.4f}"
            )
            if "event_f1" in val_m:
                print(
                    f"  Event | F1={val_m['event_f1']:.4f}"
                    f"  P={val_m['event_precision']:.4f}"
                    f"  R={val_m['event_recall']:.4f}"
                )
            if "onset_precision" in val_m:
                print(
                    f"  Onset | P={val_m['onset_precision']:.4f}  R={val_m['onset_recall']:.4f}"
                    f"  mean_err={val_m['mean_error_ms']:.2f}ms  std_err={val_m['std_error_ms']:.2f}ms  bias={val_m['bias_ms']:.2f}ms"
                )

            if val_score > self.best_f1:
                self.best_f1 = val_score
                patience_ctr = 0
                raw_model = getattr(self.model, "_orig_mod", self.model)
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.cfg,
                    ckpt_key: self.best_f1,
                }
                if self.ema_model is not None:
                    ckpt["ema_state_dict"] = self.ema_model.state_dict()
                torch.save(ckpt, self.cfg.checkpoint_dir / "best_model.pt")
                print(f"  >>> New best ({ckpt_key}={self.best_f1:.4f})")
            else:
                # Don't penalize warmup epochs — LR is still ramping
                if epoch >= self.cfg.warmup_epochs:
                    patience_ctr += 1
                print(f"  --- No improvement ({patience_ctr}/{self.cfg.patience})")
                if patience_ctr >= self.cfg.patience:
                    print(
                        f"\nEarly stopping at epoch {epoch+1}. Best {ckpt_key}: {self.best_f1:.4f}"
                    )
                    break

            if self.optimizer.param_groups[0]["lr"] <= self.cfg.min_lr * 1.01:
                print(
                    f"\nLR reached minimum ({self.cfg.min_lr:.1e}). "
                    f"Best {ckpt_key}: {self.best_f1:.4f}"
                )
                break


# ---------------------------------------------------------------------------
# Multi-seed Ensemble
# ---------------------------------------------------------------------------


class EnsemblePredictor:
    """Multi-seed ensemble predictor.

    Loads one checkpoint per seed, runs inference independently, and averages
    the per-sample probability arrays.

    Usage::

        checkpoint_map = {
            42:  Path("results/best_model_seed42.pt"),
            67:  Path("results/best_model_seed67.pt"),
            123: Path("results/best_model_seed123.pt"),
            256: Path("results/best_model_seed256.pt"),
            789: Path("results/best_model_seed789.pt"),
        }
        ep = EnsemblePredictor(cfg, checkpoint_map)
        proba = ep.predict_file(dat_path, cond)

        # Per-seed arrays are stored after each call:
        #   ep.proba_caches[42]  → np.ndarray, shape (rlen,)
    """

    SEEDS: list[int] = [42, 67, 123, 256, 789]

    def __init__(
        self,
        cfg: Config,
        checkpoint_map: dict[int, Path],
        *,
        use_ema: bool = True,
    ) -> None:
        """
        Args:
            cfg: Config shared across all seeds (device, window_samples, etc.).
            checkpoint_map: Mapping of seed → checkpoint path.
            use_ema: Load ``ema_state_dict`` from checkpoint when available;
                     fall back to ``model_state_dict`` otherwise.
        """

        self.cfg = cfg
        self.proba_caches: dict[int, np.ndarray] = {}
        self._trainers: dict[int, Trainer] = {}

        for seed, ckpt_path in checkpoint_map.items():
            torch.manual_seed(seed)
            model = StimArtifactUNet(cfg)
            trainer = Trainer(model, cfg)
            ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
            raw = getattr(trainer.model, "_orig_mod", trainer.model)
            if use_ema and "ema_state_dict" in ckpt:
                raw.load_state_dict(ckpt["ema_state_dict"])
                print(f"  Seed {seed}: loaded EMA weights from {Path(ckpt_path).name}")
            else:
                raw.load_state_dict(ckpt["model_state_dict"])
                print(
                    f"  Seed {seed}: loaded model weights from {Path(ckpt_path).name}"
                )
            self._trainers[seed] = trainer

    def predict_file(
        self,
        dat_path: Path,
        cond: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Run inference for every loaded seed and return the mean probability array.

        Per-seed arrays are cached in ``self.proba_caches`` keyed by seed.

        Args:
            dat_path: Path to the .dat recording file.
            cond: FiLM conditioning vector, shape (n_cond_features,).
            **kwargs: Forwarded to :meth:`Trainer.predict_file`
                      (e.g. ``batch_size``, ``tta``).

        Returns:
            ensemble_proba: np.ndarray, shape (rlen,) — mean over all seeds.
        """
        seeds = [s for s in self.SEEDS if s in self._trainers]
        for seed in seeds:
            self.proba_caches[seed] = self._trainers[seed].predict_file(
                dat_path, cond, **kwargs
            )
        ensemble_proba = np.mean([self.proba_caches[s] for s in seeds], axis=0)
        return ensemble_proba


# ---------------------------------------------------------------------------
# Standalone Inference Utilities
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Path | str,
    device: str = "mps",
    use_ema: bool = True,
) -> tuple[nn.Module, Config]:
    """Load a trained model from checkpoint.

    Returns (model, cfg) with model in eval mode on device.
    Handles ``_orig_mod.`` prefix from ``torch.compile``.
    Prefers EMA weights when *use_ema* is True and they exist.
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", Config())
    cfg.device = device
    model = StimArtifactUNet(cfg).to(device)

    raw = getattr(model, "_orig_mod", model)
    if use_ema and "ema_state_dict" in ckpt:
        raw.load_state_dict(ckpt["ema_state_dict"])
    else:
        raw.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def build_true_mask(
    n_samples: int,
    onset_times: list[float],
    mask_dur_ms: float,
    sr: int = 250,
    onset_offset_ms: float = 0.0,
) -> np.ndarray:
    """Build binary ground-truth mask from onset times.

    Returns int8 array of shape ``(n_samples,)`` with 1 inside artifact windows.
    """
    mask = np.zeros(n_samples, dtype=np.int8)
    dur_samp = max(int(mask_dur_ms / 1000.0 * sr), 1)
    offset_samp = int(onset_offset_ms / 1000.0 * sr)
    for t in onset_times:
        s = int(t * sr) + offset_samp
        mask[max(0, s) : min(n_samples, s + dur_samp)] = 1
    return mask


@torch.no_grad()
def predict_file_proba(
    model: nn.Module,
    cfg: Config,
    dat_path: Path | str,
    cond: np.ndarray,
    *,
    batch_size: int = 64,
    tta: bool = False,
) -> np.ndarray:
    """Run overlapping-window inference on a single .dat file.

    Standalone version of :meth:`Trainer.predict_file` that works with any
    model instance (no Trainer required).

    Returns per-sample probability array of shape ``(rlen,)``.
    """
    model.eval()
    dat_path = Path(dat_path)
    data = _load_dat_channels(dat_path, cfg.use_channels)
    median, iqr = _robust_scale_stats(data)
    med_col = median[:, np.newaxis]
    iqr_col = iqr[:, np.newaxis] + 1e-8
    rlen = data.shape[1]
    ws = cfg.window_samples
    stride = max(1, int(ws * cfg.inference_stride_ratio))

    starts = list(range(0, rlen - ws + 1, stride))
    cond_t = torch.from_numpy(cond).unsqueeze(0).to(cfg.device)

    scales = [1.0, 0.9, 1.1] if tta else [1.0]
    amp_device = (
        "mps" if cfg.device == "mps" else ("cuda" if cfg.device == "cuda" else "cpu")
    )
    amp_dtype = torch.float16 if cfg.device == "mps" else torch.bfloat16

    pred_acc = np.zeros(rlen, dtype=np.float32)
    count = np.zeros(rlen, dtype=np.float32)

    for scale in scales:
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            windows = np.stack(
                [
                    (data[:, s : s + ws] - med_col) / iqr_col * scale
                    for s in batch_starts
                ]
            )
            sig_t = torch.from_numpy(windows).to(cfg.device)
            cond_batch = cond_t.expand(len(batch_starts), -1)

            with autocast(device_type=amp_device, dtype=amp_dtype):
                logits = model(sig_t, cond_batch)
            probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)

            for i, s in enumerate(batch_starts):
                pred_acc[s : s + ws] += probs[i]
                count[s : s + ws] += 1.0

    return np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)
