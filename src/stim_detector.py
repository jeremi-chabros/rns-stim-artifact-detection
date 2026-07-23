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
"""Stimulation Artifact Detector — local training & inference.

Multi-Scale Conditional 1D U-Net with FiLM conditioning for
NeuroPace RNS ECoG stim artifact detection.

Runs on Apple Silicon M4 Pro with MPS backend.

Usage:
    uv run src/stim_detector.py                  # train
    uv run src/stim_detector.py --eval-only      # inference with best checkpoint
    uv run src/stim_detector.py --sweep           # post-processing hyperparameter sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import find_objects, label
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    FileSubsetSampler,
    StimArtifactDataset,
    StimArtifactUNet,
    Trainer,
    _load_dat_channels,
    _robust_scale_stats,
    extract_conditioning_vector,
    load_annotations,
    post_process_mask,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_device(cfg: Config) -> None:
    """Configure device-specific optimizations."""
    if cfg.device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"
        print("Device: MPS (Apple Silicon)")
    elif cfg.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {gpu_name} ({gpu_mem_gb:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        print(f"Device: {cfg.device}")


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-split by biological subject (no bilateral leakage)."""
    # Group by subject (not subject_id_lr which splits bilateral L/R)
    groups = df["subject"].values

    unique_groups = np.unique(groups)
    if len(unique_groups) == 1:
        print("  Single subject detected -> splitting by filename")
        groups = df["filename"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    print(f"  Train: {len(train_df)} recordings ({len(train_subjects)} subjects)")
    print(f"  Val:   {len(val_df)} recordings ({len(val_subjects)} subjects)")
    assert train_subjects.isdisjoint(
        val_subjects
    ), "Data leakage: overlapping subjects!"

    # Verify no bilateral leakage
    if "subject_id_lr" in df.columns:
        train_lr = set(df.iloc[train_idx]["subject_id_lr"].unique())
        val_lr = set(df.iloc[val_idx]["subject_id_lr"].unique())
        train_bases = {s.rsplit("_", 1)[0] for s in train_lr}
        val_bases = {s.rsplit("_", 1)[0] for s in val_lr}
        assert train_bases.isdisjoint(
            val_bases
        ), f"Bilateral leakage! Shared subjects: {train_bases & val_bases}"
        print(
            f"  Bilateral check passed: {len(train_lr)} train / {len(val_lr)} val devices"
        )

    return train_df, val_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: Config) -> Path:
    """Full training run. Returns path to best checkpoint."""
    setup_device(cfg)

    print("Loading annotations...")
    df = load_annotations(cfg)
    print(f"  {len(df)} annotated recordings")

    # Filter to files that exist on disk
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    n_before = len(df)
    df = df[exists_mask].reset_index(drop=True)
    print(f"  {len(df)}/{n_before} files found on disk")

    print("Splitting data...")
    train_df, val_df = split_data(df)

    print("Building datasets...")
    train_ds = StimArtifactDataset(train_df, cfg, augment=True)
    val_ds = StimArtifactDataset(val_df, cfg, augment=False)

    loader_kw = dict(
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    # File-subset sampler for per-epoch rotation (if subsample < 1.0)
    if cfg.train_file_subsample < 1.0:
        train_sampler = FileSubsetSampler(train_ds, subsample=cfg.train_file_subsample)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler, **loader_kw
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kw
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **loader_kw
    )

    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    model = StimArtifactUNet(cfg)
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)

    ckpt_path = cfg.checkpoint_dir / "best_model.pt"
    if ckpt_path.exists():
        print(f"\nBest model: {ckpt_path}")
        print(f"Best val F1: {trainer.best_f1:.4f}")
    else:
        print("WARNING: No checkpoint saved — training may not have completed.")

    return ckpt_path


# ---------------------------------------------------------------------------
# Full-dataset inference & metrics
# ---------------------------------------------------------------------------


def evaluate(cfg: Config) -> pd.DataFrame:
    """Run inference on all files and compute per-file metrics."""
    setup_device(cfg)
    ckpt_path = cfg.checkpoint_dir / "best_model.pt"
    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"

    ckpt = torch.load(str(ckpt_path), map_location=cfg.device, weights_only=False)
    best_model = StimArtifactUNet(ckpt.get("config", cfg)).to(cfg.device).eval()
    best_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    df = load_annotations(cfg)
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)
    print(f"Evaluating {len(df)} files...")

    ws = cfg.window_samples
    stride = max(1, int(ws * cfg.inference_stride_ratio))
    amp_device = "mps" if cfg.device == "mps" else "cuda"
    amp_dtype = torch.float16 if cfg.device == "mps" else torch.bfloat16

    # Cache files and build window list
    file_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    window_list: list[tuple[int, str, int]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching files"):
        fp = row["file_path"]
        if fp not in file_cache:
            data = _load_dat_channels(Path(fp), cfg.use_channels)
            med, iqr = _robust_scale_stats(data)
            file_cache[fp] = (data, med, iqr)
        rlen = file_cache[fp][0].shape[1]
        for s in range(0, rlen - ws + 1, stride):
            window_list.append((len(window_list), fp, s))

    print(f"Cached {len(file_cache)} files, {len(window_list)} windows")

    # Run batched inference
    batch_size = 1024
    pred_accs = {
        fp: np.zeros(c[0].shape[1], dtype=np.float32) for fp, c in file_cache.items()
    }
    counts = {
        fp: np.zeros(c[0].shape[1], dtype=np.float32) for fp, c in file_cache.items()
    }

    fp_to_cond: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        fp = row["file_path"]
        if fp in file_cache and fp not in fp_to_cond:
            fp_to_cond[fp] = extract_conditioning_vector(row)

    @torch.no_grad()
    def run_batch(batch_items):
        wins = np.stack(
            [
                (file_cache[fp][0][:, s : s + ws] - file_cache[fp][1][:, None])
                / (file_cache[fp][2][:, None] + 1e-8)
                for _, fp, s in batch_items
            ]
        )
        conds = np.stack([fp_to_cond[fp] for _, fp, _ in batch_items])
        with torch.autocast(amp_device, dtype=amp_dtype):
            probs = (
                torch.sigmoid(
                    best_model(
                        torch.from_numpy(wins).to(cfg.device),
                        torch.from_numpy(conds).to(cfg.device),
                    )
                )
                .float()
                .cpu()
                .numpy()
                .squeeze(1)
            )
        for idx, (_, fp, s) in enumerate(batch_items):
            pred_accs[fp][s : s + ws] += probs[idx]
            counts[fp][s : s + ws] += 1.0

    for bi in tqdm(
        range(0, len(window_list), batch_size),
        total=(len(window_list) + batch_size - 1) // batch_size,
        desc="Inference",
    ):
        run_batch(window_list[bi : bi + batch_size])

    # Average predictions
    proba_dict = {}
    for fp, (data, _, _) in file_cache.items():
        rlen = data.shape[1]
        c = counts[fp]
        proba_dict[fp] = np.divide(
            pred_accs[fp], c, out=np.zeros(rlen, dtype=np.float32), where=c > 0
        )

    del pred_accs, counts, window_list

    # Compute per-file metrics
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Metrics"):
        fp = row["file_path"]
        onsets = row["onset_times"]
        mask_dur_ms = row.get("mask_duration_ms", 1000.0)
        if pd.isna(mask_dur_ms):
            mask_dur_ms = 1000.0

        if fp not in file_cache:
            rows.append(
                dict(
                    filename=row["filename"],
                    file_path=fp,
                    mask_duration_ms=mask_dur_ms,
                    n_gt_events=len(onsets),
                    n_pred_events=0,
                    n_true_positives=0,
                    n_false_positives=0,
                    n_false_negatives=len(onsets),
                    mean_error_ms=np.nan,
                    std_error_ms=np.nan,
                    bias_ms=np.nan,
                    error="file_not_found",
                )
            )
            continue

        rlen = file_cache[fp][0].shape[1]
        pred_proba = proba_dict[fp]

        # Ground truth mask
        true_mask = np.zeros(rlen, dtype=np.int_)
        dur_s = max(int(mask_dur_ms / 1000.0 * cfg.sampling_rate), 1)
        for t in onsets:
            s = int(t * cfg.sampling_rate)
            true_mask[max(0, s) : min(rlen, s + dur_s)] = 1

        pred_bin = post_process_mask((pred_proba > 0.5).astype(np.int_), cfg)

        # Event matching (IoU >= 0.3)
        pred_lab, n_pred = label(pred_bin)
        true_lab, n_true = label(true_mask)
        tp, matched = 0, set()
        if n_pred > 0 and n_true > 0:
            ps_list = find_objects(pred_lab)
            ts_list = find_objects(true_lab)
            for i, ps in enumerate(ps_list):
                best_iou, best_j = 0.0, -1
                pm = pred_lab == (i + 1)
                for j, ts in enumerate(ts_list):
                    if (
                        j in matched
                        or ps[0].start >= ts[0].stop
                        or ps[0].stop <= ts[0].start
                    ):
                        continue
                    tm = true_lab == (j + 1)
                    iou = np.logical_and(pm, tm).sum() / (
                        np.logical_or(pm, tm).sum() + 1e-8
                    )
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.3:
                    tp += 1
                    matched.add(best_j)
        fp_count = n_pred - tp
        fn = n_true - len(matched)

        # Onset timing errors
        pred_onsets_sec = (
            [float(s[0].start) / cfg.sampling_rate for s in find_objects(pred_lab)]
            if n_pred
            else []
        )
        mean_err = std_err = bias = np.nan
        if n_pred > 0 and n_true > 0:
            p_ons = np.array([s[0].start for s in find_objects(pred_lab)])
            t_ons = np.array([s[0].start for s in find_objects(true_lab)])
            tol = int(100.0 / 1000 * cfg.sampling_rate)
            errs, mt = [], set()
            for po in p_ons:
                d = np.abs(t_ons - po)
                mi = int(np.argmin(d))
                if d[mi] <= tol and mi not in mt:
                    mt.add(mi)
                    errs.append(po - t_ons[mi])
            if errs:
                errs_ms = np.array(errs) / cfg.sampling_rate * 1000.0
                mean_err = float(np.mean(errs_ms))
                std_err = float(np.std(errs_ms))
                bias = float(np.median(errs_ms))

        rows.append(
            dict(
                filename=row["filename"],
                file_path=fp,
                mask_duration_ms=mask_dur_ms,
                pred_onset_times=json.dumps(pred_onsets_sec),
                n_gt_events=n_true,
                n_pred_events=n_pred,
                n_true_positives=tp,
                n_false_positives=fp_count,
                n_false_negatives=fn,
                mean_error_ms=mean_err,
                std_error_ms=std_err,
                bias_ms=bias,
            )
        )

    out_df = pd.DataFrame(rows)
    out_path = cfg.checkpoint_dir / "inference_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows -> {out_path}")

    # Summary
    tot_tp = out_df["n_true_positives"].sum()
    tot_fp = out_df["n_false_positives"].sum()
    tot_fn = out_df["n_false_negatives"].sum()
    p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
    r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    merr = out_df["mean_error_ms"].dropna().mean()
    mbias = out_df["bias_ms"].dropna().mean()
    print(f"  TP: {tot_tp}  FP: {tot_fp}  FN: {tot_fn}")
    print(f"  P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}")
    print(f"  Onset timing — mean error: {merr:.2f} ms, bias: {mbias:.2f} ms")

    return out_df


# ---------------------------------------------------------------------------
# Post-processing hyperparameter sweep
# ---------------------------------------------------------------------------


def sweep(cfg: Config) -> pd.DataFrame:
    """Sweep threshold × merge_gap_ms × min_dur_ms using cached predictions."""
    setup_device(cfg)
    ckpt_path = cfg.checkpoint_dir / "best_model.pt"
    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"

    df = load_annotations(cfg)
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)

    sr = cfg.sampling_rate

    # Grid
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    merge_gaps_ms = [25, 50, 100, 150, 200, 300, 500]
    min_durs_ms = [100, 150, 200, 300, 400, 600]
    n_combos = len(thresholds) * len(merge_gaps_ms) * len(min_durs_ms)
    print(
        f"Grid: {len(thresholds)}x{len(merge_gaps_ms)}x{len(min_durs_ms)} = {n_combos} combos"
    )

    # Run inference to get per-file probabilities
    print("Running inference for sweep...")
    ckpt = torch.load(str(ckpt_path), map_location=cfg.device, weights_only=False)
    best_model = StimArtifactUNet(ckpt.get("config", cfg)).to(cfg.device).eval()
    best_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    ws = cfg.window_samples
    stride = max(1, int(ws * cfg.inference_stride_ratio))
    amp_device = "mps" if cfg.device == "mps" else "cuda"
    amp_dtype = torch.float16 if cfg.device == "mps" else torch.bfloat16

    # Build per-file probabilities
    proba_dict: dict[str, np.ndarray] = {}
    max_files = min(5000, len(df))
    sweep_df_sub = (
        df.sample(n=max_files, random_state=42) if len(df) > max_files else df
    )

    for _, row in tqdm(
        sweep_df_sub.iterrows(), total=len(sweep_df_sub), desc="Inference"
    ):
        fp = row["file_path"]
        if fp in proba_dict:
            continue
        data = _load_dat_channels(Path(fp), cfg.use_channels)
        med, iqr = _robust_scale_stats(data)
        rlen = data.shape[1]
        cond = extract_conditioning_vector(row)

        pred_acc = np.zeros(rlen, dtype=np.float32)
        count = np.zeros(rlen, dtype=np.float32)

        starts = list(range(0, rlen - ws + 1, stride))
        cond_t = torch.from_numpy(cond).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            for batch_start in range(0, len(starts), 512):
                batch_starts = starts[batch_start : batch_start + 512]
                windows = np.stack(
                    [
                        (data[:, s : s + ws] - med[:, None]) / (iqr[:, None] + 1e-8)
                        for s in batch_starts
                    ]
                )
                sig_t = torch.from_numpy(windows).to(cfg.device)
                cond_batch = cond_t.expand(len(batch_starts), -1)
                with torch.autocast(amp_device, dtype=amp_dtype):
                    logits = best_model(sig_t, cond_batch)
                probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
                for i, s in enumerate(batch_starts):
                    pred_acc[s : s + ws] += probs[i]
                    count[s : s + ws] += 1.0

        proba_dict[fp] = np.divide(
            pred_acc, count, out=np.zeros(rlen, dtype=np.float32), where=count > 0
        )

    print(f"  Computed probabilities for {len(proba_dict)} files")

    # Segment utilities
    def get_segs(arr):
        p = np.empty(len(arr) + 2, dtype=np.int8)
        p[0] = 0
        p[1:-1] = arr
        p[-1] = 0
        d = np.diff(p)
        starts = np.where(d > 0)[0]
        ends = np.where(d < 0)[0]
        if len(starts) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.column_stack([starts, ends]).astype(np.int32)

    def merge_gaps(segs, gap):
        if len(segs) == 0:
            return segs
        out = [[int(segs[0, 0]), int(segs[0, 1])]]
        for i in range(1, len(segs)):
            s, e = int(segs[i, 0]), int(segs[i, 1])
            if s - out[-1][1] <= gap:
                out[-1][1] = e
            else:
                out.append([s, e])
        return np.array(out, dtype=np.int32)

    def filter_min(segs, min_s):
        if len(segs) == 0:
            return segs
        return segs[(segs[:, 1] - segs[:, 0]) >= min_s]

    def match(pred, gt, iou_thr=0.3):
        if len(pred) == 0:
            return 0, 0, len(gt)
        if len(gt) == 0:
            return 0, len(pred), 0
        tp, matched = 0, set()
        for ps, pe in pred:
            best, bj = 0.0, -1
            for j, (gs, ge) in enumerate(gt):
                if j in matched or ps >= ge or pe <= gs:
                    continue
                inter = min(pe, ge) - max(ps, gs)
                union = max(pe, ge) - min(ps, gs)
                iou = inter / (union + 1e-8)
                if iou > best:
                    best, bj = iou, j
            if best >= iou_thr:
                tp += 1
                matched.add(bj)
        return tp, len(pred) - tp, len(gt) - len(matched)

    # Build ground-truth segment lists
    print("Building ground-truth segments...")
    gt_segs: dict[str, np.ndarray] = {}
    for _, row in sweep_df_sub.iterrows():
        fp = row["file_path"]
        if fp not in proba_dict:
            continue
        rlen = len(proba_dict[fp])
        mms = row.get("mask_duration_ms", 1000.0)
        if pd.isna(mms):
            mms = 1000.0
        ds = max(int(mms / 1000.0 * sr), 1)
        m = np.zeros(rlen, dtype=np.int8)
        for t in row["onset_times"]:
            s = int(t * sr)
            m[max(0, s) : min(rlen, s + ds)] = 1
        gt_segs[fp] = get_segs(m)

    gap_samps = [max(1, int(g / 1000.0 * sr)) for g in merge_gaps_ms]
    min_samps = [max(1, int(d / 1000.0 * sr)) for d in min_durs_ms]

    # Sweep
    res = np.zeros(
        (len(thresholds), len(merge_gaps_ms), len(min_durs_ms), 3), dtype=np.int64
    )
    t0 = time.time()
    for fp, proba in tqdm(proba_dict.items(), desc="Sweep"):
        gt = gt_segs.get(fp)
        if gt is None:
            continue
        for ti, thr in enumerate(thresholds):
            raw = get_segs((proba > thr).astype(np.int8))
            for gi, gs in enumerate(gap_samps):
                merged = merge_gaps(raw, gs)
                for di, ms in enumerate(min_samps):
                    final = filter_min(merged, ms)
                    _tp, _fp, _fn = match(final, gt)
                    res[ti, gi, di, 0] += _tp
                    res[ti, gi, di, 1] += _fp
                    res[ti, gi, di, 2] += _fn

    print(f"Sweep done in {time.time() - t0:.1f}s")

    # Build results
    result_rows = []
    for ti, thr in enumerate(thresholds):
        for gi, gms in enumerate(merge_gaps_ms):
            for di, dms in enumerate(min_durs_ms):
                _tp, _fp, _fn = res[ti, gi, di]
                _p = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
                _r = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
                _f1 = 2 * _p * _r / (_p + _r) if (_p + _r) else 0.0
                result_rows.append(
                    dict(
                        threshold=thr,
                        merge_gap_ms=gms,
                        min_dur_ms=dms,
                        TP=int(_tp),
                        FP=int(_fp),
                        FN=int(_fn),
                        precision=round(_p, 6),
                        recall=round(_r, 6),
                        F1=round(_f1, 6),
                    )
                )

    sweep_df = (
        pd.DataFrame(result_rows)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 15 combinations:")
    print(sweep_df.head(15).to_string(index=False))

    pivot = sweep_df.pivot_table(
        index="merge_gap_ms", columns="threshold", values="F1", aggfunc="max"
    )
    print("\nBest F1 by merge_gap_ms x threshold (max over min_dur_ms):")
    print(pivot.round(4).to_string())

    sweep_path = cfg.checkpoint_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSaved {len(sweep_df)} rows -> {sweep_path}")

    return sweep_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stim artifact detector training")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run inference only (requires checkpoint)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run post-processing hyperparameter sweep"
    )
    parser.add_argument(
        "--catalog", type=Path, default=None, help="Path to stim_catalog.parquet"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="Checkpoint directory"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Device",
    )
    parser.add_argument("--film", action="store_true", default=None, help="Enable FiLM")
    parser.add_argument("--no-film", action="store_true", help="Disable FiLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.catalog is not None:
        cfg.catalog_path = args.catalog
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device
    if args.film is not None:
        cfg.use_film = True
    if args.no_film:
        cfg.use_film = False

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch size: {cfg.batch_size}")
    print(f"FiLM conditioning: {cfg.use_film}")

    if args.sweep:
        sweep(cfg)
    elif args.eval_only:
        evaluate(cfg)
    else:
        train(cfg)
        evaluate(cfg)


if __name__ == "__main__":
    main()
