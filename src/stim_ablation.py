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
#   "matplotlib>=3.8",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""Stim Artifact Ablation Experiments.

Systematically validates architectural choices by running inference with
specific perturbations and comparing to baseline.

Experiments:
  1. film_onoff      — FiLM conditioning on vs off (zero conditioning)
  2. film_groups     — Zero one FiLM feature group at a time
  3. film_permute    — Shuffle conditioning across files (5 repeats)
  4. postproc_sweep  — Grid search: threshold x merge_gap x min_duration
  5. channel_copy    — Copy single channel to all 4 inputs
  6. skip_ablation   — Zero out skip connections one level at a time

Usage:
    uv run src/stim_ablation.py --experiments all
    uv run src/stim_ablation.py --experiments film_onoff film_groups
    uv run src/stim_ablation.py --max-files 3000
    uv run src/stim_ablation.py --force   # re-run even if outputs exist
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from stim_detector_lib import (
    Config,
    _load_dat_channels,
    _robust_scale_stats,
    build_true_mask,
    extract_conditioning_vector,
    load_annotations,
    load_checkpoint,
    post_process_mask,
    predict_file_proba,
)

RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")

# Feature group definitions for the 32-dim conditioning vector
FEATURE_GROUPS = {
    "B1_stim_params": (0, 5),
    "B1_montage": (5, 14),
    "B2_stim_params": (14, 19),
    "B2_montage": (19, 28),
    "Lead_1": (28, 30),
    "Lead_2": (30, 32),
}

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Proportionally sample n rows stratified by subject_id_lr."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    col = "subject_id_lr" if "subject_id_lr" in df.columns else "subject"
    groups = df.groupby(col)
    counts = groups.size()
    total = counts.sum()

    alloc = (counts / total * n).astype(int).clip(lower=1)
    while alloc.sum() > n:
        biggest = alloc.idxmax()
        alloc[biggest] -= 1
    while alloc.sum() < n:
        biggest = counts.idxmax()
        alloc[biggest] += 1

    parts = []
    for gname, gdf in groups:
        k = min(alloc[gname], len(gdf))
        parts.append(gdf.sample(n=k, random_state=rng.integers(0, 2**31)))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def compute_event_f1(
    true_mask: np.ndarray,
    pred_bin: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[float, float, float, int, int, int]:
    """Returns (precision, recall, f1, tp, fp, fn)."""
    true_labeled, n_true = label(true_mask)
    pred_labeled, n_pred = label(pred_bin)

    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_true
    if n_true == 0:
        return 0.0, 0.0, 0.0, 0, n_pred, 0

    true_slices = find_objects(true_labeled)
    pred_slices = find_objects(pred_labeled)
    tp, matched = 0, set()

    for ps in pred_slices:
        p_s, p_e = ps[0].start, ps[0].stop
        best_iou, best_j = 0.0, -1
        for j, ts in enumerate(true_slices):
            if j in matched:
                continue
            g_s, g_e = ts[0].start, ts[0].stop
            if p_s >= g_e or p_e <= g_s:
                continue
            inter = min(p_e, g_e) - max(p_s, g_s)
            union = max(p_e, g_e) - min(p_s, g_s)
            iou_val = inter / (union + 1e-8)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_j)

    fp = n_pred - tp
    fn = n_true - len(matched)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp, fp, fn


def run_inference_pass(
    model,
    cfg: Config,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    cond_transform=None,
    signal_transform=None,
    desc: str = "Inference",
) -> dict:
    """Run inference over df with optional transforms, return aggregate metrics."""
    sr = cfg.sampling_rate
    total_tp, total_fp, total_fn = 0, 0, 0
    all_dice = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        fp = row["file_path"]
        if not Path(fp).exists():
            continue

        cond = extract_conditioning_vector(row)
        if cond_transform is not None:
            cond = cond_transform(cond)

        data = _load_dat_channels(Path(fp), cfg.use_channels)
        if signal_transform is not None:
            data = signal_transform(data)

        # Inline predict to apply signal_transform
        from torch.amp import autocast

        model.eval()
        import torch

        median, iqr = _robust_scale_stats(data)
        med_col = median[:, np.newaxis]
        iqr_col = iqr[:, np.newaxis] + 1e-8
        rlen = data.shape[1]
        ws = cfg.window_samples
        stride = max(1, int(ws * cfg.inference_stride_ratio))
        starts = list(range(0, rlen - ws + 1, stride))
        cond_t = torch.from_numpy(cond).unsqueeze(0).to(cfg.device)

        amp_device = (
            "mps"
            if cfg.device == "mps"
            else ("cuda" if cfg.device == "cuda" else "cpu")
        )
        amp_dtype = torch.float16 if cfg.device == "mps" else torch.bfloat16

        pred_acc = np.zeros(rlen, dtype=np.float32)
        count = np.zeros(rlen, dtype=np.float32)

        with torch.no_grad():
            for batch_start in range(0, len(starts), 64):
                batch_starts = starts[batch_start : batch_start + 64]
                windows = np.stack(
                    [(data[:, s : s + ws] - med_col) / iqr_col for s in batch_starts]
                )
                sig_t = torch.from_numpy(windows).to(cfg.device)
                cond_batch = cond_t.expand(len(batch_starts), -1)
                with autocast(device_type=amp_device, dtype=amp_dtype):
                    logits = model(sig_t, cond_batch)
                probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
                for i, s in enumerate(batch_starts):
                    pred_acc[s : s + ws] += probs[i]
                    count[s : s + ws] += 1.0

        proba = np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)
        pred_bin = post_process_mask((proba > threshold).astype(np.int_), cfg)

        onsets = row.get("onset_times", [])
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []
        if isinstance(onsets, np.ndarray):
            onsets = onsets.tolist()
        mask_dur_ms = row.get("mask_duration_ms", 1000.0)
        if pd.isna(mask_dur_ms):
            mask_dur_ms = 1000.0
        onset_offset_ms = row.get("mask_onset_offset_ms", 0.0)
        if pd.isna(onset_offset_ms):
            onset_offset_ms = 0.0

        true_mask = build_true_mask(rlen, onsets, mask_dur_ms, sr, onset_offset_ms)
        _, _, _, tp, fp, fn = compute_event_f1(true_mask, pred_bin)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        inter = np.logical_and(pred_bin, true_mask).sum()
        union = pred_bin.sum() + true_mask.sum()
        all_dice.append(2 * inter / (union + 1e-8))

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "dice": float(np.mean(all_dice)) if all_dice else 0.0,
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_film_onoff(model, cfg, df) -> pd.DataFrame:
    """Exp 1: FiLM on vs off (zero all conditioning)."""
    print("\n=== Experiment 1: FiLM On/Off ===")
    baseline = run_inference_pass(model, cfg, df, desc="Baseline")
    ablated = run_inference_pass(
        model, cfg, df, cond_transform=lambda c: np.zeros_like(c), desc="FiLM-off"
    )
    results = pd.DataFrame(
        [
            {"condition": "baseline", **baseline},
            {"condition": "film_off", **ablated},
        ]
    )
    print(
        results[["condition", "f1", "precision", "recall", "dice"]].to_string(
            index=False
        )
    )
    return results


def experiment_film_groups(model, cfg, df) -> pd.DataFrame:
    """Exp 2: Zero one feature group at a time."""
    print("\n=== Experiment 2: FiLM Feature Groups ===")
    rows = []
    baseline = run_inference_pass(model, cfg, df, desc="Baseline")
    rows.append({"group": "baseline", **baseline})

    for name, (start, end) in FEATURE_GROUPS.items():

        def make_transform(s, e):
            def transform(c):
                c = c.copy()
                c[s:e] = 0.0
                return c

            return transform

        ablated = run_inference_pass(
            model,
            cfg,
            df,
            cond_transform=make_transform(start, end),
            desc=f"Zero {name}",
        )
        rows.append({"group": name, **ablated})
        print(
            f"  {name}: F1={ablated['f1']:.4f} (Δ={ablated['f1'] - baseline['f1']:+.4f})"
        )

    return pd.DataFrame(rows)


def experiment_film_permute(model, cfg, df, n_repeats: int = 5) -> pd.DataFrame:
    """Exp 3: Shuffle conditioning vectors across files."""
    print("\n=== Experiment 3: FiLM Permutation Importance ===")
    baseline = run_inference_pass(model, cfg, df, desc="Baseline")

    # Collect all conditioning vectors
    cond_matrix = np.stack(
        [extract_conditioning_vector(row) for _, row in df.iterrows()]
    )

    rows = [{"repeat": "baseline", **baseline}]
    rng = np.random.default_rng(42)

    for rep in range(n_repeats):
        perm = rng.permutation(len(df))
        shuffled_conds = cond_matrix[perm]
        cond_idx = [0]  # mutable counter for closure

        def make_perm_transform(shuffled, counter):
            def transform(c):
                idx = counter[0]
                counter[0] += 1
                return shuffled[idx % len(shuffled)]

            return transform

        counter = [0]
        ablated = run_inference_pass(
            model,
            cfg,
            df,
            cond_transform=make_perm_transform(shuffled_conds, counter),
            desc=f"Permute {rep + 1}/{n_repeats}",
        )
        rows.append({"repeat": f"permute_{rep}", **ablated})
        print(f"  Repeat {rep + 1}: F1={ablated['f1']:.4f}")

    results = pd.DataFrame(rows)
    perm_f1s = [r["f1"] for r in rows[1:]]
    print(
        f"  Baseline F1: {baseline['f1']:.4f}, "
        f"Permuted mean F1: {np.mean(perm_f1s):.4f} ± {np.std(perm_f1s):.4f}"
    )
    return results


def experiment_postproc_sweep(model, cfg, df) -> pd.DataFrame:
    """Exp 4: Grid search over threshold x merge_gap x min_duration."""
    print("\n=== Experiment 4: Post-Processing Sweep ===")
    sr = cfg.sampling_rate

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    merge_gaps_ms = [25, 50, 100, 150, 200, 300, 500]
    min_durs_ms = [100, 150, 200, 300, 400, 600]

    # Cache probabilities and ground-truth masks
    print("Caching predictions...")
    proba_cache = {}
    gt_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        fp = row["file_path"]
        if not Path(fp).exists() or fp in proba_cache:
            continue
        cond = extract_conditioning_vector(row)
        proba = predict_file_proba(model, cfg, fp, cond)
        proba_cache[fp] = proba

        onsets = row.get("onset_times", [])
        if isinstance(onsets, str):
            onsets = json.loads(onsets) if onsets.strip() else []
        if isinstance(onsets, np.ndarray):
            onsets = onsets.tolist()
        mask_dur_ms = row.get("mask_duration_ms", 1000.0)
        if pd.isna(mask_dur_ms):
            mask_dur_ms = 1000.0
        onset_offset_ms = row.get("mask_onset_offset_ms", 0.0)
        if pd.isna(onset_offset_ms):
            onset_offset_ms = 0.0
        gt_cache[fp] = build_true_mask(
            len(proba), onsets, mask_dur_ms, sr, onset_offset_ms
        )

    print(f"Cached {len(proba_cache)} files")

    # Sweep
    results = []
    t0 = time.time()
    for thr in tqdm(thresholds, desc="Threshold sweep"):
        for mg_ms in merge_gaps_ms:
            for md_ms in min_durs_ms:
                sweep_cfg = copy.copy(cfg)
                sweep_cfg.merge_gap_ms = mg_ms
                sweep_cfg.min_artifact_samples = max(1, int(md_ms / 1000.0 * sr))

                tot_tp = tot_fp = tot_fn = 0
                for fp, proba in proba_cache.items():
                    pred_bin = post_process_mask(
                        (proba > thr).astype(np.int_), sweep_cfg
                    )
                    _, _, _, tp, _fp, fn = compute_event_f1(gt_cache[fp], pred_bin)
                    tot_tp += tp
                    tot_fp += _fp
                    tot_fn += fn

                p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
                r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                results.append(
                    {
                        "threshold": thr,
                        "merge_gap_ms": mg_ms,
                        "min_dur_ms": md_ms,
                        "precision": p,
                        "recall": r,
                        "F1": f1,
                        "TP": tot_tp,
                        "FP": tot_fp,
                        "FN": tot_fn,
                    }
                )

    sweep_df = (
        pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    )
    print(f"Sweep done in {time.time() - t0:.1f}s")
    print("\nTop 10:")
    print(sweep_df.head(10).to_string(index=False))
    return sweep_df


def experiment_channel_copy(model, cfg, df) -> pd.DataFrame:
    """Exp 5: Copy one channel to all 4 inputs."""
    print("\n=== Experiment 5: Channel Copy ===")
    baseline = run_inference_pass(model, cfg, df, desc="Baseline (all channels)")
    rows = [{"source": "all_channels", **baseline}]

    for ch in range(cfg.n_channels):

        def make_copy(ch_idx):
            def transform(data):
                out = np.empty_like(data)
                out[:] = data[ch_idx : ch_idx + 1, :]
                return out

            return transform

        ablated = run_inference_pass(
            model, cfg, df, signal_transform=make_copy(ch), desc=f"Ch{ch + 1} only"
        )
        rows.append({"source": f"ch{ch + 1}_only", **ablated})
        print(
            f"  Ch{ch + 1}: F1={ablated['f1']:.4f} (Δ={ablated['f1'] - baseline['f1']:+.4f})"
        )

    return pd.DataFrame(rows)


def experiment_skip_ablation(model, cfg, df) -> pd.DataFrame:
    """Exp 6: Zero out skip connections one level at a time."""
    print("\n=== Experiment 6: Skip Connection Ablation ===")
    import torch

    baseline = run_inference_pass(model, cfg, df, desc="Baseline")
    rows = [{"level": "baseline", **baseline}]

    raw_model = getattr(model, "_orig_mod", model)

    # Identify encoder/decoder skip connection points
    # The U-Net has 4 levels; skips connect enc_i to dec_i
    n_levels = len(cfg.channel_mult)

    for level in range(n_levels):
        # Hook to zero the skip connection at this level
        hooks = []
        skip_name = f"enc{level + 1}"

        def make_zero_hook(lv):
            def hook_fn(module, input, output):
                return torch.zeros_like(output)

            return hook_fn

        # Find the encoder block and register hook
        enc_block = getattr(raw_model, f"enc{level + 1}", None)
        if enc_block is None:
            continue

        h = enc_block.register_forward_hook(make_zero_hook(level))
        hooks.append(h)

        ablated = run_inference_pass(model, cfg, df, desc=f"Zero skip L{level + 1}")
        rows.append({"level": f"skip_L{level + 1}", **ablated})
        print(
            f"  Skip L{level + 1}: F1={ablated['f1']:.4f} "
            f"(Δ={ablated['f1'] - baseline['f1']:+.4f})"
        )

        for h in hooks:
            h.remove()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_ablation_summary(all_results: dict, save_path: Path) -> None:
    """Grouped bar chart summarizing all ablation experiments."""
    entries = []

    # Collect baseline and ablated F1 from each experiment
    for name, df in all_results.items():
        if name == "postproc_sweep":
            continue
        if "f1" not in df.columns:
            continue
        for _, row in df.iterrows():
            label = row.get(
                "condition",
                row.get(
                    "group", row.get("source", row.get("level", row.get("repeat", "")))
                ),
            )
            entries.append(
                {"experiment": name, "condition": str(label), "f1": row["f1"]}
            )

    if not entries:
        return

    edf = pd.DataFrame(entries)
    baselines = edf[edf["condition"].isin(["baseline", "all_channels"])]
    ablated = edf[~edf["condition"].isin(["baseline", "all_channels"])]

    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot baseline as horizontal line per experiment
    experiments = edf["experiment"].unique()
    x_positions = []
    labels = []
    pos = 0

    for exp in experiments:
        exp_data = edf[edf["experiment"] == exp]
        bl = exp_data[exp_data["condition"].isin(["baseline", "all_channels"])]
        bl_f1 = bl["f1"].values[0] if len(bl) > 0 else 0
        abl = exp_data[~exp_data["condition"].isin(["baseline", "all_channels"])]

        for _, row in abl.iterrows():
            delta = row["f1"] - bl_f1
            color = (
                "#F44336"
                if delta < -0.01
                else ("#4CAF50" if delta > 0.01 else "#9E9E9E")
            )
            ax.bar(pos, row["f1"], color=color, alpha=0.8, width=0.7)
            ax.hlines(bl_f1, pos - 0.35, pos + 0.35, colors="black", lw=1.5, ls="--")
            labels.append(f"{row['condition']}")
            x_positions.append(pos)
            pos += 1
        pos += 0.5  # gap between experiments

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Event F1")
    ax.set_title("Ablation Summary — Event F1 vs Baseline")
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_postproc_heatmap(sweep_df: pd.DataFrame, save_path: Path) -> None:
    """Threshold x merge_gap F1 heatmap (max over min_dur)."""
    if len(sweep_df) == 0:
        return
    pivot = sweep_df.pivot_table(
        index="merge_gap_ms", columns="threshold", values="F1", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t:.2f}" for t in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Merge Gap (ms)")
    ax.set_title("Post-Processing Sweep — Event F1")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=7
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = [
    "film_onoff",
    "film_groups",
    "film_permute",
    "postproc_sweep",
    "channel_copy",
    "skip_ablation",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stim artifact ablation experiments")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
    )
    p.add_argument("--catalog", type=Path, default=Path("data/stim_catalog.parquet"))
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=ALL_EXPERIMENTS + ["all"],
    )
    p.add_argument("--max-files", type=int, default=3000)
    p.add_argument("--split", choices=["all", "val"], default="val")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"])
    p.add_argument(
        "--checkpoint-type",
        choices=["stimask", "deployed"],
        default="stimask",
        help="Model type: stimask (original) or deployed (HP-searched)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    experiments = ALL_EXPERIMENTS if "all" in args.experiments else args.experiments

    device = args.device or "mps"
    if args.checkpoint_type == "deployed":
        from deployed_adapter import load_deployed_checkpoint

        ckpt = args.checkpoint
        if ckpt == Path("checkpoints/best_model.pt"):
            ckpt = Path("data/checkpoints/m4_unet.pt")
        print(f"Loading deployed checkpoint: {ckpt}")
        model, cfg = load_deployed_checkpoint(ckpt, device=device)
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, cfg = load_checkpoint(args.checkpoint, device=device)
    cfg.catalog_path = args.catalog

    df = load_annotations(cfg)
    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)
    print(f"Catalog: {len(df)} files on disk")

    if args.split == "val":
        if args.checkpoint_type == "deployed":
            # Fixed val split matching deployed training
            val_subjects = ["300-002", "301-003", "303-004", "305-001"]
            df = df[df["subject"].isin(val_subjects)].reset_index(drop=True)
        else:
            groups = df["subject"].values
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
            _, val_idx = next(splitter.split(df, groups=groups))
            df = df.iloc[val_idx].reset_index(drop=True)
        print(f"Val split: {len(df)} files")

    if args.max_files and len(df) > args.max_files:
        df = stratified_subsample(df, args.max_files)
        print(f"Subsampled to {len(df)} files")

    all_results = {}

    if "film_onoff" in experiments:
        out = RESULTS_DIR / "ablation_film_onoff.csv"
        if not out.exists() or args.force:
            r = experiment_film_onoff(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["film_onoff"] = r
        else:
            all_results["film_onoff"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    if "film_groups" in experiments:
        out = RESULTS_DIR / "ablation_film_groups.csv"
        if not out.exists() or args.force:
            r = experiment_film_groups(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["film_groups"] = r
        else:
            all_results["film_groups"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    if "film_permute" in experiments:
        out = RESULTS_DIR / "ablation_film_permute.csv"
        if not out.exists() or args.force:
            r = experiment_film_permute(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["film_permute"] = r
        else:
            all_results["film_permute"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    if "postproc_sweep" in experiments:
        out = RESULTS_DIR / "ablation_postproc_sweep.csv"
        if not out.exists() or args.force:
            r = experiment_postproc_sweep(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["postproc_sweep"] = r
        else:
            all_results["postproc_sweep"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    if "channel_copy" in experiments:
        out = RESULTS_DIR / "ablation_channel_copy.csv"
        if not out.exists() or args.force:
            r = experiment_channel_copy(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["channel_copy"] = r
        else:
            all_results["channel_copy"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    if "skip_ablation" in experiments:
        out = RESULTS_DIR / "ablation_skip_connections.csv"
        if not out.exists() or args.force:
            r = experiment_skip_ablation(model, cfg, df)
            r.to_csv(out, index=False)
            all_results["skip_ablation"] = r
        else:
            all_results["skip_ablation"] = pd.read_csv(out)
            print(f"Loaded cached: {out}")

    # Generate summary figures
    if all_results:
        plot_ablation_summary(all_results, FIGURES_DIR / "ablation_summary.pdf")
    if "postproc_sweep" in all_results:
        plot_postproc_heatmap(
            all_results["postproc_sweep"], FIGURES_DIR / "postproc_sweep_heatmap.pdf"
        )

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
