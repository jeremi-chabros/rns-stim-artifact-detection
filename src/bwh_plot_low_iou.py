#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "scipy>=1.10",
#   "plotly>=5.0",
#   "tqdm>=4.60",
#   "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""
Plot trace overlays for BWH files with IoU event_f1 < 0.5.

Shows raw signal (4 channels) + GT mask + model prediction probability
to visualize the mask width mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.ndimage import binary_closing, find_objects, label
from tqdm.auto import tqdm

from lgs_db import read_dat, to_microvolts

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DEPLOYED_DIR = Path("/path/to/Research/training-pipeline")
sys.path.insert(0, str(DEPLOYED_DIR))
from prepare import (
    extract_conditioning_vector,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
    STRIDE_SAMPLES,
    BOUNDARY_MARGIN_S,
)
from train_5b0d152 import StimArtifactUNet

BWH_CATALOG_PATH = Path("data/bwh_stim_catalog.parquet")
EVAL_PATH = Path("outputs/results/bwh_unet_eval_500_v2.csv")
OUT_DIR = Path("outputs/diagnostics/bwh/low_iou")

SR = SAMPLING_RATE
MIN_ARTIFACT_SAMPLES = 25
MERGE_GAP_MS = 200.0


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------


def load_model(device: str = "mps") -> StimArtifactUNet:
    """Load the training pipeline checkpoint."""
    sd = torch.load(
        str(DEPLOYED_DIR / "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = StimArtifactUNet()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params on {device}")
    return model


@torch.no_grad()
def predict_file(
    model: StimArtifactUNet,
    dat_path: str,
    cond: np.ndarray,
    device: str = "mps",
) -> np.ndarray:
    """Sliding-window inference → per-sample probability."""
    raw = read_dat(dat_path)
    data = to_microvolts(raw).astype(np.float32)
    median = np.median(data, axis=1, keepdims=True)
    q75 = np.percentile(data, 75, axis=1, keepdims=True)
    q25 = np.percentile(data, 25, axis=1, keepdims=True)
    iqr = q75 - q25 + 1e-8

    n_samples = data.shape[1]
    ws, stride = WINDOW_SAMPLES, STRIDE_SAMPLES
    starts = list(range(0, n_samples - ws + 1, stride))
    if not starts:
        return np.zeros(n_samples, dtype=np.float32)

    cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)
    pred_acc = np.zeros(n_samples, dtype=np.float32)
    count = np.zeros(n_samples, dtype=np.float32)

    for bi in range(0, len(starts), 64):
        batch_starts = starts[bi : bi + 64]
        windows = np.stack([(data[:, s : s + ws] - median) / iqr for s in batch_starts])
        sig_t = torch.from_numpy(windows).to(device)
        cond_batch = cond_t.expand(len(batch_starts), -1)
        logits, _ = model(sig_t, cond_batch)
        probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
        for i, s in enumerate(batch_starts):
            pred_acc[s : s + ws] += probs[i]
            count[s : s + ws] += 1.0

    return np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)


# ---------------------------------------------------------------------------
# GT mask
# ---------------------------------------------------------------------------


def build_gt_mask(
    n_samples: int,
    onset_times: list[float],
    mask_dur_ms: float,
) -> np.ndarray:
    """Binary GT mask from onsets."""
    mask = np.zeros(n_samples, dtype=np.int8)
    dur_samp = max(int(mask_dur_ms / 1000.0 * SR), 1)
    for t in onset_times:
        s = int(t * SR)
        mask[max(0, s) : min(n_samples, s + dur_samp)] = 1
    return mask


# ---------------------------------------------------------------------------
# Postprocessing (matching eval_bwh.py v2)
# ---------------------------------------------------------------------------


def postprocess(pred_mask: np.ndarray) -> np.ndarray:
    """Binary closing + small-component removal with lowered threshold."""
    close_samp = int(MERGE_GAP_MS / 1000 * SR)
    closed = binary_closing(
        pred_mask.astype(bool), structure=np.ones(max(close_samp, 1))
    )
    labeled_arr, _ = label(closed)
    for obj_slice in find_objects(labeled_arr):
        if obj_slice is None:
            continue
        if (obj_slice[0].stop - obj_slice[0].start) < MIN_ARTIFACT_SAMPLES:
            closed[obj_slice] = False
    return closed.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def plot_trace(
    row: pd.Series,
    proba: np.ndarray,
    uv: np.ndarray,
) -> go.Figure:
    """4-channel trace + GT mask + prediction probability + binary overlay."""
    n_ch, n_samp = uv.shape
    t = np.arange(n_samp) / SR

    gt = build_gt_mask(n_samp, row["onset_times"], row["mask_duration_ms"])
    pred_bin = postprocess((proba > 0.5).astype(np.float32))

    n_panels = n_ch + 2  # channels + GT/pred overlay + proba
    titles = [f"Ch {i+1}" for i in range(n_ch)] + [
        "GT (green) vs Predicted (blue) binary masks",
        "Model P(stim) — raw probability",
    ]
    heights = [1.0] * n_ch + [0.5, 0.5]

    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=titles,
        row_heights=heights,
    )

    # --- Channels ---
    for ch in range(n_ch):
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=uv[ch],
                mode="lines",
                line=dict(color="#333", width=0.5),
                showlegend=False,
            ),
            row=ch + 1,
            col=1,
        )
        # Shade GT (green) and pred (blue) on channels
        gt_on = np.where(np.diff(np.concatenate([[0], gt, [0]])) == 1)[0]
        gt_off = np.where(np.diff(np.concatenate([[0], gt, [0]])) == -1)[0]
        for on, off in zip(gt_on, gt_off):
            fig.add_vrect(
                x0=on / SR,
                x1=off / SR,
                fillcolor="green",
                opacity=0.25,
                line_width=0,
                row=ch + 1,
                col=1,
            )
        pred_on = np.where(np.diff(np.concatenate([[0], pred_bin, [0]])) == 1)[0]
        pred_off = np.where(np.diff(np.concatenate([[0], pred_bin, [0]])) == -1)[0]
        for on, off in zip(pred_on, pred_off):
            fig.add_vrect(
                x0=on / SR,
                x1=off / SR,
                fillcolor="blue",
                opacity=0.12,
                line_width=0,
                row=ch + 1,
                col=1,
            )
        fig.update_yaxes(title_text="uV", row=ch + 1, col=1)

    # --- Binary mask overlay panel ---
    mask_row = n_ch + 1
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=gt.astype(float) * 0.9,
            mode="lines",
            name="GT mask",
            line=dict(color="green", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,128,0,0.3)",
        ),
        row=mask_row,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=pred_bin * 1.0,
            mode="lines",
            name="Pred mask",
            line=dict(color="blue", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(0,0,255,0.1)",
        ),
        row=mask_row,
        col=1,
    )
    fig.update_yaxes(range=[-0.1, 1.2], row=mask_row, col=1)

    # --- Probability panel ---
    prob_row = n_ch + 2
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=proba,
            mode="lines",
            name="P(stim)",
            line=dict(color="blue", width=1),
            fill="tozeroy",
            fillcolor="rgba(0,0,255,0.08)",
        ),
        row=prob_row,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        row=prob_row,
        col=1,
    )
    # Shade GT on probability panel
    for on, off in zip(gt_on, gt_off):
        fig.add_vrect(
            x0=on / SR,
            x1=off / SR,
            fillcolor="green",
            opacity=0.15,
            line_width=0,
            row=prob_row,
            col=1,
        )

    # --- Boundary margins ---
    rec_len = n_samp / SR
    for r in range(1, n_panels + 1):
        fig.add_vrect(
            x0=0,
            x1=BOUNDARY_MARGIN_S,
            fillcolor="orange",
            opacity=0.05,
            line_width=0,
            row=r,
            col=1,
        )
        fig.add_vrect(
            x0=rec_len - BOUNDARY_MARGIN_S,
            x1=rec_len,
            fillcolor="orange",
            opacity=0.05,
            line_width=0,
            row=r,
            col=1,
        )

    dur = row["mask_duration_ms"]
    dur_samp = int(dur / 1000 * SR)

    # Compute pred event widths for annotation
    pred_widths = [(off - on) / SR * 1000 for on, off in zip(pred_on, pred_off)]
    gt_widths = [(off - on) / SR * 1000 for on, off in zip(gt_on, gt_off)]
    pw_str = ", ".join(f"{w:.0f}" for w in pred_widths[:5])
    gw_str = ", ".join(f"{w:.0f}" for w in gt_widths[:5])

    fig.update_layout(
        title=(
            f"<b>IoU-Failing File</b> | subj={row['subject']} | "
            f"eF1={row['event_f1']:.3f} | onset_F1={row['onset_f1']:.3f} | "
            f"sR={row['sample_recall']:.3f}<br>"
            f"GT mask: {dur:.0f}ms ({dur_samp} samp) | "
            f"GT event widths: [{gw_str}] ms<br>"
            f"Pred event widths: [{pw_str}] ms | "
            f"n_gt={row['n_gt_events']:.0f} n_pred={row['n_pred_events']:.0f}<br>"
            f"<sub>{Path(row['file_path']).stem}</sub>"
        ),
        height=140 * n_panels + 80,
        width=1400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    eval_df = pd.read_csv(EVAL_PATH)
    eval_df["subject"] = eval_df["subject"].astype(str)
    eval_df["filename"] = eval_df["filename"].astype(str)

    cat = pd.read_parquet(BWH_CATALOG_PATH)
    cat["filename"] = cat["filename"].astype(str)
    cat["subject"] = cat["subject"].astype(str)
    cat["onset_times"] = cat["onset_times"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Drop mask_duration_ms from eval (redundant with catalog)
    eval_df = eval_df.drop(columns=["mask_duration_ms"], errors="ignore")

    # Merge
    merged = eval_df.merge(
        cat[
            [
                "filename",
                "subject",
                "mask_duration_ms",
                "t1b1_ms",
                "t1b2_ms",
                "file_path",
                "onset_times",
                "n_stim_events",
                "length_sec",
            ]
        ].drop_duplicates("filename"),
        on=["filename", "subject"],
        how="left",
    )

    # Filter to IoU < 0.5 with valid file paths
    low_iou = merged[(merged["event_f1"] < 0.5) & merged["file_path"].notna()].copy()
    low_iou = low_iou[low_iou["file_path"].apply(lambda p: Path(p).exists())]

    print(f"Files with event F1 < 0.5: {len(low_iou)}")
    if len(low_iou) == 0:
        print("Nothing to plot.")
        return

    # Load model
    model = load_model()

    # Generate traces
    for idx, (_, row) in enumerate(
        tqdm(low_iou.iterrows(), total=len(low_iou), desc="Plotting")
    ):
        fp = row["file_path"]
        raw = read_dat(str(fp))
        uv = to_microvolts(raw).astype(np.float32)

        cond = extract_conditioning_vector(row)
        proba = predict_file(model, str(fp), cond)

        fig = plot_trace(row, proba, uv)

        safe_fn = str(row["filename"])[:30]
        subj = row["subject"]
        path = OUT_DIR / f"{idx:02d}_{subj}_{safe_fn}.html"
        fig.write_html(path)

    print(f"\nDone. {len(low_iou)} figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
