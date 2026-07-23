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
BWH external validation failure analysis.

Investigates why the U-Net (trained on LGS) fails on certain BWH subjects.
Produces:
  1. Failure mode classification table
  2. Mask duration distribution comparison (LGS train vs BWH pass/fail)
  3. Trace overlays: raw signal + GT mask + model prediction
  4. GT vs prediction alignment analysis

Usage:
    uv run src/bwh_failure_analysis.py                    # analysis only (no model)
    uv run src/bwh_failure_analysis.py --with-model       # + model predictions on traces
    uv run src/bwh_failure_analysis.py --n-traces 5       # more trace examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from lgs_db import read_dat, to_microvolts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BWH_CATALOG_PATH = Path("data/bwh_stim_catalog.parquet")
BWH_EVAL_PATH = Path("outputs/results/bwh_unet_eval_500_fixed.csv")
LGS_CATALOG_PATH = Path("data/stim_catalog.parquet")
OUT_DIR = Path("outputs/diagnostics/bwh")

SR = 250

# From the training pipeline
MIN_ARTIFACT_SAMPLES = 75
MERGE_GAP_MS = 300.0
BOUNDARY_MARGIN_S = 4.0
EVENT_IOU_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load BWH eval results, BWH catalog, and LGS catalog."""
    eval_df = pd.read_csv(BWH_EVAL_PATH)
    eval_df["subject"] = eval_df["subject"].astype(str)

    bwh_cat = pd.read_parquet(BWH_CATALOG_PATH)
    bwh_cat["filename"] = bwh_cat["filename"].astype(str)
    bwh_cat["subject"] = bwh_cat["subject"].astype(str)
    bwh_cat["onset_times"] = bwh_cat["onset_times"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    lgs_cat = pd.read_parquet(LGS_CATALOG_PATH)
    lgs_cat["onset_times"] = lgs_cat["onset_times"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    return eval_df, bwh_cat, lgs_cat


def merge_eval_catalog(eval_df: pd.DataFrame, bwh_cat: pd.DataFrame) -> pd.DataFrame:
    """Merge eval results with catalog metadata."""
    eval_df["filename"] = eval_df["filename"].astype(str)
    merged = eval_df.merge(
        bwh_cat[
            [
                "filename",
                "subject",
                "mask_duration_ms",
                "t1b1_ms",
                "t1b2_ms",
                "t1b1_ma",
                "t1b1_path",
                "lead_1",
                "lead_2",
                "file_path",
                "onset_times",
                "n_stim_events",
                "length_sec",
                "sampling_rate",
            ]
        ].drop_duplicates("filename"),
        on="filename",
        how="left",
        suffixes=("", "_cat"),
    )
    return merged


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def classify_failures(merged: pd.DataFrame) -> pd.DataFrame:
    """Classify each subject's dominant failure mode from per-file metrics."""
    subj_stats = merged.groupby("subject").agg(
        event_f1_mean=("event_f1", "mean"),
        sample_p_mean=("sample_precision", "mean"),
        sample_r_mean=("sample_recall", "mean"),
        n_files=("filename", "count"),
        gt_total=("n_gt_events", "sum"),
        pred_total=("n_pred_events", "sum"),
        tp_total=("event_tp", "sum"),
        fp_total=("event_fp", "sum"),
        fn_total=("event_fn", "sum"),
        mask_dur=("mask_duration_ms", "first"),
        b1_ms=("t1b1_ms", "first"),
        b2_ms=("t1b2_ms", "first"),
        b1_ma=("t1b1_ma", "first"),
    )

    def _classify(row: pd.Series) -> str:
        f1 = row["event_f1_mean"]
        sr = row["sample_r_mean"]
        sp = row["sample_p_mean"]
        pred = row["pred_total"]
        gt = row["gt_total"]
        tp = row["tp_total"]

        if f1 >= 0.7:
            return "GOOD"
        if pred == 0:
            return "ZERO_PRED"
        if sr > 0.8 and tp == 0:
            return "GT_MISALIGN"
        if sr < 0.1 and pred < gt * 0.15:
            return "MISSED"
        if sr > 0.5 and sp < 0.3 and tp == 0:
            return "FP_ONLY"
        return "MIXED"

    subj_stats["failure_mode"] = subj_stats.apply(_classify, axis=1)
    return subj_stats


# ---------------------------------------------------------------------------
# Figure 1: Failure mode summary table
# ---------------------------------------------------------------------------


def fig_failure_table(subj_stats: pd.DataFrame) -> go.Figure:
    """Colored table of per-subject failure classification."""
    df = subj_stats.sort_values("event_f1_mean").reset_index()
    mode_colors = {
        "GOOD": "#2ca02c",
        "ZERO_PRED": "#d62728",
        "GT_MISALIGN": "#ff7f0e",
        "MISSED": "#9467bd",
        "FP_ONLY": "#e377c2",
        "MIXED": "#7f7f7f",
    }

    cell_colors = [[mode_colors.get(m, "#ccc") for m in df["failure_mode"]]]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[
                    "Subject",
                    "Mode",
                    "Event F1",
                    "Sample P",
                    "Sample R",
                    "Files",
                    "GT / Pred",
                    "TP / FP / FN",
                    "Mask (ms)",
                    "B1 (ms)",
                    "B2 (ms)",
                    "B1 (mA)",
                ],
                fill_color="#333",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=[
                    df["subject"],
                    df["failure_mode"],
                    [f"{v:.3f}" for v in df["event_f1_mean"]],
                    [f"{v:.3f}" for v in df["sample_p_mean"]],
                    [f"{v:.3f}" for v in df["sample_r_mean"]],
                    df["n_files"],
                    [f"{g}/{p}" for g, p in zip(df["gt_total"], df["pred_total"])],
                    [
                        f"{t}/{fp}/{fn}"
                        for t, fp, fn in zip(
                            df["tp_total"], df["fp_total"], df["fn_total"]
                        )
                    ],
                    [f"{v:.0f}" for v in df["mask_dur"]],
                    [f"{v:.0f}" if pd.notna(v) else "?" for v in df["b1_ms"]],
                    [f"{v:.0f}" if pd.notna(v) else "?" for v in df["b2_ms"]],
                    [f"{v:.1f}" if pd.notna(v) else "?" for v in df["b1_ma"]],
                ],
                fill_color=[
                    ["white"] * len(df),
                    [
                        f"rgba({int(mode_colors.get(m, '#cccccc')[1:3], 16)},"
                        f"{int(mode_colors.get(m, '#cccccc')[3:5], 16)},"
                        f"{int(mode_colors.get(m, '#cccccc')[5:7], 16)},0.25)"
                        for m in df["failure_mode"]
                    ],
                ]
                + [["white"] * len(df)] * 10,
                align="center",
                font=dict(size=11),
                height=25,
            ),
        )
    )
    fig.update_layout(
        title="BWH External Validation: Per-Subject Failure Classification",
        height=max(400, 30 * len(df) + 100),
        width=1200,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Mask duration comparison
# ---------------------------------------------------------------------------


def fig_mask_duration_comparison(
    subj_stats: pd.DataFrame,
    bwh_cat: pd.DataFrame,
    lgs_cat: pd.DataFrame,
) -> go.Figure:
    """Compare mask duration distributions: LGS train vs BWH pass/fail."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "LGS Training Data",
            "BWH Passing (F1>0.7)",
            "BWH Failing (F1<0.7)",
        ],
        shared_yaxes=True,
    )

    # LGS
    fig.add_trace(
        go.Histogram(
            x=lgs_cat["mask_duration_ms"],
            nbinsx=30,
            marker_color="#1f77b4",
            name="LGS train",
        ),
        row=1,
        col=1,
    )

    # BWH passing subjects
    good_subjs = set(subj_stats[subj_stats["event_f1_mean"] >= 0.7].index)
    bwh_good = bwh_cat[bwh_cat["subject"].isin(good_subjs)]
    fig.add_trace(
        go.Histogram(
            x=bwh_good["mask_duration_ms"],
            nbinsx=30,
            marker_color="#2ca02c",
            name="BWH pass",
        ),
        row=1,
        col=2,
    )

    # BWH failing
    bad_subjs = set(subj_stats[subj_stats["event_f1_mean"] < 0.7].index)
    bwh_bad = bwh_cat[bwh_cat["subject"].isin(bad_subjs)]
    fig.add_trace(
        go.Histogram(
            x=bwh_bad["mask_duration_ms"],
            nbinsx=30,
            marker_color="#d62728",
            name="BWH fail",
        ),
        row=1,
        col=3,
    )

    # Mark the LGS minimum and the 200ms line
    for col in [1, 2, 3]:
        fig.add_vline(
            x=280,
            line_dash="dash",
            line_color="blue",
            annotation_text="LGS min (280ms)" if col == 1 else None,
            row=1,
            col=col,
        )
        fig.add_vline(
            x=200,
            line_dash="dot",
            line_color="red",
            annotation_text="200ms" if col == 3 else None,
            row=1,
            col=col,
        )
        # Mark MIN_ARTIFACT_SAMPLES threshold
        min_dur_ms = MIN_ARTIFACT_SAMPLES / SR * 1000
        fig.add_vline(
            x=min_dur_ms,
            line_dash="dashdot",
            line_color="orange",
            annotation_text=f"postproc min ({min_dur_ms:.0f}ms)" if col == 2 else None,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=(
            "Mask Duration Distribution: LGS Training vs BWH Pass/Fail<br>"
            "<sub>Failing BWH subjects cluster at 200ms — below LGS training minimum (280ms) "
            "and below postprocessing min-event filter (300ms)</sub>"
        ),
        height=400,
        width=1100,
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mask duration (ms)")
    fig.update_yaxes(title_text="Files", row=1, col=1)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: F1 vs mask duration scatter
# ---------------------------------------------------------------------------


def fig_f1_vs_duration(merged: pd.DataFrame) -> go.Figure:
    """Scatter: per-file event F1 vs mask_duration_ms."""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=merged["mask_duration_ms"],
            y=merged["event_f1"],
            mode="markers",
            marker=dict(
                size=5,
                color=merged["event_f1"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Event F1"),
                opacity=0.7,
            ),
            text=[
                f"subj={s}<br>dur={d:.0f}ms<br>F1={f:.3f}"
                for s, d, f in zip(
                    (
                        merged["subject_cat"]
                        if "subject_cat" in merged.columns
                        else merged["subject"]
                    ),
                    merged["mask_duration_ms"],
                    merged["event_f1"],
                )
            ],
            hoverinfo="text",
        )
    )

    # Reference lines
    fig.add_vline(x=280, line_dash="dash", line_color="blue", annotation_text="LGS min")
    fig.add_vline(
        x=MIN_ARTIFACT_SAMPLES / SR * 1000,
        line_dash="dashdot",
        line_color="orange",
        annotation_text=f"postproc min ({MIN_ARTIFACT_SAMPLES / SR * 1000:.0f}ms)",
    )
    fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.5)

    fig.update_layout(
        title=(
            "Event F1 vs Mask Duration (BWH 500-file eval)<br>"
            "<sub>All files with mask <280ms fail; postprocessing kills events <300ms</sub>"
        ),
        xaxis_title="Mask Duration (ms)",
        yaxis_title="Event F1",
        height=500,
        width=900,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Per-subject F1 bar chart colored by failure mode
# ---------------------------------------------------------------------------


def fig_subject_f1_bars(subj_stats: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of per-subject event F1, colored by failure mode."""
    df = subj_stats.sort_values("event_f1_mean").reset_index()
    mode_colors = {
        "GOOD": "#2ca02c",
        "ZERO_PRED": "#d62728",
        "GT_MISALIGN": "#ff7f0e",
        "MISSED": "#9467bd",
        "FP_ONLY": "#e377c2",
        "MIXED": "#7f7f7f",
    }

    fig = go.Figure()
    for mode, color in mode_colors.items():
        mask = df["failure_mode"] == mode
        if not mask.any():
            continue
        fig.add_trace(
            go.Bar(
                y=df.loc[mask, "subject"],
                x=df.loc[mask, "event_f1_mean"],
                orientation="h",
                marker_color=color,
                name=mode,
                text=[f"{v:.3f}" for v in df.loc[mask, "event_f1_mean"]],
                textposition="outside",
                hovertemplate=(
                    "Subject: %{y}<br>F1: %{x:.3f}<br>"
                    "Mask: %{customdata[0]:.0f}ms<br>"
                    "Files: %{customdata[1]}<extra></extra>"
                ),
                customdata=df.loc[mask, ["mask_dur", "n_files"]].values,
            )
        )

    fig.update_layout(
        title="BWH Per-Subject Event F1 (colored by failure mode)",
        xaxis_title="Mean Event F1",
        yaxis_title="Subject (patient_id)",
        height=max(500, 22 * len(df)),
        width=900,
        template="plotly_white",
        xaxis_range=[0, 1.15],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Trace overlays (raw signal + GT + optional prediction)
# ---------------------------------------------------------------------------


def _build_true_mask(
    n_samples: int,
    onset_times: list[float],
    mask_dur_ms: float,
    sr: int = SR,
) -> np.ndarray:
    """Build binary GT mask from onset times."""
    mask = np.zeros(n_samples, dtype=np.int8)
    dur_samp = max(int(mask_dur_ms / 1000.0 * sr), 1)
    for t in onset_times:
        s = int(t * sr)
        mask[max(0, s) : min(n_samples, s + dur_samp)] = 1
    return mask


def fig_trace_overlay(
    row: pd.Series,
    proba: np.ndarray | None = None,
    title_prefix: str = "",
) -> go.Figure:
    """Single file trace: 4 channels + GT mask + optional prediction."""
    fp = row["file_path"]
    raw = read_dat(str(fp))
    uv = to_microvolts(raw).astype(np.float32)
    n_ch, n_samp = uv.shape
    t = np.arange(n_samp) / SR

    gt = _build_true_mask(n_samp, row["onset_times"], row["mask_duration_ms"])

    n_panels = n_ch + 1 + (1 if proba is not None else 0)
    titles = [f"Ch {i+1}" for i in range(n_ch)] + ["Ground Truth"]
    if proba is not None:
        titles.append("Model Prediction")
    heights = [1.0] * n_ch + [0.4] * (n_panels - n_ch)

    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
        row_heights=heights,
    )

    # Channels
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
        # Shade GT on channels
        gt_on = np.where(np.diff(np.concatenate([[0], gt, [0]])) == 1)[0]
        gt_off = np.where(np.diff(np.concatenate([[0], gt, [0]])) == -1)[0]
        for on, off in zip(gt_on, gt_off):
            fig.add_vrect(
                x0=on / SR,
                x1=off / SR,
                fillcolor="green",
                opacity=0.2,
                line_width=0,
                row=ch + 1,
                col=1,
            )
        fig.update_yaxes(title_text="uV", row=ch + 1, col=1)

    # GT mask panel
    gt_row = n_ch + 1
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=gt.astype(float),
            mode="lines",
            name="GT mask",
            line=dict(color="green", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,128,0,0.2)",
        ),
        row=gt_row,
        col=1,
    )

    # Prediction panel
    if proba is not None:
        pred_row = gt_row + 1
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=proba,
                mode="lines",
                name="Model P(stim)",
                line=dict(color="blue", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,0,255,0.1)",
            ),
            row=pred_row,
            col=1,
        )
        # Threshold line
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=pred_row, col=1
        )
        # Show what postprocessing would keep
        from scipy.ndimage import binary_closing, find_objects, label

        pred_bin = (proba > 0.5).astype(float)
        close_samp = int(MERGE_GAP_MS / 1000 * SR)
        closed = binary_closing(
            pred_bin.astype(bool), structure=np.ones(max(close_samp, 1))
        )
        labeled_arr, _ = label(closed)
        for obj_slice in find_objects(labeled_arr):
            if obj_slice is None:
                continue
            s, e = obj_slice[0].start, obj_slice[0].stop
            kept = (e - s) >= MIN_ARTIFACT_SAMPLES
            color = "rgba(0,0,255,0.15)" if kept else "rgba(255,0,0,0.15)"
            fig.add_vrect(
                x0=s / SR,
                x1=e / SR,
                fillcolor=color,
                line_width=0,
                row=pred_row,
                col=1,
                annotation=(
                    dict(text="KEPT" if kept else "REMOVED", font_size=9)
                    if not kept
                    else None
                ),
            )

    # Boundary margins
    rec_len = n_samp / SR
    for r in range(1, n_panels + 1):
        fig.add_vrect(
            x0=0,
            x1=BOUNDARY_MARGIN_S,
            fillcolor="orange",
            opacity=0.06,
            line_width=0,
            row=r,
            col=1,
        )
        fig.add_vrect(
            x0=rec_len - BOUNDARY_MARGIN_S,
            x1=rec_len,
            fillcolor="orange",
            opacity=0.06,
            line_width=0,
            row=r,
            col=1,
        )

    dur = row["mask_duration_ms"]
    dur_samp = int(dur / 1000 * SR)
    subj = row.get("subject_cat", row.get("subject", "?"))
    fig.update_layout(
        title=(
            f"{title_prefix}{subj} | "
            f"mask={dur:.0f}ms ({dur_samp} samp) | "
            f"n_stim={row['n_stim_events']} | "
            f"B1={row.get('t1b1_ms', '?')}ms B2={row.get('t1b2_ms', '?')}ms | "
            f"eF1={row.get('event_f1', '?')}<br>"
            f"<sub>{Path(fp).stem}</sub>"
        ),
        height=150 * n_panels + 60,
        width=1200,
        template="plotly_white",
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
    return fig


def generate_trace_figures(
    merged: pd.DataFrame,
    subj_stats: pd.DataFrame,
    n_per_mode: int = 3,
    with_model: bool = False,
) -> list[tuple[str, str, go.Figure]]:
    """Generate trace figures for representative files from each failure mode."""
    model = None
    if with_model:
        model = _load_model()

    figs: list[tuple[str, str, go.Figure]] = []
    modes_to_show = ["GT_MISALIGN", "ZERO_PRED", "MISSED", "MIXED", "GOOD"]

    for mode in modes_to_show:
        mode_subjects = subj_stats[subj_stats["failure_mode"] == mode].index.tolist()
        if not mode_subjects:
            continue

        # Pick files from different subjects
        candidates = merged[
            merged["subject"].isin(mode_subjects) & merged["file_path"].notna()
        ].copy()

        if len(candidates) == 0:
            continue

        # Sample: pick 1 file per subject, up to n_per_mode
        sampled = (
            candidates.groupby("subject")
            .apply(lambda g: g.sample(1, random_state=42), include_groups=False)
            .reset_index(drop=True)
            .head(n_per_mode)
        )

        for _, row in sampled.iterrows():
            fp = row["file_path"]
            if not Path(fp).exists():
                continue

            proba = None
            if model is not None:
                proba = _predict_file(model, row)

            fig = fig_trace_overlay(
                row,
                proba=proba,
                title_prefix=f"[{mode}] ",
            )
            figs.append((mode, row["filename"], fig))

    return figs


# ---------------------------------------------------------------------------
# Figure 6: GT label quality analysis
# ---------------------------------------------------------------------------


def fig_gt_label_analysis(merged: pd.DataFrame) -> go.Figure:
    """Analyze GT label characteristics by failure mode."""
    # For GT_MISALIGN subjects: the model predicts ~same number of events
    # with high sample recall. The issue is spatial offset.
    # Let's show: mask_dur (samples) vs MIN_ARTIFACT_SAMPLES threshold

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Mask Duration in Samples (all subjects)",
            "GT Events vs Predicted Events",
            "Sample Recall vs Event F1",
            "Mask Duration vs Event F1 (per subject)",
        ],
    )

    # Panel 1: Mask duration in samples
    merged["mask_dur_samp"] = (merged["mask_duration_ms"] / 1000 * SR).astype(int)
    fig.add_trace(
        go.Histogram(
            x=merged["mask_dur_samp"],
            nbinsx=30,
            marker_color="#1f77b4",
            name="BWH files",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=MIN_ARTIFACT_SAMPLES,
        line_dash="dash",
        line_color="red",
        annotation_text=f"postproc min ({MIN_ARTIFACT_SAMPLES})",
        row=1,
        col=1,
    )

    # Panel 2: GT events vs predicted events
    fig.add_trace(
        go.Scattergl(
            x=merged["n_gt_events"],
            y=merged["n_pred_events"],
            mode="markers",
            marker=dict(
                size=4,
                color=merged["event_f1"],
                colorscale="RdYlGn",
                opacity=0.6,
            ),
            name="files",
        ),
        row=1,
        col=2,
    )
    max_ev = max(merged["n_gt_events"].max(), merged["n_pred_events"].max())
    fig.add_trace(
        go.Scattergl(
            x=[0, max_ev],
            y=[0, max_ev],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Panel 3: Sample recall vs event F1
    fig.add_trace(
        go.Scattergl(
            x=merged["sample_recall"],
            y=merged["event_f1"],
            mode="markers",
            marker=dict(
                size=4,
                color=merged["mask_duration_ms"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Mask (ms)", x=0.45, y=0.2, len=0.4),
                opacity=0.6,
            ),
            text=[
                f"subj={s}, dur={d:.0f}ms"
                for s, d in zip(merged["subject"], merged["mask_duration_ms"])
            ],
            hoverinfo="text",
            name="files",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Per-subject mask duration vs F1
    subj_summary = merged.groupby("subject").agg(
        f1=("event_f1", "mean"),
        dur=("mask_duration_ms", "first"),
        n=("filename", "count"),
    )
    fig.add_trace(
        go.Scatter(
            x=subj_summary["dur"],
            y=subj_summary["f1"],
            mode="markers+text",
            marker=dict(
                size=subj_summary["n"].clip(upper=30) + 5,
                color=subj_summary["f1"],
                colorscale="RdYlGn",
                opacity=0.7,
            ),
            text=subj_summary.index,
            textposition="top center",
            textfont=dict(size=8),
            name="subjects",
        ),
        row=2,
        col=2,
    )
    fig.add_vline(x=280, line_dash="dash", line_color="blue", row=2, col=2)

    fig.update_layout(
        title=(
            "GT Label Quality Analysis<br>"
            "<sub>Key insight: 200ms masks (50 samples) are below the postprocessing "
            "minimum event threshold (75 samples = 300ms)</sub>"
        ),
        height=800,
        width=1100,
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mask duration (samples)", row=1, col=1)
    fig.update_xaxes(title_text="GT events", row=1, col=2)
    fig.update_yaxes(title_text="Pred events", row=1, col=2)
    fig.update_xaxes(title_text="Sample recall", row=2, col=1)
    fig.update_yaxes(title_text="Event F1", row=2, col=1)
    fig.update_xaxes(title_text="Mask duration (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Mean event F1", row=2, col=2)
    return fig


# ---------------------------------------------------------------------------
# Model loading + inference (optional, only for --with-model)
# ---------------------------------------------------------------------------


def _load_model():
    """Load the training pipeline U-Net."""
    import torch

    DEPLOYED_DIR = Path("/path/to/Research/training-pipeline")
    sys.path.insert(0, str(DEPLOYED_DIR))
    from train_5b0d152 import StimArtifactUNet

    sd = torch.load(
        str(DEPLOYED_DIR / "best_model.pt"), map_location="mps", weights_only=False
    )
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = StimArtifactUNet()
    model.load_state_dict(sd)
    model.to("mps")
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def _predict_file(model, row: pd.Series) -> np.ndarray:
    """Run inference on a single file."""
    import torch

    sys.path.insert(0, str(Path("/path/to/Research/training-pipeline")))
    from prepare import extract_conditioning_vector, WINDOW_SAMPLES, STRIDE_SAMPLES

    raw = read_dat(str(row["file_path"]))
    data = to_microvolts(raw).astype(np.float32)
    median = np.median(data, axis=1, keepdims=True)
    q25 = np.percentile(data, 25, axis=1, keepdims=True)
    q75 = np.percentile(data, 75, axis=1, keepdims=True)
    iqr = q75 - q25 + 1e-8

    n_samples = data.shape[1]
    ws = WINDOW_SAMPLES
    stride = STRIDE_SAMPLES

    starts = list(range(0, n_samples - ws + 1, stride))
    if not starts:
        return np.zeros(n_samples, dtype=np.float32)

    cond = extract_conditioning_vector(row)
    cond_t = torch.from_numpy(cond).unsqueeze(0).to("mps")

    pred_acc = np.zeros(n_samples, dtype=np.float32)
    count = np.zeros(n_samples, dtype=np.float32)

    batch_size = 64
    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            batch_starts = starts[bi : bi + batch_size]
            windows = np.stack(
                [(data[:, s : s + ws] - median) / iqr for s in batch_starts]
            )
            sig_t = torch.from_numpy(windows).to("mps")
            cond_batch = cond_t.expand(len(batch_starts), -1)
            logits, _ = model(sig_t, cond_batch)
            probs = torch.sigmoid(logits).float().cpu().numpy().squeeze(1)
            for i, s in enumerate(batch_starts):
                pred_acc[s : s + ws] += probs[i]
                count[s : s + ws] += 1.0

    return np.divide(pred_acc, count, out=np.zeros_like(pred_acc), where=count > 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="BWH failure mode analysis")
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Include model predictions on trace figures",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=3,
        help="Number of trace examples per failure mode",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    eval_df, bwh_cat, lgs_cat = load_data()
    merged = merge_eval_catalog(eval_df, bwh_cat)

    print("Classifying failures...")
    subj_stats = classify_failures(merged)

    # Print summary
    print(f"\n{'='*70}")
    print("FAILURE MODE SUMMARY")
    print(f"{'='*70}")
    for mode in ["GOOD", "GT_MISALIGN", "ZERO_PRED", "MISSED", "FP_ONLY", "MIXED"]:
        subset = subj_stats[subj_stats["failure_mode"] == mode]
        if len(subset) == 0:
            continue
        durs = sorted(subset["mask_dur"].unique())
        print(f"\n  {mode} ({len(subset)} subjects):" f" mask_durs={durs}")
        for subj, row in subset.iterrows():
            print(
                f"    {subj}: F1={row['event_f1_mean']:.3f} "
                f"n={row['n_files']} "
                f"mask={row['mask_dur']:.0f}ms "
                f"sP={row['sample_p_mean']:.3f} sR={row['sample_r_mean']:.3f}"
            )

    # Key finding
    good_durs = subj_stats[subj_stats["failure_mode"] == "GOOD"]["mask_dur"]
    bad_durs = subj_stats[subj_stats["event_f1_mean"] < 0.5]["mask_dur"]
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    print(f"  Passing subjects mask durations: {sorted(good_durs.unique())}")
    print(f"  Failing subjects mask durations: {sorted(bad_durs.unique())}")
    print(f"  LGS training min mask: {lgs_cat['mask_duration_ms'].min():.0f}ms")
    print(
        f"  Postprocessing min event: {MIN_ARTIFACT_SAMPLES} samples = {MIN_ARTIFACT_SAMPLES/SR*1000:.0f}ms"
    )
    n_below_lgs = (
        bwh_cat["mask_duration_ms"] < lgs_cat["mask_duration_ms"].min()
    ).sum()
    n_below_postproc = (
        bwh_cat["mask_duration_ms"] < MIN_ARTIFACT_SAMPLES / SR * 1000
    ).sum()
    print(f"  BWH files below LGS training min: {n_below_lgs:,} / {len(bwh_cat):,}")
    print(f"  BWH files below postproc min: {n_below_postproc:,} / {len(bwh_cat):,}")
    print(f"{'='*70}")

    # Generate figures
    print("\nGenerating figures...")

    fig1 = fig_failure_table(subj_stats)
    fig1.write_html(OUT_DIR / "failure_table.html")
    print(f"  {OUT_DIR / 'failure_table.html'}")

    fig2 = fig_mask_duration_comparison(subj_stats, bwh_cat, lgs_cat)
    fig2.write_html(OUT_DIR / "mask_duration_comparison.html")
    print(f"  {OUT_DIR / 'mask_duration_comparison.html'}")

    fig3 = fig_f1_vs_duration(merged)
    fig3.write_html(OUT_DIR / "f1_vs_duration.html")
    print(f"  {OUT_DIR / 'f1_vs_duration.html'}")

    fig4 = fig_subject_f1_bars(subj_stats)
    fig4.write_html(OUT_DIR / "subject_f1_bars.html")
    print(f"  {OUT_DIR / 'subject_f1_bars.html'}")

    fig5 = fig_gt_label_analysis(merged)
    fig5.write_html(OUT_DIR / "gt_label_analysis.html")
    print(f"  {OUT_DIR / 'gt_label_analysis.html'}")

    # Trace figures
    print(f"\nGenerating trace figures (n_per_mode={args.n_traces})...")
    trace_figs = generate_trace_figures(
        merged,
        subj_stats,
        n_per_mode=args.n_traces,
        with_model=args.with_model,
    )
    for mode, fn, fig in trace_figs:
        safe_fn = fn[:40]
        path = OUT_DIR / f"trace_{mode}_{safe_fn}.html"
        fig.write_html(path)
        print(f"  {path}")

    print(f"\nDone. All figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
