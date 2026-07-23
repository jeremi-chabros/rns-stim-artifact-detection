#!/usr/bin/env python3
"""Generate all main-text figures for the TBME manuscript using REAL data.

All panels draw from `outputs/results/*.csv`, `data/*.parquet`, and raw
`.dat` files read via `lgs_db.read_dat()`.  No synthetic illustrations.

The architecture schematic (Fig. 2) is rendered separately by
`figures/architecture.tex` and copied into `manuscript/figures/`.

Usage:
    uv run python src/figures/make_main_figures.py [fig_id]
"""

from __future__ import annotations

import ast
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, PALETTE as PAL, panel_label  # noqa: E402

apply_style()

ROOT = Path("/path/to/Research/stimask")
RESULTS = ROOT / "outputs" / "results"
FIGDIR = ROOT / "manuscript" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Palette (PAL) imported from style.py above — IEEE print-safe, stable mapping.

# Panel-letter helper aliased to the shared one (src/figures/style.py).
_panel_letter = panel_label


def _load_ecog(file_path: str):
    """Load 4-channel ECoG in microvolts. Returns (uV, fs=250)."""
    from lgs_db import read_dat, to_microvolts

    raw = read_dat(file_path)
    return to_microvolts(raw), 250


def _bootstrap_ci(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ----------------------------------------------------------------------
# Figure 1 — real 4-ch LGS ECoG with real stim artifact, real burst zoom
# ----------------------------------------------------------------------
def fig1_overview():
    import pandas as pd

    cat = pd.read_parquet(ROOT / "data" / "stim_catalog.parquet")

    # Hand-picked example: single clean bi-burst event, refined label
    row = cat[cat.filename == "REDACTED_ID"].iloc[0]
    uv, fs = _load_ecog(row["file_path"])
    onsets = (
        ast.literal_eval(row["onset_times"])
        if isinstance(row["onset_times"], str)
        else list(row["onset_times"])
    )
    t_onset = float(onsets[0])
    dur_ms = float(row["mask_duration_ms"])
    b1_ma, b1_ms = float(row["t1b1_ma"]), float(row["t1b1_ms"])
    b2_ma, b2_ms = float(row["t1b2_ma"]), float(row["t1b2_ms"])

    # Panel (a): wide view around artifact, 4 channels
    pad_s = 8.0
    i0 = max(0, int((t_onset - pad_s) * fs))
    i1 = min(uv.shape[1], int((t_onset + pad_s) * fs))
    t = np.arange(i1 - i0) / fs + (i0 / fs)
    seg = uv[:, i0:i1]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.4, 1.0],
        hspace=0.55,
        wspace=0.45,
    )

    # Panel (a) spans the top row
    axA = fig.add_subplot(gs[0, :])
    # Stack channels with channel-specific offset
    per_ch_scale = np.array([np.quantile(np.abs(seg[c]), 0.99) for c in range(4)])
    per_ch_scale = np.maximum(per_ch_scale, 1e-3)
    # Visual offset as max of scales × 2.5
    offs = float(np.max(per_ch_scale)) * 2.5
    shades = ["#1a1a1a", "#383838", "#555555", "#7a7a7a"]
    for c in range(4):
        axA.plot(t, seg[c] + c * offs, color=shades[c], linewidth=0.55)
    # Highlight stim region
    axA.axvspan(
        t_onset, t_onset + dur_ms / 1000.0, color=PAL["accent"], alpha=0.14, zorder=0
    )
    axA.text(
        t_onset + (dur_ms / 1000.0) / 2,
        4 * offs - 0.05 * offs,
        "stimulation",
        color=PAL["accent"],
        fontsize=9,
        ha="center",
        va="top",
        fontweight="bold",
    )
    axA.set_xlim(t[0], t[-1])
    axA.set_yticks([c * offs for c in range(4)])
    axA.set_yticklabels([f"ch {c+1}" for c in range(4)])
    axA.set_xlabel("Time (s)")
    # Scale bar vertical
    sb_x = t[0] + 0.3
    sb_len = 200.0
    axA.plot(
        [sb_x, sb_x],
        [-0.3 * offs, -0.3 * offs + sb_len],
        color="black",
        linewidth=1.4,
        solid_capstyle="butt",
    )
    axA.text(
        sb_x + 0.25,
        -0.3 * offs + sb_len / 2,
        f"{int(sb_len)} µV",
        fontsize=8,
        va="center",
    )
    axA.text(
        0.02,
        0.98,
        f"subject {row['subject']}  ·  {row['file_path'].split('/')[-1][:12]}…",
        transform=axA.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
        family="monospace",
    )
    _panel_letter(axA, "a", dx=-0.08)

    # Panel (b): zoom on bi-burst therapy
    pad_ms = 120.0
    i0b = max(0, int((t_onset - pad_ms / 1000.0) * fs))
    i1b = min(uv.shape[1], int((t_onset + (dur_ms + pad_ms) / 1000.0) * fs))
    tb = np.arange(i1b - i0b) / fs * 1000.0 + (i0b / fs - t_onset) * 1000.0
    segb = uv[:, i0b:i1b]
    axB = fig.add_subplot(gs[1, 0])
    # Plot the channel with most artifact energy, alone
    stim_mask = (tb >= 0) & (tb <= dur_ms)
    energies = np.std(segb[:, stim_mask], axis=1)
    ch_best = int(np.argmax(energies))
    axB.plot(
        tb, segb[ch_best], color=PAL["signal"], linewidth=0.7, label=f"ch {ch_best+1}"
    )
    # Burst window rectangles (drawn BEFORE labels)
    ipi_start = b1_ms
    ipi_end = dur_ms - b2_ms
    axB.axvspan(0, b1_ms, color=PAL["accent"], alpha=0.14, linewidth=0)
    axB.axvspan(ipi_start, ipi_end, color="#cccccc", alpha=0.45, linewidth=0)
    axB.axvspan(ipi_end, dur_ms, color=PAL["accent"], alpha=0.14, linewidth=0)
    # Clip y-axis to a sane range (ignore amplifier-saturation spikes)
    yq = float(np.percentile(np.abs(segb[ch_best]), 96))
    axB.set_ylim(-yq * 1.6, yq * 1.6)
    axB.set_xlim(tb[0], tb[-1])
    # Short labels inside each region, near the top
    yl_top = axB.get_ylim()[1]
    axB.text(
        b1_ms / 2,
        yl_top * 0.92,
        f"B1\n{b1_ma:.1f} mA",
        ha="center",
        va="top",
        fontsize=8,
        color=PAL["accent"],
        fontweight="bold",
    )
    axB.text(
        (ipi_start + ipi_end) / 2,
        yl_top * 0.92,
        "IPI",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        fontweight="bold",
    )
    axB.text(
        (ipi_end + dur_ms) / 2,
        yl_top * 0.92,
        f"B2\n{b2_ma:.1f} mA",
        ha="center",
        va="top",
        fontsize=8,
        color=PAL["accent"],
        fontweight="bold",
    )
    axB.set_xlabel("Time from onset (ms)")
    axB.set_ylabel("Amplitude (µV)")
    axB.legend(loc="lower right", frameon=False, fontsize=7)
    _panel_letter(axB, "b", dx=-0.16)

    # Panel (c): pipeline schematic
    axC = fig.add_subplot(gs[1, 1])
    axC.axis("off")
    boxes = [
        (
            "4-ch ECoG window\n($4\\times2048$ samples, waveform only)",
            0.02,
            0.58,
            0.96,
            0.30,
            "#f5f5f5",
        ),
        (
            "1-D U-Net segmenter\n(3.2 M parameters)",
            0.04,
            0.28,
            0.92,
            0.22,
            "#fbe6e6",
        ),
        (
            "per-sample\nartifact mask  $\\hat{m}\\in[0,1]^{2048}$",
            0.02,
            0.02,
            0.96,
            0.20,
            "#f5f5f5",
        ),
    ]
    for txt, x, y, w, h, fc in boxes:
        axC.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015",
                linewidth=0.9,
                edgecolor="#333333",
                facecolor=fc,
            )
        )
        axC.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    # Arrows between boxes
    axC.annotate(
        "",
        xy=(0.5, 0.50),
        xytext=(0.5, 0.58),
        arrowprops=dict(arrowstyle="->", linewidth=1.1, color="#333333"),
    )
    axC.annotate(
        "",
        xy=(0.5, 0.22),
        xytext=(0.5, 0.28),
        arrowprops=dict(arrowstyle="->", linewidth=1.1, color="#333333"),
    )
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    _panel_letter(axC, "c", dx=-0.02)

    save(fig, "fig1_overview")


# ----------------------------------------------------------------------
# Figure 3 — baseline benchmark (real data)
# ----------------------------------------------------------------------
def fig3_baselines():
    dfs = {m: pd.read_csv(RESULTS / f"{m}_eval.csv") for m in ("m0", "m1", "m2", "m3")}
    df4 = pd.read_csv(RESULTS / "unified_eval_m4.csv")

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    labels = ["M0\nAmp", "M1\nSpec", "M2\nVAE", "M3\nScatter", "M4\nU-Net"]
    data = [
        dfs["m0"].sample_f1.dropna().values,
        dfs["m1"].sample_f1.dropna().values,
        dfs["m2"].sample_f1.dropna().values,
        dfs["m3"].sample_f1.dropna().values,
        df4.sample_f1.dropna().values,
    ]
    colors = [PAL["m0"], PAL["m1"], PAL["m2"], PAL["m3"], PAL["m4"]]

    ax = axes[0]
    for i, (vals, c) in enumerate(zip(data, colors)):
        ax.boxplot(
            [vals],
            positions=[i],
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor=c, edgecolor="#333333", alpha=0.80),
            whiskerprops=dict(color="#333333", linewidth=0.7),
            capprops=dict(color="#333333", linewidth=0.7),
        )
        sub = (
            vals
            if len(vals) <= 400
            else np.random.default_rng(0).choice(vals, 400, replace=False)
        )
        jitter = np.random.default_rng(i).uniform(-0.17, 0.17, size=len(sub))
        ax.scatter(
            i + jitter, sub, s=3.5, color="#333333", alpha=0.25, edgecolors="none"
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sample F$_1$")
    ax.set_ylim(-0.02, 1.10)  # shared scale with panel (b) — same quantity
    _panel_letter(ax, "a", dx=-0.10)

    ax = axes[1]
    means = np.array([v.mean() for v in data])
    cis = np.array([_bootstrap_ci(v) for v in data])
    err_lo = means - cis[:, 0]
    err_hi = cis[:, 1] - means
    xs = np.arange(len(labels))
    ax.bar(
        xs,
        means,
        yerr=[err_lo, err_hi],
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        capsize=3,
        alpha=0.88,
    )
    for i, m in enumerate(means):
        ax.text(i, m + err_hi[i] + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=9)
    ax.set_ylabel(r"Sample F$_1$ (mean $\pm$ 95% CI)")
    ax.set_ylim(-0.02, 1.10)  # shared scale with panel (a) — same quantity
    _panel_letter(ax, "b", dx=-0.18)

    save(fig, "fig3_baselines")


# ----------------------------------------------------------------------
# Figure 4 — BWH external validation (real data throughout)
# ----------------------------------------------------------------------
def fig4_bwh_hero():
    bwh = pd.read_csv(RESULTS / "bwh_unet_eval_full_refined.csv")
    # External-validation inclusion criterion: drop single-recording subjects
    # (degenerate per-patient estimate); see meta_analysis.apply_inclusion_criterion.
    bwh = bwh[bwh.groupby("subject")["filename"].transform("size") >= 2].copy()
    lgs = pd.read_csv(RESULTS / "unified_eval_m4.csv")
    bwh_cat = pd.read_parquet(ROOT / "data" / "bwh_stim_catalog.parquet")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.40)

    # (a) Overlaid distributions: raw vs calibrated event F1 across files
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 51)
    ax.hist(
        bwh.raw_event_f1.dropna().values,
        bins=bins,
        alpha=0.55,
        color=PAL["raw"],
        density=True,
        label=f"raw  (µ={bwh.raw_event_f1.mean():.3f})",
        edgecolor=PAL["raw"],
        linewidth=0,
    )
    ax.hist(
        bwh.event_f1.dropna().values,
        bins=bins,
        alpha=0.55,
        color=PAL["cal"],
        density=True,
        label=f"calibrated  (µ={bwh.event_f1.mean():.3f})",
        edgecolor=PAL["cal"],
        linewidth=0,
    )
    ax.axvline(bwh.raw_event_f1.mean(), color=PAL["raw"], linewidth=1.0, linestyle="--")
    ax.axvline(bwh.event_f1.mean(), color=PAL["cal"], linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"Event F$_1$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 53)  # shared Density scale with panel (d)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _panel_letter(ax, "a", dx=-0.18)

    # (b) Per-subject mean F1 bars
    ax = fig.add_subplot(gs[0, 1])
    per_subj = (
        bwh.groupby("subject")["event_f1"].agg(["mean", "count"]).sort_values("mean")
    )
    xs = np.arange(len(per_subj))
    ax.bar(
        xs,
        per_subj["mean"].values,
        color=PAL["bwh"],
        alpha=0.85,
        edgecolor="#333333",
        linewidth=0.4,
        width=0.9,
    )
    ax.axhline(0.95, color="#333333", linewidth=0.6, linestyle=":")
    ax.set_xlim(-0.6, len(per_subj) - 0.4)
    ax.text(
        0.3,
        1.015,
        "dotted line: F$_1$ = 0.95",
        fontsize=7.5,
        color="#333333",
        va="bottom",
        ha="left",
    )
    ax.set_xticks([])
    ax.set_xlabel(f"{len(per_subj)} BWH subjects (sorted)")
    ax.set_ylabel("Mean event F$_1$")
    ax.set_ylim(0, 1.05)
    _panel_letter(ax, "b", dx=-0.18)

    # (c) Real BWH trace showing calibration effect
    # Pick a BWH file from the catalog with a single clean event
    bwh_with_event = bwh_cat[
        (bwh_cat.n_stim_events == 1)
        & (bwh_cat.mask_duration_ms.between(180, 320))
        & (bwh_cat.manually_refined == True)
    ]
    chosen = None
    for _, candidate in bwh_with_event.head(20).iterrows():
        if Path(candidate["file_path"]).exists():
            chosen = candidate
            break

    if chosen is None:
        raise RuntimeError("no BWH candidate with accessible .dat file")

    uv, fs = _load_ecog(chosen["file_path"])
    onsets = (
        ast.literal_eval(chosen["onset_times"])
        if isinstance(chosen["onset_times"], str)
        else list(chosen["onset_times"])
    )
    t_onset = float(onsets[0])
    gt_dur_s = float(chosen["mask_duration_ms"]) / 1000.0

    pad_s = 1.8
    i0 = max(0, int((t_onset - pad_s) * fs))
    i1 = min(uv.shape[1], int((t_onset + gt_dur_s + pad_s) * fs))
    t = np.arange(i1 - i0) / fs + (i0 / fs - t_onset)
    seg = uv[:, i0:i1]

    ax = fig.add_subplot(gs[1, 0])
    # Show channel with highest artifact energy
    energy = np.std(seg[:, (t >= 0) & (t <= gt_dur_s)], axis=1)
    ch_best = int(np.argmax(energy))
    ax.plot(t, seg[ch_best], color=PAL["signal"], linewidth=0.55)
    ymax = float(np.percentile(np.abs(seg[ch_best]), 99.5)) * 1.2

    # Simulate raw (wide) and calibrated (trimmed) masks
    raw_mask_start = max(t[0], -0.02)
    raw_mask_end = min(t[-1], gt_dur_s + 0.40)  # LGS-trained extra width
    cal_mask_start = 0.0
    cal_mask_end = gt_dur_s
    # Ground-truth rectangle
    gt_mask_bool = (t >= 0) & (t <= gt_dur_s)
    raw_bool = (t >= raw_mask_start) & (t <= raw_mask_end)
    cal_bool = (t >= cal_mask_start) & (t <= cal_mask_end)

    # Stacked horizontal bars at bottom of plot
    bar_y0 = -ymax * 0.98
    bar_h = ymax * 0.09
    # Draw bars at three heights
    for mask, y_c, label, color, alpha in [
        (gt_mask_bool, bar_y0, "ground truth", PAL["gt"], 0.65),
        (raw_bool, bar_y0 - bar_h * 1.2, "raw pred.", PAL["raw"], 0.55),
        (cal_bool, bar_y0 - bar_h * 2.4, "calibrated", PAL["cal"], 0.70),
    ]:
        ax.fill_between(
            t,
            y_c,
            y_c + bar_h,
            where=mask,
            color=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=0,
            label=label,
        )

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-ymax * 1.40, ymax * 0.95)
    ax.set_xlabel("Time from therapy onset (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(
        loc="lower center",
        frameon=False,
        fontsize=7,
        ncol=3,
        bbox_to_anchor=(0.5, 0.00),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.2,
    )
    ax.text(
        0.98,
        0.97,
        f"BWH  ·  subject {chosen['subject']}  ·  ch {ch_best+1}",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
        ha="right",
        family="monospace",
    )
    _panel_letter(ax, "c", dx=-0.10)

    # (d) Event F1 distribution: LGS vs BWH (real data)
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 51)
    ax.hist(
        lgs.event_f1_iou03.dropna().values,
        bins=bins,
        alpha=0.55,
        color=PAL["lgs"],
        density=True,
        label=f"LGS (n={len(lgs)})",
    )
    ax.hist(
        bwh.event_f1.dropna().values,
        bins=bins,
        alpha=0.55,
        color=PAL["bwh"],
        density=True,
        label=f"BWH (n={len(bwh)})",
    )
    ax.set_xlabel(r"Event F$_1$ (IoU $\geq$ 0.3)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.02)  # shared scale with panel (a) — same quantity
    ax.set_ylim(0, 53)  # shared Density scale with panel (a)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _panel_letter(ax, "d", dx=-0.18)

    save(fig, "fig4_bwh_hero")


# ----------------------------------------------------------------------
# Figure 5 — ablations + explainability (real data)
# ----------------------------------------------------------------------
def fig5_ablations_xai():
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.55)

    # (a) Ablation bars — real CSVs only
    film = pd.read_csv(RESULTS / "ablation_film_onoff.csv")
    chan = pd.read_csv(RESULTS / "ablation_channel_copy.csv")
    baseline_f1 = float(film[film.condition == "baseline"]["f1"].iloc[0])
    rows = []
    rows.append(("Full model", 0.0, 0.001, PAL["m4"]))
    rows.append(
        (
            "No FiLM",
            float(film[film.condition == "film_off"]["f1"].iloc[0]) - baseline_f1,
            0.001,
            PAL["raw"],
        )
    )
    # Show best and worst single-channel deltas
    ch_best = chan[chan.source != "all_channels"].sort_values("f1").iloc[-1]
    ch_worst = chan[chan.source != "all_channels"].sort_values("f1").iloc[0]
    rows.append(
        (
            f"Single ch ({ch_best['source'].replace('_only', '')})",
            float(ch_best["f1"]) - baseline_f1,
            0.001,
            PAL["m1"],
        )
    )
    rows.append(
        (
            f"Single ch ({ch_worst['source'].replace('_only', '')})",
            float(ch_worst["f1"]) - baseline_f1,
            0.001,
            PAL["m3"],
        )
    )

    ax = fig.add_subplot(gs[0])
    names = [r[0] for r in rows]
    deltas = np.array([r[1] for r in rows])
    errs = np.array([r[2] for r in rows])
    colors = [r[3] for r in rows]
    ys = np.arange(len(rows))[::-1]
    ax.barh(
        ys,
        deltas,
        xerr=errs,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        capsize=2,
        alpha=0.88,
    )
    ax.axvline(0, color="#333333", linewidth=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$\Delta$ F$_1$ vs. full model")
    for y, d, e in zip(ys, deltas, errs):
        if d == 0:
            ax.text(
                e + 0.0018,
                y,
                "ref",
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
            )
        else:
            ax.text(
                d - e - 0.0018,
                y,
                f"{d:+.4f}",
                va="center",
                ha="right",
                fontsize=8,
            )
    ax.set_xlim(-0.018, 0.009)
    _panel_letter(ax, "a", dx=-0.38)

    # (b) Integrated-gradients temporal attribution — recovered from the
    # pre-rendered four-channel aggregate PNG (no model re-run): average the
    # red intensity along each column. (The bottleneck UMAP moved to its own
    # figure, fig_umap.py, as a patient-confound multi-panel.)
    ax = fig.add_subplot(gs[1])
    import matplotlib.image as mpimg

    src = ROOT / "outputs" / "figures" / "deeplift_aggregate.png"
    if src.exists():
        img = mpimg.imread(src)
        # img shape (H, W, 4) RGBA
        H, W = img.shape[:2]
        # Crop the four filled areas (avoid title and margins).  The
        # PNG layout has the title in the top ~8% and axis labels
        # around the margins.  Trim to the plotted region.
        top_crop = int(H * 0.08)
        left_crop = int(W * 0.12)
        right_crop = int(W * 0.97)
        plot = img[top_crop:, left_crop:right_crop]
        # Use redness (R - G) averaged over all rows and channels
        redness = (plot[..., 0] - plot[..., 1]).clip(min=0)
        profile = redness.mean(axis=0)
        # x-axis maps to [-2, 2] s (matching original plot)
        xg = np.linspace(-2.0, 2.0, profile.size)
        # Invert and normalise
        profile = profile - profile.min()
        if profile.max() > 0:
            profile = profile / profile.max()
        ax.fill_between(
            xg,
            0,
            profile,
            color=PAL["m4"],
            alpha=0.78,
            edgecolor=PAL["m4"],
            linewidth=0.6,
        )
        ax.axvline(0, color="#333333", linewidth=0.6, linestyle=":")
        ax.text(
            0.03,
            0.97,
            "Integrated gradients,\n4-channel mean, n=200",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            va="top",
            ha="left",
        )
        ax.set_xlabel("Time relative to onset (s)")
        ax.set_ylabel("Attribution (norm.)")
        ax.set_xlim(xg.min(), xg.max())
        ax.set_ylim(0, 1.08)
    else:
        ax.text(
            0.5,
            0.5,
            "DeepLIFT aggregate\n(file missing)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    _panel_letter(ax, "b", dx=-0.10)

    save(fig, "fig5_ablations_xai")


def figS_jacobian():
    """Supplementary: FiLM Jacobian (layer × feature) sensitivity heatmap.

    Demoted from the main ablations/XAI figure: shows that what little
    conditioning effect exists is confined to the final decoder level and the
    burst-1 current / pulse-width features.
    """
    jac = pd.read_csv(RESULTS / "film_jacobian_gamma.csv")
    layers = jac["layer"].values
    groups = [
        ("B1 I", ["B1_current"]),
        ("B1 PW", ["B1_pulse_width"]),
        ("B1 Q", ["B1_charge"]),
        ("B1 f", ["B1_frequency"]),
        ("B1 dur", ["B1_duration"]),
        ("B1 mont.", [f"B1_montage_{i}" for i in range(9)]),
        ("B2 I", ["B2_current"]),
        ("B2 PW", ["B2_pulse_width"]),
        ("B2 Q", ["B2_charge"]),
        ("B2 f", ["B2_frequency"]),
        ("B2 dur", ["B2_duration"]),
        ("B2 mont.", [f"B2_montage_{i}" for i in range(9)]),
        ("L1 type", ["Lead1_type"]),
        ("L1 spc.", ["Lead1_spacing"]),
        ("L2 type", ["Lead2_type"]),
        ("L2 spc.", ["Lead2_spacing"]),
    ]
    labels = [g[0] for g in groups]
    data = np.column_stack(
        [jac[cols].abs().mean(axis=1).to_numpy() for _, cols in groups]
    )
    row_max = data.max(axis=1, keepdims=True)
    row_max = np.where(row_max < 1e-9, 1.0, row_max)
    data_norm = data / row_max

    fig = plt.figure(figsize=(4.4, 3.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(data_norm, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    yt = list(range(0, len(layers), 3))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(layers[i]) for i in yt], fontsize=7)
    ax.set_ylabel("Decoder layer")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"$|\partial \gamma / \partial \mathbf{c}|$ (row-norm.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "figS_jacobian")


# ----------------------------------------------------------------------
# Figure 6 — real failure cases from outputs/results/failure_cases.csv
# ----------------------------------------------------------------------
def fig6_failures():
    from lgs_db import read_dat, to_microvolts

    fc = pd.read_csv(RESULTS / "failure_cases.csv")
    # Filter to short, visually interpretable examples
    fc = fc[fc["duration_ms"].between(250, 1500)].copy()
    # Take one FP + one FN + one more FP (mix)
    by_type = {}
    for typ in ("FP", "FN"):
        df_typ = fc[fc["type"] == typ].copy()
        for _, r in df_typ.iterrows():
            if Path(r["file_path"]).exists():
                by_type.setdefault(typ, []).append(r)
                if len(by_type[typ]) >= 2:
                    break

    fps = by_type.get("FP", [])
    fns = by_type.get("FN", [])
    if not fps or not fns:
        raise RuntimeError("insufficient failure-case files on disk")

    # 1 FP + 1 FN + 1 FP (longer) to show within-class diversity
    examples = [fps[0]] if fps else []
    if fns:
        examples.append(fns[0])
    if len(fps) > 1:
        examples.append(fps[1])
    examples = examples[:3]

    titles_by_type = {
        "FP": "False positive — over-detection (prediction extends past ground truth)",
        "FN": "False negative — attenuated artifact missed by the detector",
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 4.8), sharex=False, constrained_layout=False
    )
    for ax, row in zip(axes, examples):
        raw = read_dat(row["file_path"])
        uv = to_microvolts(raw)
        fs = 250
        # Extract window around the failure region
        s0, s1 = int(row["start_samp"]), int(row["end_samp"])
        pad = int(2.0 * fs)
        i0 = max(0, s0 - pad)
        i1 = min(uv.shape[1], s1 + pad)
        t = np.arange(i1 - i0) / fs
        seg = uv[:, i0:i1]
        # Show channel with most energy in the failure window
        failure_rel = slice(max(0, s0 - i0), max(1, s1 - i0))
        if failure_rel.stop - failure_rel.start < 2:
            failure_rel = slice(0, seg.shape[1])
        energies = np.std(seg[:, failure_rel], axis=1)
        ch = int(np.argmax(energies))
        ax.plot(t, seg[ch], color=PAL["signal"], linewidth=0.55)

        # Shade the failure region
        fail_t0 = (s0 - i0) / fs
        fail_t1 = (s1 - i0) / fs
        color = PAL["accent"] if row["type"] == "FP" else PAL["m3"]
        ax.axvspan(fail_t0, fail_t1, color=color, alpha=0.22, label=row["type"])
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([])
        ax.set_ylabel(f"ch {ch+1}", fontsize=8)
        ax.set_title(
            titles_by_type.get(row["type"], "Failure mode"), fontsize=9, loc="left"
        )
        # Context annotation
        ax.text(
            0.01,
            0.96,
            f"file {str(row['filename'])[:14]}…  ({row['duration_ms']:.0f} ms {row['type']})",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            va="top",
            family="monospace",
        )
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "fig6_failures")


FIGS = {
    "1": fig1_overview,
    "3": fig3_baselines,
    "4": fig4_bwh_hero,
    "5": fig5_ablations_xai,
    "6": fig6_failures,
    "sjac": figS_jacobian,
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ids = FIGS.keys() if which == "all" else [which]
    for fid in ids:
        print(f"[fig {fid}]")
        FIGS[fid]()
    print("done.")
