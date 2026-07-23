"""Random-effects meta-analysis of per-subject detector performance.

Treats each subject as a unit; pools per-subject effect sizes with the
DerSimonian-Laird estimator and reports heterogeneity (tau^2, I^2, Q) and a
95% prediction interval. Primary endpoint: onset recall on the logit scale
(clean binomial denominator). Moderators are joined from the BWH catalog.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

import sys
import matplotlib  # headless backend before pyplot is imported (incl. via style)

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from style import apply_style, PALETTE, forest_xzoom  # noqa: E402

apply_style()

DEFAULT_EVAL = Path("outputs/results/bwh_unet_eval_full_refined.csv")
DEFAULT_LGS_EVAL = Path("outputs/results/lgs_unet_eval_test.csv")
DEFAULT_CATALOG = Path("data/bwh_stim_catalog.parquet")
FIG_DIR_MANUSCRIPT = Path("manuscript/figures")
FIG_DIR_OUTPUTS = Path("outputs/figures")

# External-validation inclusion criterion (pre-specified): exclude subjects
# contributing a single recording, since a one-file subject yields a degenerate
# per-patient onset-recall estimate (the proportion is forced to 0 or 1).
MIN_RECORDINGS = 2


def apply_inclusion_criterion(
    df: pd.DataFrame, min_recordings: int = MIN_RECORDINGS
) -> pd.DataFrame:
    """Drop subjects with fewer than ``min_recordings`` files (see MIN_RECORDINGS)."""
    counts = df.groupby("subject")["filename"].transform("size")
    kept = df[counts >= min_recordings].copy()
    n_drop = df["subject"].nunique() - kept["subject"].nunique()
    if n_drop:
        dropped = sorted(set(df["subject"]) - set(kept["subject"]))
        print(f"[inclusion] dropped {n_drop} single-recording subject(s): {dropped}")
    return kept


def dersimonian_laird(yi: np.ndarray, vi: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling.

    Returns pooled mean, SE, 95% CI, tau^2, I^2 (%), Cochran's Q (+ p), and a
    95% prediction interval (Higgins-Thompson). Inputs are effect sizes ``yi``
    and their within-unit sampling variances ``vi`` (same scale, e.g. logit).
    """
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    k = len(yi)
    wi = 1.0 / vi
    y_fixed = float((wi * yi).sum() / wi.sum())
    Q = float((wi * (yi - y_fixed) ** 2).sum())
    df = k - 1
    C = float(wi.sum() - (wi**2).sum() / wi.sum())
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wi_star = 1.0 / (vi + tau2)
    mu = float((wi_star * yi).sum() / wi_star.sum())
    se = float(np.sqrt(1.0 / wi_star.sum()))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    q_p = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")
    ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se
    pi_df = k - 2
    tcrit = float(stats.t.ppf(0.975, pi_df)) if pi_df > 0 else float("inf")
    pi_se = float(np.sqrt(se**2 + tau2))
    pi_lo, pi_hi = mu - tcrit * pi_se, mu + tcrit * pi_se
    return {
        "k": k,
        "mu": mu,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "tau2": tau2,
        "I2": I2,
        "Q": Q,
        "Q_df": df,
        "Q_p": q_p,
        "pi_lo": pi_lo,
        "pi_hi": pi_hi,
    }


def subject_onset_recall_effects(df: pd.DataFrame, cc: float = 0.5) -> pd.DataFrame:
    """Per-subject onset recall as a logit effect size with Wald variance.

    recall = sum(onset_tp) / (sum(onset_tp) + sum(onset_fn)); ``cc`` is the
    continuity correction guarding 0/1 proportions.
    """
    g = df.groupby("subject")[["onset_tp", "onset_fn"]].sum().reset_index()
    tp = g["onset_tp"].to_numpy(float)
    fn = g["onset_fn"].to_numpy(float)
    recall = tp / np.maximum(tp + fn, 1.0)
    yi = logit((tp + cc) / (tp + fn + 2 * cc))
    vi = 1.0 / (tp + cc) + 1.0 / (fn + cc)
    g["recall"] = recall
    g["yi"] = yi
    g["vi"] = vi
    return g


def join_moderators(
    eff: pd.DataFrame, catalog_path: Path = DEFAULT_CATALOG
) -> pd.DataFrame:
    """Attach per-subject moderators (median therapy params, modal lead/side/site)."""
    cat = pd.read_parquet(catalog_path)
    # Coerce subject key to string in both DataFrames (catalog uses str, eval CSV uses int)
    cat = cat.copy()
    cat["subject"] = cat["subject"].astype(str)
    eff = eff.copy()
    eff["subject"] = eff["subject"].astype(str)
    num = cat.groupby("subject")[
        ["t1b1_ma", "t1b1_us", "t1b1_ms", "n_stim_events"]
    ].median()

    def _mode(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    cm = cat.groupby("subject")[["lead_1", "side", "site"]].agg(_mode)
    mods = num.join(cm).reset_index()
    return eff.merge(mods, on="subject", how="left")


def meta_regression(yi: np.ndarray, vi: np.ndarray, x: np.ndarray) -> dict:
    """Single-moderator weighted least squares on logit effects (fixed-effect weights)."""
    yi = np.asarray(yi, float)
    vi = np.asarray(vi, float)
    x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(1.0 / vi)
    XtW = X.T @ W
    A = XtW @ X
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise ValueError(
            "meta_regression: moderator x is constant or collinear with the "
            "intercept; the regression is not identified."
        )
    beta = np.linalg.solve(A, XtW @ yi)
    cov = np.linalg.inv(A)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    return {
        "intercept": beta[0],
        "slope": beta[1],
        "slope_se": se[1],
        "slope_p": float(p[1]),
    }


def subgroup_q_between(yi: np.ndarray, vi: np.ndarray, groups: np.ndarray) -> dict:
    """Fixed-effect Q-between for a categorical moderator.

    Partitions total heterogeneity (Q_total) into within-group and between-group
    components using fixed-effect weights ``wi = 1/vi``.  Only levels with ≥ 2
    subjects contribute to Q_within; the remainder are singletons.

    Args:
        yi: Array of effect sizes (e.g. logit-scale recall), shape (k,).
        vi: Array of within-unit sampling variances, shape (k,).
        groups: Array of group labels, shape (k,).

    Returns:
        Dict with keys:
            Q_between  – heterogeneity attributable to group differences.
            df         – degrees of freedom (# qualifying levels − 1).
            p_value    – chi-squared tail probability; nan if df < 1.
            n_levels   – total number of unique levels.
            n_qualifying – number of levels with ≥ 2 subjects.
            singleton_levels – list of levels excluded (< 2 subjects).
    """
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    groups = np.asarray(groups)
    wi = 1.0 / vi

    # overall fixed-effect mean and Q_total
    y_fixed = float((wi * yi).sum() / wi.sum())
    Q_total = float((wi * (yi - y_fixed) ** 2).sum())

    unique_levels = np.unique(groups)
    singleton_levels = []
    Q_within = 0.0
    qualifying = []
    for g in unique_levels:
        mask = groups == g
        if mask.sum() < 2:
            singleton_levels.append(g)
            continue
        qualifying.append(g)
        wi_g = wi[mask]
        yi_g = yi[mask]
        y_g = float((wi_g * yi_g).sum() / wi_g.sum())
        Q_within += float((wi_g * (yi_g - y_g) ** 2).sum())

    Q_between = max(0.0, Q_total - Q_within)
    df = max(0, len(qualifying) - 1)
    p_value = float(stats.chi2.sf(Q_between, df)) if df >= 1 else float("nan")
    return {
        "Q_between": Q_between,
        "df": df,
        "p_value": p_value,
        "n_levels": int(len(unique_levels)),
        "n_qualifying": int(len(qualifying)),
        "singleton_levels": list(singleton_levels),
    }


def moderator_sweep(
    eff: pd.DataFrame,
    catalog_path: Path = DEFAULT_CATALOG,
) -> pd.DataFrame:
    """Run a pre-specified moderator analysis on per-subject logit effects.

    Joins moderators via ``join_moderators``, then tests each moderator in
    two ways:

    * **Numeric** moderators ``["t1b1_ma", "t1b1_us", "t1b1_ms",
      "n_stim_events"]``: weighted-least-squares meta-regression
      (``meta_regression``), after median-imputing missing values.
      Reports slope, slope p-value, and the predicted change in
      back-transformed recall across the moderator's observed [P10, P90]
      range (``delta_recall``).

    * **Categorical** moderators ``["site", "lead_1", "side"]``: fixed-effect
      Q-between heterogeneity test (``subgroup_q_between``).  Reports
      Q_between, its p-value, the minimum DerSimonian-Laird pooled recall
      across qualifying levels, and the identity of that worst level.

    Args:
        eff: Per-subject logit effects, typically from
            ``subject_onset_recall_effects``.  Must contain columns
            ``subject``, ``yi``, ``vi``.
        catalog_path: Path to the BWH catalog parquet used by
            ``join_moderators``.

    Returns:
        DataFrame with one row per moderator and columns:

        * ``moderator`` – variable name.
        * ``type`` – ``"numeric"`` or ``"categorical"``.
        * ``stat`` – slope (numeric) or Q_between (categorical).
        * ``p_value`` – two-sided Wald p (numeric) or chi-squared p
          (categorical).
        * ``effect`` – delta_recall across P10-P90 (numeric) or min
          per-level pooled recall (categorical).
        * ``detail`` – human-readable summary string.
    """
    effm = join_moderators(eff, catalog_path)
    yi = effm["yi"].to_numpy(float)
    vi = effm["vi"].to_numpy(float)

    rows = []

    # ── Numeric moderators ────────────────────────────────────────────────────
    numeric_mods = ["t1b1_ma", "t1b1_us", "t1b1_ms", "n_stim_events"]
    for mod in numeric_mods:
        col = effm[mod].copy()
        median_val = col.median()
        col = col.fillna(median_val)
        x = col.to_numpy(float)
        try:
            mr = meta_regression(yi, vi, x)
        except ValueError:
            rows.append(
                dict(
                    moderator=mod,
                    type="numeric",
                    stat=float("nan"),
                    p_value=float("nan"),
                    effect=float("nan"),
                    detail="constant moderator — regression not identified",
                )
            )
            continue
        p10, p90 = float(np.percentile(x, 10)), float(np.percentile(x, 90))
        pred_lo = mr["intercept"] + mr["slope"] * p10
        pred_hi = mr["intercept"] + mr["slope"] * p90
        delta = float(expit(pred_hi) - expit(pred_lo))
        rows.append(
            dict(
                moderator=mod,
                type="numeric",
                stat=float(mr["slope"]),
                p_value=float(mr["slope_p"]),
                effect=delta,
                detail=f"P10={p10:.3g}, P90={p90:.3g}, delta_recall={delta:+.4f}",
            )
        )

    # ── Categorical moderators ─────────────────────────────────────────────────
    cat_mods = ["site", "lead_1", "side"]
    for mod in cat_mods:
        col = effm[mod].astype(str)
        # drop rows where the value is 'nan' or truly missing
        valid = col.notna() & (col != "nan") & (col != "")
        yi_c = yi[valid]
        vi_c = vi[valid]
        grp_c = col[valid].to_numpy()
        if len(yi_c) < 2:
            rows.append(
                dict(
                    moderator=mod,
                    type="categorical",
                    stat=float("nan"),
                    p_value=float("nan"),
                    effect=float("nan"),
                    detail="insufficient data",
                )
            )
            continue
        qb = subgroup_q_between(yi_c, vi_c, grp_c)
        # Per-level pooled recall via DL (qualifying levels only, ≥2 subjects)
        unique_levels = np.unique(grp_c)
        level_recalls = {}
        for g in unique_levels:
            mask = grp_c == g
            if mask.sum() < 2:
                continue
            dl_g = dersimonian_laird(yi_c[mask], vi_c[mask])
            level_recalls[g] = float(expit(dl_g["mu"]))
        if level_recalls:
            min_level = min(level_recalls, key=level_recalls.get)
            min_recall = level_recalls[min_level]
        else:
            min_level = None
            min_recall = float("nan")
        n_singletons = len(qb["singleton_levels"])
        detail = (
            f"n_levels={qb['n_levels']} "
            f"(qualifying={qb['n_qualifying']}, singletons={n_singletons}), "
            f"min_level={min_level!r} recall={min_recall:.4f}"
        )
        rows.append(
            dict(
                moderator=mod,
                type="categorical",
                stat=float(qb["Q_between"]),
                p_value=float(qb["p_value"]),
                effect=min_recall,
                detail=detail,
            )
        )

    return pd.DataFrame(rows)


def plot_forest(eff: pd.DataFrame, pooled: dict, stem: str = "fig8_bwh_forest") -> Path:
    """Forest plot of per-subject recall (back-transformed) with pooled line + PI.

    The x-axis is zoomed to the data band (spec rule): point estimates cluster
    near 1.0, so a 0-1 axis wastes ~80% of the panel. The few subjects whose CI
    extends below the zoom are clipped at the edge with an annotation.
    """
    import matplotlib.pyplot as plt

    e = eff.sort_values("yi").reset_index(drop=True)
    p = expit(e["yi"].to_numpy())
    sd = np.sqrt(e["vi"].to_numpy())
    lo = expit(e["yi"].to_numpy() - 1.96 * sd)
    hi = expit(e["yi"].to_numpy() + 1.96 * sd)
    y = np.arange(len(e))

    lo_bound = 0.85
    mu = expit(pooled["mu"])
    pi_lo, pi_hi = expit(pooled["pi_lo"]), expit(pooled["pi_hi"])

    fig, ax = plt.subplots(figsize=(7.0, min(8.0, max(4.2, 0.155 * len(e)))))
    ax.axvspan(
        pi_lo,
        pi_hi,
        color="#e9c46a",
        alpha=0.30,
        label="95% prediction interval",
        zorder=0,
    )
    ax.axvline(mu, color=PALETTE["accent"], lw=1.6, label=f"pooled {mu:.3f}", zorder=1)
    ax.hlines(y, np.clip(lo, lo_bound, None), hi, color="#6a6a6a", lw=1.0, zorder=2)
    ax.plot(p, y, "o", ms=3.6, color=PALETTE["bwh"], zorder=3)

    forest_xzoom(ax, lo_bound, 1.005)
    clipped = lo < lo_bound
    if clipped.any():
        ax.plot(
            np.full(int(clipped.sum()), lo_bound),
            y[clipped],
            marker="<",
            ms=5,
            ls="none",
            color="#6a6a6a",
            clip_on=False,
            zorder=4,
        )
        ax.text(
            0.02,
            0.55,
            "'<' : CI lower bound\noff-axis (few events)",
            transform=ax.transAxes,
            fontsize=7,
            color="#6a6a6a",
            va="center",
        )
    ax.set_ylim(-1, len(e))
    ax.set_xlabel("per-subject onset recall")
    ax.set_ylabel("subject (sorted by recall)")
    ax.set_title(
        f"BWH per-patient onset recall (k={pooled['k']}, $I^2$={pooled['I2']:.0f}%)"
    )
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    FIG_DIR_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    FIG_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR_MANUSCRIPT / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR_MANUSCRIPT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR_OUTPUTS / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def pool_cohorts(lgs_eval: pd.DataFrame, bwh_eval: pd.DataFrame) -> dict:
    """Pool per-subject onset recall across LGS + BWH with cohort as moderator.

    Returns overall random-effects pooled estimate, per-cohort pooled estimates,
    and the between-cohort subgroup heterogeneity test.
    """
    lgs = subject_onset_recall_effects(lgs_eval).assign(cohort="LGS")
    bwh = subject_onset_recall_effects(bwh_eval).assign(cohort="BWH")
    both = pd.concat([lgs, bwh], ignore_index=True)
    pooled = dersimonian_laird(both["yi"].to_numpy(), both["vi"].to_numpy())
    by_cohort = {}
    for name, sub in both.groupby("cohort"):
        d = dersimonian_laird(sub["yi"].to_numpy(), sub["vi"].to_numpy())
        by_cohort[name] = {
            "k": d["k"],
            "recall": float(expit(d["mu"])),
            "ci_lo": float(expit(d["ci_lo"])),
            "ci_hi": float(expit(d["ci_hi"])),
        }
    between = subgroup_q_between(
        both["yi"].to_numpy(), both["vi"].to_numpy(), both["cohort"].to_numpy()
    )
    return {
        "pooled": pooled,
        "by_cohort": by_cohort,
        "between": between,
        "effects": both,
    }


def plot_cross_cohort_forest(
    both: pd.DataFrame,
    pooled: dict,
    by_cohort: dict,
    stem: str = "fig10_cross_cohort_forest",
) -> Path:
    """Forest of per-subject onset recall coloured by cohort, with per-cohort
    and overall pooled estimates + 95% prediction interval band."""
    import matplotlib.pyplot as plt

    colors = {"LGS": PALETTE["lgs"], "BWH": PALETTE["bwh"]}
    e = both.sort_values(["cohort", "yi"]).reset_index(drop=True)
    p = expit(e["yi"].to_numpy())
    sd = np.sqrt(e["vi"].to_numpy())
    lo = expit(e["yi"].to_numpy() - 1.96 * sd)
    hi = expit(e["yi"].to_numpy() + 1.96 * sd)
    y = np.arange(len(e))

    lo_bound = 0.85
    mu = expit(pooled["mu"])
    pi_lo, pi_hi = expit(pooled["pi_lo"]), expit(pooled["pi_hi"])

    fig, ax = plt.subplots(figsize=(7.0, min(8.0, max(4.2, 0.155 * len(e)))))
    ax.axvspan(
        pi_lo, pi_hi, color="#e9c46a", alpha=0.25, label="overall 95% PI", zorder=0
    )
    ax.axvline(mu, color="#222222", lw=1.6, label=f"overall pooled {mu:.3f}", zorder=1)
    for coh in e["cohort"].unique():
        m = (e["cohort"] == coh).to_numpy()
        ax.hlines(
            y[m],
            np.clip(lo[m], lo_bound, None),
            hi[m],
            color=colors.get(coh, "#444"),
            lw=1.0,
            alpha=0.75,
            zorder=2,
        )
        ax.plot(
            p[m],
            y[m],
            "o",
            ms=3.6,
            color=colors.get(coh, "#444"),
            label=f"{coh} (k={by_cohort[coh]['k']}, pooled {by_cohort[coh]['recall']:.3f})",
            zorder=3,
        )
    forest_xzoom(ax, lo_bound, 1.005)
    clipped = lo < lo_bound
    if clipped.any():
        ax.plot(
            np.full(int(clipped.sum()), lo_bound),
            y[clipped],
            marker="<",
            ms=5,
            ls="none",
            color="#6a6a6a",
            clip_on=False,
            zorder=4,
        )
        ax.text(
            0.02,
            0.55,
            "'<' : CI lower bound\noff-axis (few events)",
            transform=ax.transAxes,
            fontsize=7,
            color="#6a6a6a",
            va="center",
        )
    ax.set_ylim(-1, len(e))
    ax.set_xlabel("per-subject onset recall")
    ax.set_ylabel("subject (sorted within cohort)")
    ax.set_title("Cross-cohort per-patient onset recall (LGS vs BWH)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    FIG_DIR_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    FIG_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR_MANUSCRIPT / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR_MANUSCRIPT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR_OUTPUTS / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def smoke_test() -> None:
    """End-to-end on the real BWH CSV if present."""
    if not DEFAULT_EVAL.exists():
        print(f"[smoke] {DEFAULT_EVAL} not found; skipping.")
        return
    df = apply_inclusion_criterion(pd.read_csv(DEFAULT_EVAL))
    eff = subject_onset_recall_effects(df)
    pooled = dersimonian_laird(eff["yi"].to_numpy(), eff["vi"].to_numpy())
    print(
        f"[smoke] pooled onset recall = {expit(pooled['mu']):.4f} "
        f"[{expit(pooled['ci_lo']):.4f}, {expit(pooled['ci_hi']):.4f}], "
        f"I²={pooled['I2']:.1f}%, tau²={pooled['tau2']:.3f}"
    )
    print(
        f"[smoke] 95% prediction interval = "
        f"[{expit(pooled['pi_lo']):.4f}, {expit(pooled['pi_hi']):.4f}]"
    )
    if DEFAULT_CATALOG.exists():
        effm = join_moderators(eff)
        mr = meta_regression(
            effm["yi"].to_numpy(),
            effm["vi"].to_numpy(),
            effm["t1b1_ms"].fillna(effm["t1b1_ms"].median()).to_numpy(),
        )
        print(
            f"[smoke] meta-regression on B1 duration: slope={mr['slope']:.4f} "
            f"(p={mr['slope_p']:.3g})"
        )
        print("\n[smoke] moderator sweep:")
        sweep = moderator_sweep(eff)
        with pd.option_context("display.max_colwidth", 80, "display.width", 120):
            print(sweep.to_string(index=False))
    print("\n[smoke] forest:", plot_forest(eff, pooled))


def make_figures() -> None:
    """Regenerate fig8 (BWH per-patient forest) and fig10 (cross-cohort forest)."""
    bwh = apply_inclusion_criterion(pd.read_csv(DEFAULT_EVAL))
    eff = subject_onset_recall_effects(bwh)
    pooled = dersimonian_laird(eff["yi"].to_numpy(), eff["vi"].to_numpy())
    print(
        f"[meta] BWH pooled onset recall = {expit(pooled['mu']):.4f} "
        f"(k={pooled['k']}, I2={pooled['I2']:.0f}%, "
        f"PI=[{expit(pooled['pi_lo']):.3f}, {expit(pooled['pi_hi']):.3f}])"
    )
    print("[fig8]", plot_forest(eff, pooled))
    if DEFAULT_LGS_EVAL.exists():
        lgs = pd.read_csv(DEFAULT_LGS_EVAL)
        cc = pool_cohorts(lgs, bwh)
        print(
            "[fig10]",
            plot_cross_cohort_forest(cc["effects"], cc["pooled"], cc["by_cohort"]),
        )
    else:
        print(f"[fig10] skipped — {DEFAULT_LGS_EVAL} not found")


if __name__ == "__main__":
    make_figures()
