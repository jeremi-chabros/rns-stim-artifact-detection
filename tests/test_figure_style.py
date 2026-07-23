"""Tests for the shared figure style module (src/figures/style.py)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "figures"))
from style import apply_style, PALETTE, panel_label, forest_xzoom  # noqa: E402

REQUIRED_KEYS = (
    "m0",
    "m1",
    "m2",
    "m3",
    "m4",
    "lgs",
    "bwh",
    "raw",
    "cal",
    "gt",
    "signal",
    "accent",
)


def test_palette_has_required_keys_as_hex():
    for k in REQUIRED_KEYS:
        assert k in PALETTE, f"missing palette key {k}"
        v = PALETTE[k]
        assert v.startswith("#") and len(v) == 7, f"{k}={v} not #rrggbb"


def test_apply_style_embeds_fonts_and_dpi():
    apply_style()
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["font.family"] == ["serif"]


def test_forest_xzoom_sets_xlim():
    apply_style()
    fig, ax = plt.subplots()
    forest_xzoom(ax, 0.85, 1.0)
    assert ax.get_xlim() == (0.85, 1.0)
    plt.close(fig)


def test_panel_label_adds_letter():
    fig, ax = plt.subplots()
    panel_label(ax, "a")
    assert any("(a)" in t.get_text() for t in ax.texts)
    plt.close(fig)
