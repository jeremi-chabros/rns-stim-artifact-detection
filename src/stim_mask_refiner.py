#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "lgs-db",
# ]
#
# [tool.uv.sources]
# lgs-db = { path = "/path/to/Research/lgs-db" }
# ///
"""
Stim Mask Refiner
=================
Web app for manually refining stim detection mask onset-offset and duration
per piecewise-constant epoch (device program period).

Each unique epoch = unique (patient_id, epoch_start_gmt, epoch_end_gmt).
Within an epoch the stim parameters — and therefore the artifact shape — are
constant, so you only need to set the mask once per epoch.

Three columns are updated in the stim_catalog.parquet:
  mask_onset_offset_ms  — ms to shift mask start from the annotated trigger time
                          (negative = start before trigger, positive = after)
  mask_duration_ms      — length of the mask window in ms
  manually_refined      — bool, set to True once you accept an epoch

Usage:
  uv run src/stim_mask_refiner.py
  uv run src/stim_mask_refiner.py --catalog data/stim_catalog.parquet --port 8766
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from lgs_db import read_dat, to_microvolts

# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────


def epoch_id(key: str) -> str:
    """URL-safe short hash for an epoch key."""
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _s(val) -> str:
    """Scalar → str, NaN/None → empty string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _f(val) -> float | None:
    """Scalar → Python float, NaN/None/non-numeric → None."""
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, bool):
        return None
    try:
        f = float(val)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _b(val) -> bool:
    """Scalar → Python bool."""
    if hasattr(val, "item"):
        val = val.item()
    return bool(val)


# ─────────────────────────────────────────────────────────────────
# Data Store
# ─────────────────────────────────────────────────────────────────


class DataStore:
    """
    Reads stim_catalog.parquet, builds one compact record per epoch,
    serves signal data from .dat files via lgs_db.
    """

    def __init__(self, catalog_path: Path, sampling_rate: int = 250):
        self.catalog_path = catalog_path
        self.sampling_rate = sampling_rate
        self._backup_dir = catalog_path.parent / "backups"
        self._backup_dir.mkdir(exist_ok=True)

        # Create timestamped backup on startup
        self._create_backup("startup")

        print(f"  Building epoch table from {catalog_path} …")
        self._epochs, self._epoch_by_id = self._build_epoch_table()

        n_ref = sum(1 for e in self._epochs if e["refined"])
        print(
            f"    {len(self._epochs)} epochs  "
            f"({n_ref} refined, {len(self._epochs) - n_ref} pending)"
        )

    def _create_backup(self, tag: str = "save") -> Path:
        """Create a timestamped backup of the catalog parquet."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self._backup_dir / f"stim_catalog_{ts}_{tag}.parquet"
        shutil.copy2(self.catalog_path, dst)
        # Keep only last 20 backups
        backups = sorted(self._backup_dir.glob("stim_catalog_*.parquet"))
        for old in backups[:-20]:
            old.unlink()
        print(f"    Backup → {dst.name}")
        return dst

    def _build_epoch_table(self) -> tuple[list[dict], dict[str, dict]]:
        """
        1. Load parquet
        2. Compute epoch keys
        3. groupby → one representative recording per epoch
        4. Return compact epoch list
        """
        df = pd.read_parquet(self.catalog_path)

        # Ensure mask columns
        for col, default in [
            ("mask_onset_offset_ms", 0.0),
            ("mask_duration_ms", 1000.0),
            ("manually_refined", False),
        ]:
            if col not in df.columns:
                df[col] = default

        df["mask_onset_offset_ms"] = pd.to_numeric(
            df["mask_onset_offset_ms"], errors="coerce"
        ).fillna(0.0)
        df["mask_duration_ms"] = pd.to_numeric(
            df["mask_duration_ms"], errors="coerce"
        ).fillna(1000.0)
        df["manually_refined"] = df["manually_refined"].fillna(False).astype(bool)

        # Epoch key: patient_id | epoch_start_gmt | epoch_end_gmt
        df["_ek"] = (
            df["patient_id"].astype(str).str.strip()
            + "|"
            + df["epoch_start_gmt"].astype(str).str.strip()
            + "|"
            + df["epoch_end_gmt"].astype(str).str.strip()
        )

        # Parse onset_times from JSON strings
        def _parse_onsets(s) -> list[float]:
            if pd.isna(s) or str(s).strip() == "":
                return []
            try:
                return sorted(json.loads(s))
            except (json.JSONDecodeError, TypeError):
                return []

        df["_onsets_parsed"] = df["onset_times"].apply(_parse_onsets)
        df["_n_events"] = df["n_stim_events"]

        # Check file existence once
        df["_file_exists"] = df["file_path"].apply(
            lambda p: Path(p).exists() if pd.notna(p) else False
        )

        epochs: list[dict] = []
        by_id: dict[str, dict] = {}

        for key, idx in df.groupby("_ek", sort=False).groups.items():
            grp = df.loc[idx]
            eid = epoch_id(key)

            # Mask values: from the last refined row if any, else first row
            if grp["manually_refined"].any():
                ref = grp[grp["manually_refined"]].iloc[-1]
                is_refined = True
            else:
                ref = grp.iloc[0]
                is_refined = False

            mask_dur = float(ref["mask_duration_ms"])
            mask_off = float(ref["mask_onset_offset_ms"])

            # Representative recording: file exists + most stim events
            avail = grp[grp["_file_exists"]]
            repr_file_path = None
            repr_onsets: list[float] = []
            repr_dur_sec = 0.0

            if not avail.empty:
                best_row = avail.loc[avail["_n_events"].idxmax()]
                repr_file_path = str(best_row["file_path"])
                repr_onsets = best_row["_onsets_parsed"]
                repr_dur_sec = float(best_row.get("length_sec") or 0)

            first = grp.iloc[0]
            epoch = {
                "epoch_id": eid,
                "patient_id": _s(first.get("patient_id")),
                "subject": _s(first.get("subject_id_lr") or first.get("subject")),
                "epoch_start": _s(first.get("epoch_start_gmt")),
                "epoch_end": _s(first.get("epoch_end_gmt")),
                "stim_current_mA": _f(first.get("t1b1_ma")),
                "stim_frequency_Hz": _f(first.get("t1b1_hz")),
                "stim_pulse_width_uS": _f(first.get("t1b1_us")),
                "stim_duration_ms": _f(first.get("t1b1_ms")),
                "mask_duration_ms": mask_dur,
                "mask_onset_offset_ms": mask_off,
                "refined": is_refined,
                "n_recordings": len(grp),
                "representative_file_path": repr_file_path,
                # private: not sent in list, sent in detail
                "_key": key,
                "_onsets": repr_onsets,
                "_duration_sec": repr_dur_sec,
            }
            epochs.append(epoch)
            by_id[eid] = epoch

        return epochs, by_id

    def get_epoch_list(self) -> list[dict]:
        """Compact list for sidebar — strips private fields."""
        return [
            {k: v for k, v in e.items() if not k.startswith("_")} for e in self._epochs
        ]

    def get_epoch_detail(self, eid: str) -> dict | None:
        e = self._epoch_by_id.get(eid)
        if not e:
            return None
        return {
            **{k: v for k, v in e.items() if not k.startswith("_")},
            "file_path": e["representative_file_path"],
            "onsets": e["_onsets"],
            "duration_sec": e["_duration_sec"],
        }

    def get_signal(self, file_path: str) -> dict | None:
        """Load ECoG signal from .dat via lgs_db."""
        p = Path(file_path)
        if not p.exists():
            return None

        MAX_SAMPLES = 3000
        try:
            raw = read_dat(str(p))  # (4, n_samples) int16
            uv = to_microvolts(raw)  # (4, n_samples) float64 µV
        except Exception as exc:
            print(f"  WARNING: read_dat failed for {p.name}: {exc}")
            return None

        n_ch, n_samp = uv.shape
        step = max(1, n_samp // MAX_SAMPLES)
        channels: dict[str, list] = {}
        for i in range(n_ch):
            data = np.nan_to_num(uv[i, ::step], nan=0.0, posinf=0.0, neginf=0.0)
            channels[str(i + 1)] = data.tolist()

        eff_sr = self.sampling_rate / step
        return {"channels": channels, "sampling_rate": eff_sr}

    def save_epoch(
        self, eid: str, onset_offset_ms: float, duration_ms: float
    ) -> tuple[bool, int]:
        """Reload parquet, update all rows for this epoch, persist."""
        epoch = self._epoch_by_id.get(eid)
        if not epoch:
            return False, 0

        key = epoch["_key"]

        # Backup before every write
        self._create_backup("save")

        df = pd.read_parquet(self.catalog_path)
        for col, default in [
            ("mask_onset_offset_ms", 0.0),
            ("mask_duration_ms", 1000.0),
            ("manually_refined", False),
        ]:
            if col not in df.columns:
                df[col] = default

        ekey = (
            df["patient_id"].astype(str).str.strip()
            + "|"
            + df["epoch_start_gmt"].astype(str).str.strip()
            + "|"
            + df["epoch_end_gmt"].astype(str).str.strip()
        )
        mask = ekey == key
        n = int(mask.sum())

        df.loc[mask, "mask_onset_offset_ms"] = onset_offset_ms
        df.loc[mask, "mask_duration_ms"] = duration_ms
        df.loc[mask, "manually_refined"] = True
        df.to_parquet(self.catalog_path, index=False, engine="pyarrow")

        # Mirror update in in-memory epoch record
        epoch["mask_onset_offset_ms"] = onset_offset_ms
        epoch["mask_duration_ms"] = duration_ms
        epoch["refined"] = True

        return True, n


# ─────────────────────────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────────────────────────


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RefineryHandler(BaseHTTPRequestHandler):
    data_store: DataStore

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._send(HTML_PAGE, "text/html; charset=utf-8")
        elif path == "/api/epochs":
            self._send_json(self.data_store.get_epoch_list())
        elif path.startswith("/api/epoch/"):
            eid = unquote(path[len("/api/epoch/") :])
            detail = self.data_store.get_epoch_detail(eid)
            if detail:
                self._send_json(detail)
            else:
                self.send_error(404, f"Epoch not found: {eid}")
        elif path.startswith("/api/signal/"):
            fp = unquote(path[len("/api/signal/") :])
            sig = self.data_store.get_signal(fp)
            if sig:
                self._send_json(sig)
            else:
                self.send_error(404, f"Signal not found: {fp}")
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            ok, n = self.data_store.save_epoch(
                body["epoch_id"],
                float(body["onset_offset_ms"]),
                float(body["duration_ms"]),
            )
            self._send_json({"ok": ok, "n_updated": n})
        else:
            self.send_error(404)

    def _send(self, body: str, content_type: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        body = json.dumps(obj, default=str)
        body = re.sub(r"\bNaN\b", "null", body)
        body = re.sub(r"\b-?Infinity\b", "null", body)
        self._send(body, "application/json")


# ─────────────────────────────────────────────────────────────────
# HTML Page
# ─────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stim Mask Refiner</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary:   #0f1117;
  --bg-secondary: #161821;
  --bg-tertiary:  #1c1e2b;
  --bg-elevated:  #222436;
  --border:       #2f3348;
  --border-acc:   #3d425a;
  --text-primary: #c8d3f5;
  --text-sec:     #828bb8;
  --text-muted:   #545c7e;
  --cyan:   #86e1fc;
  --green:  #c3e88d;
  --red:    #ff757f;
  --yellow: #ffc777;
  --blue:   #82aaff;
  --purple: #c099ff;
  --orange: #ff966c;
  --font-mono: 'JetBrains Mono', monospace;
  --font-sans: 'DM Sans', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font-sans); background: var(--bg-primary); color: var(--text-primary); overflow: hidden; height: 100vh; }
.app { display: flex; height: 100vh; }

/* Sidebar */
.sidebar { width: 320px; min-width: 320px; background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.sidebar-header { padding: 12px 14px; border-bottom: 1px solid var(--border); }
.sidebar-header h2 { font-family: var(--font-mono); font-size: 12px; font-weight: 600; color: var(--cyan); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
.global-stats { display: flex; gap: 10px; flex-wrap: wrap; font-size: 11px; font-family: var(--font-mono); }
.stat { display: flex; align-items: center; gap: 3px; }
.stat .lbl { color: var(--text-muted); }
.stat .val { font-weight: 600; }
.progress-bar-wrap { margin-top: 7px; height: 4px; background: var(--bg-primary); border-radius: 2px; overflow: hidden; }
.progress-bar { height: 100%; background: var(--green); border-radius: 2px; transition: width 0.4s; }

.filter-row { padding: 7px 12px; border-bottom: 1px solid var(--border); display: flex; gap: 6px; }
.filter-row select, .filter-row input {
  padding: 5px 7px; background: var(--bg-primary); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text-primary); font-family: var(--font-mono); font-size: 11px; outline: none;
}
.filter-row select:focus, .filter-row input:focus { border-color: var(--blue); }
.filter-row select { min-width: 100px; }
.filter-row input { flex: 1; }
.filter-row input::placeholder { color: var(--text-muted); }

.list-stats { padding: 4px 14px; font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); border-bottom: 1px solid var(--border); }

.epoch-list { flex: 1; overflow-y: auto; padding: 4px 6px; }
.epoch-item {
  padding: 6px 8px; margin: 1px 0; border-radius: 4px; cursor: pointer;
  font-family: var(--font-mono); font-size: 11px; display: flex; align-items: center; gap: 6px;
  border: 1px solid transparent; transition: background 0.1s;
}
.epoch-item:hover { background: var(--bg-tertiary); }
.epoch-item.active { background: var(--bg-elevated); border-color: var(--border-acc); }
.epoch-item .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.epoch-item.done .status-dot { background: var(--green); }
.epoch-item.pending .status-dot { background: var(--border-acc); border: 1.5px solid var(--text-muted); }
.epoch-item .ep-body { flex: 1; min-width: 0; }
.epoch-item .ep-patient { color: var(--blue); font-weight: 600; }
.epoch-item .ep-params { color: var(--text-sec); font-size: 10px; }
.epoch-item .ep-date { color: var(--text-muted); font-size: 9px; display: block; }
.epoch-item .ep-recs { font-size: 10px; color: var(--text-muted); flex-shrink: 0; }

/* Main */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.topbar {
  padding: 8px 18px; background: var(--bg-secondary); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.topbar .nav-btns { display: flex; gap: 4px; }
.topbar h1 { font-family: var(--font-mono); font-size: 13px; font-weight: 600; flex: 1; }
.status-badge { font-family: var(--font-mono); font-size: 11px; padding: 3px 10px; border-radius: 3px; font-weight: 600; flex-shrink: 0; }
.status-badge.done    { background: rgba(195,232,141,0.12); color: var(--green); border: 1px solid rgba(195,232,141,0.3); }
.status-badge.pending { background: rgba(255,199,119,0.12); color: var(--yellow); border: 1px solid rgba(255,199,119,0.3); }

.info-strip {
  padding: 5px 18px; background: var(--bg-tertiary); border-bottom: 1px solid var(--border);
  display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
}
.info-chip { display: flex; align-items: center; gap: 3px; }
.info-chip .lbl { color: var(--text-muted); font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; }
.info-chip .val { font-family: var(--font-mono); font-weight: 600; font-size: 11px; }

.event-nav-bar {
  padding: 4px 18px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px; background: var(--bg-secondary);
}
.event-nav-bar .btn { padding: 3px 9px; font-size: 11px; }
.event-label { font-family: var(--font-mono); font-size: 11px; min-width: 180px; text-align: center;
  padding: 2px 8px; border-radius: 4px; background: var(--bg-primary); border: 1px solid var(--border); color: var(--text-sec); }

.plot-area { flex: 1; padding: 4px 8px; overflow: hidden; position: relative; }
#plotDiv { width: 100%; height: 100%; }
.loading-overlay {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgba(15,17,23,0.75); z-index: 10; font-family: var(--font-mono);
  font-size: 12px; color: var(--text-sec);
}

/* Controls */
.controls-area {
  padding: 8px 18px 6px; border-top: 1px solid var(--border);
  background: var(--bg-secondary); display: flex; flex-direction: column; gap: 5px;
}
.slider-row { display: flex; align-items: center; gap: 10px; }
.slider-row label { font-family: var(--font-mono); font-size: 11px; color: var(--text-sec); width: 110px; flex-shrink: 0; }
.slider-row input[type=range] { flex: 1; accent-color: var(--cyan); cursor: pointer; height: 4px; }
.slider-row input[type=number] {
  width: 78px; padding: 3px 6px; background: var(--bg-primary);
  border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);
  font-family: var(--font-mono); font-size: 11px; text-align: right; outline: none;
}
.slider-row input[type=number]:focus { border-color: var(--cyan); }
.slider-row .unit { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); width: 20px; }
.slider-val { font-family: var(--font-mono); font-size: 11px; width: 72px; text-align: right; color: var(--cyan); flex-shrink: 0; }

.accept-row { display: flex; align-items: center; gap: 12px; margin-top: 2px; }
.accept-btn {
  padding: 6px 20px; background: rgba(134,225,252,0.1); border: 1px solid var(--cyan);
  border-radius: 4px; color: var(--cyan); font-family: var(--font-mono); font-size: 12px;
  font-weight: 600; cursor: pointer; transition: all 0.15s;
}
.accept-btn:hover { background: rgba(134,225,252,0.2); }
.accept-btn:active { transform: scale(0.97); }

/* Bottombar */
.bottombar {
  padding: 4px 18px; background: var(--bg-secondary); border-top: 1px solid var(--border);
  display: flex; align-items: center; gap: 14px; font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);
}
.shortcut { display: flex; align-items: center; gap: 3px; }
kbd {
  background: var(--bg-elevated); border: 1px solid var(--border);
  padding: 1px 5px; border-radius: 3px; font-family: var(--font-mono); font-size: 10px; color: var(--cyan);
}

/* Misc */
.btn {
  padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px;
  background: var(--bg-elevated); color: var(--text-primary); cursor: pointer;
  font-family: var(--font-sans); font-size: 12px; transition: all 0.15s;
}
.btn:hover { border-color: var(--blue); color: var(--blue); }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; color: var(--text-muted); font-size: 13px; gap: 8px; }
.empty-state .icon { font-size: 36px; opacity: 0.3; }

.toast {
  position: fixed; bottom: 18px; right: 18px; padding: 9px 16px;
  border-radius: 5px; font-size: 12px; font-family: var(--font-mono);
  opacity: 0; transform: translateY(8px); transition: all 0.22s; pointer-events: none; z-index: 999; border: 1px solid;
}
.toast.show { opacity: 1; transform: translateY(0); }
.toast.success { background: rgba(195,232,141,0.1); border-color: var(--green); color: var(--green); }
.toast.error   { background: rgba(255,117,127,0.1); border-color: var(--red);   color: var(--red);   }
.toast.info    { background: rgba(130,170,255,0.1); border-color: var(--blue);  color: var(--blue);  }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-acc); }
</style>
</head>
<body>
<div class="app">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <h2>Stim Mask Refiner</h2>
      <div class="global-stats" id="globalStats"></div>
      <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar"></div></div>
    </div>
    <div class="filter-row">
      <select id="filterStatus" onchange="applyFilters();renderSidebar();">
        <option value="pending" selected>Pending</option>
        <option value="done">Done</option>
        <option value="all">All</option>
      </select>
      <input id="searchInput" placeholder="patient / subject..." oninput="applyFilters();renderSidebar();">
    </div>
    <div class="list-stats" id="listStats">Loading...</div>
    <div class="epoch-list" id="epochList">
      <div class="empty-state"><div class="icon">&#9203;</div><div>Loading epochs...</div></div>
    </div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="topbar">
      <div class="nav-btns">
        <button class="btn" onclick="navEpoch(-1)" title="Previous epoch">&#9664;</button>
        <button class="btn" onclick="navEpoch(1)"  title="Next epoch">&#9654;</button>
      </div>
      <h1 id="epochTitle">No epoch loaded</h1>
      <div class="status-badge pending" id="statusBadge">pending</div>
    </div>

    <div class="info-strip" id="infoStrip">
      <div class="info-chip"><span class="lbl">Patient</span><span class="val" style="color:var(--blue)"   id="infoPatient">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Subject</span><span class="val" style="color:var(--purple)" id="infoSubject">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Current</span><span class="val" style="color:var(--orange)" id="infoCurrent">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Freq</span><span class="val"    style="color:var(--yellow)" id="infoFreq">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Pulse</span><span class="val"   style="color:var(--purple)" id="infoPulse">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Epoch</span><span class="val"   style="color:var(--text-sec)" id="infoEpoch">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Recordings</span><span class="val" style="color:var(--cyan)" id="infoNRec">&#8212;</span></div>
      <div class="info-chip"><span class="lbl">Events</span><span class="val" style="color:var(--cyan)" id="infoNEv">&#8212;</span></div>
    </div>

    <div class="event-nav-bar">
      <button class="btn" onclick="navEvent(-1)">&#8592; Prev</button>
      <div class="event-label" id="eventLabel">&#8212;</div>
      <button class="btn" onclick="navEvent(1)">Next &#8594;</button>
      <span style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);margin-left:8px;">
        Arrows navigate events &middot; <kbd>Z</kbd> zoom &middot; <kbd>R</kbd> reset
      </span>
    </div>

    <div class="plot-area">
      <div id="plotDiv">
        <div class="empty-state"><div class="icon">&#128202;</div><div>Select an epoch to begin</div></div>
      </div>
      <div class="loading-overlay" id="loadingOverlay" style="display:none;">Loading...</div>
    </div>

    <div class="controls-area">
      <div class="slider-row">
        <label>Onset offset</label>
        <input type="range" id="onsetSlider" min="-2000" max="2000" step="1" value="0"
               oninput="onOnsetChange(this.value,'slider')">
        <input type="number" id="onsetNum" value="0" step="1" min="-2000" max="2000"
               onchange="onOnsetChange(this.value,'num')">
        <span class="unit">ms</span>
        <span class="slider-val" id="onsetDisplay">0 ms</span>
      </div>
      <div class="slider-row">
        <label>Duration</label>
        <input type="range" id="durSlider" min="10" max="15000" step="10" value="1000"
               oninput="onDurChange(this.value,'slider')">
        <input type="number" id="durNum" value="1000" step="10" min="10" max="15000"
               onchange="onDurChange(this.value,'num')">
        <span class="unit">ms</span>
        <span class="slider-val" id="durDisplay">1000 ms</span>
      </div>
      <div class="accept-row">
        <button class="accept-btn" onclick="acceptAndNext()">Accept &amp; Next &nbsp;<kbd>&#8629;</kbd></button>
        <span style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);">
          Saves onset&nbsp;offset + duration for all <span id="acceptNRec">&#8212;</span> recordings in this epoch
        </span>
      </div>
    </div>

    <div class="bottombar">
      <div class="shortcut"><kbd>&uarr;</kbd><kbd>&darr;</kbd> epochs</div>
      <div class="shortcut"><kbd>&larr;</kbd><kbd>&rarr;</kbd> events</div>
      <div class="shortcut"><kbd>&#8629;</kbd> accept &amp; next</div>
      <div class="shortcut"><kbd>R</kbd> reset zoom</div>
      <div class="shortcut"><kbd>Z</kbd> zoom to event</div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
// State
let allEpochs      = [];
let filteredEpochs = [];
let currentIdx     = -1;
let currentEpoch   = null;
let currentSignal  = null;
let currentEventIdx = -1;
let onsetOffsetMs  = 0;
let maskDurationMs = 1000;
let signalCache    = new Map();
let plotRendered   = false;

// Init
async function init() {
    const resp = await fetch('/api/epochs');
    allEpochs = await resp.json();
    updateGlobalStats();
    applyFilters();
    renderSidebar();
    const firstPending = filteredEpochs.findIndex(e => !e.refined);
    const startIdx = firstPending >= 0 ? firstPending : 0;
    if (filteredEpochs.length > 0) loadEpoch(startIdx);
}

// Stats
function updateGlobalStats() {
    const total   = allEpochs.length;
    const refined = allEpochs.filter(e => e.refined).length;
    const pct     = total > 0 ? Math.round(100 * refined / total) : 0;
    document.getElementById('globalStats').innerHTML =
        `<div class="stat"><span class="lbl">Total:</span><span class="val" style="color:var(--blue)">${total}</span></div>` +
        `<div class="stat"><span class="lbl">Done:</span><span class="val" style="color:var(--green)">${refined}</span></div>` +
        `<div class="stat"><span class="lbl">Pending:</span><span class="val" style="color:var(--yellow)">${total - refined}</span></div>` +
        `<div class="stat"><span class="lbl">${pct}%</span></div>`;
    document.getElementById('progressBar').style.width = pct + '%';
}

// Filters
function applyFilters() {
    const status = document.getElementById('filterStatus').value;
    const q      = document.getElementById('searchInput').value.toLowerCase();
    filteredEpochs = allEpochs.filter(e => {
        if (q && !String(e.patient_id).includes(q) && !(e.subject||'').toLowerCase().includes(q))
            return false;
        if (status === 'pending') return !e.refined;
        if (status === 'done')    return  e.refined;
        return true;
    });
    const nPending = filteredEpochs.filter(e => !e.refined).length;
    document.getElementById('listStats').textContent =
        `${filteredEpochs.length} shown \u00b7 ${nPending} pending`;
}

// Sidebar
function renderSidebar() {
    const container = document.getElementById('epochList');
    if (filteredEpochs.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="icon">&#128269;</div><div>No epochs match</div></div>';
        return;
    }
    let html = '';
    for (let i = 0; i < filteredEpochs.length; i++) {
        const e = filteredEpochs[i];
        const isActive = (i === currentIdx);
        const cls = e.refined ? 'done' : 'pending';
        const current = e.stim_current_mA != null ? e.stim_current_mA + 'mA' : '?mA';
        const freq    = e.stim_frequency_Hz  != null ? e.stim_frequency_Hz  + 'Hz' : '?Hz';
        const pulse   = e.stim_pulse_width_uS != null ? e.stim_pulse_width_uS + '\u00b5s' : '?\u00b5s';
        const dateStr = (e.epoch_start || '').split(' ')[0];
        html += `<div class="epoch-item ${isActive ? 'active' : ''} ${cls}" onclick="loadEpoch(${i})">
          <div class="status-dot"></div>
          <div class="ep-body">
            <span class="ep-patient">P${e.patient_id}</span>
            <span class="ep-params"> &nbsp;${current} / ${freq} / ${pulse}</span>
            <span class="ep-date">${dateStr}</span>
          </div>
          <span class="ep-recs">${e.n_recordings}r</span>
        </div>`;
    }
    container.innerHTML = html;
    requestAnimationFrame(() => {
        const el = container.querySelector('.epoch-item.active');
        if (el) el.scrollIntoView({ block: 'nearest' });
    });
}

// Load Epoch
async function loadEpoch(idx) {
    if (idx < 0 || idx >= filteredEpochs.length) return;
    currentIdx = idx;
    currentEventIdx = -1;
    plotRendered = false;

    const slim = filteredEpochs[idx];
    document.getElementById('epochTitle').textContent =
        `Patient ${slim.patient_id}  \u00b7  ${(slim.epoch_start||'').split(' ')[0]}`;
    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('loadingOverlay').textContent = 'Loading\u2026';
    renderSidebar();

    try {
        const fp = slim.representative_file_path;
        const [detailResp, signalResp] = await Promise.all([
            fetch(`/api/epoch/${slim.epoch_id}`),
            (fp && !signalCache.has(fp))
                ? fetch(`/api/signal/${encodeURIComponent(fp)}`)
                : Promise.resolve(null),
        ]);

        if (!detailResp.ok) throw new Error(`Detail fetch failed: ${detailResp.status}`);
        currentEpoch = await detailResp.json();

        if (signalResp && signalResp.ok) {
            const sig = await signalResp.json();
            if (signalCache.size > 25) signalCache.delete(signalCache.keys().next().value);
            signalCache.set(fp, sig);
        }
        currentSignal = fp ? (signalCache.get(fp) || null) : null;

        document.getElementById('loadingOverlay').style.display = 'none';

        onsetOffsetMs  = currentEpoch.mask_onset_offset_ms || 0;
        maskDurationMs = currentEpoch.mask_duration_ms     || 1000;
        syncSliders();
        updateInfoStrip();

        if (currentSignal) {
            renderPlot();
            plotRendered = true;
        } else {
            document.getElementById('plotDiv').innerHTML =
                '<div class="empty-state"><div class="icon">&#9888;</div><div>No signal available for this epoch</div></div>';
        }
        updateEventLabel();

    } catch (err) {
        document.getElementById('loadingOverlay').style.display = 'none';
        showToast('Load error: ' + err.message, 'error');
        console.error('loadEpoch error:', err);
    }
}

// Info Strip
function updateInfoStrip() {
    const e = currentEpoch;
    if (!e) return;
    const fmt = (v, unit) => (v != null && v !== '' && !isNaN(v)) ? v + (unit||'') : '\u2014';
    document.getElementById('infoPatient').textContent = e.patient_id || '\u2014';
    document.getElementById('infoSubject').textContent = e.subject    || '\u2014';
    document.getElementById('infoCurrent').textContent = fmt(e.stim_current_mA, ' mA');
    document.getElementById('infoFreq').textContent    = fmt(e.stim_frequency_Hz, ' Hz');
    document.getElementById('infoPulse').textContent   = fmt(e.stim_pulse_width_uS, ' \u00b5s');
    const start = (e.epoch_start||'').split(' ')[0];
    const end   = (e.epoch_end  ||'').split(' ')[0];
    document.getElementById('infoEpoch').textContent   = start + (end ? ' \u2192 ' + end : '');
    document.getElementById('infoNRec').textContent    = e.n_recordings + ' recordings';
    document.getElementById('infoNEv').textContent     = (e.onsets||[]).length + ' events (repr.)';
    document.getElementById('acceptNRec').textContent  = e.n_recordings;
    const refined = e.refined;
    const badge = document.getElementById('statusBadge');
    badge.textContent = refined ? '\u2713 refined' : '\u25cb pending';
    badge.className   = 'status-badge ' + (refined ? 'done' : 'pending');
}

// Sliders
function syncSliders() {
    document.getElementById('onsetSlider').value = onsetOffsetMs;
    document.getElementById('onsetNum').value    = onsetOffsetMs;
    document.getElementById('durSlider').value   = maskDurationMs;
    document.getElementById('durNum').value      = maskDurationMs;
    document.getElementById('onsetDisplay').textContent = (onsetOffsetMs >= 0 ? '+' : '') + onsetOffsetMs + ' ms';
    document.getElementById('durDisplay').textContent   = maskDurationMs + ' ms';
}

function onOnsetChange(val, src) {
    onsetOffsetMs = parseFloat(val) || 0;
    if (src === 'slider') document.getElementById('onsetNum').value = onsetOffsetMs;
    else                  document.getElementById('onsetSlider').value = onsetOffsetMs;
    document.getElementById('onsetDisplay').textContent = (onsetOffsetMs >= 0 ? '+' : '') + onsetOffsetMs + ' ms';
    updateMaskShapes();
}

function onDurChange(val, src) {
    maskDurationMs = Math.max(10, parseFloat(val) || 10);
    if (src === 'slider') document.getElementById('durNum').value = maskDurationMs;
    else                  document.getElementById('durSlider').value = maskDurationMs;
    document.getElementById('durDisplay').textContent = maskDurationMs + ' ms';
    updateMaskShapes();
}

// Plot
function buildShapes(onsets, offsetS, durationS) {
    const shapes = [];
    for (const t of onsets) {
        shapes.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: t, y0: 0, x1: t, y1: 1,
            line: { color: 'rgba(200,200,200,0.25)', width: 1, dash: 'dot' },
        });
        shapes.push({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: t + offsetS, y0: 0, x1: t + offsetS + durationS, y1: 1,
            fillcolor: 'rgba(0,212,255,0.14)',
            line: { color: 'rgba(0,212,255,0.75)', width: 1.5 },
        });
        shapes.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: t + offsetS, y0: 0, x1: t + offsetS, y1: 1,
            line: { color: 'rgba(134,225,252,0.85)', width: 2 },
        });
        shapes.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: t + offsetS + durationS, y0: 0, x1: t + offsetS + durationS, y1: 1,
            line: { color: 'rgba(255,199,119,0.7)', width: 1.5, dash: 'dash' },
        });
    }
    return shapes;
}

function buildAnnotations(onsets, offsetS) {
    return onsets.map(t => ({
        x: t + offsetS, y: 1.02, xref: 'x', yref: 'paper',
        text: t.toFixed(2) + 's', showarrow: false,
        font: { color: 'rgba(134,225,252,0.65)', size: 8, family: 'JetBrains Mono, monospace' },
        bgcolor: 'rgba(15,17,23,0.7)', borderpad: 1,
    }));
}

function renderPlot() {
    const signal  = currentSignal;
    const sr      = signal.sampling_rate;
    const onsets  = (currentEpoch && currentEpoch.onsets) || [];
    const offsetS = onsetOffsetMs  / 1000;
    const durS    = maskDurationMs / 1000;

    const traces = [];
    const chNames = ['Ch 1', 'Ch 2', 'Ch 3', 'Ch 4'];
    const chColors = ['rgba(200,210,245,0.75)', 'rgba(150,200,245,0.7)',
                      'rgba(195,232,141,0.65)', 'rgba(255,199,119,0.65)'];
    let nCh = 0;
    for (let ch = 1; ch <= 4; ch++) {
        const data = signal.channels[String(ch)];
        if (!data || data.length === 0) continue;
        const time = Array.from({length: data.length}, (_, i) => i / sr);
        traces.push({
            x: time, y: data, type: 'scatter', mode: 'lines',
            name: chNames[ch-1],
            line: { color: chColors[nCh], width: 0.8 },
            yaxis: nCh === 0 ? 'y' : `y${nCh + 1}`,
        });
        nCh++;
    }

    const shapes      = buildShapes(onsets, offsetS, durS);
    const annotations = buildAnnotations(onsets, offsetS);

    const domStep  = nCh > 0 ? 1.0 / nCh : 1.0;
    const domGap   = 0.025;
    const yaxBase  = {
        color: '#545c7e', gridcolor: '#1c1e2b', zerolinecolor: '#2f3348',
        showticklabels: false,
    };
    const layout = {
        paper_bgcolor: '#0f1117', plot_bgcolor: '#0f1117',
        font: { color: '#828bb8', family: 'JetBrains Mono, monospace', size: 10 },
        showlegend: false,
        margin: { l: 42, r: 8, t: 18, b: 32 },
        xaxis: { color: '#545c7e', gridcolor: '#1c1e2b', zerolinecolor: '#2f3348',
                 title: { text: 'Time (s)', font: { size: 11 } } },
        hovermode: 'x', dragmode: 'pan',
        shapes: shapes, annotations: annotations,
    };
    for (let i = 0; i < nCh; i++) {
        const domBot = 1.0 - (i + 1) * domStep + domGap / 2;
        const domTop = 1.0 - i       * domStep - domGap / 2;
        const key = i === 0 ? 'yaxis' : `yaxis${i + 1}`;
        layout[key] = { ...yaxBase, domain: [Math.max(0, domBot), Math.min(1, domTop)],
            showticklabels: true, title: { text: chNames[i], font: {size: 9} } };
    }

    Plotly.react('plotDiv', traces, layout, {
        displayModeBar: true, displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        responsive: true, scrollZoom: true,
    });
}

function updateMaskShapes() {
    if (!plotRendered || !currentSignal || !currentEpoch) return;
    const onsets  = currentEpoch.onsets || [];
    const offsetS = onsetOffsetMs  / 1000;
    const durS    = maskDurationMs / 1000;
    Plotly.relayout('plotDiv', {
        shapes:      buildShapes(onsets, offsetS, durS),
        annotations: buildAnnotations(onsets, offsetS),
    });
}

// Event Navigation
function navEvent(delta) {
    const onsets = (currentEpoch && currentEpoch.onsets) || [];
    if (onsets.length === 0) return;
    if (currentEventIdx < 0) {
        currentEventIdx = delta > 0 ? 0 : onsets.length - 1;
    } else {
        currentEventIdx = Math.max(0, Math.min(onsets.length - 1, currentEventIdx + delta));
    }
    updateEventLabel();
    zoomToEvent();
}

function updateEventLabel() {
    const onsets = (currentEpoch && currentEpoch.onsets) || [];
    const el = document.getElementById('eventLabel');
    if (onsets.length === 0) { el.textContent = 'No stim events'; return; }
    if (currentEventIdx < 0) { el.textContent = onsets.length + ' stim events'; return; }
    el.textContent = `Event ${currentEventIdx + 1} / ${onsets.length}  @  ${onsets[currentEventIdx].toFixed(3)} s`;
}

function zoomToEvent() {
    if (!currentEpoch || currentEventIdx < 0) return;
    const t    = currentEpoch.onsets[currentEventIdx];
    const durS = maskDurationMs / 1000;
    const ofS  = onsetOffsetMs  / 1000;
    const pad  = Math.max(durS * 2, 1.5);
    Plotly.relayout('plotDiv', {
        'xaxis.range': [t - pad, t + ofS + durS + pad],
        'yaxis.autorange': true,
    });
}

// Epoch Navigation
function navEpoch(delta) {
    const next = currentIdx + delta;
    if (next >= 0 && next < filteredEpochs.length) loadEpoch(next);
}

// Accept & Next
async function acceptAndNext() {
    if (!currentEpoch) return;

    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('loadingOverlay').textContent = 'Saving\u2026';

    const resp = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            epoch_id:        currentEpoch.epoch_id,
            onset_offset_ms: onsetOffsetMs,
            duration_ms:     maskDurationMs,
        }),
    });
    const result = await resp.json();
    document.getElementById('loadingOverlay').style.display = 'none';

    if (!result.ok) { showToast('Save failed!', 'error'); return; }

    showToast(`Saved \u2014 ${result.n_updated} row${result.n_updated !== 1 ? 's' : ''} updated`, 'success');

    const eid = currentEpoch.epoch_id;
    [allEpochs, filteredEpochs].forEach(list => {
        const e = list.find(x => x.epoch_id === eid);
        if (e) { e.refined = true; e.mask_onset_offset_ms = onsetOffsetMs; e.mask_duration_ms = maskDurationMs; }
    });

    let nextEpochId = null;
    for (let i = currentIdx + 1; i < filteredEpochs.length; i++) {
        if (!filteredEpochs[i].refined) { nextEpochId = filteredEpochs[i].epoch_id; break; }
    }
    if (!nextEpochId) {
        for (let i = 0; i < currentIdx; i++) {
            if (!filteredEpochs[i].refined) { nextEpochId = filteredEpochs[i].epoch_id; break; }
        }
    }

    updateGlobalStats();
    applyFilters();
    renderSidebar();

    if (!nextEpochId) {
        showToast('All visible epochs refined!', 'success');
        return;
    }
    const newIdx = filteredEpochs.findIndex(e => e.epoch_id === nextEpochId);
    if (newIdx >= 0) loadEpoch(newIdx);
    else { showToast('All visible epochs refined!', 'success'); }
}

// Keyboard
document.addEventListener('keydown', e => {
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    switch (e.key) {
        case 'ArrowUp':    e.preventDefault(); navEpoch(-1);  break;
        case 'ArrowDown':  e.preventDefault(); navEpoch(1);   break;
        case 'ArrowLeft':  e.preventDefault(); navEvent(-1);  break;
        case 'ArrowRight': e.preventDefault(); navEvent(1);   break;
        case 'Enter':      e.preventDefault(); acceptAndNext(); break;
        case 'z': case 'Z': zoomToEvent(); break;
        case 'r': case 'R':
            Plotly.relayout('plotDiv', {
                'xaxis.autorange': true, 'yaxis.autorange': true,
                'yaxis2.autorange': true, 'yaxis3.autorange': true, 'yaxis4.autorange': true,
            }); break;
    }
});

// Toast
function showToast(msg, type = 'info') {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = `toast ${type} show`;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.remove('show'), 2500);
}

// Start
init();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stim Mask Refiner — manually set mask onset-offset and duration per epoch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "stim_catalog.parquet",
        help="Stim catalog parquet file to read and update in-place",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=250,
        help="ECoG sampling rate in Hz (default: 250)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Local server port (default: 8766)",
    )
    args = parser.parse_args()

    if not args.catalog.exists():
        sys.exit(f"ERROR: Catalog not found: {args.catalog}")

    print("Starting Stim Mask Refiner...")
    store = DataStore(
        catalog_path=args.catalog,
        sampling_rate=args.sampling_rate,
    )
    RefineryHandler.data_store = store

    server = ThreadedHTTPServer(("localhost", args.port), RefineryHandler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Ready at {url}")
    print("  Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
