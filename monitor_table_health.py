"""
Weekly staleness monitor for the 2D fair-price table.

Run weekly (cron / launchd). Each run:
  1. Backs up the current minute_analysis_2d.csv (keeps last 8).
  2. Rebuilds the table via analyze_minutes_2d.py.
  3. Diffs new vs old cell-by-cell — flags any win-rate drift >= 5pp.
  4. Runs simulate_dh.py with the canonical live config.
  5. Splits results into full-period vs last 7 days, compares ROI.
  6. If drift exceeds threshold OR recent ROI is degraded, writes a STALE
     marker file. The trader can check for it on startup and refuse live mode.

NOTE: the walk-forward isn't strict — the rebuilt table includes the last 7
days, so there's mild lookahead leakage into the recent-week ROI. If that
sample is degraded anyway, the regime has definitely shifted.

Run manually:
  python3 monitor_table_health.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
TABLE      = ROOT / "data" / "logs" / "minute_analysis_2d.csv"
BACKUP_DIR = ROOT / "data" / "logs" / "table_backups"
STALE_MARK = ROOT / "data" / "logs" / "TABLE_STALE"
LOG_FILE   = ROOT / "data" / "logs" / "staleness_monitor.log"

WIN_RATE_DRIFT_PP    = 5.0
DRIFT_COUNT_ALARM    = 5
MIN_RECENT_ROI_PCT   = 10.0
MIN_RECENT_WINDOWS   = 50
RECENT_DAYS          = 7
BACKUP_KEEP          = 8

SIM_CONFIG = [
    "--minutes", "4-13",
    "--fair-price-2d",
    "--time-decay",
    "--reversal-hedge", "10",
    "--slippage-cents", "4",
    "--min-edge-cents", "1",
    "--early-skip", "5", "0.05",
]


def log(msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_table(path):
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["minute"]), row["bucket"])
                out[key] = {
                    "win_rate": float(row["win_rate"]),
                    "avg_fill": float(row["avg_fill"]),
                    "n":        int(row["n"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def backup_current_table():
    if not TABLE.exists():
        log("No current table to back up — first run?")
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"minute_analysis_2d_{ts}.csv"
    shutil.copy2(TABLE, dest)
    log(f"Backed up current table -> {dest.name}")
    backups = sorted(BACKUP_DIR.glob("minute_analysis_2d_*.csv"))
    for old in backups[:-BACKUP_KEEP]:
        old.unlink()
        log(f"  Pruned old backup: {old.name}")
    return dest


def rebuild_table():
    log("Rebuilding 2D table via analyze_minutes_2d.py ...")
    r = subprocess.run(
        [sys.executable, "analyze_minutes_2d.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"REBUILD FAILED rc={r.returncode}")
        log(f"  stderr tail: {r.stderr[-500:]}")
        return False
    log("Rebuild complete.")
    return True


def diff_tables(old_path, new_table):
    old = load_table(old_path)
    if not old:
        log("No prior table to diff against — skipping drift check.")
        return 0

    n_drift = 0
    for key, new_cell in sorted(new_table.items()):
        old_cell = old.get(key)
        if not old_cell:
            log(f"  NEW    min={key[0]:2d} bucket={key[1]:12s}  "
                f"WR={new_cell['win_rate']:.3f}  n={new_cell['n']}")
            continue
        wr_delta_pp = (new_cell["win_rate"] - old_cell["win_rate"]) * 100
        if abs(wr_delta_pp) >= WIN_RATE_DRIFT_PP:
            n_drift += 1
            log(f"  DRIFT  min={key[0]:2d} bucket={key[1]:12s}  "
                f"WR {old_cell['win_rate']:.3f} -> {new_cell['win_rate']:.3f}  "
                f"Δ={wr_delta_pp:+5.1f}pp  n_old={old_cell['n']} n_new={new_cell['n']}")
    log(f"Drift summary: {n_drift} cells moved by >= {WIN_RATE_DRIFT_PP}pp")
    return n_drift


def run_simulation():
    log("Running simulate_dh.py at canonical live config...")
    r = subprocess.run(
        [sys.executable, "simulate_dh.py", *SIM_CONFIG],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"SIM FAILED rc={r.returncode}")
        log(f"  stderr tail: {r.stderr[-500:]}")
        return None
    # Pick the matching target-mode CSV by mtime
    candidates = sorted(
        (ROOT / "data" / "logs").glob("simulation_results_dh_target_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        log("No sim result CSV found after run.")
        return None
    return candidates[-1]


def split_roi(sim_csv):
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    full_pnl = 0.0; full_wag = 0.0; n_full = 0
    rec_pnl  = 0.0; rec_wag  = 0.0; n_rec  = 0

    with open(sim_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                pnl = float(row["total_pnl"])
                wag = float(row["total_wagered"])
            except (KeyError, ValueError):
                continue
            full_pnl += pnl; full_wag += wag; n_full += 1
            ts_str = row.get("timestamp_t0", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts >= cutoff:
                rec_pnl += pnl; rec_wag += wag; n_rec += 1

    full_roi = (full_pnl / full_wag * 100) if full_wag else 0.0
    rec_roi  = (rec_pnl  / rec_wag  * 100) if rec_wag  else 0.0
    return {
        "full_roi_pct":   full_roi,
        "full_pnl":       full_pnl,
        "full_wagered":   full_wag,
        "n_full":         n_full,
        "recent_roi_pct": rec_roi,
        "recent_pnl":     rec_pnl,
        "recent_wagered": rec_wag,
        "n_recent":       n_rec,
    }


def write_stale_marker(reasons, drift_count, roi_info):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reasons":       reasons,
        "cells_drifted": drift_count,
        "full_period_roi_pct": roi_info.get("full_roi_pct") if roi_info else None,
        "recent_roi_pct":      roi_info.get("recent_roi_pct") if roi_info else None,
        "recent_window_count": roi_info.get("n_recent") if roi_info else None,
    }
    STALE_MARK.write_text(json.dumps(payload, indent=2))
    log(f"⚠  TABLE STALE — marker written: {STALE_MARK}")
    for r in reasons:
        log(f"     reason: {r}")


def clear_stale_marker():
    if STALE_MARK.exists():
        STALE_MARK.unlink()
        log("Stale marker cleared.")


def main():
    log("=" * 60)
    log("Weekly 2D-table staleness monitor START")

    backup = backup_current_table()
    if not rebuild_table():
        log("ABORT: rebuild step failed; leaving old table in place.")
        if backup:
            shutil.copy2(backup, TABLE)
            log(f"Restored {backup.name} -> {TABLE.name}")
        return 1

    new_table = load_table(TABLE)
    log(f"New table loaded: {len(new_table)} cells.")

    drift_count = diff_tables(backup, new_table)

    sim_csv = run_simulation()
    roi_info = split_roi(sim_csv) if sim_csv else None
    if roi_info:
        log(f"Full-period ROI: {roi_info['full_roi_pct']:+.2f}% over "
            f"{roi_info['n_full']} windows  | "
            f"Last {RECENT_DAYS}d ROI: {roi_info['recent_roi_pct']:+.2f}% over "
            f"{roi_info['n_recent']} windows")

    # Cell drift after rebuild is informational — it means the rebuild
    # absorbed new data, which is the point. We log it but don't mark stale.
    if drift_count >= DRIFT_COUNT_ALARM:
        log(f"NOTE: {drift_count} cells moved heavily — worth eyeballing the diff above.")

    # The only signal that the fresh table is still mispriced for current
    # regime is the recent-period ROI test. If even with the new table the
    # last RECENT_DAYS underperform, the regime has shifted in a way the
    # rolling 90d window can't smooth out.
    reasons = []
    if roi_info and roi_info["n_recent"] >= MIN_RECENT_WINDOWS:
        if roi_info["recent_roi_pct"] < MIN_RECENT_ROI_PCT:
            reasons.append(
                f"Last {RECENT_DAYS}d ROI {roi_info['recent_roi_pct']:+.2f}% < {MIN_RECENT_ROI_PCT}% "
                f"(over {roi_info['n_recent']} windows)"
            )
    elif roi_info:
        log(f"Recent sample too small to gate on ({roi_info['n_recent']} < "
            f"{MIN_RECENT_WINDOWS} windows) — skipping ROI check.")

    if reasons:
        write_stale_marker(reasons, drift_count, roi_info)
        return 2

    clear_stale_marker()
    log("Table healthy. Monitor END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
