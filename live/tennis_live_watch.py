#!/usr/bin/env python3.11
"""
Focused live-match watcher over tennis_xmarket_quotes.csv (written by
tennis_xmarket_logger.py). Observation only; prints alert lines to stdout —
designed to be run under a chat Monitor.

Every POLL s:
  - "live" match = paired match whose Poly p1-mid moved >= LIVE_RANGE_C over
    the trailing WINDOW s (tennis in play reprices point-by-point; pregame
    books sit still).
  - Track the TOP_N most-active live matches. Print when the live set changes.
  - For tracked matches, alert (deduped per match per DEDUPE s) when:
      * venue mids diverge >= DIVERGE_C  (pre-lock signal: one venue lagging), or
      * best cross-venue lock edge >= EDGE_ALERT_C (from the logger's fee-adj calc).
"""
import csv
import os
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "tennis_xmarket_quotes.csv")

POLL = 20
WINDOW = 600
TOP_N = 5
LIVE_RANGE_C = 2.0
DIVERGE_C = 4.0
EDGE_ALERT_C = 1.0
DEDUPE = 180

last_alert = defaultdict(float)
tracked = set()
_pos = 0
_rows = []


def read_new():
    """Incrementally read appended CSV rows (survives logger restarts/truncation)."""
    global _pos, _rows
    try:
        size = os.path.getsize(CSV_PATH)
    except OSError:
        return
    if size < _pos:                      # truncated/rotated
        _pos = 0
    with open(CSV_PATH, newline="") as f:
        if _pos == 0:
            rd = csv.DictReader(f)
            _rows = list(rd)
            _pos = f.tell()
            return
        f.seek(0)
        header = f.readline().strip().split(",")
        f.seek(_pos)
        for line in f:
            vals = line.rstrip("\n").split(",")
            if len(vals) == len(header):
                _rows.append(dict(zip(header, vals)))
        _pos = f.tell()
    cutoff = time.time() - 2 * WINDOW
    _rows = [r for r in _rows if float(r["ts_unix"]) > cutoff]


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return 0.0


print(f"tennis live watch starting | top{TOP_N} live matches | "
      f"live={LIVE_RANGE_C}c range/{WINDOW}s | alert: diverge>={DIVERGE_C}c "
      f"or edge>={EDGE_ALERT_C}c", flush=True)

while True:
    read_new()
    now = time.time()
    by = defaultdict(list)
    for r in _rows:
        if now - float(r["ts_unix"]) <= WINDOW:
            by[r["key"]].append(r)

    ranges = {}
    for k, rs in by.items():
        mids = [(fnum(r, "pp1_bid") + fnum(r, "pp1_ask")) / 2
                for r in rs if fnum(r, "pp1_bid") > 0 and fnum(r, "pp1_ask") > 0]
        if len(mids) >= 5:
            ranges[k] = (max(mids) - min(mids)) * 100

    live = {k for k, rg in ranges.items() if rg >= LIVE_RANGE_C}
    new_tracked = set(sorted(live, key=lambda k: -ranges[k])[:TOP_N])
    if new_tracked != tracked:
        for k in sorted(new_tracked - tracked):
            r = by[k][-1]
            print(f"LIVE+ {r['p1']} vs {r['p2']} (range {ranges[k]:.1f}c/10min) [{k}]",
                  flush=True)
        for k in sorted(tracked - new_tracked):
            print(f"LIVE- {k}", flush=True)
        tracked = new_tracked

    for k in tracked:
        rs = by.get(k)
        if not rs:
            continue
        r = rs[-1]
        k1b, k1a = fnum(r, "k1_bid"), fnum(r, "k1_ask")
        p1b, p1a = fnum(r, "pp1_bid"), fnum(r, "pp1_ask")
        edge = fnum(r, "best_edge_c")
        if not (k1b and k1a and p1b and p1a):
            continue
        gap = abs((k1b + k1a) / 2 - (p1b + p1a) / 2) * 100
        if (gap >= DIVERGE_C or edge >= EDGE_ALERT_C) and now - last_alert[k] > DEDUPE:
            last_alert[k] = now
            tag = "EDGE" if edge >= EDGE_ALERT_C else "DIVERGE"
            print(f"{tag} {r['p1'][:20]} vs {r['p2'][:20]} | gap={gap:.1f}c "
                  f"edge={edge:+.1f}c | K {k1b:.2f}/{k1a:.2f} vs P {p1b:.3f}/{p1a:.3f} "
                  f"| {r['k_stem']}", flush=True)
    time.sleep(POLL)
