"""
Side-performance gate.

For each (asset, side) tracks the rolling 1-hour win rate using the V2 paper
sniper's fires (which run unconditionally — always-firing scout) joined
against settlements.csv.  Blocks live fires for sides currently bleeding.

Why V2 paper (not live) as the data source:
  - If live data drove the gate, blocking a side would freeze that side's
    sample count and the gate could never recover (feedback loop).
  - V2 paper keeps firing regardless of regime, so its win rate is an
    unbiased measure of "is this side actually working right now?"

Configuration via env vars:
  SIDE_GATE_ENABLED          (default: true)
  SIDE_GATE_LOOKBACK_MIN     (default: 60   — minutes of history to consider)
  SIDE_GATE_MIN_SAMPLES      (default: 5    — need this many settles before gating)
  SIDE_GATE_BLOCK_BELOW      (default: 0.30 — block if win rate < this)
  SIDE_GATE_REFRESH_SEC      (default: 30   — re-read CSVs at most this often)
"""
from __future__ import annotations
import csv
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
SETTLE_PATH = os.path.join(BASE, "settlements.csv")
# Read from V2 paper (always-firing scout) AND live snipes for richer samples.
# Feedback-loop concern (block → no new live data → can't recover) is mitigated
# because V2 paper never blocks, so its samples keep arriving in the rolling
# window even while live is gated.
SOURCE_PATHS = [
    os.path.join(BASE, "snipes_v2.csv"),
    os.path.join(BASE, "snipes_leo.csv"),
    os.path.join(BASE, "snipes_friend.csv"),
]

ENABLED       = os.environ.get("SIDE_GATE_ENABLED", "true").lower() == "true"
LOOKBACK_MIN  = float(os.environ.get("SIDE_GATE_LOOKBACK_MIN", "60"))
MIN_SAMPLES   = int(os.environ.get("SIDE_GATE_MIN_SAMPLES", "5"))
BLOCK_BELOW   = float(os.environ.get("SIDE_GATE_BLOCK_BELOW", "0.30"))
REFRESH_SEC   = float(os.environ.get("SIDE_GATE_REFRESH_SEC", "30"))

# Cache state
_cache = {
    "ts": 0.0,         # last refresh time (epoch seconds)
    "win_rate": {},    # (asset, side) -> float
    "samples":  {},    # (asset, side) -> int
}


def _load_settlements():
    out = {}
    if not os.path.exists(SETTLE_PATH):
        return out
    try:
        with open(SETTLE_PATH) as f:
            for r in csv.DictReader(f):
                t = (r.get("ticker") or "").strip()
                res = (r.get("result") or "").strip().lower()
                if t and res in ("yes", "no"):
                    out[t] = res
    except Exception:
        pass
    return out


def _within(ts_iso: str, minutes: float) -> bool:
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t) <= timedelta(minutes=minutes)
    except Exception:
        return False


def _refresh():
    """Recompute win rate from V2 paper settled fires in last LOOKBACK_MIN min."""
    settle = _load_settlements()
    n_by = defaultdict(int)
    w_by = defaultdict(int)
    for path in SOURCE_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                for r in csv.DictReader(f):
                    if not _within(r.get("ts_iso", ""), LOOKBACK_MIN):
                        continue
                    t = (r.get("ticker") or "").strip()
                    res = settle.get(t)
                    if res not in ("yes", "no"):
                        continue
                    a = (r.get("asset") or "").upper()
                    s = (r.get("side") or "").lower()
                    if not a or s not in ("yes", "no"):
                        continue
                    k = (a, s)
                    n_by[k] += 1
                    if s == res:
                        w_by[k] += 1
        except Exception:
            pass

    win_rate = {k: w_by[k] / n for k, n in n_by.items()}
    _cache["win_rate"] = win_rate
    _cache["samples"]  = dict(n_by)
    _cache["ts"]       = time.time()


def is_allowed(asset: str, side: str):
    """Returns (allowed: bool, reason: str)."""
    if not ENABLED:
        return True, "gate_disabled"
    if (time.time() - _cache["ts"]) > REFRESH_SEC:
        _refresh()
    k = (asset.upper(), side.lower())
    n = _cache["samples"].get(k, 0)
    if n < MIN_SAMPLES:
        return True, f"thin_data n={n}<{MIN_SAMPLES}"
    wr = _cache["win_rate"].get(k, 0.0)
    if wr < BLOCK_BELOW:
        return False, f"wr={wr * 100:.0f}% < {BLOCK_BELOW * 100:.0f}% (n={n})"
    return True, f"wr={wr * 100:.0f}% (n={n})"


def status_snapshot():
    """Returns dict of (asset, side) -> 'wr%/n' for periodic logging."""
    if (time.time() - _cache["ts"]) > REFRESH_SEC:
        _refresh()
    out = {}
    for k, wr in _cache["win_rate"].items():
        n = _cache["samples"].get(k, 0)
        out[k] = f"{wr * 100:.0f}%/{n}"
    return out


def status_string():
    """One-line human-readable summary, e.g.:
       'SOL/no:18%/112 [BLOCK]  XRP/yes:85%/40 [allow]  ...'"""
    parts = []
    for (a, s), wr in sorted(_cache["win_rate"].items()):
        n = _cache["samples"].get((a, s), 0)
        tag = "[BLOCK]" if n >= MIN_SAMPLES and wr < BLOCK_BELOW else "[allow]"
        parts.append(f"{a}/{s}:{wr * 100:.0f}%/{n} {tag}")
    return "  ".join(parts) if parts else "no data"


if __name__ == "__main__":
    # CLI for ad-hoc inspection: python3 side_performance_gate.py
    _refresh()
    print(f"ENABLED={ENABLED}  LOOKBACK_MIN={LOOKBACK_MIN}  "
          f"MIN_SAMPLES={MIN_SAMPLES}  BLOCK_BELOW={BLOCK_BELOW}")
    print()
    print(status_string())
    print()
    for asset in ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE"):
        for side in ("yes", "no"):
            ok, reason = is_allowed(asset, side)
            mark = "OK  " if ok else "BLOCK"
            print(f"  {asset:5s} {side:3s}  {mark}  {reason}")
