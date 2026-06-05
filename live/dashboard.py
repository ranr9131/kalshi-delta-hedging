"""
Live snipe dashboard.  Single-file Flask app that tails `snipes.csv` and
pushes each new row to the browser over Server-Sent Events.

Run on AWS, access via SSH tunnel:
    ssh -i KEY -L 8080:localhost:8080 ec2-user@HOST
    open http://localhost:8080

Routes:
  GET /            HTML dashboard
  GET /api/recent  JSON: last 100 snipes + aggregate stats (initial load)
  GET /events      SSE stream: one event per new snipe row + heartbeats
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from typing import Dict, List

from flask import Flask, Response, jsonify

LOG_PATH = os.environ.get(
    "LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "snipes.csv"),
)
LOG_V2_PATH = os.environ.get(
    "LOG_V2_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "snipes_v2.csv"),
)
SETTLE_PATH = os.environ.get(
    "SETTLE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "settlements.csv"),
)
PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")

app = Flask(__name__)
# Don't try to sort_keys — our row dicts can have heterogeneous keys
# (the _pnl/_status decorations added at API time), and sort_keys=True
# crashes if any None ever sneaks in.  Order is irrelevant for JS clients.
app.json.sort_keys = False


# ── CSV helpers ────────────────────────────────────────────────────────────

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Read CSV, strip out any None keys (csv.DictReader puts extra columns
    from malformed rows into key=None, which then crashes Flask's
    sort_keys=True JSON encoding).  Also skip rows missing critical fields."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            # Drop the None key (csv puts extra fields here)
            r.pop(None, None)
            # Skip rows that lost their primary key
            if not r.get("ts_iso") or not r.get("ticker"):
                continue
            out.append(r)
    return out


def _read_all_rows() -> List[Dict[str, str]]:
    return _read_csv_rows(LOG_PATH)


def _read_v2_rows() -> List[Dict[str, str]]:
    return _read_csv_rows(LOG_V2_PATH)


def _read_settlements() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(SETTLE_PATH):
        return {}
    with open(SETTLE_PATH, newline="") as f:
        return {r["ticker"]: r for r in csv.DictReader(f) if r.get("ticker")}


def _pnl_series(rows: List[Dict[str, str]],
                settlements: Dict[str, Dict[str, str]]) -> List[dict]:
    """Time-ordered cumulative realized PnL.  One point per settled snipe.

    Sort by settlement timestamp (falling back to snipe timestamp).  Walk
    in order and accumulate.  Returns [{t, pnl}].  Open / unknown rows are
    skipped — only realized cashflows show on the chart."""
    events: List[Tuple[str, float]] = []
    for r in rows:
        s = settlements.get(r.get("ticker", ""))
        if not s:
            continue
        result = (s.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue
        side = (r.get("side") or "").lower()
        try:
            qty   = float(r.get("qty") or 0)
            stake = float(r.get("stake_dollars") or 0)
        except Exception:
            continue
        pnl = (qty - stake) if side == result else (-stake)
        t = s.get("settled_at_iso") or r.get("ts_iso", "")
        if not t:
            continue
        events.append((t, pnl))
    events.sort(key=lambda x: x[0])
    series = []
    running = 0.0
    for t, p in events:
        running += p
        series.append({"t": t, "pnl": round(running, 4)})
    return series


def _row_pnl(row: Dict[str, str], settle: Dict[str, str] | None):
    """Per-snipe realized PnL in dollars.  Returns (pnl, status) where
    status ∈ {"open", "win", "loss", "unknown"} and pnl is None for open."""
    if settle is None:
        return None, "open"
    result = (settle.get("result") or "").lower()
    if result not in ("yes", "no"):
        return 0.0, "unknown"   # un-settleable (e.g. recycled ticker)
    side = (row.get("side") or "").lower()
    try:
        qty   = float(row.get("qty") or 0)
        stake = float(row.get("stake_dollars") or 0)
    except Exception:
        return 0.0, "unknown"
    won = (side == result)
    pnl = (qty * 1.0 - stake) if won else (-stake)
    return round(pnl, 4), ("win" if won else "loss")


def _aggregate(rows: List[Dict[str, str]],
               settlements: Dict[str, Dict[str, str]]) -> dict:
    """Compute aggregate stats over a list of snipe rows (joined w/ settlements)."""
    n = len(rows)
    by_asset: Dict[str, dict] = {}
    realized_pnl = 0.0
    open_count = 0
    open_stake = 0.0
    win_count = 0
    loss_count = 0
    unk_count  = 0
    settled_stake = 0.0

    total_stake = 0.0
    edge_sum = 0.0

    for r in rows:
        stake = float(r.get("stake_dollars") or 0)
        ec    = float(r.get("edge_cents")   or 0)
        total_stake += stake
        edge_sum    += ec

        pnl, st = _row_pnl(r, settlements.get(r.get("ticker", "")))
        if st == "open":
            open_count += 1
            open_stake += stake
        elif st == "win":
            win_count += 1
            realized_pnl += pnl or 0
            settled_stake += stake
        elif st == "loss":
            loss_count += 1
            realized_pnl += pnl or 0
            settled_stake += stake
        else:
            unk_count += 1

        a = r.get("asset") or "?"
        d = by_asset.setdefault(a, {
            "count": 0, "stake": 0.0, "edge_sum": 0.0,
            "pnl": 0.0, "wins": 0, "losses": 0,
            "last_edge": 0.0, "last_fill_cents": 0.0,
            "last_ts": "", "last_side": "",
        })
        d["count"] += 1
        d["stake"] += stake
        d["edge_sum"] += ec
        d["last_edge"] = ec
        d["last_fill_cents"] = float(r.get("fill_cents_est") or 0)
        d["last_ts"] = r.get("ts_iso", "")
        d["last_side"] = r.get("side", "")
        if st in ("win", "loss"):
            d["pnl"] += (pnl or 0)
            if st == "win":  d["wins"] += 1
            else:            d["losses"] += 1

    for d in by_asset.values():
        d["avg_edge"] = round(d["edge_sum"] / max(1, d["count"]), 2)
        d["pnl"]      = round(d["pnl"], 2)
        d["stake"]    = round(d["stake"], 2)

    # snipes/min over last 5 min
    now = time.time()
    recent = 0
    for r in rows:
        ts = r.get("ts_iso")
        if not ts: continue
        try:
            import datetime as _dt
            t = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            if now - t <= 300:
                recent += 1
        except Exception:
            pass

    settled_total = win_count + loss_count
    win_rate = (win_count / settled_total) if settled_total > 0 else 0.0
    roi = (realized_pnl / settled_stake) if settled_stake > 0 else 0.0
    return {
        "total": n,
        "total_stake": round(total_stake, 2),
        "avg_edge": round(edge_sum / max(1, n), 2),
        "by_asset": by_asset,
        "snipes_per_min_recent": round(recent / 5.0, 2),
        "realized_pnl": round(realized_pnl, 2),
        "open_count": open_count,
        "open_stake": round(open_stake, 2),
        "settled_count": settled_total,
        "win_count": win_count,
        "loss_count": loss_count,
        "unknown_count": unk_count,
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/recent")
def api_recent():
    rows = _read_all_rows()
    settlements = _read_settlements()
    last = rows[-200:][::-1]   # most recent first
    # Decorate each visible row with pnl/status for the frontend.
    for r in last:
        pnl, st = _row_pnl(r, settlements.get(r.get("ticker", "")))
        r["_pnl"]    = pnl
        r["_status"] = st
    return jsonify({
        "rows":       last,
        "stats":      _aggregate(rows, settlements),
        "pnl_series": _pnl_series(rows, settlements),
    })


@app.route("/api/settlements")
def api_settlements():
    """Return {ticker: {result, raw_status, settled_at_iso}} for the JS to
    join into the live DOM rows on each poll."""
    return jsonify(_read_settlements())


@app.route("/api/v2")
def api_v2():
    """Same shape as /api/recent but reads snipes_v2.csv.  Used by the
    dashboard to render the v2 comparison panel + chart line."""
    rows = _read_v2_rows()
    settlements = _read_settlements()
    last = rows[-200:][::-1]
    for r in last:
        pnl, st = _row_pnl(r, settlements.get(r.get("ticker", "")))
        r["_pnl"]    = pnl
        r["_status"] = st
    return jsonify({
        "rows":       last,
        "stats":      _aggregate(rows, settlements),
        "pnl_series": _pnl_series(rows, settlements),
    })


@app.route("/events")
def events():
    """SSE stream: emit each newly-appended CSV row.

    We track byte offset and re-poll the file every 500 ms.  When new bytes
    appear, parse them as CSV continuation and emit each row as one event.
    A heartbeat is sent every 15 s to keep the connection from idling out.
    """
    def stream():
        # Seek to end so we only emit *new* rows from now on.
        try:
            size = os.path.getsize(LOG_PATH)
        except OSError:
            size = 0
        last_heartbeat = time.time()
        # Snapshot field order from header line (or fallback)
        header = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH) as f:
                    header = f.readline().strip().split(",")
            except Exception:
                header = []
        # Send an initial event so the page knows the stream is live.
        yield "event: ready\ndata: {}\n\n"

        while True:
            try:
                cur_size = os.path.getsize(LOG_PATH)
            except OSError:
                cur_size = 0
            if cur_size < size:
                # File truncated/rotated — reset.
                size = 0
                header = []
            if cur_size > size:
                with open(LOG_PATH, "rb") as f:
                    f.seek(size)
                    chunk = f.read(cur_size - size).decode("utf-8", errors="replace")
                    size = cur_size
                # Make sure we have the header
                if not header:
                    nl = chunk.find("\n")
                    if nl != -1:
                        header = chunk[:nl].strip().split(",")
                        chunk = chunk[nl + 1:]
                lines = [ln for ln in chunk.splitlines() if ln.strip()]
                for ln in lines:
                    try:
                        row = dict(zip(header, next(csv.reader([ln]))))
                    except Exception:
                        continue
                    if row.get("ts_iso"):
                        yield f"event: snipe\ndata: {json.dumps(row)}\n\n"
                last_heartbeat = time.time()
            elif time.time() - last_heartbeat > 15:
                yield "event: heartbeat\ndata: {}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    })


# ── HTML ───────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kalshi Sniper</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {
  --bg-0: #07080d;
  --bg-1: #0e1118;
  --bg-2: #161b25;
  --bg-3: #1f2735;
  --txt: #e6ebf2;
  --txt-dim: #7a8597;
  --acc: #4cf0c2;
  --warn: #ffb73b;
  --danger: #ff5677;
  --btc: #f7931a;
  --eth: #8c8cff;
  --sol: #14f195;
  --xrp: #45c4ff;
  --yes: #4cf0c2;
  --no:  #ff7b95;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; height: 100%;
  font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace;
  background: radial-gradient(ellipse at top, var(--bg-1), var(--bg-0) 80%);
  color: var(--txt);
  -webkit-font-smoothing: antialiased;
}
.app { display: flex; flex-direction: column; height: 100vh; }

header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 22px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  background: rgba(0,0,0,0.2);
  backdrop-filter: blur(10px);
}
header .brand {
  font-size: 20px; font-weight: 700; letter-spacing: 2px;
  background: linear-gradient(90deg, var(--acc), #b59bff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
header .brand .sub {
  font-size: 11px; color: var(--txt-dim); letter-spacing: 1px;
  margin-left: 8px; font-weight: 400;
}
header .right { display: flex; align-items: center; gap: 16px; font-size: 12px; }
.led {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--danger); box-shadow: 0 0 8px var(--danger);
}
.led.live { background: var(--acc); box-shadow: 0 0 12px var(--acc); }
.clock { color: var(--txt-dim); font-variant-numeric: tabular-nums; }

.stats-row {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 10px;
  padding: 12px 22px;
}
.stat {
  background: var(--bg-2);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 10px;
  padding: 12px 14px;
}
.stat .label { color: var(--txt-dim); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; }
.stat .value { font-size: 24px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.stat .sub   { color: var(--txt-dim); font-size: 11px; margin-top: 2px; }
.stat.pnl .value.pos { color: var(--acc); text-shadow: 0 0 12px rgba(76,240,194,0.4); }
.stat.pnl .value.neg { color: var(--danger); text-shadow: 0 0 12px rgba(255,86,119,0.4); }

.chart-row {
  margin: 0 22px 12px 22px;
  background: var(--bg-2);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 10px;
  padding: 10px 14px 8px 14px;
}
.chart-row .head {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 4px;
}
.chart-row .head .label {
  color: var(--txt-dim); font-size: 10px; letter-spacing: 2px;
  text-transform: uppercase;
}
.chart-row .head .meta {
  color: var(--txt-dim); font-size: 10px; font-variant-numeric: tabular-nums;
}
#pnl-chart {
  display: block;
  width: 100%;
  height: 160px;
}

.asset-row {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
  padding: 0 22px 12px 22px;
}
.asset {
  background: var(--bg-2);
  border-radius: 10px;
  padding: 10px 14px;
  position: relative;
  border: 1px solid rgba(255,255,255,0.04);
  overflow: hidden;
}
.asset::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--bar);
}
.asset.BTC  { --bar: var(--btc); }
.asset.ETH  { --bar: var(--eth); }
.asset.SOL  { --bar: var(--sol); }
.asset.XRP  { --bar: var(--xrp); }
.asset.HYPE { --bar: #ff77c6; }
.asset .name { font-weight: 700; letter-spacing: 1px; color: var(--bar); }
.asset .grid { display: grid; grid-template-columns: 1fr 1fr; margin-top: 4px; font-size: 11px; }
.asset .grid div { color: var(--txt-dim); }
.asset .grid div b { color: var(--txt); font-weight: 600; font-variant-numeric: tabular-nums; }

main {
  flex: 1; overflow-y: auto;
  padding: 0 22px 22px 22px;
}
.feed-head {
  display: grid;
  grid-template-columns: 84px 56px 54px 1fr 68px 60px 60px 70px 56px 56px 90px;
  gap: 8px; padding: 8px 12px;
  font-size: 10px; letter-spacing: 2px; color: var(--txt-dim);
  text-transform: uppercase; border-bottom: 1px solid rgba(255,255,255,0.05);
  position: sticky; top: 0;
  background: var(--bg-0);
  z-index: 10;
}
.row {
  display: grid;
  grid-template-columns: 84px 56px 54px 1fr 68px 60px 60px 70px 56px 56px 90px;
  gap: 8px; padding: 10px 12px;
  align-items: center;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  animation: slideIn 350ms ease-out;
  position: relative;
}
@keyframes slideIn {
  from { opacity: 0; transform: translateY(-8px); background: rgba(76,240,194,0.10); }
  to   { opacity: 1; transform: translateY(0);    background: transparent; }
}
.row .ts  { color: var(--txt-dim); }
.row .asset-tag {
  font-weight: 700;
  padding: 3px 6px; border-radius: 4px;
  text-align: center; font-size: 11px;
  background: rgba(255,255,255,0.03);
}
.row.BTC  .asset-tag { color: var(--btc); border: 1px solid var(--btc); }
.row.ETH  .asset-tag { color: var(--eth); border: 1px solid var(--eth); }
.row.SOL  .asset-tag { color: var(--sol); border: 1px solid var(--sol); }
.row.XRP  .asset-tag { color: var(--xrp); border: 1px solid var(--xrp); }
.row.HYPE .asset-tag { color: #ff77c6;    border: 1px solid #ff77c6; }
.row .side {
  font-weight: 700; text-align: center; padding: 3px;
  border-radius: 4px; font-size: 11px;
}
.row .side.yes { color: var(--yes); background: rgba(76,240,194,0.08); }
.row .side.no  { color: var(--no);  background: rgba(255,123,149,0.08); }
.row .ticker { color: var(--txt-dim); font-size: 11px; overflow: hidden; text-overflow: ellipsis; }
.row .ticker b { color: var(--txt); }
.row .edge {
  font-weight: 700;
  color: var(--acc);
  text-align: right;
}
.row .edge.big   { color: #b59bff; text-shadow: 0 0 8px rgba(181,155,255,0.5); }
.row .edge.huge  { color: var(--warn); text-shadow: 0 0 10px rgba(255,183,59,0.7); }
.row .fill, .row .qty, .row .stake, .row .age, .row .mv, .row .pnl {
  text-align: right; color: var(--txt);
}
.row .age.fresh { color: var(--danger); }
.row .age.stale { color: var(--acc); }
.row .pnl.win  { color: var(--acc);    font-weight: 700; }
.row .pnl.loss { color: var(--danger); font-weight: 700; }
.row .pnl.open { color: var(--txt-dim); font-style: italic; }
.row .pnl.unknown { color: var(--warn); }

.empty {
  padding: 60px;
  text-align: center; color: var(--txt-dim);
  font-size: 14px;
}
.pulse {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--acc); margin-right: 6px;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50%       { opacity: 1; }
}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="brand">KALSHI SNIPER<span class="sub">• stale-order monitor • paper mode</span></div>
    <div class="right">
      <span class="clock" id="clock">--:--:--</span>
      <span><span class="led" id="led"></span> <span id="conn">connecting…</span></span>
    </div>
  </header>

  <div class="stats-row">
    <div class="stat pnl"><div class="label">Realized PnL</div>
      <div class="value" id="s-pnl">$0</div>
      <div class="sub" id="s-pnl-sub">0 settled • 0% ROI</div></div>
    <div class="stat"><div class="label">Win rate</div>
      <div class="value" id="s-wr">—</div>
      <div class="sub" id="s-wr-sub">0 W / 0 L</div></div>
    <div class="stat"><div class="label">Open exposure</div>
      <div class="value" id="s-open">$0</div>
      <div class="sub" id="s-open-sub">0 unsettled</div></div>
    <div class="stat"><div class="label">Total snipes</div>
      <div class="value" id="s-total">0</div>
      <div class="sub">since service start</div></div>
    <div class="stat"><div class="label">Rate</div>
      <div class="value" id="s-rate">0.0</div>
      <div class="sub">snipes / min (last 5)</div></div>
    <div class="stat"><div class="label">Avg edge</div>
      <div class="value" id="s-edge">0¢</div>
      <div class="sub">model vs market</div></div>
    <div class="stat"><div class="label">Next settle</div>
      <div class="value" id="s-eta">—</div>
      <div class="sub" id="s-eta-sub">15-min cycle</div></div>
  </div>

  <div class="stats-row" style="border-top: 1px dashed rgba(255,255,255,0.04); padding-top: 8px;">
    <div class="stat pnl"><div class="label" style="color:#b59bff">V2 Realized PnL</div>
      <div class="value" id="s-pnl-v2">$0</div>
      <div class="sub" id="s-pnl-v2-sub">v2: realized σ + directional gate</div></div>
    <div class="stat"><div class="label">V2 Win rate</div>
      <div class="value" id="s-wr-v2">—</div>
      <div class="sub" id="s-wr-v2-sub">0 W / 0 L</div></div>
    <div class="stat"><div class="label">V2 Open</div>
      <div class="value" id="s-open-v2">$0</div>
      <div class="sub" id="s-open-v2-sub">0 unsettled</div></div>
    <div class="stat"><div class="label">V2 Snipes</div>
      <div class="value" id="s-total-v2">0</div>
      <div class="sub">since v2 launched</div></div>
    <div class="stat"><div class="label">V2 Rate</div>
      <div class="value" id="s-rate-v2">0.0</div>
      <div class="sub">snipes / min (last 5)</div></div>
    <div class="stat"><div class="label">V2 Avg edge</div>
      <div class="value" id="s-edge-v2">0¢</div>
      <div class="sub">model vs market</div></div>
    <div class="stat"><div class="label">V2 ROI</div>
      <div class="value" id="s-roi-v2">—</div>
      <div class="sub" id="s-roi-v2-sub">on wagered</div></div>
  </div>

  <div class="chart-row">
    <div class="head">
      <span class="label">Cumulative realized PnL</span>
      <span class="meta" id="chart-meta">—</span>
    </div>
    <svg id="pnl-chart" preserveAspectRatio="none"></svg>
  </div>

  <div class="asset-row" id="asset-row">
    <div class="asset BTC" data-asset="BTC"><div class="name">BTC</div>
      <div class="grid"><div>count <b data-k="count">0</b></div><div>edge <b data-k="avg_edge">0¢</b></div>
      <div>pnl <b data-k="pnl">$0</b></div><div>W/L <b data-k="wl">0/0</b></div></div></div>
    <div class="asset ETH" data-asset="ETH"><div class="name">ETH</div>
      <div class="grid"><div>count <b data-k="count">0</b></div><div>edge <b data-k="avg_edge">0¢</b></div>
      <div>pnl <b data-k="pnl">$0</b></div><div>W/L <b data-k="wl">0/0</b></div></div></div>
    <div class="asset SOL" data-asset="SOL"><div class="name">SOL</div>
      <div class="grid"><div>count <b data-k="count">0</b></div><div>edge <b data-k="avg_edge">0¢</b></div>
      <div>pnl <b data-k="pnl">$0</b></div><div>W/L <b data-k="wl">0/0</b></div></div></div>
    <div class="asset XRP" data-asset="XRP"><div class="name">XRP</div>
      <div class="grid"><div>count <b data-k="count">0</b></div><div>edge <b data-k="avg_edge">0¢</b></div>
      <div>pnl <b data-k="pnl">$0</b></div><div>W/L <b data-k="wl">0/0</b></div></div></div>
    <div class="asset HYPE" data-asset="HYPE"><div class="name">HYPE</div>
      <div class="grid"><div>count <b data-k="count">0</b></div><div>edge <b data-k="avg_edge">0¢</b></div>
      <div>pnl <b data-k="pnl">$0</b></div><div>W/L <b data-k="wl">0/0</b></div></div></div>
  </div>

  <main>
    <div class="feed-head">
      <div>TIME UTC</div><div>ASSET</div><div>SIDE</div><div>MARKET</div>
      <div style="text-align:right">EDGE</div>
      <div style="text-align:right">FILL¢</div>
      <div style="text-align:right">QTY</div>
      <div style="text-align:right">STAKE</div>
      <div style="text-align:right">AGE</div>
      <div style="text-align:right">MV bps</div>
      <div style="text-align:right">PNL</div>
    </div>
    <div id="feed"><div class="empty">Waiting for first snipe…</div></div>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
const fmt$  = (v) => "$" + Number(v || 0).toFixed(2);
const fmtC  = (v) => Number(v || 0).toFixed(1) + "¢";
const fmtN  = (v, d=1) => Number(v || 0).toFixed(d);
const recent60 = [];   // timestamps of snipes in past minute, for the small counter

// Settler waits SETTLEMENT_DELAY_SEC after each 15-min boundary before it
// queries Kalshi.  Keep this in sync with the systemd unit env var so the
// countdown reflects reality.
const SETTLEMENT_DELAY_SEC = 5;

function tickClock(){
  const d = new Date();
  $("clock").textContent = d.toISOString().slice(11,19) + "Z";
  // Prune recent60 (kept around for future use even though it's no longer
  // displayed — could come back as a sparkline)
  const cutoff = Date.now() - 60_000;
  while (recent60.length && recent60[0] < cutoff) recent60.shift();

  // Next settlement = next 15-min UTC boundary + delay
  const m = d.getUTCMinutes();
  const s = d.getUTCSeconds();
  const minsToBoundary = (15 - (m % 15)) % 15;
  const boundarySec = (minsToBoundary === 0 && s === 0)
    ? 0 : (minsToBoundary * 60 - s);
  const settleIn = boundarySec + SETTLEMENT_DELAY_SEC;
  const mm = Math.floor(settleIn / 60);
  const ss = settleIn % 60;
  $("s-eta").textContent = `${mm}m ${String(ss).padStart(2,"0")}s`;
}
setInterval(tickClock, 1000); tickClock();

let totalRows = 0;
const feed = $("feed");
const maxRows = 300;

function tickerShort(t){
  if (!t) return "";
  const parts = t.split("-");
  return parts.length >= 3 ? `<b>${parts[2]}</b>` : t;
}
function edgeClass(c){
  c = Number(c);
  if (c >= 20) return "huge";
  if (c >= 10) return "big";
  return "";
}
function ageClass(s){
  s = Number(s);
  return s < 2.5 ? "fresh" : "stale";
}

function pnlCell(status, pnl){
  if (status === "open" || status == null)
    return `<div class="pnl open">open</div>`;
  if (status === "unknown")
    return `<div class="pnl unknown">—</div>`;
  const cls = status === "win" ? "win" : "loss";
  const sign = pnl > 0 ? "+" : "";
  return `<div class="pnl ${cls}">${sign}${fmt$(pnl)}</div>`;
}

function addRow(r, animate=true){
  if (totalRows === 0) feed.innerHTML = "";
  totalRows++;
  // Only count *live* fires toward the rolling-60s counter — initial-load
  // rows already have timestamps stamped in the past, so use those instead.
  if (animate) {
    recent60.push(Date.now());
  }

  const ts = (r.ts_iso || "").slice(11,19);
  const asset = (r.asset || "?").toUpperCase();
  const side  = (r.side || "").toLowerCase();
  const ec = r.edge_cents;
  const status = r._status || "open";
  const pnl    = r._pnl;

  const div = document.createElement("div");
  div.className = "row " + asset;
  div.dataset.ticker = r.ticker || "";
  div.dataset.side   = side;
  div.dataset.qty    = r.qty || "0";
  div.dataset.stake  = r.stake_dollars || "0";
  if (!animate) div.style.animation = "none";
  div.innerHTML = `
    <div class="ts">${ts}</div>
    <div class="asset-tag">${asset}</div>
    <div class="side ${side}">${side.toUpperCase()}</div>
    <div class="ticker">${tickerShort(r.ticker)}</div>
    <div class="edge ${edgeClass(ec)}">${fmtC(ec)}</div>
    <div class="fill">${fmtN(r.fill_cents_est)}¢</div>
    <div class="qty">${fmtN(r.qty, 0)}</div>
    <div class="stake">${fmt$(r.stake_dollars)}</div>
    <div class="age ${ageClass(r.lvl_age_sec)}">${fmtN(r.lvl_age_sec)}s</div>
    <div class="mv">${fmtN(r.move_bps)}</div>
    ${pnlCell(status, pnl)}
  `;
  feed.insertBefore(div, feed.firstChild);
  while (feed.children.length > maxRows) feed.removeChild(feed.lastChild);
}

function applySettlements(map){
  // For each open row in the DOM, if its ticker is now settled, update PnL.
  const rows = feed.querySelectorAll(".row");
  rows.forEach(row => {
    const ticker = row.dataset.ticker;
    if (!ticker) return;
    const s = map[ticker];
    if (!s) return;
    const pnlCellEl = row.querySelector(".pnl");
    if (!pnlCellEl || (!pnlCellEl.classList.contains("open") &&
                       !pnlCellEl.classList.contains("unknown"))) return;
    const result = (s.result || "").toLowerCase();
    if (result !== "yes" && result !== "no") {
      pnlCellEl.outerHTML = pnlCell("unknown", 0);
      return;
    }
    const side  = row.dataset.side;
    const qty   = Number(row.dataset.qty || 0);
    const stake = Number(row.dataset.stake || 0);
    const won = (side === result);
    const pnl = won ? (qty - stake) : -stake;
    pnlCellEl.outerHTML = pnlCell(won ? "win" : "loss", pnl);
  });
}

function renderPnlChart(seriesV1, seriesV2){
  const svg = $("pnl-chart");
  const meta = $("chart-meta");
  if (!svg) return;

  const VW = 1000, VH = 160;
  svg.setAttribute("viewBox", `0 0 ${VW} ${VH}`);

  const sV1 = seriesV1 || [];
  const sV2 = seriesV2 || [];
  if (sV1.length === 0 && sV2.length === 0){
    svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle"
        fill="#7a8597" font-size="13" font-family="inherit">
        Waiting for first settled snipe…</text>`;
    meta.textContent = "0 settled";
    return;
  }

  const pad = {l: 56, r: 70, t: 14, b: 22};
  const W = VW - pad.l - pad.r;
  const H = VH - pad.t - pad.b;

  // Shared y-axis range: min of all troughs, max of all peaks, plus zero
  const allPnls = [...sV1.map(d=>d.pnl), ...sV2.map(d=>d.pnl), 0];
  const peak = Math.max(...allPnls);
  const trough = Math.min(...allPnls);
  const range = (peak - trough) || 1;

  // Each series uses its OWN x-axis (its own time span).  We map each
  // series's own length to [pad.l, VW-pad.r] for shape clarity.
  function xFor(series){
    return (i) => pad.l + (series.length > 1 ? (i / (series.length - 1)) * W : W/2);
  }
  const y = (p) => pad.t + H - ((p - trough) / range) * H;
  const zeroY = y(0);
  const fmt$ = (v) => (v >= 0 ? "+$" : "-$") + Math.abs(v).toFixed(2);

  function renderLine(series, lineCol, fillCol, isV1){
    if (series.length === 0) return "";
    const x = xFor(series);
    const ptStr = series.map((d, i) => `${x(i).toFixed(1)},${y(d.pnl).toFixed(1)}`).join(" ");
    const finalPnl = series[series.length - 1].pnl;
    const polyFill = series.length >= 2
      ? `${x(0).toFixed(1)},${zeroY.toFixed(1)} ${ptStr} ${x(series.length-1).toFixed(1)},${zeroY.toFixed(1)}`
      : "";
    const lastX = x(series.length - 1).toFixed(1);
    const lastY = y(finalPnl).toFixed(1);
    const labelX = (parseFloat(lastX) + 8).toFixed(1);
    const labelY = (parseFloat(lastY) + 4 + (isV1 ? 0 : 14)).toFixed(1);
    return `
      ${polyFill ? `<polygon points="${polyFill}" fill="${fillCol}" />` : ""}
      <polyline points="${ptStr}" fill="none" stroke="${lineCol}"
                stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />
      <circle cx="${lastX}" cy="${lastY}" r="4" fill="${lineCol}" />
      <text x="${labelX}" y="${labelY}" text-anchor="start" fill="${lineCol}"
            font-size="12" font-family="inherit" font-weight="700">
        ${isV1 ? 'V1 ' : 'V2 '}${fmt$(finalPnl)}
      </text>`;
  }

  const ticks = [];
  if (peak > 0)  ticks.push(peak);
  ticks.push(0);
  if (trough < 0) ticks.push(trough);

  let labels = "";
  for (const v of ticks){
    const ty = y(v).toFixed(1);
    labels += `
      <line x1="${pad.l}" x2="${VW - pad.r}" y1="${ty}" y2="${ty}"
            stroke="rgba(255,255,255,${v === 0 ? 0.10 : 0.04})"
            stroke-dasharray="${v === 0 ? '4,4' : '0'}" />
      <text x="${pad.l - 8}" y="${(parseFloat(ty) + 4).toFixed(1)}"
            text-anchor="end" fill="#7a8597" font-size="11" font-family="inherit">
        ${v >= 0 ? '$' : '-$'}${Math.abs(v).toFixed(v % 1 === 0 ? 0 : 2)}
      </text>`;
  }

  // v1 = green/red, v2 = purple
  const v1FinalPos = sV1.length > 0 ? sV1[sV1.length-1].pnl >= 0 : true;
  const v1Col = v1FinalPos ? "#4cf0c2" : "#ff5677";
  const v1Fill = v1FinalPos ? "rgba(76,240,194,0.13)" : "rgba(255,86,119,0.13)";
  const v2Col = "#b59bff";
  const v2Fill = "rgba(181,155,255,0.10)";

  svg.innerHTML = `
    ${labels}
    ${renderLine(sV1, v1Col, v1Fill, true)}
    ${renderLine(sV2, v2Col, v2Fill, false)}
  `;

  const parts = [];
  if (sV1.length) parts.push(`V1: ${sV1.length} settled`);
  if (sV2.length) parts.push(`V2: ${sV2.length} settled`);
  meta.textContent = parts.join("  •  ");
}

function applyStats(stats){
  $("s-total").textContent = stats.total;
  $("s-rate").textContent  = fmtN(stats.snipes_per_min_recent, 1);
  $("s-edge").textContent  = fmtC(stats.avg_edge);

  // PnL tile
  const pnlEl = $("s-pnl");
  const pnl   = Number(stats.realized_pnl || 0);
  pnlEl.textContent = (pnl >= 0 ? "+" : "") + fmt$(pnl);
  pnlEl.classList.toggle("pos", pnl > 0);
  pnlEl.classList.toggle("neg", pnl < 0);
  $("s-pnl-sub").textContent =
    `${stats.settled_count||0} settled • ${((stats.roi||0)*100).toFixed(1)}% ROI`;

  // Win rate tile
  const wr = stats.settled_count > 0
    ? ((stats.win_rate || 0) * 100).toFixed(1) + "%"
    : "—";
  $("s-wr").textContent = wr;
  $("s-wr-sub").textContent =
    `${stats.win_count||0} W / ${stats.loss_count||0} L` +
    (stats.unknown_count ? ` • ${stats.unknown_count} ?` : "");

  // Open exposure tile
  $("s-open").textContent = fmt$(stats.open_stake);
  $("s-open-sub").textContent = `${stats.open_count||0} unsettled`;

  for (const a of ["BTC","ETH","SOL","XRP","HYPE"]){
    const card = document.querySelector(`.asset[data-asset="${a}"]`);
    const d = stats.by_asset?.[a];
    card.querySelector('[data-k="count"]').textContent = d ? d.count : 0;
    card.querySelector('[data-k="avg_edge"]').textContent = d ? fmtC(d.avg_edge) : "0¢";
    const pnlB = card.querySelector('[data-k="pnl"]');
    if (d) {
      const p = Number(d.pnl || 0);
      pnlB.textContent = (p >= 0 ? "+" : "") + fmt$(p);
      pnlB.style.color = p > 0 ? "var(--acc)" : (p < 0 ? "var(--danger)" : "");
    } else {
      pnlB.textContent = "$0";
      pnlB.style.color = "";
    }
    card.querySelector('[data-k="wl"]').textContent =
      d ? `${d.wins}/${d.losses}` : "0/0";
  }
}

function applyStatsV2(stats){
  $("s-total-v2").textContent = stats.total;
  $("s-rate-v2").textContent  = fmtN(stats.snipes_per_min_recent, 1);
  $("s-edge-v2").textContent  = fmtC(stats.avg_edge);

  const pnl = Number(stats.realized_pnl || 0);
  const pnlEl = $("s-pnl-v2");
  pnlEl.textContent = (pnl >= 0 ? "+" : "") + fmt$(pnl);
  pnlEl.classList.toggle("pos", pnl > 0);
  pnlEl.classList.toggle("neg", pnl < 0);
  $("s-pnl-v2-sub").textContent =
    `${stats.settled_count||0} settled • ${((stats.roi||0)*100).toFixed(1)}% ROI`;

  const wr = stats.settled_count > 0
    ? ((stats.win_rate || 0) * 100).toFixed(1) + "%" : "—";
  $("s-wr-v2").textContent = wr;
  $("s-wr-v2-sub").textContent =
    `${stats.win_count||0} W / ${stats.loss_count||0} L` +
    (stats.unknown_count ? ` • ${stats.unknown_count} ?` : "");

  $("s-open-v2").textContent = fmt$(stats.open_stake);
  $("s-open-v2-sub").textContent = `${stats.open_count||0} unsettled`;

  $("s-roi-v2").textContent = stats.settled_count > 0
    ? ((stats.roi||0) * 100).toFixed(1) + "%" : "—";
}

let lastSeries = {v1: [], v2: []};

function fetchAll(initial){
  Promise.all([
    fetch("/api/recent").then(r => r.json()),
    fetch("/api/v2").then(r => r.json()),
  ]).then(([d1, d2]) => {
    applyStats(d1.stats);
    applyStatsV2(d2.stats);
    lastSeries.v1 = d1.pnl_series || [];
    lastSeries.v2 = d2.pnl_series || [];
    renderPnlChart(lastSeries.v1, lastSeries.v2);
    if (initial){
      // First load only: replay last 200 v1 rows (no animation)
      for (const r of d1.rows.slice().reverse()) addRow(r, false);
    }
  });
}

// 1) Initial load
fetchAll(true);

// 2) Live stream
let es = null, statsTimer = null;
function connect(){
  es = new EventSource("/events");
  es.addEventListener("ready", () => {
    $("led").classList.add("live");
    $("conn").textContent = "live";
  });
  es.addEventListener("snipe", (ev) => {
    try { addRow(JSON.parse(ev.data), true); } catch(e){}
  });
  es.addEventListener("heartbeat", () => {});
  es.onerror = () => {
    $("led").classList.remove("live");
    $("conn").textContent = "reconnecting…";
    es.close();
    setTimeout(connect, 1500);
  };
}
connect();

// 3) Periodically refresh stats + chart (covers both v1 and v2)
setInterval(() => { fetchAll(false); }, 5000);

// 4) Periodically refresh settlement map and patch any settled rows.
// 2-second cadence so the row flips within ~2s of the settler writing to
// settlements.csv.  Settler itself targets <10s from market close.
setInterval(() => {
  fetch("/api/settlements").then(r => r.json()).then(applySettlements);
}, 2000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


if __name__ == "__main__":
    # threaded=True so SSE can hold a connection while other reqs land.
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
