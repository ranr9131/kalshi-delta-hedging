"""
TRUE out-of-sample test of the ~4c 2D-table edge on NEVER-SEEN data
(KXBTC15M markets settled after 2026-07-03 16:00 UTC, the end of every
training/refit cache), with AGGRESSIVE fee + slippage modeling.

Methodology mirrors research/MATH_STRATEGY_RESEARCH_5_WALKFORWARD.md:
  - frozen production table data/logs/minute_analysis_2d.csv (fit Feb23-May2)
  - per (minute, |move| bucket) cell: trade continuation side iff the frozen
    table says net edge > gate under that scenario's cost model
  - fill price = Kalshi candle close at decision time (+ slippage stress)
  - win iff continuation held to settlement (direction_up == resolved_yes)

Fee models:
  exact : 0.07 * c * (1-c) per contract (sub-cent, as confirmed from real fills)
  ceil  : ceil to next cent per contract (worst-case rounding; brutal for 1-lots)

Slippage stress: +0c / +1c / +2c / +3c added to the fill price (live measured
~+2.4c from intended price on n=11). Fee is computed on the SLIPPED price.

Gates:
  naive    : frozen-table net edge > 0 at exact fee, no slip (walk-forward gate)
  aware    : frozen-table net edge > 0 under the scenario's OWN costs
             (what a cost-aware live trader would do: fewer trades)

Run: python3 oos_test.py   (self-contained caches in scratchpad/oos_cache/)
"""

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "oos_cache")
os.makedirs(CACHE, exist_ok=True)

REPO = "/Users/leolee/Desktop/kalshi-delta-hedging"
TABLE_CSV = os.path.join(REPO, "data/logs/minute_analysis_2d.csv")

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Unseen period: strictly after the last cached close_time
START = datetime(2026, 7, 3, 16, 0, tzinfo=timezone.utc)
START_TS = int(START.timestamp())

BUCKETS = [
    ("0.00-0.05%", 0.000, 0.05),
    ("0.05-0.10%", 0.050, 0.10),
    ("0.10-0.20%", 0.100, 0.20),
    ("0.20-0.50%", 0.200, 0.50),
    ("0.50%+",     0.500, float("inf")),
]


def get_bucket(pct):
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo <= pct < hi:
            return i
    return len(BUCKETS) - 1


def fee_exact(c):
    return 0.07 * c * (1.0 - c)


def fee_ceil(c):
    return math.ceil(0.07 * c * (1.0 - c) * 100.0) / 100.0


def _get(path, params, retries=4):
    for a in range(retries):
        try:
            r = requests.get(f"{KALSHI}{path}", params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** a)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + a)
    raise RuntimeError(f"failed: {path}")


def fetch_markets():
    p = os.path.join(CACHE, "markets_oos.json")
    if os.path.exists(p):
        return json.load(open(p))
    out, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "settled",
                  "min_close_ts": START_TS, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        batch = d.get("markets", [])
        out.extend(batch)
        print(f"  markets: {len(out)}", flush=True)
        cursor = d.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(0.25)
    json.dump(out, open(p, "w"))
    return out


def fetch_candles(ticker, open_iso, close_iso):
    p = os.path.join(CACHE, f"candles_{ticker}.json")
    if os.path.exists(p):
        return json.load(open(p))
    o = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    c = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    params = {"start_ts": int((o - timedelta(minutes=1)).timestamp()),
              "end_ts": int((c + timedelta(minutes=1)).timestamp()),
              "period_interval": 1}
    try:
        d = _get(f"/series/{SERIES}/markets/{ticker}/candlesticks", params)
    except Exception as e:
        print(f"  candles fail {ticker}: {e}", flush=True)
        return []
    res = []
    for k in d.get("candlesticks", []):
        pr = k.get("price", {})
        yo = pr.get("open_dollars") or pr.get("open")
        yc = pr.get("close_dollars") or pr.get("close")
        if yo is None or yc is None:
            continue
        res.append({"ts": k["end_period_ts"], "yes_close": float(yc)})
    res.sort(key=lambda x: x["ts"])
    json.dump(res, open(p, "w"))
    time.sleep(0.12)
    return res


def yes_at(candles, t):
    price = None
    for k in candles:
        if k["ts"] <= t:
            price = k["yes_close"]
        else:
            break
    return price


def fetch_btc(start_ts, end_ts):
    prices = {}
    day = (start_ts // 86400) * 86400
    while day < end_ts + 86400:
        ds = datetime.fromtimestamp(day, tz=timezone.utc).strftime("%Y%m%d")
        p = os.path.join(CACHE, f"btc_{ds}.json")
        if os.path.exists(p):
            prices.update(json.load(open(p)))
            day += 86400
            continue
        dp = {}
        cs = day
        while cs < day + 86400:
            ce = min(cs + 300 * 60, day + 86400)
            si = datetime.fromtimestamp(cs, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            ei = datetime.fromtimestamp(ce, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            for a in range(4):
                try:
                    r = requests.get(CB_URL, params={"granularity": 60, "start": si, "end": ei}, timeout=15)
                    if r.status_code == 429:
                        time.sleep(5 * (a + 1))
                        continue
                    r.raise_for_status()
                    for row in r.json():
                        ts = int(row[0])
                        if day <= ts < day + 86400:
                            dp[str(ts)] = float(row[4])
                    break
                except Exception:
                    time.sleep(2 * (a + 1))
            cs += 300 * 60
            time.sleep(0.15)
        json.dump(dp, open(p, "w"))
        print(f"  btc {ds}: {len(dp)}", flush=True)
        prices.update(dp)
        day += 86400
    return prices


def lookup(prices, ts):
    m = (ts // 60) * 60
    for off in (0, 60, -60, 120, -120):
        v = prices.get(str(m + off))
        if v is not None:
            return v
    return None


def load_table():
    tbl = {}
    labels = [b[0] for b in BUCKETS]
    for row in csv.DictReader(open(TABLE_CSV)):
        key = (int(row["minute"]), labels.index(row["bucket"]))
        tbl[key] = {"win_rate": float(row["win_rate"]),
                    "avg_fill": float(row["avg_fill"]),
                    "n": int(row["n"])}
    return tbl


def main():
    tbl = load_table()
    print("frozen table cells:", len(tbl), flush=True)

    markets = fetch_markets()
    print(f"unseen settled markets since {START.isoformat()}: {len(markets)}", flush=True)

    ts_all = []
    for m in markets:
        try:
            ts_all.append(int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()))
        except Exception:
            pass
    btc = fetch_btc(min(ts_all) - 600, max(ts_all) + 1800)
    print("btc minutes:", len(btc), flush=True)

    # collect raw observations
    obs = []      # (market_idx, minute, bucket, fill, win)
    skipped = 0
    for i, m in enumerate(markets):
        oi, ci, res = m.get("open_time", ""), m.get("close_time", ""), m.get("result", "")
        if not oi or not ci or res not in ("yes", "no"):
            skipped += 1
            continue
        t0 = int(datetime.fromisoformat(oi.replace("Z", "+00:00")).timestamp())
        ry = res == "yes"
        b0 = lookup(btc, t0)
        if b0 is None:
            skipped += 1
            continue
        candles = fetch_candles(m["ticker"], oi, ci)
        if not candles:
            skipped += 1
            continue
        for minute in range(1, 15):
            t = t0 + minute * 60
            bt = lookup(btc, t)
            ky = yes_at(candles, t)
            if bt is None or ky is None or not (0.01 < ky < 0.99):
                continue
            pct = abs(bt - b0) / b0 * 100.0
            up = bt > b0
            bi = get_bucket(pct)
            fill = ky if up else 1.0 - ky
            win = (up == ry)
            obs.append((i, minute, bi, fill, win))
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(markets)}", flush=True)

    print(f"observations: {len(obs)}  (skipped {skipped} markets)", flush=True)
    json.dump(obs, open(os.path.join(CACHE, "obs.json"), "w"))

    # ── scenarios ──────────────────────────────────────────────────────────
    scenarios = []
    for slip in (0.00, 0.01, 0.02, 0.03):
        for fname, ffn in (("exact", fee_exact), ("ceil", fee_ceil)):
            scenarios.append((slip, fname, ffn))

    def cell_gate(cell, slip, ffn, min_edge=0.0):
        # frozen-table expected net edge under this cost model
        c = cell["avg_fill"] + slip
        return cell["win_rate"] - c - ffn(c) > min_edge

    print("\n=== TRUE OOS (Jul 3 16:00Z -> now), frozen Feb-May table ===")
    hdr = f"{'slip':>5} {'fee':>6} {'gate':>6} {'trades':>7} {'win%':>6} {'edge/ct':>8} {'SEwin':>6} {'ROI%':>7}"
    print(hdr)
    print("-" * len(hdr))
    results = []
    for slip, fname, ffn in scenarios:
        for gate_name in ("naive", "aware"):
            pnl, costs, wins, n = [], [], 0, 0
            by_market = {}
            for (mi, minute, bi, fill, win) in obs:
                cell = tbl.get((minute, bi))
                if cell is None or cell["n"] < 30:
                    continue
                if gate_name == "naive":
                    if not cell_gate(cell, 0.0, fee_exact):
                        continue
                else:
                    if not cell_gate(cell, slip, ffn):
                        continue
                f = fill + slip
                if f >= 0.99:
                    continue
                fee = ffn(f)
                p = (1.0 - f - fee) if win else (-f - fee)
                pnl.append(p)
                costs.append(f + fee)
                wins += int(win)
                n += 1
                by_market.setdefault(mi, []).append(p)
            if n == 0:
                print(f"{slip*100:>4.0f}c {fname:>6} {gate_name:>6} {'0':>7}")
                continue
            edge = sum(pnl) / n
            wr = wins / n
            # cluster-robust SE at the market (settlement) level
            mk = [sum(v) / len(v) for v in by_market.values()]
            mu = sum(mk) / len(mk)
            se = (sum((x - mu) ** 2 for x in mk) / max(len(mk) - 1, 1)) ** 0.5 / max(len(mk), 1) ** 0.5
            roi = edge / (sum(costs) / n) * 100.0
            results.append({"slip": slip, "fee": fname, "gate": gate_name, "n": n,
                            "win": wr, "edge": edge, "se_mkt": se,
                            "markets": len(by_market), "roi": roi})
            print(f"{slip*100:>4.0f}c {fname:>6} {gate_name:>6} {n:>7} {wr*100:>5.1f} "
                  f"{edge*100:>+7.2f}c {se*100:>5.2f} {roi:>+6.1f}", flush=True)

    json.dump(results, open(os.path.join(CACHE, "results.json"), "w"), indent=1)

    # per-day breakdown at the walk-forward baseline (slip 0, exact, naive gate)
    print("\nper-day, baseline scenario (slip=0, exact fee, naive gate):")
    day_pnl = {}
    for (mi, minute, bi, fill, win) in obs:
        cell = tbl.get((minute, bi))
        if cell is None or cell["n"] < 30 or not cell_gate(cell, 0.0, fee_exact):
            continue
        m = markets[mi]
        day = m["close_time"][:10]
        fee = fee_exact(fill)
        p = (1.0 - fill - fee) if win else (-fill - fee)
        d = day_pnl.setdefault(day, [0, 0, 0.0])
        d[0] += 1
        d[1] += int(win)
        d[2] += p
    for day in sorted(day_pnl):
        n, w, p = day_pnl[day]
        print(f"  {day}: n={n:5d} win={w/n*100:5.1f}% edge={p/n*100:+6.2f}c/ct")


if __name__ == "__main__":
    main()
