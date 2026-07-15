"""
Honest-execution replay: same frozen-table strategy, same unseen Jul 3-7 data,
but fills at the REAL top-of-book price from Kalshi candlesticks:
  buy YES (up-continuation)  -> yes_ask close of the decision minute
  buy NO  (down-continuation)-> 1 - yes_bid close
This removes the optimistic trade-print fill assumption entirely. Remaining
slippage scenarios (+1c/+2c) represent walking depth beyond top-of-book.

Outputs: adverse-selection diagnostic on executable prices, scenario grid,
and the two-arm bankroll sim (flat $4 / quarter-Kelly on blended p).
"""

import csv
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta

import requests

import oos_test as base

CACHE = base.CACHE
START_BANK = 120.0
MAX_STAKE = 15.0
MAX_WINDOW = 8.0
KELLY_FRAC = 0.25
FLAT_STAKE = 4.0


def fetch_candles2(ticker, open_iso, close_iso):
    p = os.path.join(CACHE, f"candles2_{ticker}.json")
    if os.path.exists(p):
        return json.load(open(p))
    o = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    c = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    params = {"start_ts": int((o - timedelta(minutes=1)).timestamp()),
              "end_ts": int((c + timedelta(minutes=1)).timestamp()),
              "period_interval": 1}
    try:
        d = base._get(f"/series/{base.SERIES}/markets/{ticker}/candlesticks", params)
    except Exception as e:
        print(f"  candles2 fail {ticker}: {e}", flush=True)
        return []
    res = []
    for k in d.get("candlesticks", []):
        pr = k.get("price", {})
        ask = k.get("yes_ask", {})
        bid = k.get("yes_bid", {})
        yc = pr.get("close_dollars") or pr.get("close")
        ac = ask.get("close_dollars") or ask.get("close")
        bc = bid.get("close_dollars") or bid.get("close")
        if yc is None or ac is None or bc is None:
            continue
        res.append({"ts": k["end_period_ts"], "yes_close": float(yc),
                    "ask": float(ac), "bid": float(bc)})
    res.sort(key=lambda x: x["ts"])
    json.dump(res, open(p, "w"))
    time.sleep(0.12)
    return res


def at_ts(candles, t):
    row = None
    for k in candles:
        if k["ts"] <= t:
            row = k
        else:
            break
    return row


def build_obs():
    markets = base.fetch_markets()
    ts_all = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
              for m in markets if m.get("open_time")]
    btc = base.fetch_btc(min(ts_all) - 600, max(ts_all) + 1800)
    obs = []
    for i, m in enumerate(markets):
        oi, ci, res = m.get("open_time", ""), m.get("close_time", ""), m.get("result", "")
        if not oi or not ci or res not in ("yes", "no"):
            continue
        t0 = int(datetime.fromisoformat(oi.replace("Z", "+00:00")).timestamp())
        ry = res == "yes"
        # NO-LOOKAHEAD: Coinbase candles are keyed by START ts, so the close
        # keyed at t is the price at t+60. Price knowable AT t = close of the
        # candle keyed t-60. (The original pipeline used t — 60s of future.)
        b0 = base.lookup(btc, t0 - 60)
        if b0 is None:
            continue
        candles = fetch_candles2(m["ticker"], oi, ci)
        if not candles:
            continue
        for minute in range(1, 15):
            t = t0 + minute * 60
            bt = base.lookup(btc, t - 60)
            row = at_ts(candles, t)
            if bt is None or row is None:
                continue
            ky = row["yes_close"]
            if not (0.01 < ky < 0.99):
                continue
            up = bt > b0
            pct = abs(bt - b0) / b0 * 100.0
            fill_trade = ky if up else 1.0 - ky
            fill_exec = row["ask"] if up else 1.0 - row["bid"]
            spread = row["ask"] - row["bid"]
            obs.append({"mi": i, "t": t, "day": ci[:10], "minute": minute,
                        "bucket": base.get_bucket(pct), "fill": fill_trade,
                        "exec": fill_exec, "spread": spread,
                        "win": (up == ry)})
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(markets)}", flush=True)
    obs.sort(key=lambda o: o["t"])
    json.dump(obs, open(os.path.join(CACHE, "obs2.json"), "w"))
    return obs


def simulate(obs, tbl, sizing, slip):
    bank = START_BANK
    peak, max_dd = START_BANK, 0.0
    window_wagered = {}
    trades = wins = contracts = 0
    edge_sum = 0.0
    days = {}
    for o in obs:
        cell = tbl.get((o["minute"], o["bucket"]))
        if cell is None or cell["n"] < 30:
            continue
        f = o["exec"] + slip
        if f >= 0.99 or f <= 0.01:
            continue
        fee = base.fee_ceil(f)
        c_eff = f + fee
        p_est = (cell["win_rate"] + o["fill"]) / 2.0
        if p_est - c_eff <= 0:
            continue
        if sizing == "kelly":
            k = (p_est - c_eff) / (1.0 - c_eff)
            stake = min(k * KELLY_FRAC * bank, MAX_STAKE,
                        MAX_WINDOW - window_wagered.get(o["mi"], 0.0))
        else:
            stake = min(FLAT_STAKE, MAX_WINDOW - window_wagered.get(o["mi"], 0.0))
        n_ct = int(stake // c_eff)
        if n_ct < 1:
            continue
        window_wagered[o["mi"]] = window_wagered.get(o["mi"], 0.0) + n_ct * c_eff
        pnl_ct = (1.0 - c_eff) if o["win"] else (-c_eff)
        bank += n_ct * pnl_ct
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        trades += 1
        wins += int(o["win"])
        contracts += n_ct
        edge_sum += pnl_ct
        d = days.setdefault(o["day"], [0, 0, 0.0])
        d[0] += 1; d[1] += int(o["win"]); d[2] += n_ct * pnl_ct
        if bank <= 1.0:
            break
    return {"bank": bank, "dd": max_dd, "n": trades, "wins": wins,
            "ct": contracts, "edge": edge_sum / trades if trades else 0.0, "days": days}


def main():
    tbl = base.load_table()
    obs = build_obs()
    print(f"obs with executable quotes: {len(obs)}", flush=True)
    sp = sorted(o["spread"] for o in obs)
    print(f"spread: median={sp[len(sp)//2]*100:.1f}c p90={sp[int(len(sp)*0.9)]*100:.1f}c")

    # 1) adverse-selection diagnostic at EXECUTABLE prices
    passing = {k for k, v in tbl.items()
               if v["n"] >= 30 and v["win_rate"] - (v["avg_fill"] + 0.02)
               - base.fee_ceil(v["avg_fill"] + 0.02) > 0}
    print(f"\n{'trade-fill bin':>14} {'n':>6} {'real_win':>8} {'tbl_win':>8} "
          f"{'avg_exec':>8} {'net@exec+ceil':>13}")
    for lo, hi in [(0, 0.40), (0.40, 0.48), (0.48, 0.55), (0.55, 0.65),
                   (0.65, 0.80), (0.80, 1.0)]:
        sel = [o for o in obs if (o["minute"], o["bucket"]) in passing
               and lo <= o["fill"] < hi and o["exec"] < 0.99]
        if not sel:
            continue
        n = len(sel)
        rw = sum(o["win"] for o in sel) / n
        tw = sum(tbl[(o["minute"], o["bucket"])]["win_rate"] for o in sel) / n
        ae = sum(o["exec"] for o in sel) / n
        net = rw - ae - base.fee_ceil(ae)
        print(f"  {lo:.2f}-{hi:.2f}   {n:>6} {rw*100:>7.1f}% {tw*100:>7.1f}% "
              f"{ae*100:>7.1f}c {net*100:>+12.2f}c")

    # 2) scenario grid at executable prices
    print(f"\n{'slip':>5} {'gate':>6} {'trades':>7} {'win%':>6} {'edge/ct':>8}")
    for slip in (0.00, 0.01, 0.02):
        for gate in ("naive", "aware"):
            pnl, w, n = 0.0, 0, 0
            for o in obs:
                cell = tbl.get((o["minute"], o["bucket"]))
                if cell is None or cell["n"] < 30:
                    continue
                c_t = cell["avg_fill"] + (0.0 if gate == "naive" else slip + 0.02)
                if cell["win_rate"] - c_t - base.fee_ceil(c_t) <= 0:
                    continue
                f = o["exec"] + slip
                if f >= 0.99:
                    continue
                fee = base.fee_ceil(f)
                pnl += (1.0 - f - fee) if o["win"] else (-f - fee)
                w += int(o["win"]); n += 1
            if n:
                print(f"{slip*100:>4.0f}c {gate:>6} {n:>7} {w/n*100:>5.1f} {pnl/n*100:>+7.2f}c")

    # 3) bankroll sims
    print()
    for sizing in ("flat", "kelly"):
        r = simulate(obs, tbl, sizing, slip=0.0)
        wr = r["wins"] / r["n"] * 100 if r["n"] else 0
        print(f"[{sizing}] n={r['n']} ({r['ct']}ct) win={wr:.1f}% "
              f"edge={r['edge']*100:+.2f}c bank=${r['bank']:.2f} "
              f"({(r['bank']/START_BANK-1)*100:+.1f}%) DD={r['dd']*100:.1f}%")
        for day in sorted(r["days"]):
            n, w, p = r["days"][day]
            print(f"    {day}: n={n:4d} win={w/max(n,1)*100:5.1f}% pnl=${p:+8.2f}")


if __name__ == "__main__":
    main()
