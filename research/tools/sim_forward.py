"""
Forward paper-sim of the cost-aware 4c strategy on never-seen KXBTC15M data.

Re-runnable: each run refetches newly settled markets since 2026-07-03 16:00Z
(candles/BTC cached incrementally), replays the FROZEN Feb-May production table
with AGGRESSIVE costs, and simulates a bankroll:

  - gate  : frozen-table net edge > 0 under stressed costs (fill+2c slip, ceil fee)
  - fill  : candle close + 2c slippage, fee = ceil(0.07*c*(1-c)) per contract
  - sizing: quarter-Kelly f = (p - c_eff)/(1 - c_eff)/4 on current bankroll,
            capped at $15/trade and $8 wagered per 15-min window (live caps)
  - bank  : starts at $120 (current account)

Usage:
  python3 sim_forward.py            # one summary line (for the recurring loop)
  python3 sim_forward.py --detail   # + per-day table and equity stats
"""

import json
import os
import sys
from datetime import datetime, timezone

import oos_test as base

CACHE = base.CACHE
START_BANK = 120.0
SLIP = 0.02
MAX_STAKE = 15.0
MAX_WINDOW = 8.0
KELLY_FRAC = 0.25


def fetch_markets_fresh():
    """Always refetch the settled-market list (it grows); overwrite cache."""
    p = os.path.join(CACHE, "markets_oos.json")
    if os.path.exists(p):
        os.remove(p)
    return base.fetch_markets()


def refresh_recent_btc():
    """Drop cached BTC files for the last 2 UTC days (they were partial)."""
    now = int(datetime.now(timezone.utc).timestamp())
    for d in (0, 1):
        ds = datetime.fromtimestamp(now - d * 86400, tz=timezone.utc).strftime("%Y%m%d")
        p = os.path.join(CACHE, f"btc_{ds}.json")
        if os.path.exists(p):
            os.remove(p)


def build_obs(markets, btc):
    obs = []
    for i, m in enumerate(markets):
        oi, ci, res = m.get("open_time", ""), m.get("close_time", ""), m.get("result", "")
        if not oi or not ci or res not in ("yes", "no"):
            continue
        t0 = int(datetime.fromisoformat(oi.replace("Z", "+00:00")).timestamp())
        ry = res == "yes"
        b0 = base.lookup(btc, t0)
        if b0 is None:
            continue
        candles = base.fetch_candles(m["ticker"], oi, ci)
        if not candles:
            continue
        for minute in range(1, 15):
            t = t0 + minute * 60
            bt = base.lookup(btc, t)
            ky = base.yes_at(candles, t)
            if bt is None or ky is None or not (0.01 < ky < 0.99):
                continue
            pct = abs(bt - b0) / b0 * 100.0
            up = bt > b0
            fill = ky if up else 1.0 - ky
            win = (up == ry)
            obs.append({"mi": i, "t": t, "day": ci[:10], "minute": minute,
                        "bucket": base.get_bucket(pct), "fill": fill, "win": win})
    obs.sort(key=lambda o: o["t"])
    return obs


FLAT_STAKE = 4.0


def simulate(obs, tbl, sizing):
    """sizing: 'flat' ($4/trade) or 'kelly' (1/4-Kelly on BLENDED probability).

    Blend: p_est = (table win_rate + market-implied fill) / 2 — first-order
    correction for the measured adverse selection (table is ~right when fill
    agrees with it, market is ~right when fill is far from the table).
    Gate and sizing both use p_est, so cheap-fill cells where the market
    strongly disagrees get small stakes instead of table-Kelly oversizing.
    """
    bank = START_BANK
    peak = START_BANK
    max_dd = 0.0
    window_wagered = {}
    trades = wins = contracts_total = 0
    edge_sum = 0.0
    day_stats = {}

    for o in obs:
        cell = tbl.get((o["minute"], o["bucket"]))
        if cell is None or cell["n"] < 30:
            continue
        f = o["fill"] + SLIP
        if f >= 0.99:
            continue
        fee = base.fee_ceil(f)
        c_eff = f + fee
        p_est = (cell["win_rate"] + o["fill"]) / 2.0
        if p_est - c_eff <= 0:
            continue
        if sizing == "kelly":
            kelly = (p_est - c_eff) / (1.0 - c_eff)
            stake = min(kelly * KELLY_FRAC * bank, MAX_STAKE,
                        MAX_WINDOW - window_wagered.get(o["mi"], 0.0))
        else:
            stake = min(FLAT_STAKE,
                        MAX_WINDOW - window_wagered.get(o["mi"], 0.0))
        n_ct = int(stake // c_eff)
        if n_ct < 1:
            continue
        cost = n_ct * c_eff
        window_wagered[o["mi"]] = window_wagered.get(o["mi"], 0.0) + cost
        pnl_ct = (1.0 - c_eff) if o["win"] else (-c_eff)
        pnl = n_ct * pnl_ct
        bank += pnl
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        trades += 1
        wins += int(o["win"])
        contracts_total += n_ct
        edge_sum += pnl_ct
        d = day_stats.setdefault(o["day"], [0, 0, 0.0])
        d[0] += 1
        d[1] += int(o["win"])
        d[2] += pnl
        if bank <= 1.0:
            break
    return {"bank": bank, "max_dd": max_dd, "trades": trades, "wins": wins,
            "contracts": contracts_total, "edge": (edge_sum / trades if trades else 0.0),
            "days": day_stats}


def main():
    detail = "--detail" in sys.argv
    tbl = base.load_table()
    refresh_recent_btc()
    markets = fetch_markets_fresh()
    ts_all = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
              for m in markets if m.get("open_time")]
    btc = base.fetch_btc(min(ts_all) - 600, max(ts_all) + 1800)
    obs = build_obs(markets, btc)

    now = datetime.now(timezone.utc).strftime("%m-%d %H:%MZ")
    line = f"SIM {now} | mkts={len(markets)} obs={len(obs)}"
    results = {}
    for sizing in ("flat", "kelly"):
        r = simulate(obs, tbl, sizing)
        results[sizing] = r
        wr = r["wins"] / r["trades"] * 100 if r["trades"] else 0.0
        ret = (r["bank"] / START_BANK - 1) * 100
        line += (f" | {sizing}: n={r['trades']} win={wr:.1f}% "
                 f"edge={r['edge']*100:+.2f}c bank=${r['bank']:.2f} "
                 f"({ret:+.1f}%) DD={r['max_dd']*100:.1f}%")
    print(line, flush=True)

    if detail:
        for sizing in ("flat", "kelly"):
            r = results[sizing]
            print(f"\nper-day [{sizing}] (blend gate, 2c slip, ceil fee, $120 start):")
            for day in sorted(r["days"]):
                n, w, p = r["days"][day]
                print(f"  {day}: trades={n:4d} win={w/max(n,1)*100:5.1f}% pnl=${p:+8.2f}")


if __name__ == "__main__":
    main()
