"""
Strategies 5, 6, 7 on the full corrected KXBTC15M corpus (Apr 26 - Jul 3,
~5.8k markets, executable ask/bid fills, exact fee, no lookahead).

  S5  systematic NO-side  — is NO structurally cheap (retail bullish bias)?
      Buy NO at exec ask at fixed minutes, unconditionally. + YES mirror.
  S6  hour-of-day gating  — continuation taker net edge by 4h UTC block.
  S7  vol clustering      — early-window continuation entries conditioned on
      the PREVIOUS window's |move| quartile.
"""

import math
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/Users/leolee/Desktop/kalshi-delta-hedging")
import btc_data
import kalshi_client


def fee(c):
    return 0.07 * c * (1.0 - c)


def cluster_se(by_mkt):
    vals = [sum(v) / len(v) for v in by_mkt.values()]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return (sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 / len(vals) ** 0.5


def main():
    markets = kalshi_client.fetch_settled_markets(days=90)
    markets = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    markets.sort(key=lambda m: m["open_time"])
    ts = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in markets]
    btc = btc_data.fetch_btc_prices(min(ts) - 600, max(ts) + 1800)

    obs = []           # dicts
    prev_move = {}     # window open ts -> previous window |move| pct
    for m in markets:
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        a = btc_data.lookup(btc, t0 - 900)
        b = btc_data.lookup(btc, t0)
        if a and b:
            prev_move[t0] = abs(b - a) / a * 100

    for mi, m in enumerate(markets):
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        ry = m["result"] == "yes"
        b0 = btc_data.lookup(btc, t0)
        if b0 is None:
            continue
        candles = kalshi_client.fetch_candlesticks(m["ticker"], m["open_time"], m["close_time"])
        if not candles:
            continue
        for minute in range(1, 14):
            t = t0 + minute * 60
            bt = btc_data.lookup(btc, t)
            row = kalshi_client.get_quotes_at(candles, t)
            if bt is None or row is None:
                continue
            ky = row["yes_close"]
            if not (0.01 < ky < 0.99):
                continue
            ask, bid = row.get("ask"), row.get("bid")
            if ask is None or bid is None:
                continue
            yes_exec = ask
            no_exec = 1.0 - bid
            obs.append({"mi": mi, "minute": minute,
                        "hour": datetime.utcfromtimestamp(t).hour,
                        "up": bt > b0, "yes_won": ry,
                        "yes_exec": yes_exec, "no_exec": no_exec,
                        "pmove": prev_move.get(t0)})
    print(f"obs: {len(obs):,}")

    def pnl(cost, won):
        return (1 - cost - fee(cost)) if won else (-cost - fee(cost))

    # ── S5: unconditional side bets at fixed minutes ────────────────────────
    print("\n── S5: unconditional side taker (buy at exec ask)")
    print(f"{'minute':>7} {'NO edge':>9} {'SE':>5} {'YES edge':>9} {'SE':>5} {'n':>6}")
    for minute in (1, 3, 5, 7, 9, 11, 13):
        sel = [o for o in obs if o["minute"] == minute
               and 0.03 < o["no_exec"] < 0.97 and 0.03 < o["yes_exec"] < 0.97]
        if len(sel) < 200:
            continue
        bw_n, bw_y = defaultdict(list), defaultdict(list)
        for o in sel:
            bw_n[o["mi"]].append(pnl(o["no_exec"], not o["yes_won"]))
            bw_y[o["mi"]].append(pnl(o["yes_exec"], o["yes_won"]))
        en = sum(x for v in bw_n.values() for x in v) / len(sel)
        ey = sum(x for v in bw_y.values() for x in v) / len(sel)
        print(f"  T+{minute:>3} {en*100:>+8.2f}c {cluster_se(bw_n)*100:>4.1f} "
              f"{ey*100:>+8.2f}c {cluster_se(bw_y)*100:>4.1f} {len(sel):>6}")

    # ── S6: continuation taker by hour block ────────────────────────────────
    print("\n── S6: continuation taker by UTC hour block (all minutes)")
    blocks = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    for lo, hi in blocks:
        sel = [o for o in obs if lo <= o["hour"] < hi]
        bw = defaultdict(list)
        for o in sel:
            c = o["yes_exec"] if o["up"] else o["no_exec"]
            if not (0.03 < c < 0.97):
                continue
            won = o["yes_won"] == o["up"]
            bw[o["mi"]].append(pnl(c, won))
        n = sum(len(v) for v in bw.values())
        if n < 300:
            continue
        e = sum(x for v in bw.values() for x in v) / n
        print(f"  {lo:02d}-{hi:02d}Z: n={n:>6} edge={e*100:+6.2f}c ±{cluster_se(bw)*100:.2f}")

    # ── S7: early continuation by previous-window vol quartile ─────────────
    print("\n── S7: minute 1-2 continuation by PREV window |move| quartile")
    early = [o for o in obs if o["minute"] <= 2 and o["pmove"] is not None]
    pm = sorted(o["pmove"] for o in early)
    qs = [pm[int(len(pm) * q)] for q in (0.25, 0.5, 0.75)]
    for qi, (lo, hi) in enumerate([(0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 99)]):
        sel = [o for o in early if lo <= o["pmove"] < hi]
        bw = defaultdict(list)
        for o in sel:
            c = o["yes_exec"] if o["up"] else o["no_exec"]
            if not (0.03 < c < 0.97):
                continue
            bw[o["mi"]].append(pnl(c, o["yes_won"] == o["up"]))
        n = sum(len(v) for v in bw.values())
        if n < 100:
            continue
        e = sum(x for v in bw.values() for x in v) / n
        print(f"  Q{qi+1} (prev |move| {lo:.2f}-{hi:.2f}%): n={n:>5} "
              f"edge={e*100:+6.2f}c ±{cluster_se(bw)*100:.2f}")


if __name__ == "__main__":
    main()
