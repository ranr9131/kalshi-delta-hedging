"""
Classic TA indicators as Kalshi KXBTC15M entry signals, honest harness:
lag-corrected BTC minutes, executable ask/bid fills, exact fee, cluster SE.

Signals at decision minute t (indicator computed on candles CLOSED by t):
  RSI(14) 1-min   — reversion: RSI<30 -> buy YES, RSI>70 -> buy NO
                  — momentum:  RSI>70 -> buy YES, RSI<30 -> buy NO
  Bollinger z(20) — reversion: z<-2 -> YES, z>+2 -> NO
  EMA9 vs EMA21   — trend: EMA9>EMA21 -> YES else NO (all obs)
  MACD(12,26,9)   — histogram sign -> side
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/Users/leolee/Desktop/kalshi-delta-hedging")
import btc_data
import kalshi_client


def fee(c):
    return 0.07 * c * (1 - c)


def cluster_se(bw):
    v = [sum(x) / len(x) for x in bw.values()]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5


def build_indicators(btc):
    """Indicator value keyed by candle END ts (start_key + 60)."""
    keys = sorted(int(k) for k in btc)
    closes = [btc[str(k)] for k in keys]
    n = len(keys)
    rsi = {}
    bbz = {}
    ema_tr = {}
    macd_h = {}
    # RSI(14) Wilder
    au = ad = None
    e9 = e21 = e12 = e26 = sig = None
    a9, a21, a12, a26, asig = 2 / 10, 2 / 22, 2 / 13, 2 / 27, 2 / 10
    window = []
    for i in range(n):
        c = closes[i]
        end_ts = keys[i] + 60
        contig = i > 0 and keys[i] - keys[i - 1] == 60
        if i > 0 and contig:
            ch = c - closes[i - 1]
            u, d = max(ch, 0.0), max(-ch, 0.0)
            if au is None:
                au, ad = u, d
            else:
                au = (au * 13 + u) / 14
                ad = (ad * 13 + d) / 14
            if i >= 15:
                rsi[end_ts] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
        window.append(c)
        if len(window) > 20:
            window.pop(0)
        if len(window) == 20:
            m = sum(window) / 20
            sd = (sum((x - m) ** 2 for x in window) / 20) ** 0.5
            if sd > 0:
                bbz[end_ts] = (c - m) / sd
        e9 = c if e9 is None else e9 + a9 * (c - e9)
        e21 = c if e21 is None else e21 + a21 * (c - e21)
        e12 = c if e12 is None else e12 + a12 * (c - e12)
        e26 = c if e26 is None else e26 + a26 * (c - e26)
        if i >= 26:
            ema_tr[end_ts] = e9 > e21
            macd = e12 - e26
            sig = macd if sig is None else sig + asig * (macd - sig)
            macd_h[end_ts] = macd - sig
    return rsi, bbz, ema_tr, macd_h


def main():
    markets = kalshi_client.fetch_settled_markets(days=90)
    markets = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    ts = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in markets]
    btc = btc_data.fetch_btc_prices(min(ts) - 90000, max(ts) + 1800)
    rsi, bbz, ema_tr, macd_h = build_indicators(btc)
    print(f"indicator points: rsi={len(rsi):,} bbz={len(bbz):,}")

    obs = []
    for mi, m in enumerate(markets):
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        ry = m["result"] == "yes"
        candles = kalshi_client.fetch_candlesticks(m["ticker"], m["open_time"], m["close_time"])
        if not candles:
            continue
        for minute in range(1, 14):
            t = t0 + minute * 60
            row = kalshi_client.get_quotes_at(candles, t)
            if row is None or row.get("ask") is None or row.get("bid") is None:
                continue
            if not (0.01 < row["yes_close"] < 0.99):
                continue
            obs.append({"mi": mi, "yes_won": ry,
                        "ye": row["ask"], "ne": 1 - row["bid"],
                        "rsi": rsi.get(t), "bbz": bbz.get(t),
                        "ema": ema_tr.get(t), "mh": macd_h.get(t)})
    print(f"obs: {len(obs):,}")

    def pnl(c, won):
        return (1 - c - fee(c)) if won else (-c - fee(c))

    def run(name, side_fn):
        bw = defaultdict(list)
        for o in obs:
            s = side_fn(o)
            if s is None:
                continue
            c = o["ye"] if s else o["ne"]
            if not (0.03 < c < 0.97):
                continue
            bw[o["mi"]].append(pnl(c, o["yes_won"] == s))
        n = sum(len(v) for v in bw.values())
        if n < 100:
            print(f"  {name:<34} n={n} (too few)")
            return
        e = sum(x for v in bw.values() for x in v) / n
        print(f"  {name:<34} n={n:>6} edge={e*100:+6.2f}c ±{cluster_se(bw)*100:.2f}")

    print("\nindicator strategies (buy at executable ask, exact fee):")
    run("RSI<30 rev->YES / >70 rev->NO",
        lambda o: True if (o["rsi"] is not None and o["rsi"] < 30) else
                  (False if (o["rsi"] is not None and o["rsi"] > 70) else None))
    run("RSI>70 mom->YES / <30 mom->NO",
        lambda o: True if (o["rsi"] is not None and o["rsi"] > 70) else
                  (False if (o["rsi"] is not None and o["rsi"] < 30) else None))
    run("RSI extreme 20/80 reversion",
        lambda o: True if (o["rsi"] is not None and o["rsi"] < 20) else
                  (False if (o["rsi"] is not None and o["rsi"] > 80) else None))
    run("Bollinger z<-2 ->YES / z>2 ->NO",
        lambda o: True if (o["bbz"] is not None and o["bbz"] < -2) else
                  (False if (o["bbz"] is not None and o["bbz"] > 2) else None))
    run("Bollinger momentum (z>2->YES)",
        lambda o: True if (o["bbz"] is not None and o["bbz"] > 2) else
                  (False if (o["bbz"] is not None and o["bbz"] < -2) else None))
    run("EMA9>EMA21 trend follow",
        lambda o: o["ema"])
    run("MACD hist sign follow",
        lambda o: (o["mh"] > 0) if o["mh"] is not None else None)
    run("MACD hist sign fade",
        lambda o: (o["mh"] < 0) if o["mh"] is not None else None)


if __name__ == "__main__":
    main()
