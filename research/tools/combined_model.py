"""
The 'combine everything' test, done right: one model, all features, all
interactions, chronological 70/30 split, executable fills, exact fee.

Features (all computable at decision time, lag-corrected):
  signed move since open (%), minute, RSI(14), Bollinger z(20),
  MACD histogram, EMA9>EMA21, hour sin/cos, prev-window signed move (%)
Models: logistic regression (linear combine) and HistGradientBoosting
(nonlinear interactions). Predict P(yes wins); trade whichever side has
model edge > buffer vs its executable ask.
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
import math

import numpy as np

sys.path.insert(0, "/Users/leolee/Desktop/kalshi-delta-hedging")
import btc_data
import kalshi_client
sys.path.insert(0, "/private/tmp/claude-501/-Users-leolee-Desktop-kalshi-delta-hedging/ab8d688d-a288-495d-9b30-2feb237354aa/scratchpad")
from indicator_tests import build_indicators

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def fee(c):
    return 0.07 * c * (1 - c)


def cluster_se(bw):
    v = [sum(x) / len(x) for x in bw.values()]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5


def main():
    markets = kalshi_client.fetch_settled_markets(days=90)
    markets = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    markets.sort(key=lambda m: m["open_time"])
    ts = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in markets]
    btc = btc_data.fetch_btc_prices(min(ts) - 90000, max(ts) + 1800)
    rsi, bbz, ema_tr, macd_h = build_indicators(btc)

    X, y, meta = [], [], []
    for mi, m in enumerate(markets):
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        ry = 1 if m["result"] == "yes" else 0
        b0 = btc_data.lookup(btc, t0)
        pa = btc_data.lookup(btc, t0 - 900)
        if b0 is None:
            continue
        pmove = ((b0 - pa) / pa * 100) if pa else 0.0
        candles = kalshi_client.fetch_candlesticks(m["ticker"], m["open_time"], m["close_time"])
        if not candles:
            continue
        for minute in range(1, 14):
            t = t0 + minute * 60
            bt = btc_data.lookup(btc, t)
            row = kalshi_client.get_quotes_at(candles, t)
            if bt is None or row is None or row.get("ask") is None or row.get("bid") is None:
                continue
            if not (0.01 < row["yes_close"] < 0.99):
                continue
            r = rsi.get(t)
            z = bbz.get(t)
            mh = macd_h.get(t)
            et = ema_tr.get(t)
            if r is None or z is None or mh is None or et is None:
                continue
            hour = datetime.fromtimestamp(t, tz=timezone.utc).hour
            X.append([
                (bt - b0) / b0 * 100,          # signed move since open
                minute / 13.0,
                (r - 50) / 50.0,               # RSI centered
                z,                             # bollinger z
                math.tanh(mh / 20.0),          # macd hist squashed
                1.0 if et else 0.0,            # ema trend
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                pmove,                         # prev window signed move
            ])
            y.append(ry)
            meta.append((mi, row["ask"], 1 - row["bid"]))
    X = np.array(X)
    y = np.array(y)
    n_mkts = meta[-1][0]
    split_mi = None
    # chronological split at 70% of MARKETS (not obs)
    cut = int(n_mkts * 0.70)
    tr = np.array([m[0] <= cut for m in meta])
    te = ~tr
    print(f"obs: {len(y):,}  train {tr.sum():,}  test {te.sum():,}")

    for name, model in (
        ("logistic", LogisticRegression(C=1.0, max_iter=2000)),
        ("gbm", HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                               learning_rate=0.05,
                                               validation_fraction=0.15,
                                               early_stopping=True,
                                               random_state=0)),
    ):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        # calibration on test
        print(f"\n[{name}] test calibration:")
        bins = np.linspace(0, 1, 6)
        for i in range(5):
            m2 = (p >= bins[i]) & (p < bins[i + 1])
            if m2.sum() > 200:
                print(f"  pred {bins[i]:.1f}-{bins[i+1]:.1f}: n={m2.sum():>6} "
                      f"avg_pred={p[m2].mean():.3f} actual={y[te][m2].mean():.3f}")
        # trading vs executable asks
        idx = np.where(te)[0]
        for buf in (0.0, 0.01, 0.02, 0.03):
            bw = defaultdict(list)
            n = w = 0
            for j, pi in zip(idx, p):
                mi, ye, ne = meta[j]
                for side, pw, c in ((1, pi, ye), (0, 1 - pi, ne)):
                    if not (0.03 < c < 0.97):
                        continue
                    if pw - c - fee(c) <= buf:
                        continue
                    won = (y[j] == side)
                    bw[mi].append((1 - c - fee(c)) if won else (-c - fee(c)))
                    n += 1
                    w += int(won)
            if n == 0:
                print(f"  buf {buf*100:.0f}c: no trades")
                continue
            e = sum(x for v in bw.values() for x in v) / n
            print(f"  buf {buf*100:.0f}c: trades={n:>6} win={w/n*100:5.1f}% "
                  f"edge={e*100:+6.2f}c ±{cluster_se(bw)*100:.2f}")


if __name__ == "__main__":
    main()
