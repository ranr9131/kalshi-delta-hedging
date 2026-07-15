"""
Corrected walk-forward on the 2D table: no lookahead, executable ask/bid
fills, ceil fee. Train table on first 70% of markets, freeze, evaluate the
train-flagged cells on the last 30%.

Answers: does ANY cell of the (minute, |move|) table retain real,
out-of-sample, executable edge once the pipeline is honest?
"""

import math
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, "/Users/leolee/Desktop/kalshi-delta-hedging")
import btc_data
import kalshi_client

BUCKETS = [(0.000, 0.05), (0.050, 0.10), (0.100, 0.20), (0.200, 0.50), (0.500, 1e9)]
BUCKET_LABELS = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]


def bucket(pct):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= pct < hi:
            return i
    return len(BUCKETS) - 1


def fee_ceil(c):
    return math.ceil(0.07 * c * (1.0 - c) * 100.0) / 100.0


def build_obs(markets, btc):
    obs = []
    for i, m in enumerate(markets):
        oi, ci, res = m.get("open_time", ""), m.get("close_time", ""), m.get("result", "")
        if not oi or not ci or res not in ("yes", "no"):
            continue
        t0 = int(datetime.fromisoformat(oi.replace("Z", "+00:00")).timestamp())
        ry = res == "yes"
        b0 = btc_data.lookup(btc, t0)
        if b0 is None:
            continue
        candles = kalshi_client.fetch_candlesticks(m["ticker"], oi, ci)
        if not candles:
            continue
        for minute in range(1, 15):
            t = t0 + minute * 60
            bt = btc_data.lookup(btc, t)
            row = kalshi_client.get_quotes_at(candles, t)
            if bt is None or row is None:
                continue
            ky = row["yes_close"]
            if not (0.01 < ky < 0.99):
                continue
            up = bt > b0
            pct = abs(bt - b0) / b0 * 100.0
            f_exec = row.get("ask", ky) if up else 1.0 - row.get("bid", ky)
            if not (0.01 < f_exec < 0.99):
                continue
            obs.append((i, minute, bucket(pct), f_exec, up == ry))
    return obs


def main():
    markets = kalshi_client.fetch_settled_markets(days=90)
    markets = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    markets.sort(key=lambda m: m["open_time"])
    ts = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in markets]
    btc = btc_data.fetch_btc_prices(min(ts) - 600, max(ts) + 1800)

    n_train = int(len(markets) * 0.70)
    print(f"markets: {len(markets)}  train: {n_train}  test: {len(markets) - n_train}")
    print(f"train: {markets[0]['open_time'][:10]} -> {markets[n_train-1]['open_time'][:10]}")
    print(f"test:  {markets[n_train]['open_time'][:10]} -> {markets[-1]['open_time'][:10]}", flush=True)

    train_obs = build_obs(markets[:n_train], btc)
    test_obs = build_obs(markets[n_train:], btc)
    print(f"obs: train {len(train_obs)}  test {len(test_obs)}", flush=True)

    # frozen train table: win rate + avg executable fill per cell
    cell = defaultdict(lambda: [0, 0, 0.0])   # n, wins, fill_sum
    for (_, minute, bi, f, win) in train_obs:
        c = cell[(minute, bi)]
        c[0] += 1
        c[1] += int(win)
        c[2] += f

    def train_net(k):
        n, w, fs = cell[k]
        if n < 100:
            return None
        wr, af = w / n, fs / n
        return wr - af - fee_ceil(af)

    for hurdle in (0.00, 0.01, 0.02):
        flagged = {k for k in cell if (train_net(k) or -1) > hurdle}
        pnl, n, wins = 0.0, 0, 0
        by_mkt = defaultdict(float)
        for (mi, minute, bi, f, win) in test_obs:
            if (minute, bi) not in flagged:
                continue
            fee = fee_ceil(f)
            p = (1.0 - f - fee) if win else (-f - fee)
            pnl += p
            n += 1
            wins += int(win)
            by_mkt[mi] += p
        if n == 0:
            print(f"hurdle {hurdle*100:.0f}c: flagged cells {len(flagged)}, no trades")
            continue
        vals = list(by_mkt.values())
        mu = sum(vals) / len(vals)
        se = (sum((x - mu) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5 / len(vals) ** 0.5
        print(f"hurdle {hurdle*100:>2.0f}c: cells {len(flagged):>2}  test trades {n:>6}  "
              f"win {wins/n*100:5.1f}%  net edge {pnl/n*100:+6.2f}c/ct  "
              f"(mkt-SE {se*100:.2f}c, {len(vals)} mkts)", flush=True)

    # per-cell detail for cells flagged at the 1c hurdle
    print("\nper-cell (train-flagged > +1c):")
    print(f"{'cell':>18} {'train_n':>7} {'train_net':>9} | {'test_n':>6} {'test_win':>8} {'test_net':>8}")
    tstat = defaultdict(lambda: [0, 0, 0.0])
    for (_, minute, bi, f, win) in test_obs:
        t = tstat[(minute, bi)]
        t[0] += 1
        t[1] += int(win)
        t[2] += (1.0 - f - fee_ceil(f)) if win else (-f - fee_ceil(f))
    for k in sorted(cell, key=lambda k: -(train_net(k) or -1)):
        tn = train_net(k)
        if tn is None or tn <= 0.01:
            continue
        n, w, p = tstat[k]
        tw = f"{w/n*100:7.1f}%" if n else "      -"
        tp = f"{p/n*100:+7.2f}c" if n else "       -"
        print(f"T+{k[0]:>2} {BUCKET_LABELS[k[1]]:>12} {cell[k][0]:>7} {tn*100:>+8.2f}c | {n:>6} {tw} {tp}")


if __name__ == "__main__":
    main()
