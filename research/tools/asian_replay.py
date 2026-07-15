"""
STRATEGY 1 REPLAY — Asian settlement snipe on KXBTC15M, Jul 2-7 2026.

Data (all real, all recorded):
  book_log.jsonl      — Kalshi top-of-book + ladders every ~2s (AWS-local feed)
  btc1s/*             — Binance BTCUSDT 1s closes (daily zips + API for Jul 7)
  replay_markets.json — settled markets: floor_strike, result, expiration_value

Model: live/asian_pricer.py math, replayed offline. P(settle>strike) from
banked partial average + remaining-time variance, sigma from trailing 1s diffs.

Basis handling (Binance USDT vs CF BRTI USD): rolling median of
(expiration_value − binance 60s avg) over the K prior SETTLED windows —
information a live bot would have. No lookahead.

Execution realism:
  - decisions only at recorded book snapshots in the final LOOKBACK seconds
  - buy at the BEST ASK of the chosen side, fee = ceil(0.07*c*(1-c)) per ct
  - "delayed" fill mode: signal at snapshot t, fill at the NEXT snapshot's ask
    (~2s later) if still profitable-signal side — models latency + stale quotes
  - one entry per (window, side): first trigger wins; no pyramiding
Report per entry-buffer: n, win rate, net edge/contract, opps/day, and
breakdowns; cluster SE by window.
"""

import csv
import io
import json
import math
import os
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_LOG = "/Users/leolee/Desktop/kalshi-delta-hedging/live/book_log.jsonl"

LOOKBACK_S = 45           # deep lock only: banked fraction >= 25%
MIN_TTC = 3.0             # can't execute in the last 3s
BASIS_K = 8               # rolling windows for basis correction
SIGMA_WIN = 900           # trailing seconds for sigma estimate
MIN_VOL_SAMPLES = 120


def fee_ceil(c):
    # exact sub-cent fee, as confirmed from real V2 fills (4dp rounding)
    return 0.07 * c * (1.0 - c)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── BTC 1s series ───────────────────────────────────────────────────────────
def _to_sec(raw):
    t = int(raw)
    if t > 10 ** 14:      # microseconds (newer Binance dumps)
        return t // 10 ** 6
    if t > 10 ** 11:      # milliseconds
        return t // 10 ** 3
    return t


def load_btc():
    px = {}
    d = os.path.join(HERE, "btc1s")
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    for row in csv.reader(io.TextIOWrapper(f)):
                        if not row or not row[0].isdigit():
                            continue
                        px[_to_sec(row[0])] = float(row[4])
        elif fn.endswith(".csv"):
            for row in csv.reader(open(p)):
                if not row:
                    continue
                px[_to_sec(row[0])] = float(row[4])
    ts = sorted(px)
    prices = [px[t] for t in ts]
    print(f"btc 1s: {len(ts):,} seconds  {datetime.fromtimestamp(ts[0],tz=timezone.utc)} -> "
          f"{datetime.fromtimestamp(ts[-1],tz=timezone.utc)}")
    return ts, prices


class Btc:
    """O(1) rolling sigma + banked mean via prefix sums over the 1s grid."""

    def __init__(self, ts, prices):
        self.ts = ts
        self.p = prices
        n = len(ts)
        self.psum = [0.0] * (n + 1)
        for i in range(n):
            self.psum[i + 1] = self.psum[i] + prices[i]
        # 1s diffs (only across contiguous seconds)
        self.d = [0.0] * n
        self.dok = [0] * n
        for i in range(1, n):
            if ts[i] - ts[i - 1] == 1:
                self.d[i] = prices[i] - prices[i - 1]
                self.dok[i] = 1
        self.dsum = [0.0] * (n + 1)
        self.d2sum = [0.0] * (n + 1)
        self.doksum = [0] * (n + 1)
        for i in range(n):
            self.dsum[i + 1] = self.dsum[i] + self.d[i]
            self.d2sum[i + 1] = self.d2sum[i] + self.d[i] * self.d[i]
            self.doksum[i + 1] = self.doksum[i] + self.dok[i]

    def idx_le(self, t):
        i = bisect_right(self.ts, t) - 1
        return i if i >= 0 else None

    def spot(self, t):
        i = self.idx_le(t)
        if i is None or t - self.ts[i] > 5:
            return None
        return self.p[i]

    def sigma1(self, t):
        j = self.idx_le(t)
        if j is None:
            return None
        i = bisect_left(self.ts, t - SIGMA_WIN)
        n = self.doksum[j + 1] - self.doksum[i]
        if n < MIN_VOL_SAMPLES:
            return None
        s = self.dsum[j + 1] - self.dsum[i]
        s2 = self.d2sum[j + 1] - self.d2sum[i]
        m = s / n
        var = max(0.0, (s2 - n * m * m) / (n - 1))
        return max(1e-6, math.sqrt(var))

    def mean_range(self, t_lo, t_hi):
        """Mean of 1s closes with ts in [t_lo, t_hi]; (mean, count)."""
        i = bisect_left(self.ts, t_lo)
        j = bisect_right(self.ts, t_hi)
        if j <= i:
            return None, 0
        return (self.psum[j] - self.psum[i]) / (j - i), j - i


def p_up(btc, strike, close_ts, now):
    a, b = close_ts - 60.0, close_ts
    if now >= b:
        return None
    spot = btc.spot(now)
    s1 = btc.sigma1(now)
    if spot is None or s1 is None:
        return None
    if now < a:
        mu = spot
        var = (s1 ** 2) * ((a - now) + 20.0)
    else:
        banked_mean, n_banked = btc.mean_range(a, now)
        elapsed = now - a
        if elapsed >= 1.0 and n_banked < 0.8 * elapsed:
            return None
        f = min(1.0, elapsed / 60.0)
        if banked_mean is None:
            banked_mean = spot
        r = b - now
        mu = f * banked_mean + (1.0 - f) * spot
        var = (s1 ** 2) * (r ** 3) / (3.0 * 3600.0)
    if var <= 0:
        return 1.0 if mu > strike else 0.0
    return min(0.999, max(0.001, phi((mu - strike) / math.sqrt(var))))


# ── main replay ─────────────────────────────────────────────────────────────
def main():
    ts, prices = load_btc()
    btc = Btc(ts, prices)

    markets = {}
    for m in json.load(open(os.path.join(HERE, "replay_markets.json"))):
        if m.get("result") not in ("yes", "no"):
            continue
        try:
            close_ts = datetime.fromisoformat(
                m["close_time"].replace("Z", "+00:00")).timestamp()
            markets[m["ticker"]] = {
                "close": close_ts,
                "strike": float(m["floor_strike"]),
                "yes": m["result"] == "yes",
                "exp": float(m["expiration_value"]) if m.get("expiration_value") else None,
            }
        except (KeyError, TypeError, ValueError):
            continue
    print(f"settled markets with strikes: {len(markets)}")

    # rolling basis: for each window (by close), median of (exp - binance60avg)
    # over the K prior settled windows. computed once, keyed by ticker.
    closes = sorted((v["close"], k) for k, v in markets.items())
    resid = []           # (close_ts, exp - binance_avg)
    basis_for = {}
    for close_ts, tk in closes:
        prior = [r for ct, r in resid if ct < close_ts - 1][-BASIS_K:]
        basis_for[tk] = sorted(prior)[len(prior) // 2] if len(prior) >= 3 else 0.0
        v = markets[tk]
        if v["exp"] and v["exp"] > 1000:
            avg, n = btc.mean_range(close_ts - 60, close_ts - 1)
            if avg and n >= 50:
                resid.append((close_ts, v["exp"] - avg))
    bs = sorted(r for _, r in resid)
    print(f"basis (BRTI - binance60avg): median {bs[len(bs)//2]:+.2f} $  "
          f"p5 {bs[int(len(bs)*.05)]:+.2f}  p95 {bs[int(len(bs)*.95)]:+.2f}  (n={len(bs)})")

    # walk the book log; group snapshots by ticker
    snaps = defaultdict(list)
    with open(BOOK_LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tk = r.get("ticker")
            if tk not in markets:
                continue
            t = datetime.fromisoformat(r["ts"]).timestamp()
            close = markets[tk]["close"]
            if close - LOOKBACK_S <= t <= close - MIN_TTC:
                ya = r["yes_asks"][0] if r["yes_asks"] else None
                na = r["no_asks"][0] if r["no_asks"] else None
                age = r.get("book_age")
                if age is not None and age > 10:
                    continue
                snaps[tk].append((t, ya, na))
    print(f"windows with final-{LOOKBACK_S}s book coverage: {len(snaps)}")
    n_days = (closes[-1][0] - closes[0][0]) / 86400.0

    # trade log per buffer
    BUFFERS = (0.005, 0.01, 0.02, 0.03)
    results = {b: [] for b in BUFFERS}          # (ticker, side, cost, win, ttc, p, size)
    for tk, rows in snaps.items():
        v = markets[tk]
        strike_adj = v["strike"] - basis_for[tk]   # shift strike into binance terms
        rows.sort()
        done = {b: set() for b in BUFFERS}
        for i, (t, ya, na) in enumerate(rows):
            p = p_up(btc, strike_adj, v["close"], t)
            if p is None:
                continue
            for side, ask, pw in (("yes", ya, p), ("no", na, 1.0 - p)):
                if ask is None:
                    continue
                price, size = ask[0], ask[1]
                if not (0.01 <= price <= 0.99) or size < 1:
                    continue
                edge = pw - price - fee_ceil(price)
                for b in BUFFERS:
                    if side in done[b] or edge <= b:
                        continue
                    # delayed fill: next snapshot's ask on the same side
                    fill = None
                    for t2, ya2, na2 in rows[i + 1:]:
                        if t2 - t < 1.0:
                            continue
                        if t2 - t > 8.0:
                            break
                        a2 = ya2 if side == "yes" else na2
                        if a2 is None:
                            break
                        p2 = p_up(btc, strike_adj, v["close"], t2)
                        if p2 is None:
                            break
                        pw2 = p2 if side == "yes" else 1.0 - p2
                        if pw2 - a2[0] - fee_ceil(a2[0]) > 0:   # still +EV at fill time
                            fill = a2[0]
                        break
                    if fill is None:
                        continue
                    won = v["yes"] if side == "yes" else not v["yes"]
                    results[b].append((tk, side, fill, won, v["close"] - t, pw, size))
                    done[b].add(side)

    print(f"\n=== Asian snipe, delayed-fill replay (Jul 2-7, {n_days:.1f} days) ===")
    print(f"{'buffer':>7} {'trades':>7} {'opp/day':>8} {'win%':>6} {'edge/ct':>8} {'SE':>5} {'avg cost':>9}")
    for b in BUFFERS:
        rr = results[b]
        if not rr:
            print(f"{b*100:>6.0f}c {'0':>7}")
            continue
        pnl = [(1.0 - c - fee_ceil(c)) if w else (-c - fee_ceil(c)) for _, _, c, w, _, _, _ in rr]
        by_w = defaultdict(list)
        for (tk, _, c, w, _, _, _), pl in zip(rr, pnl):
            by_w[tk].append(pl)
        mk = [sum(v) / len(v) for v in by_w.values()]
        mu = sum(mk) / len(mk)
        se = (sum((x - mu) ** 2 for x in mk) / max(len(mk) - 1, 1)) ** 0.5 / len(mk) ** 0.5
        wins = sum(1 for r in rr if r[3])
        cost = sum(r[2] for r in rr) / len(rr)
        print(f"{b*100:>6.0f}c {len(rr):>7} {len(rr)/n_days:>8.1f} {wins/len(rr)*100:>5.1f} "
              f"{sum(pnl)/len(pnl)*100:>+7.2f}c {se*100:>4.1f} {cost*100:>8.1f}c")

    # detail at the 2c buffer
    rr = results[0.02]
    if rr:
        print("\n2c-buffer trades by time-to-close:")
        for lo, hi in ((3, 10), (10, 20), (20, 30), (30, 45), (45, 60)):
            sel = [r for r in rr if lo <= r[4] < hi]
            if not sel:
                continue
            pnl = [(1.0 - c - fee_ceil(c)) if w else (-c - fee_ceil(c)) for _, _, c, w, _, _, _ in sel]
            wins = sum(1 for r in sel if r[3])
            print(f"  ttc {lo:>3}-{hi:<3}s: n={len(sel):>4} win={wins/len(sel)*100:5.1f}% "
                  f"edge={sum(pnl)/len(pnl)*100:+6.2f}c  avg_size={sum(r[6] for r in sel)/len(sel):,.0f}")
        json.dump([{"ticker": r[0], "side": r[1], "cost": r[2], "won": r[3],
                    "ttc": round(r[4], 1), "p_model": round(r[5], 4), "size": r[6]}
                   for r in rr], open(os.path.join(HERE, "asian_trades_2c.json"), "w"), indent=1)
        print("saved asian_trades_2c.json")


if __name__ == "__main__":
    main()
