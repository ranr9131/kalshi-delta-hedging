"""
Strategies 2, 3, 4, 10 on the real KXBTC15M book log (Jul 2-7).

  S2  book-imbalance taker  — does ladder depth predict settlement beyond mid?
  S3  BTC-impulse snipe     — after a >=8bp/10s Binance impulse, is the ask at
                              the NEXT snapshot (~2s later) still cheap?
  S4  maker at touch        — join best bid both sides, fill iff a later ask
                              trades STRICTLY through our price within 60s,
                              hold to settlement. ttc in [120, 780]s only.
  S10 favorite-side maker   — S4 restricted to the side priced >= 0.85.

Fees: taker legs pay exact 0.07*c*(1-c); maker fills assume fee 0 (flagged —
must be confirmed with a real 1-lot GTC fill).
"""

import csv
import io
import json
import math
import os
import zipfile
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def fee(c):
    return 0.07 * c * (1.0 - c)


def load_markets():
    mk = {}
    for m in json.load(open(os.path.join(HERE, "replay_markets.json"))):
        if m.get("result") in ("yes", "no"):
            mk[m["ticker"]] = (m["result"] == "yes",
                               datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
    return mk


def load_book(mk):
    snaps = defaultdict(list)
    for line in open("/Users/leolee/Desktop/kalshi-delta-hedging/live/book_log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tk = r.get("ticker")
        if tk not in mk:
            continue
        t = datetime.fromisoformat(r["ts"]).timestamp()
        ya = [(p, s) for p, s in r["yes_asks"][:5] if 0 < p < 1]
        na = [(p, s) for p, s in r["no_asks"][:5] if 0 < p < 1]
        yb, yk = r.get("yes_bid"), r.get("yes_ask")
        if yb is not None and not (0 < yb < 1):
            yb = None
        if yk is not None and not (0 < yk < 1):
            yk = None
        snaps[tk].append((t, yb, yk, ya, na))
    for tk in snaps:
        snaps[tk].sort()
    return snaps


def load_btc():
    px = {}
    d = os.path.join(HERE, "btc1s")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(d, fn)) as z:
            with z.open(z.namelist()[0]) as f:
                for row in csv.reader(io.TextIOWrapper(f)):
                    if row and row[0].isdigit():
                        t = int(row[0])
                        t = t // 10**6 if t > 10**14 else t // 10**3
                        px[t] = float(row[4])
    ts = sorted(px)
    return ts, [px[t] for t in ts]


def cluster_se(by_window):
    vals = [sum(v) / len(v) for v in by_window.values()]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return (sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 / len(vals) ** 0.5


def s2_imbalance(mk, snaps):
    print("\n── S2: book-imbalance predictiveness (one obs per window per ttc band)")
    rows = []
    for tk, rr in snaps.items():
        yes_won, close = mk[tk]
        for target in (600, 300, 150):
            best = None
            for t, yb, yk, ya, na in rr:
                ttc = close - t
                if abs(ttc - target) < 15 and yb and yk and ya and na:
                    best = (yb, yk, ya, na)
                    break
            if not best:
                continue
            yb, yk, ya, na = best
            mid = (yb + yk) / 2
            if not (0.10 < mid < 0.90):
                continue
            dy = sum(s for _, s in ya)
            dn = sum(s for _, s in na)
            imb = (dn - dy) / (dn + dy) if dn + dy > 0 else 0.0
            rows.append((mid, imb, 1.0 if yes_won else 0.0))
    # within mid terciles, split by imbalance sign
    rows.sort()
    k = len(rows) // 3
    for name, chunk in (("low-mid", rows[:k]), ("mid", rows[k:2 * k]), ("high-mid", rows[2 * k:])):
        pos = [r for r in chunk if r[1] > 0.1]
        neg = [r for r in chunk if r[1] < -0.1]
        if len(pos) < 20 or len(neg) < 20:
            continue
        wp = sum(r[2] for r in pos) / len(pos) - sum(r[0] for r in pos) / len(pos)
        wn = sum(r[2] for r in neg) / len(neg) - sum(r[0] for r in neg) / len(neg)
        print(f"  {name:>9}: imb>0 excess win {wp*100:+5.1f}pp (n={len(pos)}) | "
              f"imb<0 {wn*100:+5.1f}pp (n={len(neg)})")


def s3_impulse(mk, snaps, bts, btp):
    print("\n── S3: BTC-impulse stale-quote snipe (fill at NEXT snapshot ask)")

    def spot(t):
        i = bisect_right(bts, t) - 1
        return btp[i] if i >= 0 and t - bts[i] <= 3 else None

    res = defaultdict(list)
    by_w = defaultdict(lambda: defaultdict(list))
    for tk, rr in snaps.items():
        yes_won, close = mk[tk]
        traded = set()
        for i, (t, yb, yk, ya, na) in enumerate(rr[:-1]):
            ttc = close - t
            if not (90 <= ttc <= 800):
                continue
            p_now, p_10 = spot(t), spot(t - 10)
            if p_now is None or p_10 is None:
                continue
            imp = (p_now - p_10) / p_10 * 1e4
            if abs(imp) < 8:
                continue
            side = "yes" if imp > 0 else "no"
            if (tk, side) in traded:
                continue
            t2, yb2, yk2, ya2, na2 = rr[i + 1]
            if t2 - t > 6:
                continue
            lad = ya2 if side == "yes" else na2
            if not lad:
                continue
            ask = lad[0]
            if not (0.03 <= ask[0] <= 0.95):
                continue
            c = ask[0]
            won = yes_won if side == "yes" else not yes_won
            pnl = (1 - c - fee(c)) if won else (-c - fee(c))
            b = "8-15bp" if abs(imp) < 15 else "15-30bp" if abs(imp) < 30 else "30bp+"
            res[b].append((pnl, won))
            res["ALL"].append((pnl, won))
            by_w[b][tk].append(pnl)
            by_w["ALL"][tk].append(pnl)
            traded.add((tk, side))
    for b in ("8-15bp", "15-30bp", "30bp+", "ALL"):
        rr2 = res.get(b, [])
        if not rr2:
            continue
        pnl = [x for x, _ in rr2]
        w = sum(1 for _, x in rr2 if x)
        print(f"  {b:>8}: n={len(rr2):>4} win={w/len(rr2)*100:5.1f}% "
              f"edge={sum(pnl)/len(pnl)*100:+6.2f}c ±{cluster_se(by_w[b])*100:.2f}")


def s4_maker(mk, snaps, favorite_only=False):
    tag = "S10: favorite-side maker (price>=0.85)" if favorite_only else "S4: maker at touch, both sides"
    print(f"\n── {tag}  [maker fee assumed 0 — UNCONFIRMED]")
    res = []
    by_w = defaultdict(list)
    for tk, rr in snaps.items():
        yes_won, close = mk[tk]
        filled = set()
        for i, (t, yb, yk, ya, na) in enumerate(rr):
            ttc = close - t
            if not (120 <= ttc <= 780):
                continue
            for side in ("yes", "no"):
                if (tk, side) in filled:
                    continue
                if side == "yes":
                    if yb is None or yk is None or yk - yb < 0.015:
                        continue
                    q = yb
                    if favorite_only and q < 0.85:
                        continue
                    if not (0.03 <= q <= 0.96):
                        continue
                    # fill iff later yes ask trades strictly through our bid
                    for t2, _, _, ya2, _ in rr[i + 1:]:
                        if t2 - t > 60 or close - t2 < 60:
                            break
                        if ya2 and ya2[0][0] < q:
                            won = yes_won
                            pnl = (1 - q) if won else -q
                            res.append((pnl, won, q))
                            by_w[tk].append(pnl)
                            filled.add((tk, side))
                            break
                else:
                    if yb is None or yk is None or yk - yb < 0.015:
                        continue
                    q = 1 - yk          # our NO bid
                    if favorite_only and q < 0.85:
                        continue
                    if not (0.03 <= q <= 0.96):
                        continue
                    for t2, _, _, _, na2 in rr[i + 1:]:
                        if t2 - t > 60 or close - t2 < 60:
                            break
                        if na2 and na2[0][0] < q:
                            won = not yes_won
                            pnl = (1 - q) if won else -q
                            res.append((pnl, won, q))
                            by_w[tk].append(pnl)
                            filled.add((tk, side))
                            break
    if not res:
        print("  no fills")
        return
    pnl = [x for x, _, _ in res]
    w = sum(1 for _, x, _ in res if x)
    q = sum(x for _, _, x in res) / len(res)
    print(f"  fills={len(res)} ({len(by_w)} windows) win={w/len(res)*100:.1f}% "
          f"avg_price={q*100:.1f}c edge={sum(pnl)/len(pnl)*100:+.2f}c ±{cluster_se(by_w)*100:.2f}")


def main():
    mk = load_markets()
    snaps = load_book(mk)
    print(f"windows: {len(snaps)}")
    bts, btp = load_btc()
    s2_imbalance(mk, snaps)
    s3_impulse(mk, snaps, bts, btp)
    s4_maker(mk, snaps, favorite_only=False)
    s4_maker(mk, snaps, favorite_only=True)


if __name__ == "__main__":
    main()
