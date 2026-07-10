"""
Race clock v3 — per-ticker book reconstruction (v2 merged concurrent
markets' deltas into one book: bug). Reaction = first top-of-book change
(>=1c move or >=25% best-depth drop) on ANY KXBTC15M ticker after a Kraken
1s impulse. Incremental: rerun any time after pulling new slices.
"""

import glob
import gzip
import json
import os
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TR_CACHE = os.path.join(HERE, "kraken_cache.json")


def load_events():
    seen = set()
    events = []
    files = glob.glob(f"{HERE}/book_archive/*.jsonl.gz") + [f"{HERE}/btc0709.jsonl.gz"]
    for fp in files:
        with gzip.open(fp, "rt") as f:
            for line in f:
                if '"KXBTC15M' not in line:
                    continue
                h = hash(line)
                if h in seen:
                    continue
                seen.add(h)
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = r.get("msg", {})
                tk = m.get("market_ticker")
                ts = m.get("ts")
                if ts is None or not tk:
                    continue
                if isinstance(ts, (int, float)):
                    t = float(ts) / (1000.0 if ts > 10**11 else 1.0)
                else:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                events.append((t, r.get("seq", 0), r["type"], tk, m))
    events.sort(key=lambda x: (x[0], x[1]))
    return events


def coverage_segments(times, gap=60):
    segs = []
    s0 = times[0]
    prev = s0
    for t in times:
        if t - prev > gap:
            segs.append((s0, prev))
            s0 = t
        prev = t
    segs.append((s0, prev))
    return segs


def fetch_trades(segs):
    cache = json.load(open(TR_CACHE)) if os.path.exists(TR_CACHE) else []
    have = set()
    for t, _ in cache:
        have.add(int(t // 300))
    out = list(cache)
    for a, b in segs:
        cur = a
        while cur < b:
            blk = int(cur // 300)
            if blk in have:
                cur += 300
                continue
            since = int(cur * 1e9)
            r = {}
            for att in range(5):
                try:
                    r = requests.get("https://api.kraken.com/0/public/Trades",
                                     params={"pair": "XBTUSD", "since": since, "count": 1000},
                                     timeout=15).json()
                    if not r.get("error"):
                        break
                except requests.RequestException:
                    pass
                time.sleep(1 + att)
            res = r.get("result", {})
            key = next((k for k in res if k != "last"), None)
            rows = res.get(key, []) if key else []
            if not rows:
                have.add(blk)
                cur += 300
                continue
            for tr in rows:
                out.append((float(tr[2]), float(tr[0])))
            last_t = float(rows[-1][2])
            for bb in range(int(cur // 300), int(last_t // 300)):
                have.add(bb)
            cur = max(last_t, cur + 1)
            time.sleep(0.55)
    out = sorted(set(map(tuple, out)))
    json.dump(out, open(TR_CACHE, "w"))
    return out


def main():
    events = load_events()
    times = [e[0] for e in events]
    segs = coverage_segments(times)
    tot = sum(b - a for a, b in segs)
    print(f"book events: {len(events):,}  coverage: {tot/60:.0f} min in {len(segs)} segments")

    trades = fetch_trades(segs)
    trades = [x for x in trades if any(a - 2 <= x[0] <= b for a, b in segs)]
    print(f"kraken trades in coverage: {len(trades):,}")
    tts = [x[0] for x in trades]

    def impulses(bp):
        out = []
        last = 0.0
        for i, (t, p) in enumerate(trades):
            if t - last < 20:
                continue
            j = bisect_left(tts, t - 1.0)
            if j >= i:
                continue
            p0 = trades[j][1]
            if abs(p - p0) / p0 * 1e4 >= bp:
                out.append(t)
                last = t
        return out

    # per-ticker books -> unified best-change event list
    books = defaultdict(lambda: {"yes": defaultdict(float), "no": defaultdict(float)})
    state = {}          # ticker -> (best_yes, best_no, qty_y, qty_n)
    changes = []        # (t, ticker) whenever a ticker's top-of-book changes materially

    def best(tk, side):
        d = books[tk][side]
        bp = 0.0
        for p, q in d.items():
            if q > 0.5 and p > bp:
                bp = p
        return bp, d.get(bp, 0.0)

    for t, _, kind, tk, m in events:
        if kind == "orderbook_snapshot":
            books[tk] = {"yes": defaultdict(float), "no": defaultdict(float)}
            for key, side in (("yes_dollars_fp", "yes"), ("no_dollars_fp", "no"),
                              ("yes", "yes"), ("no", "no")):
                for pair in m.get(key, []) or []:
                    try:
                        books[tk][side][float(pair[0])] = float(pair[1])
                    except (TypeError, ValueError, IndexError):
                        pass
        elif kind == "orderbook_delta":
            side = m.get("side")
            try:
                p = float(m["price_dollars"])
                dq = float(m["delta_fp"])
            except (KeyError, TypeError, ValueError):
                continue
            if side in ("yes", "no"):
                books[tk][side][p] += dq
        else:
            continue
        by, qy = best(tk, "yes")
        bn, qn = best(tk, "no")
        prev = state.get(tk)
        state[tk] = (by, bn, qy, qn)
        if prev is None:
            continue
        pby, pbn, pqy, pqn = prev
        if abs(by - pby) >= 0.01 or abs(bn - pbn) >= 0.01 \
           or (pqy > 0 and qy < 0.75 * pqy) or (pqn > 0 and qn < 0.75 * pqn):
            changes.append(t)

    print(f"material top-of-book changes: {len(changes):,}")

    # null baseline: time-to-next-change from deterministic grid points
    null_lat = []
    for a, b in segs:
        t = a + 7.0
        while t < b - 10:
            i = bisect_right(changes, t)
            if i < len(changes) and changes[i] - t <= 10:
                null_lat.append(changes[i] - t)
            t += 11.0
    null_lat.sort()
    if null_lat:
        nn = len(null_lat)
        print(f"NULL baseline (random moments, n={nn}): median "
              f"{null_lat[nn//2]*1000:.0f}ms  p25 {null_lat[nn//4]*1000:.0f}ms")

    # ── STALE-QUOTE EXPOSURE: rebuild book states at impulse±lags ──────────
    # For each signed impulse, the continuation side's pre-impulse best ask:
    # up-impulse -> YES ask (= 1 - best NO bid); down -> NO ask (= 1 - best
    # YES bid). Exposure(lag) = stale price still available at T+lag.
    def imp_signed(bp):
        out = []
        last = 0.0
        for i, (t, p) in enumerate(trades):
            if t - last < 20:
                continue
            j = bisect_left(tts, t - 1.0)
            if j >= i:
                continue
            mv = (p - trades[j][1]) / trades[j][1] * 1e4
            if abs(mv) >= bp:
                out.append((t, mv > 0))
                last = t
        return out

    # time-indexed book state per ticker: replay once, snapshot at query times
    queries = []      # (t_query, impulse_id, lag_label)
    LAGS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    imps = imp_signed(3)
    for iid, (T, up) in enumerate(imps):
        queries.append((T - 0.05, iid, "pre"))
        for L in LAGS:
            queries.append((T + L, iid, L))
    queries.sort()

    books2 = defaultdict(lambda: {"yes": defaultdict(float), "no": defaultdict(float)})
    snap_out = defaultdict(dict)      # impulse_id -> label -> {tk: (yes_ask, no_ask)}
    qi = 0
    for t, _, kind, tk, m in events:
        while qi < len(queries) and queries[qi][0] < t:
            tq, iid, lab = queries[qi]
            best_asks = {}
            for tk2, bd in books2.items():
                bn = max((p for p, q in bd["no"].items() if q > 0.5), default=None)
                by = max((p for p, q in bd["yes"].items() if q > 0.5), default=None)
                if bn is not None and by is not None:
                    best_asks[tk2] = (1.0 - bn, 1.0 - by)   # (yes_ask, no_ask)
            snap_out[iid][lab] = best_asks
            qi += 1
        if kind == "orderbook_snapshot":
            books2[tk] = {"yes": defaultdict(float), "no": defaultdict(float)}
            for key, side in (("yes_dollars_fp", "yes"), ("no_dollars_fp", "no"),
                              ("yes", "yes"), ("no", "no")):
                for pair in m.get(key, []) or []:
                    try:
                        books2[tk][side][float(pair[0])] = float(pair[1])
                    except (TypeError, ValueError, IndexError):
                        pass
        elif kind == "orderbook_delta":
            side = m.get("side")
            try:
                p = float(m["price_dollars"])
                dq = float(m["delta_fp"])
            except (KeyError, TypeError, ValueError):
                continue
            if side in ("yes", "no"):
                books2[tk][side][p] += dq

    print(f"\nstale-quote exposure after >=3bp impulses (n={len(imps)}):")
    print(f"{'lag':>7} {'still liftable at pre-impulse ask':>36}")
    for L in LAGS:
        alive = tot2 = 0
        for iid, (T, up) in enumerate(imps):
            pre = snap_out.get(iid, {}).get("pre", {})
            post = snap_out.get(iid, {}).get(L, {})
            for tk2, (ya0, na0) in pre.items():
                a0 = ya0 if up else na0
                if not (0.05 < a0 < 0.95):
                    continue
                if tk2 not in post:
                    continue
                a1 = post[tk2][0] if up else post[tk2][1]
                tot2 += 1
                if a1 <= a0 + 1e-9:
                    alive += 1
        if tot2:
            print(f"{L*1000:>5.0f}ms {alive:>5}/{tot2:<5} = {alive/tot2*100:5.1f}%")


if __name__ == "__main__":
    main()
