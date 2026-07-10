"""
Replicate the Reddit claim: maker on >=90c favorites in the FINAL 10-5 MIN of
Kalshi hourly crypto ladders (KXBTCD / KXETHD) has positive edge, while the
same trade as taker is ~breakeven.

Method (honest harness): 7 days x 24 hourly events x 2 assets; strikes within
+-1.2% of the hour's settlement; minute candles with bid/ask closes.
Maker: at each minute in the zone, if a side's BID >= 0.90, join it; filled
iff a LATER minute in the zone prints a trade at/below our price (conservative
trade-through proxy). Taker: buy at the ask at the first minute the ask is in
tier. Mark everything to settlement. Cluster SE by EVENT (strikes within an
hour share one settlement). Maker fee assumed 0 (Kalshi taker-fee schedule);
sensitivity noted.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hourly_cache")
os.makedirs(CACHE, exist_ok=True)
BASE = "https://api.elections.kalshi.com/trade-api/v2"
DAYS = 30
ZONE = (300, 600)          # 5..10 minutes before close
TIERS = [(0.90, 0.94), (0.94, 0.97), (0.97, 0.99)]


def get(path, params, retries=6):
    for a in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** a)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + a)
    return None


def markets(series):
    p = os.path.join(CACHE, f"mk_{series}_{DAYS}d.json")
    if os.path.exists(p):
        return json.load(open(p))
    out, cursor = [], None
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp())
    while True:
        params = {"series_ticker": series, "status": "settled",
                  "min_close_ts": cutoff, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = get("/markets", params)
        b = d.get("markets", []) if d else []
        out.extend(b)
        cursor = d.get("cursor") if d else None
        if not cursor or not b:
            break
        time.sleep(0.2)
    json.dump(out, open(p, "w"))
    return out


def candles(series, tk, oi, ci):
    p = os.path.join(CACHE, f"c_{tk}.json")
    if os.path.exists(p):
        return json.load(open(p))
    o = datetime.fromisoformat(oi.replace("Z", "+00:00"))
    c = datetime.fromisoformat(ci.replace("Z", "+00:00"))
    # only need the final 15 minutes
    d = get(f"/series/{series}/markets/{tk}/candlesticks",
            {"start_ts": int(c.timestamp()) - 900, "end_ts": int(c.timestamp()) + 60,
             "period_interval": 1})
    res = []
    if d:
        for k in d.get("candlesticks", []):
            pr, ask, bid = k.get("price", {}), k.get("yes_ask", {}), k.get("yes_bid", {})
            yc = pr.get("close_dollars")
            ac, bc = ask.get("close_dollars"), bid.get("close_dollars")
            res.append({"ts": k["end_period_ts"],
                        "yes": float(yc) if yc else None,
                        "ask": float(ac) if ac else None,
                        "bid": float(bc) if bc else None,
                        "vol": float(k.get("volume_fp") or 0)})
        res.sort(key=lambda x: x["ts"])
    json.dump(res, open(p, "w"))
    time.sleep(0.12)
    return res


def cluster_se(by_ev):
    v = [sum(x) / len(x) for x in by_ev.values()]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5


def main():
    for series in ("KXDOGED",):
        mk = [m for m in markets(series) if m.get("result") in ("yes", "no")]
        # settlement value per event, for strike filtering
        ev_settle = {}
        for m in mk:
            ev = m["event_ticker"]
            try:
                x = float(m.get("expiration_value") or 0)
            except (TypeError, ValueError):
                x = 0
            if x > 0.001:
                ev_settle[ev] = x
        sel = []
        for m in mk:
            s = ev_settle.get(m["event_ticker"])
            try:
                strike = float(m.get("floor_strike") or m.get("cap_strike") or 0)
            except (TypeError, ValueError):
                strike = 0
            if not strike and "-T" in m["ticker"]:
                # strike_type=custom (e.g. DOGE): strike only in ticker suffix
                try:
                    strike = float(m["ticker"].rsplit("-T", 1)[1])
                except ValueError:
                    continue
            if s and strike and abs(strike - s) / s <= 0.05:
                sel.append(m)
        print(f"\n=== {series}: {len(mk)} settled, {len(sel)} near-settlement strikes "
              f"({len(ev_settle)} events)")

        maker = defaultdict(lambda: (list(), defaultdict(list)))   # tier -> (pnl list, by_ev)
        taker = defaultdict(lambda: (list(), defaultdict(list)))
        quotes = fills = 0
        for i, m in enumerate(sel):
            tk = m["ticker"]
            ev = m["event_ticker"]
            close = int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
            ry = m["result"] == "yes"
            cs = candles(series, tk, m["open_time"], m["close_time"])
            zone = [c for c in cs if ZONE[0] <= close - c["ts"] <= ZONE[1]]
            if not zone:
                continue
            done_mk, done_tk = set(), set()
            for ci_, c in enumerate(zone):
                for side in ("yes", "no"):
                    if side == "yes":
                        b, a = c["bid"], c["ask"]
                        win = ry
                    else:
                        b = 1 - c["ask"] if c["ask"] is not None else None
                        a = 1 - c["bid"] if c["bid"] is not None else None
                        win = not ry
                    # MAKER: join bid >= 0.90
                    if side not in done_mk and b is not None and b >= 0.90:
                        quotes += 1
                        tier = next((t for t in TIERS if t[0] <= b < t[1]), None)
                        if tier:
                            # conservative fill: later trade prints at/below our bid
                            for c2 in zone[ci_ + 1:]:
                                tr = c2["yes"] if side == "yes" else (1 - c2["yes"] if c2["yes"] else None)
                                if tr is not None and c2["vol"] > 0 and tr <= b:
                                    fills += 1
                                    pnl = (1 - b) if win else -b
                                    maker[tier][0].append(pnl)
                                    maker[tier][1][ev].append(pnl)
                                    done_mk.add(side)
                                    break
                    # TAKER: cross the ask in tier
                    if side not in done_tk and a is not None:
                        tier = next((t for t in TIERS if t[0] <= a < t[1]), None)
                        if tier:
                            fee = 0.07 * a * (1 - a)
                            pnl = (1 - a - fee) if win else (-a - fee)
                            taker[tier][0].append(pnl)
                            taker[tier][1][ev].append(pnl)
                            done_tk.add(side)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(sel)}", flush=True)

        print(f"  maker quotes {quotes}, fills {fills} "
              f"(fill rate {fills/max(quotes,1)*100:.0f}%)")
        # deep-tier loser autopsy + weekly stability
        deep = maker[TIERS[2]]
        losers = [(ev, x) for ev, v in deep[1].items() for x in v if x < 0]
        print(f"  DEEP-TIER losers: {len(losers)} of {len(deep[0])} fills; "
              f"events with a loss: {len(set(e for e,_ in losers))}")
        for ev, x in sorted(losers):
            print(f"    LOSS {ev}: {x*100:+.0f}c")
        byweek = defaultdict(list)
        for ev, v in deep[1].items():
            wk = ev.split("-")[1][:5]
            byweek[wk[:5]].extend(v)
        # group by calendar chunk via event date prefix (crude weekly-ish)

        print(f"  {'tier':>10} {'side':>6} {'n':>5} {'win%':>6} {'edge/ct':>8} {'SE':>5} {'events':>7}")
        for t in TIERS:
            for name, d in (("MAKER", maker), ("taker", taker)):
                pnl, by_ev = d[t]
                if len(pnl) < 20:
                    continue
                wins = sum(1 for x in pnl if x > 0)
                print(f"  {t[0]:.2f}-{t[1]:.2f} {name:>6} {len(pnl):>5} "
                      f"{wins/len(pnl)*100:>5.1f} {sum(pnl)/len(pnl)*100:>+7.2f}c "
                      f"{cluster_se(by_ev)*100:>4.1f} {len(by_ev):>7}")


if __name__ == "__main__":
    main()
