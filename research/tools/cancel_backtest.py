"""
Counterfactual fast-cancel backtest on the month of real shadow maker fills.

For each settlement-matched fill (DOGE/SOL/XRP 15m, Jun 8 - Jul 7):
  trigger = any trailing-1s spot move >= X bp ending within [fill-10s, fill]
  the cancel pulls quotes at trigger_time + LAMBDA and stays out 10s,
  so the fill is AVOIDED iff a trigger fired at least LAMBDA before the fill.
Sweep (X, LAMBDA); recompute maker economics on the surviving fills.
Also applies the static policy layer first (no final 180s, no near-strike
in last 7 min) — the engineering stack as designed.

Fill timestamps are ms; spot bars are 1s (Binance) — trigger times are
known to the second, so LAMBDA below ~200ms is extrapolation, flagged.
"""

import csv
import io
import os
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/leolee/Desktop/kalshi-delta-hedging"


def load_spot(sym):
    """second-ts -> close; plus sorted trigger arrays built later."""
    px = {}
    for fn in sorted(os.listdir(f"{HERE}/alt1s")):
        if not fn.startswith(sym):
            continue
        try:
            with zipfile.ZipFile(f"{HERE}/alt1s/{fn}") as z:
                with z.open(z.namelist()[0]) as f:
                    for row in csv.reader(io.TextIOWrapper(f)):
                        if row and row[0].isdigit():
                            t = int(row[0])
                            t = t // 10**6 if t > 10**14 else t // 10**3
                            px[t] = float(row[4])
        except zipfile.BadZipFile:
            continue
    return px


def triggers(px, bp):
    """sorted list of trigger times: end-of-second where |1s move| >= bp."""
    out = []
    for t in sorted(px):
        if t - 1 in px:
            if abs(px[t] - px[t - 1]) / px[t - 1] * 1e4 >= bp:
                out.append(t + 1.0)   # bar [t, t+1) closes at t+1: info time
    return out


def main():
    res = {}
    for r in csv.DictReader(open(f"{REPO}/data/server_logs/settlements.csv")):
        if r["result"] in ("yes", "no"):
            res[r["ticker"]] = r["result"] == "yes"

    fills = []
    for f in csv.DictReader(open(f"{REPO}/data/server_logs/conv_maker_fills.csv")):
        tk = f["ticker"]
        if tk not in res:
            continue
        yes_won = res[tk]
        won = yes_won if f["action"] == "buy_yes" else not yes_won
        p = float(f["price_c"])
        fair = float(f["fair_c"])
        imm = (fair - p) if f["action"] == "buy_yes" else ((100 - p) - (100 - fair))
        fills.append({"ts": float(f["ts"]), "asset": f["asset"], "ttc": float(f["ttc_sec"]),
                      "fair": fair, "imm": imm,
                      "pnl": (100 - p) if won else -p})
    print(f"settlement-matched fills: {len(fills):,}")

    # static policy layer
    fills = [f for f in fills if f["ttc"] >= 180
             and not (f["ttc"] < 420 and abs(f["fair"] - 50) < 20)]
    base_n = len(fills)
    base_pnl = sum(f["pnl"] for f in fills) / base_n
    base_po = sum(1 for f in fills if f["imm"] < 0) / base_n
    print(f"after policy layer: n={base_n:,}  net={base_pnl:+.2f}c/fill  pickoff={base_po*100:.1f}%")

    spot = {a: load_spot(s) for a, s in (("DOGE", "DOGEUSDT"), ("SOL", "SOLUSDT"), ("XRP", "XRPUSDT"))}
    for a, px in spot.items():
        print(f"  {a}: {len(px):,} spot seconds")

    print(f"\n{'X (bp)':>7} {'lambda':>8} {'kept':>6} {'kept%':>6} {'pickoff%':>8} "
          f"{'noise/fill':>10} {'NET c/fill':>10} {'net $/day @10ct':>15}")
    days = 30.0
    for X in (3, 5, 8):
        trg = {a: triggers(px, X) for a, px in spot.items()}
        for lam in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            kept = []
            for f in fills:
                tl = trg[f["asset"]]
                i = bisect_right(tl, f["ts"] - lam)
                # a trigger in [fill-10s, fill-lam] => quote already cancelled
                if i > 0 and tl[i - 1] >= f["ts"] - 10.0:
                    continue
                kept.append(f)
            if not kept:
                continue
            n = len(kept)
            po = sum(1 for f in kept if f["imm"] < 0) / n
            net = sum(f["pnl"] for f in kept) / n
            noise = [f["pnl"] for f in kept if f["imm"] >= 0]
            na = sum(noise) / len(noise) if noise else 0.0
            perday = net / 100.0 * 10 * n / days     # $, 10 contracts per fill
            flag = " *" if lam < 0.2 else ""
            print(f"{X:>7} {lam*1000:>6.0f}ms {n:>6} {n/base_n*100:>5.1f} {po*100:>7.1f} "
                  f"{na:>+9.2f} {net:>+9.2f} {perday:>+14.2f}{flag}")
    print("\n* sub-200ms lambdas are extrapolation (1s spot bars); treat as bound")


if __name__ == "__main__":
    main()
