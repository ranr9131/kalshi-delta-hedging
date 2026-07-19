import csv, math, collections

def f(x):
    try: return float(x)
    except: return None

res = {r["ticker"]: r["result"] for r in csv.DictReader(open("settlements.csv"))}

def fee(p):
    p = p / 100.0
    return 7.0 * p * (1 - p)

def enrich(path):
    """Return list of dicts with net cents/contract, dollar pnl, fair_p, win."""
    out = []
    for r in csv.DictReader(open(path)):
        fill = f(r["fill_cents_est"]) or f(r["limit_cents"])
        rr = res.get(r["ticker"])
        fp = f(r["fair_p"]); qty = f(r["qty"])
        if fill is None or rr not in ("yes", "no") or fp is None:
            continue
        yes = rr == "yes"
        g = ((100.0 if yes else 0.0) - fill) if r["side"] == "yes" else ((100.0 if not yes else 0.0) - fill)
        net = g - fee(fill)
        out.append({"net": net, "fp": fp, "qty": qty or 0.0,
                    "usd": (qty or 0.0) * net / 100.0, "win": net > 0,
                    "ts": r["ts_iso"], "asset": r["asset"]})
    return out

def summ(rows, label):
    if not rows:
        print("%-26s  (no settled)" % label); return
    n = len(rows)
    nets = [x["net"] for x in rows]
    mean = sum(nets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in nets) / n) if n > 1 else 0
    se = sd / math.sqrt(n) if n else 0
    win = 100.0 * sum(1 for x in rows if x["win"]) / n
    usd = sum(x["usd"] for x in rows)
    sig = abs(mean) / se if se else 0
    print("%-26s n=%5d  win=%4.1f%%  net=%+5.2f¢ ±%4.2f (%.1fσ)  paper$=%+7.2f" %
          (label, n, win, mean, se, sig, usd))

V1 = enrich("snipes.csv")
V2 = enrich("snipes_v2.csv")

print("=" * 78)
print("HEAD-TO-HEAD (all settled)")
print("=" * 78)
summ(V1, "V1 all")
summ(V2, "V2 all")
print("\nlast day (6/07):")
summ([r for r in V1 if r["ts"] >= "2026-06-07"], "V1 6/07")
summ([r for r in V2 if r["ts"] >= "2026-06-07"], "V2 6/07")

def band(rows, lo, hi):
    return [r for r in rows if lo <= r["fp"] < hi]

print("\n" + "=" * 78)
print("NARROWING TEST — V1 net edge by fair_p band")
print("=" * 78)
summ(band(V1, 0.15, 0.30), "V1 low wing  .15-.30")
summ(band(V1, 0.30, 0.50), "V1 mid-lo    .30-.50")
summ(band(V1, 0.50, 0.70), "V1 mid-hi    .50-.70")
summ(band(V1, 0.70, 0.86), "V1 high wing .70-.85")
print("--- grouped ---")
summ(band(V1, 0.30, 0.70), "V1 NARROW    .30-.70")
summ([r for r in V1 if r["fp"] < 0.30 or r["fp"] >= 0.70], "V1 WINGS only")

print("\n(same split for V2, for reference)")
summ(band(V2, 0.30, 0.70), "V2 NARROW    .30-.70")
summ([r for r in V2 if r["fp"] < 0.30 or r["fp"] >= 0.70], "V2 WINGS only")
