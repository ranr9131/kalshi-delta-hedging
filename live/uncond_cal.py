import csv, collections, sys, re
sys.path.insert(0, ".")
from fair_price_model_v2 import fair_p_yes_v2

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def asset_of(tkr):
    m = re.match(r"KX([A-Z]+)15M", tkr)
    return m.group(1) if m else None

rows = list(csv.DictReader(open("window_log.csv")))
# unconditional: every window. strike ~ price at open (btc_t0); current = btc_t5; 10 min left.
pairs = []
for r in rows:
    a = asset_of(r["ticker"])
    strike = f(r["btc_t0"]); cur = f(r["btc_t5"]); w = r.get("market_winner")
    if not a or strike is None or cur is None or w not in ("yes", "no"):
        continue
    fp = fair_p_yes_v2(cur, strike, 10.0, a)
    pairs.append((fp, 1 if w == "yes" else 0))

print("UNCONDITIONAL 15M calibration (all windows, fair_p@t5 vs winner), n=%d" % len(pairs))
print("%9s %6s %10s %10s %7s" % ("bin", "n", "mean_fp", "realized", "err"))
B = collections.defaultdict(list)
for fp, y in pairs:
    B[min(9, int(fp * 10))].append((fp, y))
for b in range(10):
    it = B[b]
    if not it:
        continue
    mp = sum(p for p, _ in it) / len(it)
    rw = sum(y for _, y in it) / len(it)
    print("%3d-%3d%% %6d %10.3f %10.3f %+7.3f" % (b * 10, (b + 1) * 10, len(it), mp, rw, rw - mp))

# Brier vs a flat 0.5 baseline
br = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
br0 = sum((0.5 - y) ** 2 for p, y in pairs) / len(pairs)
print("\nBrier(model)=%.4f  Brier(flat .5)=%.4f  -> model %s" %
      (br, br0, "beats coinflip" if br < br0 else "WORSE than coinflip"))
