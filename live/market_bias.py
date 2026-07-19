import csv, collections, os, sys, requests
sys.path.insert(0, ".")
from dotenv import dotenv_values
import kalshi_auth
env = dotenv_values(".env")
kid = env.get("KALSHI_API_KEY_ID") or env.get("LEO_KEY_ID")
pk = kalshi_auth.load_private_key(env.get("KALSHI_PRIVATE_KEY") or env.get("LEO_PRIVATE_KEY"))
BASE = "https://api.elections.kalshi.com"

def f(x):
    try: return float(x)
    except: return None

# 1) pick, per ticker, the snapshot nearest a target horizon (mins_left ~ 20)
TARGET_ML = 20.0
best = {}  # ticker -> (abs_dist, mid_c, mins_left)
for r in csv.DictReader(open("mm_shadow_snapshots.csv")):
    bb = f(r["bb_c"]); ba = f(r["ba_c"]); ml = f(r["mins_left"])
    if bb is None or ba is None or ml is None or ba <= bb:
        continue
    mid = (bb + ba) / 2.0
    d = abs(ml - TARGET_ML)
    t = r["ticker"]
    if t not in best or d < best[t][0]:
        best[t] = (d, mid, ml)
print("distinct tickers w/ a usable snapshot near %.0fm: %d" % (TARGET_ML, len(best)))

# 2) fetch settlement result per ticker (cached)
def result_of(t):
    p = "/trade-api/v2/markets/" + t
    h = kalshi_auth.make_auth_headers(pk, kid, "GET", p)
    try:
        m = requests.get(BASE + p, headers=h, timeout=15).json().get("market", {})
        return m.get("result")
    except Exception:
        return None

pairs = []  # (market_mid_prob, outcome_yes)
for t, (d, mid, ml) in best.items():
    res = result_of(t)
    if res in ("yes", "no"):
        pairs.append((mid / 100.0, 1 if res == "yes" else 0))
print("settled tickers: %d" % len(pairs))

# 3) MARKET-price calibration: is the market mid biased?
print("\n=== MARKET mid-price calibration (XRP daily, ~%.0fm to close) ===" % TARGET_ML)
print("%9s %6s %10s %10s %8s  %s" % ("mid_bin", "n", "mean_mid", "realized", "edge", "read"))
B = collections.defaultdict(list)
for mp, y in pairs:
    B[min(9, int(mp * 10))].append((mp, y))
for b in range(10):
    it = B[b]
    if not it:
        continue
    mm = sum(p for p, _ in it) / len(it)
    rw = sum(y for _, y in it) / len(it)
    edge = rw - mm  # >0 => market UNDERprices yes here (buy yes); <0 => OVERprices (sell yes)
    read = "buy YES" if edge > 0.03 else "sell YES" if edge < -0.03 else "fair"
    print("%3d-%3d%% %6d %10.3f %10.3f %+8.3f  %s" % (b*10, (b+1)*10, len(it), mm, rw, edge, read))

br = sum((p - y)**2 for p, y in pairs) / len(pairs)
print("\nMarket Brier=%.4f (lower=sharper). If every bin's edge ~0, the market is" % br)
print("efficient and there's no passive bias to fade. Systematic +/- = exploitable.")
