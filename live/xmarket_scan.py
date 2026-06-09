"""
Quantify the cross-venue (Kalshi<->Polymarket) sports mispricing from the
collected CSV. Characterises: edge distribution, frequency, persistence,
executable depth, WHICH venue is systematically wrong, and per-game $ size.

Only counts CROSS-venue locks (legs bought on different venues) on rows where
every outcome is quoted on >=1 venue. Kalshi taker fee modelled; Poly fee 0.
Depth is top-of-book only (lower bound on size). Fills are NOT proven — this
sizes the *paper* opportunity to decide whether a live fill-test is worth it.
"""
import csv, math, os, statistics, sys
from collections import defaultdict, Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "xmarket_quotes.csv"


def kfee(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0 if p else 0.0


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def leg(k_ask, p_ask, k_sz, p_sz):
    opts = []
    if k_ask and k_ask > 0:
        opts.append((k_ask + kfee(k_ask), "K", k_ask, k_sz))
    if p_ask and p_ask > 0:
        opts.append((p_ask, "P", p_ask, p_sz))
    return min(opts) if opts else None


rows = list(csv.DictReader(open(PATH)))
n_out = max(int(r.get("n_outcomes") or 0) for r in rows)

# only live, fully-2-sided in-progress rows
def live(r):
    s = (r.get("status") or "").lower()
    return ("final" not in s and "over" not in s and "postponed" not in s)

net_all = []                 # net edge for every fully-quoted cross row
pos_rows = 0
fav_on_kalshi = 0            # of 2-way positive rows, Kalshi-bought leg is the favorite (poly>0.5)
fav_total = 0
buy_venue = Counter()       # which venue the cheap legs land on
per_game = defaultdict(lambda: {"rows": 0, "pos": 0, "peak": 0.0, "best_sz": 0.0, "dollars": 0.0})
# episode tracking at thresholds
THRESH = [0.0, 0.01, 0.05, 0.10, 0.20]
episodes = {t: [] for t in THRESH}       # list of (game, dur, peak, minsize)
cur = {t: defaultdict(list) for t in THRESH}

def close_eps(t, game):
    ep = cur[t][game]
    if ep:
        t0 = f(ep[0][0]); t1 = f(ep[-1][0])
        dur = (t1 - t0) if (t0 and t1) else 0.0
        peak = max(e[1] for e in ep)
        szs = [e[2] for e in ep if e[2] is not None]
        msz = min(szs) if szs else None
        episodes[t].append((game, dur, peak, msz))
        cur[t][game] = []

for r in rows:
    if not live(r):
        continue
    legs = []
    ok = True
    for i in range(1, n_out + 1):
        if not r.get(f"o{i}_label"):
            continue
        b = leg(f(r.get(f"o{i}_k_ask")), f(r.get(f"o{i}_p_ask")),
                f(r.get(f"o{i}_k_ask_sz")), f(r.get(f"o{i}_p_ask_sz")))
        if not b:
            ok = False; break
        legs.append((b, i))
    game = f"{r['league']} {r['game_label']}"
    if not ok or not legs:
        for t in THRESH: close_eps(t, game)
        continue
    venues = "".join(b[1] for b, _ in legs)
    is_cross = len(set(venues)) > 1
    cost = sum(b[0] for b, _ in legs)
    net = round(1 - cost, 4)
    szs = [b[3] for b, _ in legs]
    exec_sz = min(szs) if all(s is not None for s in szs) else None
    ts = r.get("ts_unix")

    pg = per_game[game]; pg["rows"] += 1
    if is_cross:
        net_all.append(net)
    if is_cross and net > 0:
        pos_rows += 1
        pg["pos"] += 1
        pg["peak"] = max(pg["peak"], net)
        for b, _ in legs:
            buy_venue[b[1]] += 1
        # 2-way favourite-direction: is the Kalshi-bought team the favourite?
        if len(legs) == 2:
            for b, i in legs:
                if b[1] == "K":
                    pp = f(r.get(f"o{i}_p_ask"))   # poly ask ~ true prob of that team
                    if pp is not None:
                        fav_total += 1
                        if pp > 0.5:
                            fav_on_kalshi += 1
    # episodes per threshold
    for t in THRESH:
        if is_cross and net > t:
            cur[t][game].append((ts, net, exec_sz))
        else:
            close_eps(t, game)

for t in THRESH:
    for game in list(cur[t].keys()):
        close_eps(t, game)

# ---- report ----
print(f"\n========== CROSS-VENUE MISPRICING SCAN : {os.path.basename(PATH)} ==========")
live_rows = sum(per_game[g]['rows'] for g in per_game)
print(f"live 2-sided rows: {live_rows}   games: {len(per_game)}   cross-quoted rows: {len(net_all)}")

if net_all:
    net_all.sort()
    def pct(x): return f"{100*sum(1 for v in net_all if v> x)/len(net_all):.1f}%"
    print(f"\n--- net cross-venue edge distribution (after Kalshi fee) ---")
    print(f"  median={statistics.median(net_all)*100:+.2f}c   "
          f"p90={net_all[int(.9*len(net_all))]*100:+.2f}c   "
          f"p99={net_all[int(.99*len(net_all))]*100:+.2f}c   max={net_all[-1]*100:+.2f}c")
    print(f"  % of cross rows with net >0c: {pct(0)}   >1c: {pct(.01)}   "
          f">5c: {pct(.05)}   >10c: {pct(.10)}   >20c: {pct(.20)}")

print(f"\n--- persistence: episodes (consecutive rows above threshold, per game) ---")
print(f"  {'thresh':>7} {'episodes':>9} {'med_dur_s':>10} {'p90_dur_s':>10} {'med_size':>9} {'tot_$est':>9}")
for t in THRESH:
    eps = episodes[t]
    if not eps:
        print(f"  {t*100:6.0f}c {0:>9}"); continue
    durs = sorted(e[1] for e in eps)
    szs = [e[3] for e in eps if e[3] is not None]
    dollars = sum(e[2] * (e[3] or 0) for e in eps)   # peak edge x min size, per episode
    med_sz = statistics.median(szs) if szs else 0
    print(f"  {t*100:6.0f}c {len(eps):>9} {durs[len(durs)//2]:>10.1f} "
          f"{durs[int(.9*len(durs))]:>10.1f} {med_sz:>9.0f} {dollars:>9.0f}")

print(f"\n--- WHICH venue is mispriced (cheap leg you buy) ---")
tot = sum(buy_venue.values()) or 1
print(f"  buy-legs landing on Kalshi: {buy_venue['K']} ({100*buy_venue['K']/tot:.0f}%)   "
      f"Polymarket: {buy_venue['P']} ({100*buy_venue['P']/tot:.0f}%)")
if fav_total:
    print(f"  of 2-way locks, Kalshi-bought leg is the FAVOURITE (Poly>0.5): "
          f"{fav_on_kalshi}/{fav_total} = {100*fav_on_kalshi/fav_total:.0f}%")
    print(f"  -> high % means Kalshi systematically LAGS pricing the late favourite (Poly leads).")

print(f"\n--- top games by paper $ opportunity (sum peak-edge x size over >5c episodes) ---")
g_dollars = defaultdict(float)
for game, dur, peak, msz in episodes[0.05]:
    g_dollars[game] += peak * (msz or 0)
for game, d in sorted(g_dollars.items(), key=lambda kv: -kv[1])[:12]:
    pg = per_game[game]
    print(f"  {game[:24]:24} ${d:8.0f}   peak={pg['peak']*100:.0f}c  pos_rows={pg['pos']}")

print(f"\nCAVEATS: top-of-book depth only (real size may differ); episodes across "
      f"different games overlap in time (need capital on all at once); fills UNPROVEN.")
