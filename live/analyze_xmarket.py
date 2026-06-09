"""
Analyse xmarket_quotes.csv for fee-adjusted, cross-venue executable arbs.

For each poll, an arb locks $1 of payout by buying one contract of EVERY
outcome on whichever venue offers it cheapest:
    cost = Σ_outcomes ( best_ask_i + fee(venue_i, best_ask_i) )
    net_edge = $1 - cost          (>0  =>  risk-free profit per $1 contract set)

Repeated polls sample the SAME opportunity, so positive rows are grouped into
"episodes" (consecutive positive rows per game) — episode count + duration is
what tells you whether an arb was actually executable, not the raw row count.

Fees:
  - Kalshi taker fee  ≈ 0.07 * price * (1-price)  per contract (rounded up to 1c).
  - Polymarket        = 0 on standard markets (override with --poly-fee).

LIMITATION: the logger stores top-of-book only (no size). A lock here proves
the prices touched; it does NOT prove depth. Treat edges as upper bounds.

Usage:
  python3 analyze_xmarket.py [--csv xmarket_quotes.csv] [--min-edge 0.0]
                             [--poly-fee 0.0] [--cross-only]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))


def kalshi_fee(price: float) -> float:
    """Kalshi taker fee per 1 contract, dollars. ceil(0.07*p*(1-p)) to the cent."""
    if price is None:
        return 0.0
    return math.ceil(0.07 * price * (1.0 - price) * 100) / 100.0


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def leg_best(k_ask, p_ask, poly_fee, k_sz=None, p_sz=None):
    """Return (price_incl_fee, venue, raw_ask, size) for the cheaper venue."""
    opts = []
    if k_ask is not None and k_ask > 0:
        opts.append((k_ask + kalshi_fee(k_ask), "K", k_ask, k_sz))
    if p_ask is not None and p_ask > 0:
        opts.append((p_ask + poly_fee * p_ask, "P", p_ask, p_sz))
    return min(opts) if opts else None


def analyze(path, min_edge, poly_fee, cross_only):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("no rows in", path)
        return
    n_out = max(int(r.get("n_outcomes") or 0) for r in rows)

    # episodes[game] = list of episodes; each = list of (row, net, venues, size)
    episodes = defaultdict(list)
    cur = defaultdict(list)
    positive_rows = 0
    all_nets = []   # (net, game, venues, row) for every fully-quoted row

    for r in rows:
        legs = []
        ok = True
        for i in range(1, n_out + 1):
            lbl = r.get(f"o{i}_label")
            if not lbl:
                continue
            best = leg_best(_f(r.get(f"o{i}_k_ask")), _f(r.get(f"o{i}_p_ask")),
                            poly_fee, _f(r.get(f"o{i}_k_ask_sz")), _f(r.get(f"o{i}_p_ask_sz")))
            if best is None:
                ok = False
                break
            legs.append(best)
        game = f"{r['league']} {r['game_label']}"
        if not ok or not legs:
            if cur[game]:
                episodes[game].append(cur[game]); cur[game] = []
            continue
        cost = sum(p for p, _, _, _ in legs)
        net = round(1.0 - cost, 4)
        venues = "".join(v for _, v, _, _ in legs)
        # executable size = min ask-size across legs (contracts/shares, each a
        # $1 payout unit). None if any leg lacks depth (REST fallback).
        sizes = [s for _, _, _, s in legs]
        exec_sz = min(sizes) if all(s is not None for s in sizes) else None
        all_nets.append((net, game, venues, r))
        is_cross = len(set(venues)) > 1          # legs span both venues
        hit = net > min_edge and (is_cross or not cross_only)
        if hit:
            positive_rows += 1
            cur[game].append((r, net, venues, exec_sz))
        else:
            if cur[game]:
                episodes[game].append(cur[game]); cur[game] = []
    for game, c in cur.items():
        if c:
            episodes[game].append(c)

    # ── report ──
    all_eps = [(g, e) for g, eps in episodes.items() for e in eps]
    all_eps.sort(key=lambda ge: max(n for _, n, _, _ in ge[1]), reverse=True)

    print(f"\n=== xmarket arb analysis : {os.path.basename(path)} ===")
    print(f"rows={len(rows)}  outcomes/game≤{n_out}  poly_fee={poly_fee:.2%}  "
          f"min_edge={min_edge*100:.2f}c  cross_only={cross_only}")
    print(f"positive rows: {positive_rows}   distinct episodes: {len(all_eps)}\n")

    if not all_eps:
        print("No fee-adjusted arbs found at this threshold.\n")
        # closest misses per game, so a null result still calibrates the gap
        best_per_game = {}
        for net, game, ven, r in all_nets:
            if game not in best_per_game or net > best_per_game[game][0]:
                best_per_game[game] = (net, ven, r)
        print("closest misses (best net edge seen per game, incl. fees):")
        print(f"  {'game':22} {'net¢':>7} {'venues':>7}  when")
        for game, (net, ven, r) in sorted(best_per_game.items(),
                                          key=lambda kv: -kv[1][0]):
            print(f"  {game[:22]:22} {net*100:7.2f} {ven:>7}  {r['ts_iso'][11:19]}")
        return

    # size = min executable across legs (contracts); est$ = peak_net * size
    print(f"{'game':22} {'peak¢':>6} {'dur(s)':>7} {'polls':>5} {'venues':>7} "
          f"{'size':>6} {'est$':>7}  window")
    for game, ep in all_eps[:25]:
        peakrow = max(ep, key=lambda x: x[1])
        peak = peakrow[1] * 100
        szs = [s for _, _, _, s in ep if s is not None]
        sz = min(szs) if szs else None                 # conservative across the window
        est = (peakrow[1] * peakrow[3]) if peakrow[3] is not None else None
        t0 = _f(ep[0][0]["ts_unix"]); t1 = _f(ep[-1][0]["ts_unix"])
        dur = round((t1 - t0), 1) if (t0 and t1) else 0
        ven = ep[len(ep) // 2][2]
        start = ep[0][0]["ts_iso"][11:19]
        print(f"{game[:22]:22} {peak:6.2f} {dur:7.1f} {len(ep):5d} {ven:>7} "
              f"{(f'{sz:.0f}' if sz is not None else '?'):>6} "
              f"{(f'{est:.2f}' if est is not None else '?'):>7}  {start}")

    # best single moment, fully itemised
    bg, bep = max(all_eps, key=lambda ge: max(n for _, n, _, _ in ge[1]))
    br, bn, _, bsz = max(bep, key=lambda x: x[1])
    estd = f"  est ${bn*bsz:.2f} on {bsz:.0f} units" if bsz is not None else "  (size unknown — REST leg)"
    print(f"\n--- best moment: {bg}  net={bn*100:.2f}c @ {br['ts_iso'][11:19]} "
          f"(score {br['away_score']}-{br['home_score']}){estd} ---")
    for i in range(1, n_out + 1):
        if not br.get(f"o{i}_label"):
            continue
        ka, pa = _f(br.get(f"o{i}_k_ask")), _f(br.get(f"o{i}_p_ask"))
        ksz, psz = _f(br.get(f"o{i}_k_ask_sz")), _f(br.get(f"o{i}_p_ask_sz"))
        best = leg_best(ka, pa, poly_fee, ksz, psz)
        if not best:
            print(f"  {br[f'o{i}_label'][:20]:20} no quote")
            continue
        sz = f"x{best[3]:.0f}" if best[3] is not None else "x?"
        print(f"  {br[f'o{i}_label'][:20]:20} K={ka}(sz{ksz}) P={pa}(sz{psz})  "
              f"-> buy {best[1]} @ {best[2]} {sz} (+fee {best[0]-best[2]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "xmarket_quotes.csv"))
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="min net edge in DOLLARS (e.g. 0.01 = 1 cent)")
    ap.add_argument("--poly-fee", type=float, default=0.0,
                    help="Polymarket fee fraction (0.02 = 2%% intl)")
    ap.add_argument("--cross-only", action="store_true",
                    help="only count arbs whose legs span BOTH venues")
    a = ap.parse_args()
    analyze(a.csv, a.min_edge, a.poly_fee, a.cross_only)
