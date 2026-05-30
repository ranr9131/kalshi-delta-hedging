"""
1000-strategy zoo backtest on KXBTC15M.

15 archetypes — each conceptually distinct, not just knob-tweaks of one idea:

  1.  mom_btc         — buy trending side on BTC velocity
  2.  rev_btc         — buy fading side on BTC velocity
  3.  strike_mom      — at minute M, if BTC above/below strike by X%, bet correct side
  4.  strike_rev      — same trigger, bet opposite side
  5.  fav_follow      — at minute M, bet whichever side has YES > thresh (favorite chasing)
  6.  fav_fade        — same condition, bet the underdog instead
  7.  extreme_fav     — only enter if YES > 0.85 or < 0.15, bet favorite
  8.  extreme_fade    — same threshold, bet underdog (mean-reversion)
  9.  early_yes       — fixed YES bet at minute K (test pure bull bias)
  10. early_no        — fixed NO bet at minute K
  11. late_hodl       — only enter in last few minutes with side/band filter
  12. pin_fade        — when BTC close to strike near close, bet against favorite
  13. consec_trend    — bet same direction as PREVIOUS window's result
  14. consec_rev      — bet opposite of previous window's result
  15. kalshi_drift    — bet on side whose Kalshi YES has trended over last K minutes

Tuning happens on TRAIN (oldest 80%). The honest verdict is TEST (newest 20%)
performance of the train-best in each archetype, plus the "positive in both"
filter at the end.

Run:  python3 simulate_strategy_zoo.py
"""
import os
import sys
import time
import itertools
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import DATA_DAYS, FEE_RATE

# ── Common scaffolding (not swept) ───────────────────────────────────────────
STAKE_DOLLARS = 3.0
COOLDOWN_MIN  = 1
MAX_ENTRIES   = 4
MAX_WAGERED   = 20.0
STOP_LAST_MIN = 2
SLIPPAGE_C    = 1
LOOKBACK_OPTIONS = [1, 2, 3, 5, 8]  # precompute these on every window
KALSHI_LOOKBACK_OPTIONS = [1, 2, 3, 5]


def _btc_close_at(prices, t):
    """Close of the candle ending at t (Coinbase keys candles by start time)."""
    return btc_data.lookup(prices, t - 60)


# ── Event preprocessing ──────────────────────────────────────────────────────
def build_window_events(market, btc_prices):
    """Returns (events, result, strike) or (None, None, None)."""
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    strike    = market.get("floor_strike")
    if result not in ("yes", "no") or not open_iso or not close_iso or strike is None:
        return None, None, None
    try:
        t0      = int(datetime.fromisoformat(open_iso .replace("Z","+00:00")).timestamp())
        t_close = int(datetime.fromisoformat(close_iso.replace("Z","+00:00")).timestamp())
    except Exception:
        return None, None, None
    candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
    if not candles:
        return None, None, None

    # Pre-build Kalshi yes_close timeline keyed by exact minute end ts
    yc_at = {c["ts"]: float(c["yes_close"]) for c in candles}

    events = []
    for k in range(1, 15):
        t = t0 + k * 60
        if t > t_close - STOP_LAST_MIN * 60:
            break
        btc_now = _btc_close_at(btc_prices, t)
        if btc_now is None:
            continue
        # Kalshi yes_close at t (end of minute k inside window)
        yc = yc_at.get(t)
        if yc is None:
            yc = kalshi_client.get_yes_price_at(candles, t)
        if yc is None or not (0.01 < yc < 0.99):
            continue
        btc_lb = {}
        for L in LOOKBACK_OPTIONS:
            v = _btc_close_at(btc_prices, t - L * 60)
            if v is not None:
                btc_lb[L] = v
        if not btc_lb:
            continue
        # Kalshi lookback (yes_close N min ago)
        yc_lb = {}
        for L in KALSHI_LOOKBACK_OPTIONS:
            yc_lb[L] = yc_at.get(t - L * 60)
        # Local vol — std of 1-min BTC returns over past 5 min
        # Use the 5 lookbacks if available
        events.append({
            "k":       k,
            "t":       t,
            "yes":     yc,
            "btc":     btc_now,
            "btc_lb":  btc_lb,
            "yc_lb":   yc_lb,
            "hod":     datetime.fromtimestamp(t, tz=timezone.utc).hour,
            "to_close_min": (t_close - t) / 60.0,
        })
    if not events:
        return None, None, None
    return events, result, float(strike)


# ── Backtest harness ─────────────────────────────────────────────────────────
def run_strategy(window_data, decide_fn):
    """
    window_data: list of (events, result, strike, prev_result)
    decide_fn(events, strike, prev_result) -> list of (k, side) entries
                                              (entries are evaluated against
                                               that minute's yes_close + slippage,
                                               then filtered by cooldown + caps)
    """
    total_w = 0.0
    total_p = 0.0
    total_tix = 0
    active_w = 0
    winning_w = 0
    yes_tix = 0
    no_tix  = 0

    for events, result, strike, prev_result in window_data:
        raw = decide_fn(events, strike, prev_result)
        if not raw:
            continue
        # Translate (k, side) entries -> actual ticket with band gate handled inside fn
        yes_c = 0.0
        no_c  = 0.0
        wag   = 0.0
        n     = 0
        last_entry_t = -10**9
        ev_by_k = {ev["k"]: ev for ev in events}

        for (k, side, pay_c) in raw:
            ev = ev_by_k.get(k)
            if ev is None:
                continue
            t = ev["t"]
            if n   >= MAX_ENTRIES: continue
            if wag >= MAX_WAGERED: continue
            if t - last_entry_t < COOLDOWN_MIN * 60: continue
            if not (2 <= pay_c <= 98): continue
            contracts = STAKE_DOLLARS / (pay_c / 100.0)
            cost      = contracts * pay_c / 100.0
            if side == "yes":
                yes_c += contracts
                yes_tix += 1
            else:
                no_c  += contracts
                no_tix += 1
            wag += cost
            n   += 1
            last_entry_t = t

        if n == 0:
            continue
        active_w += 1
        gross = (yes_c if result == "yes" else 0.0) + (no_c if result == "no" else 0.0)
        pnl_pre = gross - wag
        fee = FEE_RATE * pnl_pre if pnl_pre > 0 else 0.0
        pnl = pnl_pre - fee
        total_w  += wag
        total_p  += pnl
        total_tix += n
        if pnl > 0:
            winning_w += 1

    return {
        "wagered": total_w, "pnl": total_p, "tickets": total_tix,
        "active_w": active_w, "win_w": winning_w,
        "yes_tix": yes_tix, "no_tix": no_tix,
        "roi": (total_p / total_w * 100) if total_w > 0 else 0.0,
    }


# ── Price helpers ────────────────────────────────────────────────────────────
def _pay_c(yes_close, side):
    """Cost in cents to take 'side' as a market taker at this yes_close."""
    if side == "yes":
        return round(yes_close * 100) + SLIPPAGE_C
    else:
        return 100 - round(yes_close * 100) + SLIPPAGE_C


def _gate(pay_c, band_low, band_high):
    return band_low <= pay_c <= band_high


# ── Strategy archetypes — each returns list of (k, side, pay_c) ──────────────

def mk_mom(trig, lb, blo, bhi, sf):
    def decide(events, strike, prev_result):
        out = []
        for ev in events:
            if lb not in ev["btc_lb"]: continue
            d = ev["btc"] - ev["btc_lb"][lb]
            if abs(d) < trig: continue
            side = "yes" if d > 0 else "no"
            if sf != "both" and side != sf: continue
            pc = _pay_c(ev["yes"], side)
            if not _gate(pc, blo, bhi): continue
            out.append((ev["k"], side, pc))
        return out
    return decide

def mk_rev(trig, lb, blo, bhi, sf):
    def decide(events, strike, prev_result):
        out = []
        for ev in events:
            if lb not in ev["btc_lb"]: continue
            d = ev["btc"] - ev["btc_lb"][lb]
            if abs(d) < trig: continue
            side = "no" if d > 0 else "yes"   # FADE the move
            if sf != "both" and side != sf: continue
            pc = _pay_c(ev["yes"], side)
            if not _gate(pc, blo, bhi): continue
            out.append((ev["k"], side, pc))
        return out
    return decide

def mk_strike_mom(minute, pct, sf):
    """At minute M, if BTC > strike+pct%, bet YES; if < strike-pct%, bet NO."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            diff = (ev["btc"] - strike) / strike * 100
            if diff > pct:
                side = "yes"
            elif diff < -pct:
                side = "no"
            else:
                return []
            if sf != "both" and side != sf: return []
            return [(ev["k"], side, _pay_c(ev["yes"], side))]
        return []
    return decide

def mk_strike_rev(minute, pct, sf):
    """Opposite: bet AGAINST the side BTC has moved toward."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            diff = (ev["btc"] - strike) / strike * 100
            if diff > pct:
                side = "no"
            elif diff < -pct:
                side = "yes"
            else:
                return []
            if sf != "both" and side != sf: return []
            return [(ev["k"], side, _pay_c(ev["yes"], side))]
        return []
    return decide

def mk_fav_follow(minute, thresh):
    """At minute M, if max(YES, 1-YES) > thresh, bet the favorite side."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            yc = ev["yes"]
            if yc >= thresh:
                side = "yes"
            elif 1 - yc >= thresh:
                side = "no"
            else:
                return []
            return [(ev["k"], side, _pay_c(yc, side))]
        return []
    return decide

def mk_fav_fade(minute, thresh):
    """Same trigger, fade the favorite (bet the underdog)."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            yc = ev["yes"]
            if yc >= thresh:
                side = "no"
            elif 1 - yc >= thresh:
                side = "yes"
            else:
                return []
            return [(ev["k"], side, _pay_c(yc, side))]
        return []
    return decide

def mk_extreme_fav(minute, thresh):
    """Only enter when one side >= thresh; bet that side."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            yc = ev["yes"]
            if yc >= thresh:
                side = "yes"
            elif 1 - yc >= thresh:
                side = "no"
            else:
                return []
            return [(ev["k"], side, _pay_c(yc, side))]
        return []
    return decide

def mk_extreme_fade(minute, thresh):
    """Same gate, bet the underdog (mean-reversion bet)."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            yc = ev["yes"]
            if yc >= thresh:
                side = "no"
            elif 1 - yc >= thresh:
                side = "yes"
            else:
                return []
            return [(ev["k"], side, _pay_c(yc, side))]
        return []
    return decide

def mk_early(minute, side, blo, bhi):
    """Pure directional: always enter `side` at minute, IF inside band."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] != minute: continue
            pc = _pay_c(ev["yes"], side)
            if not _gate(pc, blo, bhi): return []
            return [(ev["k"], side, pc)]
        return []
    return decide

def mk_late_hodl(min_k, side, blo, bhi):
    """At first event at or after min_k, take side if in band."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["k"] < min_k: continue
            pc = _pay_c(ev["yes"], side)
            if not _gate(pc, blo, bhi): continue
            return [(ev["k"], side, pc)]
        return []
    return decide

def mk_pin_fade(dollar_proximity, min_to_close, sf):
    """If |BTC - strike| < $X with <= min_to_close minutes left, bet the UNDERDOG."""
    def decide(events, strike, prev_result):
        for ev in events:
            if ev["to_close_min"] > min_to_close: continue
            if abs(ev["btc"] - strike) > dollar_proximity: continue
            yc = ev["yes"]
            # Fade the favorite
            side = "no" if yc >= 0.5 else "yes"
            if sf != "both" and side != sf: continue
            return [(ev["k"], side, _pay_c(yc, side))]
        return []
    return decide

def mk_consec_trend(blo, bhi, only_if_prev):
    """Bet SAME direction as previous window's result. only_if_prev∈{yes,no,any}."""
    def decide(events, strike, prev_result):
        if prev_result is None: return []
        if only_if_prev != "any" and prev_result != only_if_prev: return []
        side = prev_result  # 'yes' or 'no'
        # take at first qualifying minute
        for ev in events:
            pc = _pay_c(ev["yes"], side)
            if _gate(pc, blo, bhi):
                return [(ev["k"], side, pc)]
        return []
    return decide

def mk_consec_rev(blo, bhi, only_if_prev):
    """Bet OPPOSITE of previous window's result."""
    def decide(events, strike, prev_result):
        if prev_result is None: return []
        if only_if_prev != "any" and prev_result != only_if_prev: return []
        side = "no" if prev_result == "yes" else "yes"
        for ev in events:
            pc = _pay_c(ev["yes"], side)
            if _gate(pc, blo, bhi):
                return [(ev["k"], side, pc)]
        return []
    return decide

def mk_kalshi_drift(min_move_c, lb, sf):
    """If Kalshi YES has drifted by min_move_c cents over last lb minutes,
    follow the drift (bet the side it's moving toward)."""
    def decide(events, strike, prev_result):
        out = []
        for ev in events:
            past = ev["yc_lb"].get(lb)
            if past is None: continue
            d_c = round((ev["yes"] - past) * 100)
            if abs(d_c) < min_move_c: continue
            side = "yes" if d_c > 0 else "no"
            if sf != "both" and side != sf: continue
            out.append((ev["k"], side, _pay_c(ev["yes"], side)))
        return out
    return decide


# ── Strategy registry ────────────────────────────────────────────────────────
def build_registry():
    """Return list of (name_str, decide_fn) covering ~1000 unique configs."""
    R = []

    # 1. Momentum on BTC: 5 trig × 4 lb × 5 bands × 3 sides = 300
    for trig in [25, 50, 75, 100, 150]:
        for lb in [1, 2, 3, 5]:
            for (blo, bhi) in [(50,65),(55,70),(60,75),(65,80),(55,80)]:
                for sf in ["both", "yes_only", "no_only"]:
                    name = f"mom_btc[trig={trig},lb={lb}m,band={blo}-{bhi},sf={sf}]"
                    R.append((name, mk_mom(trig, lb, blo, bhi, sf)))

    # 2. Reversion on BTC: same grid = 300
    for trig in [25, 50, 75, 100, 150]:
        for lb in [1, 2, 3, 5]:
            for (blo, bhi) in [(50,65),(55,70),(60,75),(65,80),(55,80)]:
                for sf in ["both", "yes_only", "no_only"]:
                    name = f"rev_btc[trig={trig},lb={lb}m,band={blo}-{bhi},sf={sf}]"
                    R.append((name, mk_rev(trig, lb, blo, bhi, sf)))

    # 3. Strike-momentum: 5 minutes × 4 thresholds × 3 sides = 60
    for m in [2, 4, 6, 8, 10]:
        for pct in [0.05, 0.1, 0.2, 0.3]:
            for sf in ["both", "yes_only", "no_only"]:
                R.append((f"strike_mom[m={m},pct={pct}%,sf={sf}]",
                          mk_strike_mom(m, pct, sf)))

    # 4. Strike-reversion: 60
    for m in [2, 4, 6, 8, 10]:
        for pct in [0.05, 0.1, 0.2, 0.3]:
            for sf in ["both", "yes_only", "no_only"]:
                R.append((f"strike_rev[m={m},pct={pct}%,sf={sf}]",
                          mk_strike_rev(m, pct, sf)))

    # 5. Favorite-follow: 5 minutes × 5 thresholds = 25
    for m in [3, 5, 7, 9, 11]:
        for th in [0.55, 0.60, 0.65, 0.70, 0.80]:
            R.append((f"fav_follow[m={m},th={th}]", mk_fav_follow(m, th)))

    # 6. Favorite-fade: 25
    for m in [3, 5, 7, 9, 11]:
        for th in [0.55, 0.60, 0.65, 0.70, 0.80]:
            R.append((f"fav_fade[m={m},th={th}]", mk_fav_fade(m, th)))

    # 7. Extreme favorite: 5 minutes × 3 thresholds = 15
    for m in [4, 6, 8, 10, 12]:
        for th in [0.80, 0.85, 0.90]:
            R.append((f"extreme_fav[m={m},th={th}]", mk_extreme_fav(m, th)))

    # 8. Extreme fade: 15
    for m in [4, 6, 8, 10, 12]:
        for th in [0.80, 0.85, 0.90]:
            R.append((f"extreme_fade[m={m},th={th}]", mk_extreme_fade(m, th)))

    # 9. Early-YES: 5 minutes × 3 bands = 15
    for m in [1, 2, 3, 5, 7]:
        for (blo, bhi) in [(30,70),(40,70),(45,65)]:
            R.append((f"early_yes[m={m},band={blo}-{bhi}]", mk_early(m, "yes", blo, bhi)))

    # 10. Early-NO: 15
    for m in [1, 2, 3, 5, 7]:
        for (blo, bhi) in [(30,70),(40,70),(45,65)]:
            R.append((f"early_no[m={m},band={blo}-{bhi}]", mk_early(m, "no", blo, bhi)))

    # 11. Late-hodl: 3 min_k × 2 sides × 3 bands = 18
    for min_k in [10, 11, 12]:
        for side in ["yes", "no"]:
            for (blo, bhi) in [(50,70),(60,80),(70,90)]:
                R.append((f"late_hodl[min_k={min_k},side={side},band={blo}-{bhi}]",
                          mk_late_hodl(min_k, side, blo, bhi)))

    # 12. Pin-fade: 3 prox × 3 min_to_close × 3 sf = 27
    for prox in [25, 50, 100]:
        for mtc in [3, 5, 8]:
            for sf in ["both", "yes_only", "no_only"]:
                R.append((f"pin_fade[prox=${prox},mtc={mtc}m,sf={sf}]",
                          mk_pin_fade(prox, mtc, sf)))

    # 13. Consecutive-trend: 4 bands × 3 prev_filter = 12
    for (blo, bhi) in [(50,70),(55,75),(60,80),(45,85)]:
        for prev in ["any", "yes", "no"]:
            R.append((f"consec_trend[band={blo}-{bhi},prev_was={prev}]",
                      mk_consec_trend(blo, bhi, prev)))

    # 14. Consecutive-reversal: 12
    for (blo, bhi) in [(50,70),(55,75),(60,80),(45,85)]:
        for prev in ["any", "yes", "no"]:
            R.append((f"consec_rev[band={blo}-{bhi},prev_was={prev}]",
                      mk_consec_rev(blo, bhi, prev)))

    # 15. Kalshi drift: 4 thresh × 3 lb × 3 sf = 36
    for mc in [3, 5, 8, 12]:
        for lb in [1, 2, 3]:
            for sf in ["both", "yes_only", "no_only"]:
                R.append((f"kalshi_drift[move>={mc}c,lb={lb}m,sf={sf}]",
                          mk_kalshi_drift(mc, lb, sf)))

    return R


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading settled markets…")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    print(f"  {len(markets)} markets")

    dated = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z","+00:00"))
            dated.append((dt, m))
        except Exception:
            pass
    dated.sort(key=lambda x: x[0])

    timestamps = [int(d.timestamp()) for d,_ in dated]
    print("Loading BTC prices…")
    btc_prices = btc_data.fetch_btc_prices(min(timestamps) - 600, max(timestamps) + 1800)
    print(f"  {len(btc_prices)} BTC minute-prices")

    sorted_markets = [m for _,m in dated]
    split = int(len(sorted_markets) * 0.8)
    train_m = sorted_markets[:split]
    test_m  = sorted_markets[split:]
    print(f"Train: {len(train_m)} windows  ({dated[0][0].date()} → {dated[split-1][0].date()})")
    print(f"Test:  {len(test_m)} windows  ({dated[split][0].date()} → {dated[-1][0].date()})")

    print("\nPre-computing events for all windows…")
    t0 = time.time()
    train_data = []
    test_data  = []

    # Track previous result chronologically
    prev_result = None
    for m in train_m:
        events, result, strike = build_window_events(m, btc_prices)
        if events is None:
            train_data.append((None, None, None, prev_result))
            continue
        train_data.append((events, result, strike, prev_result))
        prev_result = result
    # The boundary: test starts with prev_result = last train window's result
    for m in test_m:
        events, result, strike = build_window_events(m, btc_prices)
        if events is None:
            test_data.append((None, None, None, prev_result))
            continue
        test_data.append((events, result, strike, prev_result))
        prev_result = result

    # Filter out non-events
    train_data = [w for w in train_data if w[0] is not None]
    test_data  = [w for w in test_data  if w[0] is not None]
    print(f"  built {len(train_data)} train + {len(test_data)} test windows in {time.time()-t0:.1f}s")

    # Build registry
    registry = build_registry()
    print(f"\nStrategy zoo: {len(registry)} unique strategies")
    print("Running sweep…")
    t1 = time.time()

    results = []
    for ci, (name, fn) in enumerate(registry):
        train_r = run_strategy(train_data, fn)
        test_r  = run_strategy(test_data,  fn)
        results.append({"name": name, "train": train_r, "test": test_r})
        if (ci+1) % 100 == 0:
            print(f"  {ci+1}/{len(registry)}  ({time.time()-t1:.1f}s)")
    print(f"Sweep done in {time.time()-t1:.1f}s")

    # ── Reports ──────────────────────────────────────────────────────────────
    def fmt_row(r, label):
        x = r[label]
        if x["tickets"] == 0:
            return f"    {label:>5}: (no tickets)"
        win_pct = x["win_w"] / x["active_w"] * 100 if x["active_w"] else 0
        return (f"    {label:>5}: ROI={x['roi']:+6.2f}%  P&L=${x['pnl']:+8.2f}  "
                f"wag=${x['wagered']:8.2f}  tix={x['tickets']:>5}  "
                f"win_w={x['win_w']}/{x['active_w']} ({win_pct:.0f}%)")

    MIN_TX_TRAIN = 150
    MIN_TX_TEST  = 40

    qual_train = [r for r in results if r["train"]["tickets"] >= MIN_TX_TRAIN]
    qual_train.sort(key=lambda r: r["train"]["roi"], reverse=True)

    print("\n" + "="*100)
    print(f"TOP 10 BY TRAIN ROI  (≥{MIN_TX_TRAIN} train tix)")
    print("="*100)
    for r in qual_train[:10]:
        print(f"  {r['name']}")
        print(fmt_row(r, "train"))
        print(fmt_row(r, "test"))
        print()

    # Honest: positive in BOTH with decent test sample
    honest = [r for r in qual_train
              if r["train"]["roi"] > 0
              and r["test"]["tickets"] >= MIN_TX_TEST
              and r["test"]["roi"] > 0]
    honest.sort(key=lambda r: min(r["train"]["roi"], r["test"]["roi"]), reverse=True)
    print("="*100)
    print(f"POSITIVE IN BOTH TRAIN AND TEST  (train+ AND test+, ≥{MIN_TX_TEST} test tix)  n={len(honest)}")
    print("Sorted by MIN(train_roi, test_roi) — the most conservative read.")
    print("="*100)
    for r in honest[:15]:
        print(f"  {r['name']}")
        print(fmt_row(r, "train"))
        print(fmt_row(r, "test"))
        print()

    # Bucket by archetype: how does each family do?
    def archetype(name):
        return name.split("[")[0]
    by_arch = {}
    for r in results:
        a = archetype(r["name"])
        by_arch.setdefault(a, []).append(r)
    print("="*100)
    print("ARCHETYPE PERFORMANCE  (count of train+, test+, both+ in each family)")
    print("="*100)
    print(f"  {'archetype':<20}  {'n':>5}  {'q_train':>8}  {'pos_train':>9}  {'pos_test':>9}  {'pos_both':>9}")
    for a in sorted(by_arch.keys()):
        rs = by_arch[a]
        n = len(rs)
        q = sum(1 for r in rs if r["train"]["tickets"] >= MIN_TX_TRAIN)
        pt = sum(1 for r in rs if r["train"]["tickets"] >= MIN_TX_TRAIN and r["train"]["roi"] > 0)
        ps = sum(1 for r in rs if r["test"]["tickets"]  >= MIN_TX_TEST  and r["test"]["roi"]  > 0)
        pb = sum(1 for r in rs if r["train"]["tickets"] >= MIN_TX_TRAIN
                              and r["test"]["tickets"]  >= MIN_TX_TEST
                              and r["train"]["roi"] > 0 and r["test"]["roi"] > 0)
        print(f"  {a:<20}  {n:>5}  {q:>8}  {pt:>9}  {ps:>9}  {pb:>9}")

    # Aggregate
    print("="*100)
    n_qual = len(qual_train)
    pt = sum(1 for r in qual_train if r["train"]["roi"] > 0)
    ps = sum(1 for r in results if r["test"]["tickets"] >= MIN_TX_TEST and r["test"]["roi"] > 0)
    pb = len(honest)
    print(f"  Strategies:               {len(results)}")
    print(f"  Qualified train (≥{MIN_TX_TRAIN}):     {n_qual}")
    print(f"  Positive train:           {pt}/{n_qual} ({pt/max(n_qual,1)*100:.1f}%)")
    print(f"  Positive test:            {ps} (with ≥{MIN_TX_TEST} test tix)")
    print(f"  Positive in BOTH:         {pb}")
    print(f"  Persistence (train+ → both+): {pb/max(pt,1)*100:.1f}%")

    # Save full CSV for inspection
    csv_path = os.path.join(os.path.dirname(__file__), "data", "logs", "strategy_zoo_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("name,train_roi,train_pnl,train_wagered,train_tix,train_win_w,train_active_w,"
                "test_roi,test_pnl,test_wagered,test_tix,test_win_w,test_active_w\n")
        for r in results:
            tr, te = r["train"], r["test"]
            f.write(f"\"{r['name']}\","
                    f"{tr['roi']:.4f},{tr['pnl']:.2f},{tr['wagered']:.2f},{tr['tickets']},{tr['win_w']},{tr['active_w']},"
                    f"{te['roi']:.4f},{te['pnl']:.2f},{te['wagered']:.2f},{te['tickets']},{te['win_w']},{te['active_w']}\n")
    print(f"\nFull results CSV: {csv_path}")


if __name__ == "__main__":
    main()
