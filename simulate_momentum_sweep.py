"""
Parameter sweep over the momentum-scalper strategy family.

Search space (1176 configs):
  trigger_usd     ∈ {20, 30, 40, 50, 75, 100, 150}
  lookback_min    ∈ {1, 2, 3, 5}
  (band_low_c, band_high_c) ∈ 7 pairs
  mode            ∈ {momentum, reversion}
  side_filter     ∈ {both, yes_only, no_only}

For each config we report: TRAIN ROI, TEST ROI, ticket count, win rate.

Honest methodology:
  - 80/20 chrono split (oldest 80% = train, newest 20% = test)
  - Pick winners by TRAIN performance, show their TEST result alongside
  - Also report the OOS-best config — but with a caveat: top-by-test is
    cherry-picked, so it's selection bias unless TRAIN also liked it

Pre-loads all 1-min Kalshi candles + BTC prices into memory once, then
walks every window emitting a (minute, btc_now, btc_lb1, btc_lb2, btc_lb3,
btc_lb5, yes_close) tuple list. Each config then just filters these tuples
— no I/O in the inner loops.

Run:  python3 simulate_momentum_sweep.py
"""
import os
import sys
import time
import itertools
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import DATA_DAYS, FEE_RATE

# Fixed scaffolding (don't sweep — these don't affect edge)
STAKE_DOLLARS = 3.0
COOLDOWN_MIN  = 1
MAX_ENTRIES   = 4
MAX_WAGERED   = 20.0
STOP_LAST_MIN = 2
SLIPPAGE_C    = 1

LOOKBACK_OPTIONS = [1, 2, 3, 5]   # we precompute all of these per window

# ── Sweep grid ───────────────────────────────────────────────────────────────
TRIGGER_GRID     = [20, 30, 40, 50, 75, 100, 150]
BAND_GRID        = [(50,65),(55,70),(60,75),(65,80),(70,85),(55,80),(60,80)]
MODE_GRID        = ["momentum", "reversion"]
SIDE_GRID        = ["both", "yes_only", "no_only"]

# Minimum tickets to take a config seriously (avoid lucky 3-trade winners)
MIN_TICKETS_TRAIN = 100
MIN_TICKETS_TEST  = 30


def _btc_close_at(prices, t):
    """Price at time t = close of candle ending at t (Coinbase keys by start)."""
    return btc_data.lookup(prices, t - 60)


def _build_window_events(market, btc_prices):
    """
    Pre-compute the decision-point series for one window.
    Returns list of dicts: {k, t, yes_close, btc_now, btc_lb: {1: x, 2: y, ...}}
    Returns None if window is unusable.
    """
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    if result not in ("yes", "no") or not open_iso or not close_iso:
        return None, None
    try:
        t0      = int(datetime.fromisoformat(open_iso .replace("Z","+00:00")).timestamp())
        t_close = int(datetime.fromisoformat(close_iso.replace("Z","+00:00")).timestamp())
    except Exception:
        return None, None
    candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
    if not candles:
        return None, None

    events = []
    for k in range(1, 15):
        t = t0 + k * 60
        if t > t_close - STOP_LAST_MIN * 60:
            break
        btc_now = _btc_close_at(btc_prices, t)
        if btc_now is None:
            continue
        yes_close = kalshi_client.get_yes_price_at(candles, t)
        if yes_close is None or not (0.01 < yes_close < 0.99):
            continue
        lb = {}
        for L in LOOKBACK_OPTIONS:
            v = _btc_close_at(btc_prices, t - L * 60)
            if v is not None:
                lb[L] = v
        if not lb:
            continue
        events.append({"k": k, "t": t, "yes_close": yes_close, "btc_now": btc_now, "lb": lb})
    if not events:
        return None, None
    return events, result


# ── Per-config inner loop ────────────────────────────────────────────────────
def run_config(window_data, trigger, lookback, band_low, band_high, mode, side_filter):
    """Sum (wagered, payout, n_tix, n_winning_windows, n_total_active_windows)."""
    total_wagered = 0.0
    total_payout  = 0.0
    total_tix     = 0
    total_pnl     = 0.0     # for stats
    winning_w     = 0
    active_w      = 0
    yes_tix       = 0
    no_tix        = 0

    for events, result in window_data:
        if lookback not in events[0]["lb"]:
            # Skip windows missing this lookback's btc data
            # (rare; usually only first few minutes)
            pass

        yes_c = 0.0
        no_c  = 0.0
        wag_w = 0.0
        n_w   = 0
        last_entry_t = -10**9

        for ev in events:
            if lookback not in ev["lb"]:
                continue
            delta = ev["btc_now"] - ev["lb"][lookback]
            if abs(delta) < trigger:
                continue

            if mode == "momentum":
                signal_side = "yes" if delta > 0 else "no"
            else:  # reversion
                signal_side = "no" if delta > 0 else "yes"

            if side_filter == "yes_only" and signal_side != "yes":
                continue
            if side_filter == "no_only"  and signal_side != "no":
                continue

            if signal_side == "yes":
                pay_c = round(ev["yes_close"] * 100) + SLIPPAGE_C
            else:
                pay_c = 100 - round(ev["yes_close"] * 100) + SLIPPAGE_C
            if not (band_low <= pay_c <= band_high):
                continue

            if n_w   >= MAX_ENTRIES: continue
            if wag_w >= MAX_WAGERED: continue
            if ev["t"] - last_entry_t < COOLDOWN_MIN * 60: continue

            contracts = STAKE_DOLLARS / (pay_c / 100.0)
            cost      = contracts * pay_c / 100.0
            if signal_side == "yes":
                yes_c += contracts
                yes_tix += 1
            else:
                no_c  += contracts
                no_tix += 1
            wag_w += cost
            n_w   += 1
            last_entry_t = ev["t"]

        if n_w == 0:
            continue
        active_w += 1
        # Settlement
        gross = (yes_c if result == "yes" else 0.0) + (no_c if result == "no" else 0.0)
        pnl_pre = gross - wag_w
        fee = FEE_RATE * pnl_pre if pnl_pre > 0 else 0.0
        pnl_net = pnl_pre - fee

        total_wagered += wag_w
        total_payout  += gross
        total_tix     += n_w
        total_pnl     += pnl_net
        if pnl_net > 0:
            winning_w += 1

    return {
        "wagered":  total_wagered,
        "payout":   total_payout,
        "pnl":      total_pnl,
        "tickets":  total_tix,
        "yes_tix":  yes_tix,
        "no_tix":   no_tix,
        "active_w": active_w,
        "win_w":    winning_w,
        "roi":      (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0,
    }


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

    print("\nBuilding event series for all windows (this is the slow part)…")
    t0 = time.time()
    train_data = []
    test_data  = []
    skipped = 0
    for i, m in enumerate(train_m):
        events, result = _build_window_events(m, btc_prices)
        if events is None:
            skipped += 1
            continue
        train_data.append((events, result))
        if (i+1) % 1000 == 0:
            print(f"  train {i+1}/{len(train_m)}…")
    for i, m in enumerate(test_m):
        events, result = _build_window_events(m, btc_prices)
        if events is None:
            skipped += 1
            continue
        test_data.append((events, result))
    print(f"  built {len(train_data)} train + {len(test_data)} test windows "
          f"(skipped {skipped}) in {time.time()-t0:.1f}s")

    # Build grid
    grid = list(itertools.product(TRIGGER_GRID, LOOKBACK_OPTIONS, BAND_GRID, MODE_GRID, SIDE_GRID))
    print(f"\nSweeping {len(grid)} configs across {len(train_data)} train + {len(test_data)} test windows…")

    results = []
    sweep_t0 = time.time()
    for ci, (trig, lb, (blo, bhi), mode, sfilt) in enumerate(grid):
        train_r = run_config(train_data, trig, lb, blo, bhi, mode, sfilt)
        test_r  = run_config(test_data,  trig, lb, blo, bhi, mode, sfilt)
        results.append({
            "cfg":  (trig, lb, blo, bhi, mode, sfilt),
            "train": train_r,
            "test":  test_r,
        })
        if (ci+1) % 100 == 0:
            print(f"  {ci+1}/{len(grid)} configs ({time.time()-sweep_t0:.1f}s)")
    print(f"Sweep done in {time.time()-sweep_t0:.1f}s")

    # ── Reports ──────────────────────────────────────────────────────────────
    def fmt_cfg(c):
        trig, lb, blo, bhi, mode, sf = c
        return (f"trig=${trig:>3} lb={lb}m band={blo:>2}-{bhi:>2}c "
                f"mode={mode:<9} side={sf:<8}")

    def fmt_row(r, label):
        x = r[label]
        if x["tickets"] == 0:
            return f"  {label:>5}: (no tickets)"
        win_pct = x["win_w"] / x["active_w"] * 100 if x["active_w"] else 0
        return (f"  {label:>5}: ROI={x['roi']:+6.2f}%  "
                f"P&L=${x['pnl']:+7.2f}  wag=${x['wagered']:7.2f}  "
                f"tix={x['tickets']:>4}  win_w={x['win_w']}/{x['active_w']} ({win_pct:.0f}%)")

    # Sort by TRAIN ROI (only meaningful sample sizes)
    qualified_train = [r for r in results if r["train"]["tickets"] >= MIN_TICKETS_TRAIN]
    qualified_train.sort(key=lambda r: r["train"]["roi"], reverse=True)

    print("\n" + "="*100)
    print(f"TOP 10 BY TRAIN ROI  (require ≥{MIN_TICKETS_TRAIN} train tickets)")
    print("="*100)
    for r in qualified_train[:10]:
        print(fmt_cfg(r["cfg"]))
        print(fmt_row(r, "train"))
        print(fmt_row(r, "test"))
        print()

    # Bottom 5 (most negative train) — sanity check the "opposite" strategy
    print("="*100)
    print(f"BOTTOM 5 BY TRAIN ROI (the worst — useful as a sanity check)")
    print("="*100)
    for r in qualified_train[-5:][::-1]:
        print(fmt_cfg(r["cfg"]))
        print(fmt_row(r, "train"))
        print(fmt_row(r, "test"))
        print()

    # Top by TEST (caveat: selection bias)
    qualified_test = [r for r in results if r["test"]["tickets"] >= MIN_TICKETS_TEST]
    qualified_test.sort(key=lambda r: r["test"]["roi"], reverse=True)
    print("="*100)
    print(f"TOP 10 BY TEST ROI  (≥{MIN_TICKETS_TEST} test tickets) — CHERRY-PICKED, beware overfit")
    print("="*100)
    for r in qualified_test[:10]:
        print(fmt_cfg(r["cfg"]))
        print(fmt_row(r, "train"))
        print(fmt_row(r, "test"))
        print()

    # Best honest config: positive in BOTH and ranked by min(train, test) ROI
    honest = [r for r in qualified_train
              if r["train"]["roi"] > 0
              and r["test"]["tickets"] >= MIN_TICKETS_TEST
              and r["test"]["roi"] > 0]
    honest.sort(key=lambda r: min(r["train"]["roi"], r["test"]["roi"]), reverse=True)
    print("="*100)
    print(f"BEST CONFIGS POSITIVE IN *BOTH* TRAIN AND TEST  (n={len(honest)})")
    print("="*100)
    if not honest:
        print("  (none — strategy family has no robustly positive config in this data)")
    else:
        for r in honest[:10]:
            print(fmt_cfg(r["cfg"]))
            print(fmt_row(r, "train"))
            print(fmt_row(r, "test"))
            print()

    # Summary: of all configs, how many positive train? positive test? both?
    pos_train = sum(1 for r in results if r["train"]["tickets"] >= MIN_TICKETS_TRAIN and r["train"]["roi"] > 0)
    pos_test  = sum(1 for r in results if r["test"]["tickets"]  >= MIN_TICKETS_TEST  and r["test"]["roi"]  > 0)
    pos_both  = len(honest)
    n_qual    = len(qualified_train)
    print("="*100)
    print("AGGREGATE")
    print("="*100)
    print(f"  Qualified configs (train tix ≥ {MIN_TICKETS_TRAIN}): {n_qual}/{len(results)}")
    print(f"  Positive in train: {pos_train}/{n_qual} ({pos_train/max(n_qual,1)*100:.1f}%)")
    print(f"  Positive in test:  {pos_test} configs")
    print(f"  Positive in BOTH:  {pos_both}")
    if pos_train > 0:
        ratio = pos_both / pos_train
        print(f"  Persistence rate (train+ that survive test+): {ratio*100:.1f}%")


if __name__ == "__main__":
    main()
