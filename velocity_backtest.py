"""
Velocity-based latency-arb backtest with chronological train/test split.

CAVEAT: cached BTC data is 1-min granularity, so the "velocity" we test here
is over 1+ minute windows — much slower than the 5–10s velocity the live
latency-arb mode actually uses. This test only tells us if velocity *as a
direction signal* is even useful at minute resolution. Real edge from a
true latency-arb strategy needs sub-second feed latency, which this can't
simulate.

Strategies tested:
    "v1m"  — direction = sign of (btc_now − btc_1min_ago), single bet at fixed minute
    "v2m"  — same with 2-min lookback (more stable signal)
    "v_thresh" — only trade if |velocity| > threshold

Comparison baseline:
    "magnitude" — direction = sign of (btc_now − btc_t0), as current dh-target uses
"""

import math
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import btc_data
import kalshi_client
from config import DATA_DAYS, KALSHI_SERIES, BTC_SYMBOL, FEE_RATE

BASE_STAKE = 20.0
TRAIN_FRAC = 0.70


def load_data():
    print(f"Loading markets...")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    markets = [m for m in markets if m.get("result") in ("yes","no") and m.get("open_time")]
    markets.sort(key=lambda m: m["open_time"])
    print(f"  {len(markets)} markets")
    print("Loading BTC...")
    timestamps = [datetime.fromisoformat(m["open_time"].replace("Z","+00:00")).timestamp() for m in markets]
    btc = btc_data.fetch_btc_prices(int(min(timestamps))-600, int(max(timestamps))+1800)
    print(f"  {len(btc)} BTC samples (1-min)")
    return markets, btc


def simulate(markets, btc, signal_fn, params, label):
    """Each strategy: signal_fn(features, params) → (side, stake) at entry minute."""
    pnl_total = 0; wagered_sum = 0; pnls = []
    for m in markets:
        try:
            t0 = int(datetime.fromisoformat(m["open_time"].replace("Z","+00:00")).timestamp())
        except: continue
        resolved = m["result"] == "yes"
        btc_t0 = btc_data.lookup(btc, t0)
        if btc_t0 is None: continue

        candles = kalshi_client.fetch_candlesticks(m["ticker"], m["open_time"], m["close_time"])
        if not candles: continue

        # Single-bet strategy: enter at params['minute']
        entry_min = params["entry_minute"]
        t_entry   = t0 + entry_min * 60
        btc_entry = btc_data.lookup(btc, t_entry)
        if btc_entry is None: continue

        # Velocity at entry: difference btc_entry vs btc_(entry - lookback_min)
        lookback_min = params.get("lookback_min", 1)
        t_prior = t_entry - lookback_min * 60
        btc_prior = btc_data.lookup(btc, t_prior)
        if btc_prior is None: continue
        velocity = (btc_entry - btc_prior) / btc_prior * 100.0   # % over lookback period
        magnitude = (btc_entry - btc_t0) / btc_t0 * 100.0        # % since window open
        quote = kalshi_client.get_quotes_at(candles, t_entry)
        if quote is None: continue
        kalshi_yes = quote["yes_close"]
        if not (0.01 < kalshi_yes < 0.99): continue

        features = {
            "btc_t0": btc_t0, "btc_entry": btc_entry, "btc_prior": btc_prior,
            "velocity": velocity, "magnitude": magnitude,
            "kalshi_yes": kalshi_yes, "resolved": resolved,
        }
        side = signal_fn(features, params)
        if side is None: continue

        # Place single $20 bet at the EXECUTABLE price (cross the book)
        if side == "yes":
            fill = quote.get("ask", kalshi_yes)
        else:
            fill = 1 - quote.get("bid", kalshi_yes)
        if not (0.02 < fill < 0.98): continue
        stake = params["stake"]
        wagered_sum += stake
        # corrected Kalshi fee: ceil(0.07*C*P*(1-P)) per fill, charged win or lose
        fee = math.ceil(FEE_RATE * stake * (1 - fill) * 100) / 100
        if (side == "yes") == resolved:
            pnl = stake * (1 - fill) / fill - fee
        else:
            pnl = -stake - fee
        pnl_total += pnl
        pnls.append(pnl)

    if not pnls: return None
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    avg = pnl_total / n
    std = (sum((p-avg)**2 for p in pnls)/n)**0.5 if n>1 else 0
    return {
        "label": label, "n": n, "pnl": round(pnl_total, 2),
        "roi_pct": round(pnl_total/wagered_sum*100, 2) if wagered_sum else 0,
        "win_rate": round(wins/n, 3), "sharpe": round(avg/std, 3) if std>0 else 0,
        "worst": round(min(pnls), 2), "best": round(max(pnls), 2),
    }


# ── Signal functions ────────────────────────────────────────────────────────

def magnitude_signal(f, p):
    """Baseline: direction = since-window-open."""
    if abs(f["magnitude"]) < p.get("min_pct", 0): return None
    return "yes" if f["btc_entry"] > f["btc_t0"] else "no"

def velocity_signal(f, p):
    """Bet on direction of recent velocity (last lookback_min minutes)."""
    if abs(f["velocity"]) < p.get("min_velocity", 0): return None
    return "yes" if f["velocity"] > 0 else "no"

def velocity_aligned(f, p):
    """Only bet if velocity AND magnitude agree (filtering)."""
    if abs(f["velocity"]) < p.get("min_velocity", 0): return None
    if (f["velocity"] > 0) != (f["magnitude"] > 0): return None
    return "yes" if f["velocity"] > 0 else "no"

def velocity_contra_magnitude(f, p):
    """Bet on velocity direction even when it disagrees with magnitude (mean revert)."""
    if abs(f["velocity"]) < p.get("min_velocity", 0): return None
    return "yes" if f["velocity"] > 0 else "no"


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    markets, btc = load_data()
    n_train = int(len(markets) * TRAIN_FRAC)
    train, test = markets[:n_train], markets[n_train:]
    print(f"\nTrain: {train[0]['open_time'][:10]} → {train[-1]['open_time'][:10]} ({len(train)} mkts)")
    print(f"Test:  {test[0]['open_time'][:10]} → {test[-1]['open_time'][:10]} ({len(test)} mkts)\n")

    cases = [
        # ── Baselines: magnitude-only (current strategy proxy) ──
        ("MAGNITUDE T+10",          magnitude_signal,            {"entry_minute": 10, "stake": BASE_STAKE, "min_pct": 0.0}),
        ("MAGNITUDE T+12",          magnitude_signal,            {"entry_minute": 12, "stake": BASE_STAKE, "min_pct": 0.0}),
        ("MAGNITUDE T+12 ≥0.10%",   magnitude_signal,            {"entry_minute": 12, "stake": BASE_STAKE, "min_pct": 0.10}),
        # ── Velocity over 1-min lookback ──
        ("VELOCITY 1m T+5",         velocity_signal,             {"entry_minute": 5,  "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.0}),
        ("VELOCITY 1m T+5 ≥0.05%",  velocity_signal,             {"entry_minute": 5,  "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.05}),
        ("VELOCITY 1m T+5 ≥0.10%",  velocity_signal,             {"entry_minute": 5,  "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.10}),
        ("VELOCITY 1m T+8",         velocity_signal,             {"entry_minute": 8,  "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.0}),
        ("VELOCITY 1m T+10",        velocity_signal,             {"entry_minute": 10, "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.0}),
        ("VELOCITY 1m T+10 ≥0.05%", velocity_signal,             {"entry_minute": 10, "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.05}),
        # ── Velocity over 2-min ──
        ("VELOCITY 2m T+10",        velocity_signal,             {"entry_minute": 10, "stake": BASE_STAKE, "lookback_min": 2, "min_velocity": 0.0}),
        ("VELOCITY 2m T+10 ≥0.10%", velocity_signal,             {"entry_minute": 10, "stake": BASE_STAKE, "lookback_min": 2, "min_velocity": 0.10}),
        # ── Filters ──
        ("V+M aligned T+10 ≥0.05%", velocity_aligned,            {"entry_minute": 10, "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.05}),
        ("V+M aligned T+12 ≥0.10%", velocity_aligned,            {"entry_minute": 12, "stake": BASE_STAKE, "lookback_min": 1, "min_velocity": 0.10}),
    ]

    print(f"{'strategy':<28} {'split':<6} {'n':>5} {'pnl':>8} {'roi%':>7} {'win%':>6} {'sharpe':>7} {'worst':>7} {'best':>7}")
    print("-"*92)
    for label, fn, params in cases:
        tr = simulate(train, btc, fn, params, label)
        te = simulate(test,  btc, fn, params, label)
        if tr: print(f"{label:<28} {'train':<6} {tr['n']:>5} {tr['pnl']:>+8.0f} {tr['roi_pct']:>+7.2f} {tr['win_rate']*100:>6.1f} {tr['sharpe']:>+7.3f} {tr['worst']:>+7.0f} {tr['best']:>+7.0f}")
        if te: print(f"{label:<28} {'TEST ':<6} {te['n']:>5} {te['pnl']:>+8.0f} {te['roi_pct']:>+7.2f} {te['win_rate']*100:>6.1f} {te['sharpe']:>+7.3f} {te['worst']:>+7.0f} {te['best']:>+7.0f}")
        if tr and te:
            print(f"{'':28} {'gap':<6} {'':>5} {'':>8} {te['roi_pct']-tr['roi_pct']:>+7.2f}")
        print()


if __name__ == "__main__":
    main()
