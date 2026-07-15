"""
Strategy lab: 5 distinct trading strategies, each with parameter variants,
backtested with a chronological train/test split for honest OOS evaluation.

Strategies:
  S1 — single late-window bet (no scaling, fixed minute, fixed stake)
  S2 — momentum DH (current strategy) with refit sigmoid params
  S3 — mispricing-only entry (no BTC scaling; bet only when |mispricing| > threshold)
  S4 — volatility-gated entry (only trade if BTC moved > threshold by T+5; ride continuation)
  S5 — extreme-price fade (when kalshi_yes > 0.85 or < 0.15, fade it)

Each strategy returns (side, stake) per tick. Strategies are pure: same input → same decision.
PnL is computed from actual settlement.

Run:
    python strategy_lab.py
"""

import json
import math
import os
from datetime import datetime, timezone
from collections import defaultdict

import btc_data
import kalshi_client
from config import CACHE_DIR, DATA_DAYS, KALSHI_SERIES, BTC_SYMBOL, FEE_RATE

BASE_STAKE = 20.0   # match live config
MIN_BET    = 1.0
TRAIN_FRAC = 0.70


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading markets ({DATA_DAYS} days)...")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    markets = [m for m in markets if m.get("result") in ("yes", "no")
               and m.get("open_time") and m.get("close_time")]
    markets.sort(key=lambda m: m["open_time"])
    print(f"  {len(markets)} settled markets")

    print("Loading BTC prices...")
    timestamps = [datetime.fromisoformat(m["open_time"].replace("Z","+00:00")).timestamp() for m in markets]
    btc_prices = btc_data.fetch_btc_prices(int(min(timestamps)) - 600, int(max(timestamps)) + 1800)
    print(f"  {len(btc_prices)} BTC price points")
    return markets, btc_prices


def load_2d_fair_price():
    """2D table: (minute, mag_bucket) -> win_rate. Used by S2 momentum strategy."""
    path = os.path.join("data", "logs", "minute_analysis_2d.csv")
    table = {}
    if not os.path.exists(path): return table
    with open(path) as f:
        lines = f.readlines()
    bucket_map = {"0.00-0.05%": 0, "0.05-0.10%": 1, "0.10-0.20%": 2, "0.20-0.50%": 3, "0.50%+": 4}
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 4: continue
        try:
            minute = int(parts[0]); bi = bucket_map.get(parts[1])
            n = int(parts[2]); wr = float(parts[3])
            if bi is not None and n >= 30:
                table[(minute, bi)] = wr
        except: continue
    return table

FAIR_2D = load_2d_fair_price()

def get_fair(minute, abs_pct):
    """Bucket abs_pct, look up fair price from 2D table."""
    if abs_pct < 0.05:    bi = 0
    elif abs_pct < 0.10:  bi = 1
    elif abs_pct < 0.20:  bi = 2
    elif abs_pct < 0.50:  bi = 3
    else:                  bi = 4
    return FAIR_2D.get((minute, bi), 0.65)  # fallback


# ────────────────────────────────────────────────────────────────────────────
# Per-market simulator: walks ticks, calls strategy, computes PnL
# ────────────────────────────────────────────────────────────────────────────

def simulate_market(market, btc_prices, strategy_fn, params):
    """Returns (pnl, wagered, n_trades) for one market under given strategy."""
    open_dt   = datetime.fromisoformat(market["open_time"].replace("Z","+00:00"))
    t0        = int(open_dt.timestamp())
    resolved  = market["result"] == "yes"
    btc_t0    = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None: return None
    candles = kalshi_client.fetch_candlesticks(market["ticker"], market["open_time"], market["close_time"])
    if not candles: return None
    kalshi_t0 = candles[0].get("yes_open")
    if kalshi_t0 is None or not (0.01 < kalshi_t0 < 0.99): return None

    # Build per-minute snapshot
    yes_exposure = 0.0
    no_exposure  = 0.0
    yes_bets = []  # list of (stake, fill_price)
    no_bets  = []

    # Tick range: T+1 .. T+14 minutes
    for minute in range(1, 15):
        t = t0 + minute * 60
        btc_t = btc_data.lookup(btc_prices, t)
        row = kalshi_client.get_quotes_at(candles, t)
        if btc_t is None or row is None: continue
        kalshi_yes = row["yes_close"]
        if not (0.01 < kalshi_yes < 0.99): continue
        # Executable taker fills: cross the book, not the last trade print
        # (signals still see the trade print — that's observable info).
        yes_fill = row.get("ask", kalshi_yes)
        no_fill = 1.0 - row.get("bid", kalshi_yes)

        abs_pct = abs(btc_t - btc_t0) / btc_t0 * 100
        direction_up = btc_t > btc_t0
        features = {
            "minute": minute, "btc_t0": btc_t0, "btc_t": btc_t,
            "abs_pct": abs_pct, "direction_up": direction_up,
            "kalshi_yes": kalshi_yes, "kalshi_yes_t0": kalshi_t0,
            "yes_fill": yes_fill, "no_fill": no_fill,
            "yes_exposure": yes_exposure, "no_exposure": no_exposure,
        }
        side, stake = strategy_fn(features, params)
        if stake < MIN_BET or side is None: continue

        if side == "yes":
            if not (0.01 < yes_fill < 0.99): continue
            yes_bets.append((stake, yes_fill)); yes_exposure += stake
        else:  # no
            if not (0.01 < no_fill < 0.99): continue
            no_bets.append((stake, no_fill)); no_exposure += stake

    # PnL = payouts - costs - fees.
    # Corrected Kalshi fee: ceil(FEE_RATE * C * P * (1-P)) per fill, charged on
    # EVERY fill regardless of outcome (C = stake/fill contracts). In stake
    # terms the fee reduces to ceil(FEE_RATE * stake * (1-fill)), rounded up to
    # the cent. Peaks at fill=0.50, unlike the old "7% of winning profit" model.
    pnl = 0; wagered = 0
    for stake, fill in yes_bets:
        wagered += stake
        fee = math.ceil(FEE_RATE * stake * (1 - fill) * 100) / 100
        pnl += (stake * (1 - fill) / fill) if resolved else -stake
        pnl -= fee
    for stake, fill in no_bets:
        wagered += stake
        fee = math.ceil(FEE_RATE * stake * (1 - fill) * 100) / 100
        pnl += (stake * (1 - fill) / fill) if not resolved else -stake
        pnl -= fee

    return (pnl, wagered, len(yes_bets) + len(no_bets))


# ────────────────────────────────────────────────────────────────────────────
# Strategies — each takes features dict + params dict, returns (side, stake)
# ────────────────────────────────────────────────────────────────────────────

def S1_late_single_bet(f, p):
    """Single bet at fixed late minute, BASE_STAKE, direction = current BTC direction."""
    if f["minute"] != p["minute"]: return None, 0
    if f["yes_exposure"] + f["no_exposure"] > 0: return None, 0  # already bet
    if f["abs_pct"] < p.get("min_pct", 0): return None, 0
    side = "yes" if f["direction_up"] else "no"
    return side, p["stake"]


def S2_momentum_dh(f, p):
    """Current bot: bet target = stake * f_btc * g_misprice, scale into target."""
    minute = f["minute"]
    abs_pct = f["abs_pct"]
    f_btc = p["max_mult"] / (1 + math.exp(-p["k"] * (abs_pct - p["center"])))
    fair = get_fair(minute, abs_pct)
    if f["direction_up"]:
        misp = fair - f["kalshi_yes"]
    else:
        misp = (1 - f["kalshi_yes"]) - (1 - fair)  # equivalent: kalshi_yes - (1 - fair); but stay symmetric
        misp = f["kalshi_yes"] - (1 - fair)
    g = p["mm"] / (1 + math.exp(-p["mk"] * misp))
    target = p["stake"] * f_btc * g
    if f["direction_up"]:
        gap = max(0.0, target - f["yes_exposure"])
        return ("yes", gap) if gap >= MIN_BET else (None, 0)
    else:
        gap = max(0.0, target - f["no_exposure"])
        return ("no", gap) if gap >= MIN_BET else (None, 0)


def S3_mispricing_only(f, p):
    """Only bet if |mispricing| > threshold, ignore BTC magnitude."""
    if f["minute"] < p["min_minute"] or f["minute"] > p["max_minute"]: return None, 0
    fair = get_fair(f["minute"], f["abs_pct"])
    if f["direction_up"]:
        misp = fair - f["kalshi_yes"]
        if misp > p["threshold"] and f["yes_exposure"] < p["max_per_side"]:
            return "yes", min(p["stake"], p["max_per_side"] - f["yes_exposure"])
    else:
        misp = f["kalshi_yes"] - (1 - fair)
        if misp > p["threshold"] and f["no_exposure"] < p["max_per_side"]:
            return "no", min(p["stake"], p["max_per_side"] - f["no_exposure"])
    return None, 0


def S4_vol_gated(f, p):
    """Only trade if BTC has moved > threshold by entry minute. Bet on continuation, single shot."""
    if f["minute"] != p["entry_minute"]: return None, 0
    if f["yes_exposure"] + f["no_exposure"] > 0: return None, 0
    if f["abs_pct"] < p["vol_threshold"]: return None, 0
    side = "yes" if f["direction_up"] else "no"
    return side, p["stake"]


def S5_fade_extremes(f, p):
    """When kalshi_yes is extreme (>p['hi'] or <p['lo']), fade it (bet against)."""
    if f["minute"] < p["min_minute"]: return None, 0
    if f["yes_exposure"] + f["no_exposure"] > 0: return None, 0
    ky = f["kalshi_yes"]
    if ky > p["hi"]:
        # market saying YES is near-certain — fade by betting NO
        return "no", p["stake"]
    if ky < p["lo"]:
        return "yes", p["stake"]
    return None, 0


# ────────────────────────────────────────────────────────────────────────────
# Backtest harness
# ────────────────────────────────────────────────────────────────────────────

def backtest(markets, btc_prices, strategy_fn, params, label):
    pnls = []; wagered_sum = 0; n_trades_sum = 0; n_mkts_traded = 0
    for m in markets:
        r = simulate_market(m, btc_prices, strategy_fn, params)
        if r is None: continue
        pnl, wagered, n = r
        if n > 0:
            pnls.append(pnl); wagered_sum += wagered; n_trades_sum += n; n_mkts_traded += 1
    if not pnls:
        return None
    total_pnl = sum(pnls)
    n = len(pnls)
    losers = sum(1 for p in pnls if p < -0.01)
    win_rate = sum(1 for p in pnls if p > 0.01) / n
    roi = total_pnl / wagered_sum * 100 if wagered_sum else 0
    avg_pnl = total_pnl / n
    std_pnl = (sum((p - avg_pnl)**2 for p in pnls) / n) ** 0.5 if n > 1 else 0
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
    worst = min(pnls)
    return {
        "label": label, "n_mkts": n, "trades": n_trades_sum,
        "total_pnl": round(total_pnl, 2), "wagered": round(wagered_sum, 2),
        "roi_pct": round(roi, 2), "win_rate": round(win_rate, 3),
        "sharpe": round(sharpe, 3), "worst": round(worst, 2),
    }


def run_strategy(name, fn, variants, train_mkts, test_mkts, btc):
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  {name}")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'variant':<40} {'split':<6} {'n':>5} {'pnl':>9} {'roi%':>7} {'win%':>6} {'sharpe':>7} {'worst':>8}")
    for variant in variants:
        params, vlabel = variant
        train = backtest(train_mkts, btc, fn, params, vlabel)
        test  = backtest(test_mkts,  btc, fn, params, vlabel)
        if train: print(f"  {vlabel:<40} {'train':<6} {train['n_mkts']:>5} {train['total_pnl']:>+9.0f} {train['roi_pct']:>+7.2f} {train['win_rate']*100:>6.1f} {train['sharpe']:>+7.3f} {train['worst']:>+8.0f}")
        if test:  print(f"  {vlabel:<40} {'TEST ':<6} {test['n_mkts']:>5} {test['total_pnl']:>+9.0f} {test['roi_pct']:>+7.2f} {test['win_rate']*100:>6.1f} {test['sharpe']:>+7.3f} {test['worst']:>+8.0f}")
        if train and test:
            gap = test['roi_pct'] - train['roi_pct']
            print(f"  {'':<40} {'gap  ':<6} {'':>5} {'':>9} {gap:>+7.2f} {'':>6} {'':>7} {'':>8}")
        print()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    markets, btc = load_data()

    # Chronological 70/30 split
    n_train = int(len(markets) * TRAIN_FRAC)
    train_mkts = markets[:n_train]
    test_mkts  = markets[n_train:]
    print(f"\nTrain: {train_mkts[0]['open_time'][:10]} → {train_mkts[-1]['open_time'][:10]}  ({len(train_mkts)} mkts)")
    print(f"Test:  {test_mkts[0]['open_time'][:10]} → {test_mkts[-1]['open_time'][:10]}  ({len(test_mkts)} mkts)")
    print(f"Base stake: ${BASE_STAKE} | Fee rate: {FEE_RATE}\n")

    # ── S1: late single bet ───────────────────────────────────────────────
    s1_variants = [
        ({"minute": 7,  "stake": BASE_STAKE, "min_pct": 0.0}, "T+7 always"),
        ({"minute": 10, "stake": BASE_STAKE, "min_pct": 0.0}, "T+10 always"),
        ({"minute": 12, "stake": BASE_STAKE, "min_pct": 0.0}, "T+12 always"),
        ({"minute": 10, "stake": BASE_STAKE, "min_pct": 0.05}, "T+10 (≥0.05% move)"),
        ({"minute": 12, "stake": BASE_STAKE, "min_pct": 0.10}, "T+12 (≥0.10% move)"),
    ]

    # ── S2: momentum DH (varied sigmoid) ──────────────────────────────────
    s2_variants = [
        ({"stake": BASE_STAKE, "k": 20, "center": 0.10, "max_mult": 3.0, "mk": 8,  "mm": 2.0}, "current (k20 c0.10 max3 mk8 mm2)"),
        ({"stake": BASE_STAKE, "k": 30, "center": 0.05, "max_mult": 3.0, "mk": 12, "mm": 2.5}, "refit-defensive (k30 c0.05 mk12 mm2.5)"),
        ({"stake": BASE_STAKE, "k": 20, "center": 0.05, "max_mult": 3.0, "mk": 8,  "mm": 2.5}, "refit-mid"),
        ({"stake": BASE_STAKE, "k": 15, "center": 0.05, "max_mult": 2.0, "mk": 12, "mm": 2.0}, "refit-conservative (max2)"),
    ]

    # ── S3: mispricing-only ───────────────────────────────────────────────
    s3_variants = [
        ({"min_minute": 4, "max_minute": 13, "threshold": 0.05, "stake": BASE_STAKE, "max_per_side": 25}, "thresh 0.05"),
        ({"min_minute": 4, "max_minute": 13, "threshold": 0.10, "stake": BASE_STAKE, "max_per_side": 25}, "thresh 0.10"),
        ({"min_minute": 4, "max_minute": 13, "threshold": 0.15, "stake": BASE_STAKE, "max_per_side": 25}, "thresh 0.15"),
        ({"min_minute": 6, "max_minute": 13, "threshold": 0.10, "stake": BASE_STAKE, "max_per_side": 25}, "thresh 0.10 (T+6+)"),
        ({"min_minute": 8, "max_minute": 13, "threshold": 0.10, "stake": BASE_STAKE, "max_per_side": 25}, "thresh 0.10 (T+8+)"),
    ]

    # ── S4: vol-gated ─────────────────────────────────────────────────────
    s4_variants = [
        ({"entry_minute": 5,  "vol_threshold": 0.05, "stake": BASE_STAKE}, "T+5  vol≥0.05%"),
        ({"entry_minute": 5,  "vol_threshold": 0.10, "stake": BASE_STAKE}, "T+5  vol≥0.10%"),
        ({"entry_minute": 7,  "vol_threshold": 0.05, "stake": BASE_STAKE}, "T+7  vol≥0.05%"),
        ({"entry_minute": 7,  "vol_threshold": 0.10, "stake": BASE_STAKE}, "T+7  vol≥0.10%"),
        ({"entry_minute": 10, "vol_threshold": 0.10, "stake": BASE_STAKE}, "T+10 vol≥0.10%"),
    ]

    # ── S5: fade extremes ─────────────────────────────────────────────────
    s5_variants = [
        ({"min_minute": 4,  "hi": 0.85, "lo": 0.15, "stake": BASE_STAKE}, "T+4+  fade >0.85 / <0.15"),
        ({"min_minute": 7,  "hi": 0.85, "lo": 0.15, "stake": BASE_STAKE}, "T+7+  fade >0.85 / <0.15"),
        ({"min_minute": 10, "hi": 0.90, "lo": 0.10, "stake": BASE_STAKE}, "T+10+ fade >0.90 / <0.10"),
        ({"min_minute": 12, "hi": 0.95, "lo": 0.05, "stake": BASE_STAKE}, "T+12+ fade >0.95 / <0.05"),
    ]

    run_strategy("S1: Single late-window bet", S1_late_single_bet, s1_variants, train_mkts, test_mkts, btc)
    run_strategy("S2: Momentum DH (vary sigmoid params)", S2_momentum_dh, s2_variants, train_mkts, test_mkts, btc)
    run_strategy("S3: Mispricing-only entry", S3_mispricing_only, s3_variants, train_mkts, test_mkts, btc)
    run_strategy("S4: Volatility-gated entry", S4_vol_gated, s4_variants, train_mkts, test_mkts, btc)
    run_strategy("S5: Fade Kalshi extremes", S5_fade_extremes, s5_variants, train_mkts, test_mkts, btc)


if __name__ == "__main__":
    main()
