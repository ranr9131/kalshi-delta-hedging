"""
Backtest the momentum-scalper strategy on historical KXBTC15M windows.

Models the Polymarket wallet 0xce25e2…7fdc:
  - Pure taker, no quoting
  - Trigger on BTC move ≥ TRIGGER_USD over LOOKBACK_MIN minutes
  - Buy trending side IF its YES price sits in [BAND_LOW_C, BAND_HIGH_C]
  - Hold to settlement (no within-window stop)
  - Cooldown + per-window caps

80/20 split: oldest 80% = train, newest 20% = test. We report both so we
can see whether the param choice held out-of-sample (no peeking — the
backtest is path-independent so the split is just for honesty).

Run:  python3 simulate_momentum.py
"""
import os
import sys
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import DATA_DAYS, FEE_RATE

# ── Strategy params (mirror momentum_scalper.py defaults) ────────────────────
TRIGGER_USD       = 50.0      # BTC must move ≥ this many $
LOOKBACK_MIN      = 1         # …over the prior N minutes (1-min granularity)
BAND_LOW_C        = 60        # enter only when trending YES is in 60–75¢
BAND_HIGH_C       = 75
STAKE_DOLLARS     = 3.0       # per entry
COOLDOWN_MIN      = 1         # min minutes between entries
MAX_ENTRIES       = 4
MAX_WAGERED       = 20.0
STOP_LAST_MIN     = 2         # don't enter inside last N minutes
SLIPPAGE_C        = 1         # cross the spread: pay close + 1c (approx ask)

# Decision times: we act AT THE START of each minute inside the window, with
# data through the close of the prior minute. So decision_t = t0 + k*60 for
# k = 1..14, and we read:
#   btc_close_at(t)   = btc_data.lookup(t - 60)   # close of candle ending at t
#   yes_close_at(t)   = get_yes_price_at(candles, t)  # already end-aligned
EVAL_K = list(range(1, 15))   # k = 1..14 → decision_t = t0+60 .. t0+14m


def _btc_close_at(prices, t):
    """Price at time t = close of the candle ending at t. Coinbase keys candles
    by start time, so prices[t-60] = close of [t-60, t]."""
    return btc_data.lookup(prices, t - 60)


# ── Per-window backtest ──────────────────────────────────────────────────────
def backtest_window(market, btc_prices):
    """Run momentum scalper through one 15-min window. Returns dict or None."""
    open_iso  = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result    = market.get("result", "")
    if result not in ("yes", "no") or not open_iso or not close_iso:
        return None
    try:
        t0      = int(datetime.fromisoformat(open_iso .replace("Z","+00:00")).timestamp())
        t_close = int(datetime.fromisoformat(close_iso.replace("Z","+00:00")).timestamp())
    except Exception:
        return None

    candles = kalshi_client.fetch_candlesticks(market["ticker"], open_iso, close_iso)
    if not candles:
        return None

    yes_contracts  = 0.0
    no_contracts   = 0.0
    wagered        = 0.0
    n_entries      = 0
    last_entry_t   = -10**9
    sides_taken    = []     # debug: list of (minute, side, pay_cents)

    for k in EVAL_K:
        t = t0 + k * 60      # decision time (real-time clock)
        minute = k
        if t > t_close - STOP_LAST_MIN * 60:
            break
        btc_t  = _btc_close_at(btc_prices, t)
        btc_lb = _btc_close_at(btc_prices, t - LOOKBACK_MIN * 60)
        if btc_t is None or btc_lb is None:
            continue
        yes_close = kalshi_client.get_yes_price_at(candles, t)
        if yes_close is None or not (0.01 < yes_close < 0.99):
            continue

        delta = btc_t - btc_lb
        if abs(delta) < TRIGGER_USD:
            continue
        side = "yes" if delta > 0 else "no"

        # Trending-side price, with slippage on the cross
        if side == "yes":
            pay_c = round(yes_close * 100) + SLIPPAGE_C
        else:
            pay_c = 100 - round(yes_close * 100) + SLIPPAGE_C
        if not (BAND_LOW_C <= pay_c <= BAND_HIGH_C):
            continue

        # Per-window caps + cooldown
        if n_entries >= MAX_ENTRIES:           continue
        if wagered  >= MAX_WAGERED:            continue
        if t - last_entry_t < COOLDOWN_MIN*60: continue

        contracts = STAKE_DOLLARS / (pay_c / 100.0)
        cost      = contracts * pay_c / 100.0
        if side == "yes":
            yes_contracts += contracts
        else:
            no_contracts  += contracts
        wagered      += cost
        n_entries    += 1
        last_entry_t  = t
        sides_taken.append((minute, side, pay_c))

    # Settlement
    yes_payout = yes_contracts if result == "yes" else 0.0
    no_payout  = no_contracts  if result == "no"  else 0.0
    gross      = yes_payout + no_payout
    pnl_pre_fee = gross - wagered
    fee = FEE_RATE * pnl_pre_fee if pnl_pre_fee > 0 else 0.0
    pnl = pnl_pre_fee - fee

    return {
        "ticker":     market["ticker"],
        "result":     result,
        "n_entries":  n_entries,
        "wagered":    wagered,
        "payout":     gross,
        "pnl":        pnl,
        "sides":      sides_taken,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────
def summarize(label, rows):
    rows = [r for r in rows if r is not None and r["n_entries"] > 0]
    if not rows:
        print(f"\n{label}: no windows with entries.")
        return
    n        = len(rows)
    total_w  = sum(r["wagered"] for r in rows)
    total_p  = sum(r["pnl"]     for r in rows)
    n_tix    = sum(r["n_entries"] for r in rows)
    wins     = sum(1 for r in rows if r["pnl"] > 0)
    pnls     = sorted(r["pnl"] for r in rows)
    roi      = total_p / total_w * 100 if total_w > 0 else 0.0

    # Per-ticket P&L (assumes equal stake per ticket — true by construction here)
    per_tix_pnl = total_p / n_tix if n_tix else 0.0

    print(f"\n══ {label} ══  windows_with_entries={n}, total_tickets={n_tix}")
    print(f"  Wagered:       ${total_w:,.2f}")
    print(f"  Net P&L:       ${total_p:+,.2f}   (after {FEE_RATE*100:.0f}% fee on profit)")
    print(f"  ROI/wagered:   {roi:+.2f}%")
    print(f"  Edge/ticket:   ${per_tix_pnl:+.3f}")
    print(f"  Win windows:   {wins}/{n} ({wins/n*100:.1f}%)")
    print(f"  Worst window:  ${pnls[0]:+,.2f}")
    print(f"  Best window:   ${pnls[-1]:+,.2f}")
    # Direction breakdown
    yes_tix = sum(1 for r in rows for _,s,_ in r["sides"] if s == "yes")
    no_tix  = sum(1 for r in rows for _,s,_ in r["sides"] if s == "no")
    print(f"  YES tickets:   {yes_tix}   NO tickets: {no_tix}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading settled markets…")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    print(f"  {len(markets)} markets")

    # Chronological sort
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
    train = sorted_markets[:split]
    test  = sorted_markets[split:]
    print(f"\nTrain: {len(train)} windows  ({dated[0][0].date()} → {dated[split-1][0].date()})")
    print(f"Test:  {len(test)} windows  ({dated[split][0].date()} → {dated[-1][0].date()})")

    print(f"\nParams: trigger=${TRIGGER_USD:.0f}/{LOOKBACK_MIN}m  band={BAND_LOW_C}-{BAND_HIGH_C}c  "
          f"stake=${STAKE_DOLLARS}  max_entries={MAX_ENTRIES}  max_wagered=${MAX_WAGERED}  "
          f"cooldown={COOLDOWN_MIN}m  stop_last={STOP_LAST_MIN}m  slippage={SLIPPAGE_C}c")

    print("\nBacktesting TRAIN…")
    train_rows = [backtest_window(m, btc_prices) for m in train]
    print("Backtesting TEST…")
    test_rows  = [backtest_window(m, btc_prices) for m in test]

    summarize("TRAIN  (in-sample window range)", train_rows)
    summarize("TEST   (out-of-sample, newest 20%)", test_rows)


if __name__ == "__main__":
    main()
