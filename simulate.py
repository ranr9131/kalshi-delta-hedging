"""
Runs the strategy simulation on historical Kalshi KXBTC15M data.

Strategy:
  T+0  : Bet $STAKE on Yes (BTC will go up) at Kalshi's opening Yes price.
  T+5  : Check REAL BTC price vs BTC price at T+0.
         - If BTC went UP  -> size up: bet $STAKE more on Yes at current Kalshi price.
         - If BTC went DOWN -> hedge: bet $STAKE on No at current Kalshi price.
  T+15 : Kalshi market settles. Collect P&L.

BTC prices come from Kraken 1-minute OHLCV (cached per day).
Kalshi odds at T+0 and T+5 come from Kalshi's own candlestick API.
"""

import os
import csv
import time
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import STAKE, FEE_RATE, DATA_DAYS, LOGS_DIR, CACHE_DIR


# ── P&L helpers ──────────────────────────────────────────────────────────────

def yes_pnl(stake, yes_price, resolved_yes):
    """P&L for a Yes bet. yes_price is the probability (0-1)."""
    if resolved_yes:
        gross = stake * (1 - yes_price) / yes_price
        return gross * (1 - FEE_RATE)
    return -stake


def no_pnl(stake, yes_price, resolved_yes):
    """P&L for a No bet (bought at price = 1 - yes_price)."""
    no_price = 1 - yes_price
    if not resolved_yes:
        gross = stake * (1 - no_price) / no_price
        return gross * (1 - FEE_RATE)
    return -stake


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_market(market, btc_prices):
    """
    Simulate one 15-minute Kalshi window. Returns a result dict or None if data is missing.
    btc_prices: dict of {str(unix_second) -> float} for the relevant day(s).
    """
    ticker = market["ticker"]
    open_iso = market.get("open_time", "")
    close_iso = market.get("close_time", "")
    result_field = market.get("result", "")

    if not open_iso or not close_iso or result_field not in ("yes", "no"):
        return None

    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    t5 = t0 + 5 * 60

    resolved_yes = result_field == "yes"

    # ── Real BTC price signal ─────────────────────────────────────────────────
    btc_at_t0 = btc_data.lookup(btc_prices, t0)
    btc_at_t5 = btc_data.lookup(btc_prices, t5)

    if btc_at_t0 is None or btc_at_t5 is None:
        return None

    btc_direction_up = btc_at_t5 > btc_at_t0  # True = BTC price rose from T+0 to T+5

    # ── Kalshi candlesticks ───────────────────────────────────────────────────
    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles:
        return None

    kalshi_t0_price = candles[0]["yes_open"] if candles else None
    kalshi_t5_price = kalshi_client.get_yes_price_at(candles, t5)

    if kalshi_t0_price is None or kalshi_t5_price is None:
        return None

    if not (0.01 < kalshi_t0_price < 0.99) or not (0.01 < kalshi_t5_price < 0.99):
        return None

    # ── Strategy execution ────────────────────────────────────────────────────
    pnl_initial = yes_pnl(STAKE, kalshi_t0_price, resolved_yes)

    if btc_direction_up:   # BTC went up at T+5 -> size up (buy more Yes)
        decision = "size_up"
        pnl_decision = yes_pnl(STAKE, kalshi_t5_price, resolved_yes)
    else:                  # BTC went down at T+5 -> hedge (buy No)
        decision = "hedge"
        pnl_decision = no_pnl(STAKE, kalshi_t5_price, resolved_yes)

    total_pnl = pnl_initial + pnl_decision
    total_wagered = STAKE * 2

    # Did BTC direction at T+5 correctly predict the Kalshi 15-min resolution?
    signal_correct = btc_direction_up == resolved_yes

    return {
        "timestamp_t0": open_iso,
        "ticker": ticker,
        "resolved_yes": resolved_yes,
        "btc_t0": round(btc_at_t0, 2),
        "btc_t5": round(btc_at_t5, 2),
        "btc_direction_up": btc_direction_up,
        "signal_correct": signal_correct,
        "kalshi_yes_t0": round(kalshi_t0_price, 4),
        "kalshi_yes_t5": round(kalshi_t5_price, 4),
        "kalshi_t5_vs_t0_shift": round(kalshi_t5_price - kalshi_t0_price, 4),
        "decision": decision,
        "pnl_initial": round(pnl_initial, 4),
        "pnl_decision": round(pnl_decision, 4),
        "total_pnl": round(total_pnl, 4),
        "total_wagered": total_wagered,
        "roi_pct": round(total_pnl / total_wagered * 100, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    if not markets:
        print("No Kalshi markets found.")
        return

    # Find the date range of all markets
    timestamps = []
    for m in markets:
        try:
            open_dt = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            timestamps.append(int(open_dt.timestamp()))
        except Exception:
            pass

    if not timestamps:
        print("Could not parse market timestamps.")
        return

    range_start = min(timestamps) - 600   # a little buffer
    range_end   = max(timestamps) + 1800  # T+15 buffer

    # Pre-fetch all BTC price data (cached per day)
    print(f"Fetching BTC price data for {DATA_DAYS} days from Kraken...")
    btc_prices = btc_data.fetch_btc_prices(range_start, range_end)
    print(f"Loaded {len(btc_prices)} BTC minute-prices.\n")

    if len(btc_prices) < 100:
        print("WARNING: Very few BTC prices loaded. Check Kraken connectivity.")

    print(f"Simulating {len(markets)} Kalshi windows...\n")

    results = []
    skipped = 0

    for i, market in enumerate(markets):
        ticker = market.get("ticker", "?")
        result = simulate_market(market, btc_prices)
        if result is None:
            skipped += 1
            continue

        results.append(result)
        if (i + 1) % 100 == 0 or i < 5:
            print(f"[{i+1}/{len(markets)}] {ticker}: PnL={result['total_pnl']:+.2f}  {result['decision']}  signal={'ok' if result['signal_correct'] else 'ng'}")

    if not results:
        print(f"\nNo valid results (skipped {skipped}).")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = os.path.join(LOGS_DIR, "simulation_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")
    print(f"Simulated: {len(results)}  Skipped: {skipped}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    total_pnl = sum(r["total_pnl"] for r in results)
    total_wagered = sum(r["total_wagered"] for r in results)
    signal_hits = sum(1 for r in results if r["signal_correct"])
    size_ups = [r for r in results if r["decision"] == "size_up"]
    hedges   = [r for r in results if r["decision"] == "hedge"]

    print(f"\n{'='*50}")
    print(f"SIMULATION SUMMARY ({len(results)} windows)")
    print(f"{'='*50}")
    print(f"Total P&L:          ${total_pnl:+.2f}")
    print(f"Total wagered:      ${total_wagered:.2f}")
    print(f"Overall ROI:        {total_pnl/total_wagered*100:+.2f}%")
    print(f"Signal accuracy:    {signal_hits/len(results)*100:.1f}%  (BTC T+5 dir predicts Kalshi 15-min)")
    print(f"Size-up windows:    {len(size_ups)}")
    print(f"Hedge windows:      {len(hedges)}")
    if size_ups:
        su_pnl = sum(r["pnl_decision"] for r in size_ups)
        print(f"Size-up leg P&L:    ${su_pnl:+.2f}")
    if hedges:
        h_pnl = sum(r["pnl_decision"] for r in hedges)
        print(f"Hedge leg P&L:      ${h_pnl:+.2f}")
    print(f"Initial bet P&L:    ${sum(r['pnl_initial'] for r in results):+.2f}")
    print(f"{'='*50}")
    print(f"\nRun analyze.py for charts and deeper breakdown.")


if __name__ == "__main__":
    run()
