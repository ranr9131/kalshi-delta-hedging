"""
Replay reversal-handling strategies against the actual live trader bet log to
estimate how the new overlays would have changed P&L on real observed windows.

Reads:
  live/window_log.csv  — one row per settled window (winner + totals)
  live/trade_log.csv   — one row per bet placed. The current trader writes
                          dh-target rows under an OLD-format header due to
                          a column-shift bug; this script decodes around it.

For each window:
  1. Reconstruct the sequence of (minute, btc_now, yes_bid/ask, side, stake)
     bets the live trader actually placed.
  2. Re-score under each strategy variant:
       - baseline       : all bets as placed
       - ncs-MIN-THR    : drop bets at minute >= MIN with |move| < THR%
       - rh-MIN         : at every observed minute >= MIN, if direction is
                          opposite the current larger side, ADD a hedge bet
                          sized to neutralize that side's contracts
       - ncs + rh       : both overlays composed
  3. Recompute P&L using the actual market_winner.

Outputs:
  data/logs/replay_live_summary.csv   — per-strategy aggregates
  data/logs/replay_live_windows.csv   — per-window per-strategy P&L
  console table of comparative results
"""
import csv
import os
import sys
from collections import defaultdict

FEE_RATE = 0.07
MIN_BET  = 5.0

LIVE_TRADE_LOG  = "live/trade_log.csv"
LIVE_WINDOW_LOG = "live/window_log.csv"
OUT_DIR         = "data/logs"

# Mapping from the OLD header column name (what DictReader thinks the column
# is called) to the NEW field name (what the value actually represents in
# dh-target rows). Corrects the column-shift bug in trader._append_csv.
DH_FIELD_MAP = {
    "window_ts":          "window_ts",
    "timestamp":          "mode",
    "mode":               "ticker",
    "ticker":             "close_time",
    "close_time":         "dh_minute",
    "direction":          "btc_t0",
    "btc_t0":             "btc_now",
    "btc_t5":             "btc_price_age_secs",
    "btc_price_age_secs": "abs_pct_move",
    "abs_pct_move":       "yes_bid",
    "yes_bid":            "yes_ask",
    "yes_ask":            "spread",
    "spread":             "kalshi_yes_mid",
    "kalshi_yes_t5":      "direction",
    "mispricing":         "yes_target",
    "f_btc":              "no_target",
    "g_misprice":         "yes_exposure_before",
    "stake":              "no_exposure_before",
    "fill_price":         "bet_side",
    "count":              "mispricing",
    "order_id":           "f_btc",
    "order_result":       "g_misprice",
    "settlement_ts":      "stake",
    "market_winner":      "fill_price",
    "outcome":            "count",
    "pnl":                "order_id",
    "cumulative_pnl":     "order_result",
}


def decode_dh_row(raw: dict) -> dict:
    """Translate a misaligned dh-target trade_log row into proper fields."""
    return {DH_FIELD_MAP[k]: v for k, v in raw.items() if k in DH_FIELD_MAP}


def load_windows() -> dict:
    """Returns {window_ts → window record}."""
    out = {}
    for r in csv.DictReader(open(LIVE_WINDOW_LOG)):
        ts = r["window_ts"]
        out[ts] = {
            "ticker":     r["ticker"],
            "btc_t0":     float(r["btc_t0"]),
            "btc_t5":     float(r["btc_t5"])  if r["btc_t5"]  else None,
            "btc_t10":    float(r["btc_t10"]) if r["btc_t10"] else None,
            "winner":     r["market_winner"],
            "actual_pnl": float(r["total_pnl"]),
            "actual_yes": float(r["total_yes_stake"]),
            "actual_no":  float(r["total_no_stake"]),
        }
    return out


def load_bets(windows: dict) -> dict:
    """Returns {window_ts → [bet, ...]} for windows we know about."""
    bets_by_window = defaultdict(list)
    for raw in csv.DictReader(open(LIVE_TRADE_LOG)):
        # Old paper-mode rows have mode == "PAPER"; dh rows have mode == ticker.
        if raw["mode"] == "PAPER":
            continue
        if raw["window_ts"] not in windows:
            continue
        d = decode_dh_row(raw)
        try:
            bet = {
                "window_ts":  d["window_ts"],
                "minute":     int(d["dh_minute"]),
                "btc_now":    float(d["btc_now"]),
                "yes_bid":    float(d["yes_bid"]),
                "yes_ask":    float(d["yes_ask"]),
                "kalshi_mid": float(d["kalshi_yes_mid"]),
                "side":       d["bet_side"],
                "stake":      float(d["stake"]),
                "fill_price": float(d["fill_price"]),
            }
        except (KeyError, ValueError):
            continue
        bets_by_window[d["window_ts"]].append(bet)
    for ts in bets_by_window:
        bets_by_window[ts].sort(key=lambda b: (b["minute"], b["side"]))
    return bets_by_window


def pnl_for(side: str, stake: float, fill_price: float, winner: str) -> float:
    """Net P&L on one bet after Kalshi's fee."""
    contracts = stake / fill_price
    if side == winner:
        return (contracts - stake) * (1 - FEE_RATE)
    return -stake


# ── Strategy variants ─────────────────────────────────────────────────────────

def apply_strategy(bets, btc_t0, winner, *,
                   ncs_minute=None, ncs_threshold=0.0,
                   rh_minute=None, rh_trigger=10.0):
    """
    Replay `bets` (in placement order) under the given overlays.
    Returns (kept_bets, hedge_bets, total_pnl).
    """
    kept = []
    hedges = []
    yes_exp = 0.0
    no_exp  = 0.0
    yes_ctr = 0.0
    no_ctr  = 0.0

    for b in bets:
        abs_move = abs(b["btc_now"] - btc_t0) / btc_t0 * 100.0

        # ── Near-Cutoff Skip ──
        if (ncs_minute is not None and b["minute"] >= ncs_minute
                and abs_move < ncs_threshold):
            pass  # drop this bet
        else:
            kept.append(b)
            if b["side"] == "yes":
                yes_exp += b["stake"]
                yes_ctr += b["stake"] / b["fill_price"]
            else:
                no_exp  += b["stake"]
                no_ctr  += b["stake"] / b["fill_price"]

        # ── Reversal Hedge ──
        # We only have market quotes at minutes where a bet was placed, so
        # this is a per-observed-minute hook.
        if rh_minute is not None and b["minute"] >= rh_minute:
            direction_up = b["btc_now"] > btc_t0
            kalshi_yes = b["kalshi_mid"]

            if direction_up and no_exp >= rh_trigger and no_ctr > 0:
                fill = b["yes_ask"]
                hedge_stake = no_ctr * fill
                if hedge_stake >= MIN_BET:
                    hedges.append({**b, "side": "yes",
                                   "stake": hedge_stake, "fill_price": fill,
                                   "hedge": True})
                    yes_exp += hedge_stake
                    yes_ctr += hedge_stake / fill
            elif (not direction_up) and yes_exp >= rh_trigger and yes_ctr > 0:
                fill = 1.0 - b["yes_bid"]
                hedge_stake = yes_ctr * fill
                if hedge_stake >= MIN_BET:
                    hedges.append({**b, "side": "no",
                                   "stake": hedge_stake, "fill_price": fill,
                                   "hedge": True})
                    no_exp += hedge_stake
                    no_ctr += hedge_stake / fill

    total_pnl = sum(pnl_for(x["side"], x["stake"], x["fill_price"], winner)
                    for x in kept + hedges)
    return kept, hedges, total_pnl


STRATEGIES = [
    ("baseline",           dict()),
    ("ncs-11-0.08",        dict(ncs_minute=11, ncs_threshold=0.08)),
    ("ncs-10-0.10",        dict(ncs_minute=10, ncs_threshold=0.10)),
    ("rh-11",              dict(rh_minute=11)),
    ("rh-10",              dict(rh_minute=10)),
    ("rh-12",              dict(rh_minute=12)),
    ("ncs-11-0.08+rh-11",  dict(ncs_minute=11, ncs_threshold=0.08, rh_minute=11)),
    ("ncs-10-0.10+rh-10",  dict(ncs_minute=10, ncs_threshold=0.10, rh_minute=10)),
]


def main():
    if not os.path.exists(LIVE_TRADE_LOG):
        print(f"ERROR: {LIVE_TRADE_LOG} not found.")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    windows  = load_windows()
    bets_map = load_bets(windows)

    eligible = [ts for ts in windows
                if ts in bets_map and windows[ts]["winner"] in ("yes", "no")]
    skipped = len(windows) - len(eligible)
    if skipped:
        print(f"Skipped {skipped} windows (no bet data or no winner).")

    summary_rows = []
    per_window_rows = []

    actual_total = sum(windows[ts]["actual_pnl"] for ts in eligible)
    print(f"Eligible windows: {len(eligible)}  Actual logged P&L: ${actual_total:+.2f}\n")

    for label, kwargs in STRATEGIES:
        total_pnl = 0.0
        total_wagered = 0.0
        n_hedges = 0
        n_skipped = 0
        n_wins = 0
        for ts in eligible:
            w = windows[ts]
            bets = bets_map[ts]
            kept, hedges, pnl = apply_strategy(
                bets, w["btc_t0"], w["winner"], **kwargs,
            )
            wagered = sum(x["stake"] for x in kept + hedges)
            n_hedges += len(hedges)
            n_skipped += (len(bets) - len(kept))
            n_wins += int(pnl > 0)
            total_pnl += pnl
            total_wagered += wagered
            per_window_rows.append({
                "strategy":      label,
                "window_ts":     ts,
                "winner":        w["winner"],
                "actual_pnl":    round(w["actual_pnl"], 4),
                "replay_pnl":    round(pnl, 4),
                "delta_vs_actual": round(pnl - w["actual_pnl"], 4),
                "n_kept":        len(kept),
                "n_hedges":      len(hedges),
                "n_dropped":     len(bets) - len(kept),
                "wagered":       round(wagered, 4),
            })
        roi = total_pnl / total_wagered * 100 if total_wagered else 0.0
        summary_rows.append({
            "strategy":        label,
            "n_windows":       len(eligible),
            "total_pnl":       round(total_pnl, 4),
            "total_wagered":   round(total_wagered, 4),
            "roi_pct":         round(roi, 2),
            "win_rate_pct":    round(n_wins / len(eligible) * 100, 1),
            "n_hedge_bets":    n_hedges,
            "n_dropped_bets":  n_skipped,
            "vs_baseline_pnl": 0.0,
        })

    base_pnl = summary_rows[0]["total_pnl"]
    for row in summary_rows:
        row["vs_baseline_pnl"] = round(row["total_pnl"] - base_pnl, 4)

    with open(os.path.join(OUT_DIR, "replay_live_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(os.path.join(OUT_DIR, "replay_live_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_window_rows[0].keys()))
        w.writeheader(); w.writerows(per_window_rows)

    print(f"{'strategy':<22} {'pnl':>9} {'roi%':>6} {'wins%':>6} {'wagered':>9} {'hedges':>7} {'dropped':>8} {'vs base':>9}")
    print("-" * 86)
    for r in summary_rows:
        print(f"{r['strategy']:<22} {r['total_pnl']:>+9.2f} "
              f"{r['roi_pct']:>+6.2f} {r['win_rate_pct']:>6.1f} "
              f"{r['total_wagered']:>9.2f} {r['n_hedge_bets']:>7d} "
              f"{r['n_dropped_bets']:>8d} {r['vs_baseline_pnl']:>+9.2f}")

    print("\nReplay on the 5 worst-loss windows (baseline pnl):")
    base_per = [r for r in per_window_rows if r["strategy"] == "baseline"]
    base_per.sort(key=lambda r: r["replay_pnl"])
    worst = [r["window_ts"] for r in base_per[:5]]

    print(f"{'window':<22}", end="")
    for label, _ in STRATEGIES:
        print(f"{label:>22}", end="")
    print()
    for ts in worst:
        print(f"{ts[5:16]:<22}", end="")
        for label, _ in STRATEGIES:
            row = next(r for r in per_window_rows if r["window_ts"] == ts and r["strategy"] == label)
            print(f"{row['replay_pnl']:>+22.2f}", end="")
        print()

    print("\nSaved:")
    print(f"  {os.path.join(OUT_DIR, 'replay_live_summary.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'replay_live_windows.csv')}")


if __name__ == "__main__":
    main()
