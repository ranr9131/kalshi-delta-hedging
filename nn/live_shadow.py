"""
Live shadow sim for the NN strategy. Runs alongside (not interfering with) the
DH trader. Watches live/window_log.csv for newly-settled windows, reconstructs
the T+10 feature state from cached Kalshi candle + BTC data, runs the trained
NN, and records what the NN strategy would have done.

Logs to nn/shadow_log.csv:
    window_ts, ticker, winner, p_yes_nn, direction, side, fair, fill, mispr,
    stake, contracts, pnl, cum_pnl

State file: nn/shadow_state.json (tracks last processed window so restarts
don't double-count).

No live trading. Pure paper sidecar.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 nn/live_shadow.py [--interval 30]

CTRL-C to stop.
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import signal
import numpy as np
import torch
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import kalshi_client
import btc_data
from model import TSWinPredictor

WINDOW_LOG_PATH = os.path.join(os.path.dirname(ROOT), "live", "window_log.csv")
CKPT_PATH       = os.path.join(ROOT, "checkpoints", "best.pt")
SHADOW_LOG      = os.path.join(ROOT, "shadow_log.csv")
STATE_PATH      = os.path.join(ROOT, "shadow_state.json")

DECISION_MINUTE  = 10
BASE_STAKE       = 10.0      # match live trader
SLIP_CENTS       = 4
MIN_EDGE_CENTS   = 5         # tuned from backtest sweep — peak ROI region
FEE_KEEP         = 0.93
MAX_FILL         = 0.97

SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0

N_FEATURES = 7
WINDOW_MINUTES = 15


def f_btc(p):
    return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))


def g_mispr(m):
    return 1.0 + (MISPR_MAX - 1) / (1.0 + math.exp(-MISPR_K * m)) - (MISPR_MAX - 1) / 2


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_ticker": "", "cum_pnl": 0.0}


def save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def ensure_log_header():
    if os.path.exists(SHADOW_LOG):
        return
    with open(SHADOW_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "window_ts", "ticker", "winner", "p_yes_nn", "direction",
            "side", "fair", "fill", "mispricing", "stake", "contracts",
            "pnl", "cum_pnl", "outcome",
        ])


def build_features_at_t10(market_open_iso, market_close_iso, btc_prices):
    """Return (X: (15,7) float32, mask: (15,) bool) truncated at T+10, or None."""
    try:
        open_dt = datetime.fromisoformat(market_open_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    t0 = int(open_dt.timestamp())
    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None:
        return None

    candles = kalshi_client.fetch_candlesticks(
        ticker=None, open_time_iso=market_open_iso, close_time_iso=market_close_iso,
    )  # ticker arg ignored if fetch is by time? double-check
    # We need the ticker, so caller passes it. Refactor:
    return None  # never used; see build_for_ticker below


def build_for_ticker(ticker, open_iso, close_iso, btc_prices):
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None:
        return None
    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles:
        return None
    kalshi_t0 = candles[0].get("yes_open")
    if kalshi_t0 is None or not (0.01 < kalshi_t0 < 0.99):
        return None

    X = np.zeros((WINDOW_MINUTES, N_FEATURES), dtype=np.float32)
    mask = np.zeros(WINDOW_MINUTES, dtype=bool)
    last_btc = btc_t0
    hour = open_dt.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    for m in range(WINDOW_MINUTES):
        t = t0 + m * 60
        btc_t = btc_data.lookup(btc_prices, t)
        kalshi_yes = kalshi_client.get_yes_price_at(candles, t)
        if btc_t is None or kalshi_yes is None or not (0.01 < kalshi_yes < 0.99):
            continue
        X[m, 0] = (btc_t / btc_t0) - 1.0
        X[m, 1] = (btc_t / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc_t
        X[m, 2] = kalshi_yes
        X[m, 3] = kalshi_yes - kalshi_t0
        X[m, 4] = m / float(WINDOW_MINUTES - 1)
        X[m, 5] = hour_sin
        X[m, 6] = hour_cos
        mask[m] = True

    # Truncate at decision minute
    X[DECISION_MINUTE + 1:, :] = 0.0
    mask[DECISION_MINUTE + 1:]  = False
    if not mask[DECISION_MINUTE]:
        return None
    return X, mask, btc_t0


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = TSWinPredictor(n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"],  dtype=np.float32)
    return model, mean, std


def decide_and_score(X, mask, mean, std, model, winner):
    """Return dict with NN prediction, decision, and P&L if it had bet."""
    Xn = (X - mean) / std
    with torch.no_grad():
        logit = model(torch.from_numpy(Xn[None]), torch.from_numpy(mask[None]))
        p_yes = torch.sigmoid(logit).item()

    ret_t = float(X[DECISION_MINUTE, 0])
    kal_t = float(X[DECISION_MINUTE, 2])
    direction_up = ret_t > 0
    pct_abs = abs(ret_t) * 100.0

    fair = p_yes if direction_up else (1.0 - p_yes)
    fill = min(MAX_FILL, (kal_t if direction_up else 1.0 - kal_t) + SLIP_CENTS / 100.0)
    mispr = fair - fill
    side = "yes" if direction_up else "no"

    if mispr * 100.0 < MIN_EDGE_CENTS:
        return {"p_yes_nn": p_yes, "direction": "up" if direction_up else "down",
                "side": "skip", "fair": fair, "fill": fill, "mispricing": mispr,
                "stake": 0.0, "contracts": 0.0, "pnl": 0.0, "outcome": "no_bet"}

    f = f_btc(pct_abs); g = g_mispr(mispr)
    stake = BASE_STAKE * f * g
    if stake < 1.0:
        return {"p_yes_nn": p_yes, "direction": "up" if direction_up else "down",
                "side": "skip", "fair": fair, "fill": fill, "mispricing": mispr,
                "stake": 0.0, "contracts": 0.0, "pnl": 0.0, "outcome": "tiny"}

    contracts = stake / fill
    won = (side == "yes" and winner == "yes") or (side == "no" and winner == "no")
    pnl = contracts * (1.0 - fill) * FEE_KEEP if won else -contracts * fill
    return {"p_yes_nn": p_yes, "direction": "up" if direction_up else "down",
            "side": side, "fair": fair, "fill": fill, "mispricing": mispr,
            "stake": stake, "contracts": contracts, "pnl": pnl,
            "outcome": "net_win" if pnl > 0 else "net_loss"}


def read_new_settled_windows(state):
    """Read window_log.csv and return entries newer than last_ticker."""
    if not os.path.exists(WINDOW_LOG_PATH):
        return []
    last = state.get("last_ticker", "")
    rows = []
    seen_last = (last == "")
    with open(WINDOW_LOG_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if not seen_last:
                if r["ticker"] == last:
                    seen_last = True
                continue
            if not r.get("settlement_ts"):
                continue
            rows.append(r)
    return rows


def run_loop(interval):
    print(f"NN live shadow starting | decision T+{DECISION_MINUTE} | "
          f"min_edge={MIN_EDGE_CENTS}c | base_stake=${BASE_STAKE}")
    ensure_log_header()
    state = load_state()
    print(f"  last_ticker: {state['last_ticker']!r}  cum_pnl: ${state['cum_pnl']:+.2f}")

    model, mean, std = load_model()
    print(f"  model loaded: {CKPT_PATH}")

    # Cache BTC prices for ~last 30 days (covers any new windows)
    now = int(time.time())
    print("  fetching recent BTC prices...")
    btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
    last_btc_refresh = time.time()
    print(f"  loaded {len(btc_prices)} BTC minute prices\n")

    stop = {"v": False}
    def _sigint(*_): stop["v"] = True
    signal.signal(signal.SIGINT, _sigint)

    while not stop["v"]:
        try:
            # Refresh BTC every hour
            if time.time() - last_btc_refresh > 3600:
                now = int(time.time())
                btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
                last_btc_refresh = time.time()

            new_rows = read_new_settled_windows(state)
            for r in new_rows:
                ticker = r["ticker"]
                winner = r.get("market_winner", "")
                if winner not in ("yes", "no"):
                    state["last_ticker"] = ticker
                    continue
                ts = r["window_ts"]
                # Reconstruct close time from ticker or window_ts+15min
                try:
                    close_dt = datetime.fromisoformat(r["close_time"].replace("Z","+00:00"))
                    close_iso = close_dt.isoformat()
                    open_dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                    open_iso = open_dt.isoformat()
                except Exception:
                    state["last_ticker"] = ticker
                    continue

                built = build_for_ticker(ticker, open_iso, close_iso, btc_prices)
                if built is None:
                    print(f"  [skip] {ticker}: feature build failed")
                    state["last_ticker"] = ticker
                    continue
                X, mask, _ = built
                d = decide_and_score(X, mask, mean, std, model, winner)
                state["cum_pnl"] += d["pnl"]
                with open(SHADOW_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([
                        ts, ticker, winner,
                        f"{d['p_yes_nn']:.4f}", d["direction"], d["side"],
                        f"{d['fair']:.4f}", f"{d['fill']:.4f}",
                        f"{d['mispricing']:+.4f}", f"{d['stake']:.4f}",
                        f"{d['contracts']:.4f}", f"{d['pnl']:+.4f}",
                        f"{state['cum_pnl']:+.4f}", d["outcome"],
                    ])
                print(f"  {ts}  {ticker}  win={winner}  side={d['side']:>4}  "
                      f"NN P(YES)={d['p_yes_nn']:.3f}  fair={d['fair']:.3f}  "
                      f"fill={d['fill']:.3f}  mispr={d['mispricing']:+.3f}  "
                      f"stake=${d['stake']:.2f}  P&L=${d['pnl']:+.2f}  "
                      f"cum=${state['cum_pnl']:+.2f}")
                state["last_ticker"] = ticker
                save_state(state)
        except Exception as e:
            print(f"  [error in loop] {type(e).__name__}: {e}")

        # Sleep up to interval, checking stop flag every second
        for _ in range(int(interval)):
            if stop["v"]: break
            time.sleep(1)

    print("\nNN live shadow stopped.")
    save_state(state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=30)
    args = p.parse_args()
    run_loop(args.interval)
