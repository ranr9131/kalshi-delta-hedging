"""
Multi-bet live shadow for the NN strategy. Same DH template as live trader
(target mode, multi-bet, RH overlay, leg cap, time decay, early skip) but
uses the multi-minute NN (`best_multi.pt`) as the fair-price source instead
of the 2D table.

Polls live/window_log.csv every interval, processes newly-settled windows,
runs the full multi-minute strategy on cached candle data, writes per-window
results to nn/shadow_log.csv.

Config matches LIVE trader settings (not backtest defaults):
  - BASE_STAKE = $10
  - RH_TRIGGER = $1
  - All other knobs identical to canonical DH

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 nn/live_shadow_multi.py [--interval 30]
"""

import os
import sys
import csv
import json
import math
import time
import signal
import argparse
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
CKPT_PATH       = os.path.join(ROOT, "checkpoints", "best_multi.pt")
SHADOW_LOG      = os.path.join(ROOT, "shadow_log.csv")        # new multi-bet log
STATE_PATH      = os.path.join(ROOT, "shadow_state.json")     # new state file

# ── Strategy config (late-multi-bet, tuned from backtest_late.py sweep) ───────
# T+10..T+13 only — early minutes drag down ROI. 10c min edge — backtest
# showed peak ROI/win-rate at this threshold.
MINUTES        = list(range(10, 14))   # T+10..T+13 (was 4..13)
BASE_STAKE     = 10.0
MIN_BET        = 1.0
SLIP_C         = 4
MIN_EDGE_C     = 10                    # was 1
RH_MIN         = 10
RH_TRIGGER     = 1.0
MAX_HEDGE_F    = 0.80
MAX_FILL       = 0.97
MAX_LEGS       = 2
EARLY_SKIP_M   = 5
EARLY_SKIP_P   = 0.05
FEE_KEEP       = 0.93

SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0

N_FEATURES     = 7
WINDOW_MINUTES = 15


def sigmoid_btc(p):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))
def sigmoid_misp(m): return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))
def time_decay(minute):
    if minute < 7:  return 0.4
    if minute < 10: return 0.8
    return 1.2


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"last_ticker": "", "cum_pnl": 0.0}


def save_state(s):
    json.dump(s, open(STATE_PATH, "w"), indent=2)


def ensure_log_header():
    if os.path.exists(SHADOW_LOG):
        return
    with open(SHADOW_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "window_ts", "ticker", "winner",
            "n_yes_bets", "n_no_bets", "total_bets",
            "yes_wagered", "no_wagered", "total_wagered",
            "yes_pnl", "no_pnl", "total_pnl", "cum_pnl", "outcome",
            "first_minute", "first_direction",
        ])


def build_features(ticker, open_iso, close_iso, btc_prices):
    """Same as live_shadow.py — full 15-minute feature tensor + mask."""
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None: return None
    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles: return None
    kalshi_t0 = candles[0].get("yes_open")
    if kalshi_t0 is None or not (0.01 < kalshi_t0 < 0.99): return None

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
    return X, mask


def nn_predict_at(model, mean, std, X_raw, mask, minute):
    """Truncate at minute, run NN, return P(YES wins)."""
    X = X_raw.copy(); mk = mask.copy()
    X[minute + 1:, :] = 0.0
    mk[minute + 1:]   = False
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logit = model(torch.from_numpy(Xn[None]), torch.from_numpy(mk[None]))
        return float(torch.sigmoid(logit).item())


def simulate_one(X_raw, mask, winner, model, mean, std):
    """Run multi-bet target-mode DH on one window using NN at each minute."""
    resolved_yes = (winner == "yes")
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c   = no_c   = 0.0
    legs = 0
    first_minute = None; first_dir = None

    for m in MINUTES:
        if legs >= MAX_LEGS:
            break
        if not mask[m]:
            continue
        ret_t = float(X_raw[m, 0])
        kal_t = float(X_raw[m, 2])
        if not (0.01 < kal_t < 0.99): continue
        if ret_t == 0: continue

        direction_up = ret_t > 0
        pct_abs = abs(ret_t) * 100.0

        # NN fair price at this minute
        p_yes = nn_predict_at(model, mean, std, X_raw, mask, m)
        fair  = p_yes if direction_up else (1.0 - p_yes)

        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal_t + slip)
        no_fill  = min(MAX_FILL, (1.0 - kal_t) + slip)

        if direction_up:
            mispr = fair - yes_fill; fill_use = yes_fill
        else:
            mispr = fair - no_fill;  fill_use = no_fill

        if mispr * 100 < MIN_EDGE_C:
            target = 0.0
        else:
            target = BASE_STAKE * sigmoid_btc(pct_abs) * sigmoid_misp(mispr) * time_decay(m)

        if m <= EARLY_SKIP_M and pct_abs < EARLY_SKIP_P:
            target = 0.0

        if direction_up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and fill_use < MAX_FILL:
                yes_bets.append((gap, yes_fill))
                yes_exp += gap; yes_c += gap / yes_fill; legs += 1
                if first_minute is None:
                    first_minute = m; first_dir = "yes"
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and fill_use < MAX_FILL:
                no_bets.append((gap, no_fill))
                no_exp += gap; no_c += gap / no_fill; legs += 1
                if first_minute is None:
                    first_minute = m; first_dir = "no"

        # RH at T+RH_MIN+
        if m >= RH_MIN and legs < MAX_LEGS:
            if direction_up:
                if no_exp >= RH_TRIGGER and no_c > 0 and yes_fill <= MAX_HEDGE_F:
                    hedge = no_c * yes_fill
                    if hedge >= MIN_BET:
                        yes_bets.append((hedge, yes_fill))
                        yes_exp += hedge; yes_c += hedge / yes_fill; legs += 1
            else:
                if yes_exp >= RH_TRIGGER and yes_c > 0 and no_fill <= MAX_HEDGE_F:
                    hedge = yes_c * no_fill
                    if hedge >= MIN_BET:
                        no_bets.append((hedge, no_fill))
                        no_exp += hedge; no_c += hedge / no_fill; legs += 1

    # P&L — fp stored is the actual fill price for each side.
    # For a side that wins: profit = stake * (1 - fp) / fp * FEE_KEEP
    # For a side that loses: lose -stake
    pnl_yes = sum((1 - fp) * (s / fp) * FEE_KEEP if resolved_yes else -s
                  for s, fp in yes_bets)
    pnl_no  = sum((1 - fp) * (s / fp) * FEE_KEEP if not resolved_yes else -s
                  for s, fp in no_bets)
    return {
        "n_yes_bets": len(yes_bets), "n_no_bets": len(no_bets),
        "yes_wagered": yes_exp, "no_wagered": no_exp,
        "total_wagered": yes_exp + no_exp,
        "yes_pnl": pnl_yes, "no_pnl": pnl_no,
        "total_pnl": pnl_yes + pnl_no,
        "first_minute": first_minute, "first_direction": first_dir,
    }


def read_new_settled(state):
    if not os.path.exists(WINDOW_LOG_PATH):
        return []
    last = state.get("last_ticker", "")
    rows, seen = [], (last == "")
    with open(WINDOW_LOG_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if not seen:
                if r["ticker"] == last: seen = True
                continue
            if not r.get("settlement_ts"): continue
            rows.append(r)
    return rows


def run_loop(interval):
    print(f"NN multi-bet live shadow starting | base_stake=${BASE_STAKE} "
          f"min_edge={MIN_EDGE_C}c RH=T+{RH_MIN} leg_cap={MAX_LEGS}")
    ensure_log_header()
    state = load_state()
    print(f"  last_ticker={state['last_ticker']!r}  cum_pnl=${state['cum_pnl']:+.2f}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"], dtype=np.float32)
    model = TSWinPredictor(n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  model loaded: {CKPT_PATH}")

    print("  fetching recent BTC prices...")
    # Always delete today's cache before initial fetch — btc_data caches the
    # incomplete day and otherwise we'd read a stale snapshot from a prior run.
    from config import CACHE_DIR as _CD
    _today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    _today_cache = os.path.join(os.path.dirname(ROOT), _CD, f"btc_cb_{_today_str}.json")
    try:
        os.remove(_today_cache)
        print(f"  cleared stale today-cache: {os.path.basename(_today_cache)}")
    except FileNotFoundError:
        pass
    now = int(time.time())
    btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
    last_btc = time.time()
    print(f"  {len(btc_prices)} BTC minute prices loaded\n")

    stop = {"v": False}
    def _sig(*_): stop["v"] = True
    signal.signal(signal.SIGINT, _sig)

    while not stop["v"]:
        try:
            if time.time() - last_btc > 3600:
                now = int(time.time())
                # Delete today's cache file before re-fetching — btc_data caches
                # by day and won't refresh today's file otherwise, so we'd keep
                # reading a stale snapshot.
                from config import CACHE_DIR as _CD
                from datetime import datetime as _dt, timezone as _tz
                _today_str = _dt.now(_tz.utc).strftime("%Y%m%d")
                _today_cache = os.path.join(os.path.dirname(ROOT), _CD,
                                            f"btc_cb_{_today_str}.json")
                try:
                    os.remove(_today_cache)
                except FileNotFoundError:
                    pass
                btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
                last_btc = time.time()

            new_rows = read_new_settled(state)
            for r in new_rows:
                ticker = r["ticker"]
                winner = r.get("market_winner", "")
                if winner not in ("yes", "no"):
                    state["last_ticker"] = ticker
                    continue
                ts = r["window_ts"]
                try:
                    open_iso  = datetime.fromisoformat(ts.replace("Z","+00:00")).isoformat()
                    close_iso = datetime.fromisoformat(r["close_time"].replace("Z","+00:00")).isoformat()
                except Exception:
                    state["last_ticker"] = ticker
                    continue

                built = build_features(ticker, open_iso, close_iso, btc_prices)
                if built is None:
                    print(f"  [skip] {ticker}: feature build failed")
                    state["last_ticker"] = ticker
                    continue
                X, mask = built

                d = simulate_one(X, mask, winner, model, mean, std)
                state["cum_pnl"] += d["total_pnl"]

                if d["total_wagered"] > 0:
                    outcome = "net_win" if d["total_pnl"] > 0 else "net_loss"
                else:
                    outcome = "no_bet"

                with open(SHADOW_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([
                        ts, ticker, winner,
                        d["n_yes_bets"], d["n_no_bets"],
                        d["n_yes_bets"] + d["n_no_bets"],
                        f"{d['yes_wagered']:.4f}", f"{d['no_wagered']:.4f}",
                        f"{d['total_wagered']:.4f}",
                        f"{d['yes_pnl']:+.4f}", f"{d['no_pnl']:+.4f}",
                        f"{d['total_pnl']:+.4f}", f"{state['cum_pnl']:+.4f}", outcome,
                        d["first_minute"] if d["first_minute"] is not None else "",
                        d["first_direction"] or "",
                    ])

                tag = f"bets={d['n_yes_bets']}Y/{d['n_no_bets']}N"
                print(f"  {ts}  {ticker}  win={winner}  {tag}  "
                      f"wag=${d['total_wagered']:.2f}  P&L=${d['total_pnl']:+.2f}  "
                      f"cum=${state['cum_pnl']:+.2f}")
                state["last_ticker"] = ticker
                save_state(state)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

        for _ in range(int(interval)):
            if stop["v"]: break
            time.sleep(1)

    save_state(state)
    print("\nNN multi-bet shadow stopped.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=30)
    a = p.parse_args()
    run_loop(a.interval)
