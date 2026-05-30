"""
Live shadow for v2_small (14 features, v1-sized architecture).

Runs alongside the existing v1 shadow (live_shadow_multi.py). Polls
live/window_log.csv for settled windows, builds the 14-feature tensor
matching build_dataset_v2.py, runs NN at each decision minute T+10..T+13,
and logs results to nn/shadow_log_v2_small.csv.

Same DH template as live_shadow_multi.py (multi-bet, target mode, RH=T+10,
leg cap=2, edge=10c, base stake $10).
"""

import os, sys, csv, json, math, time, signal, argparse
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
CKPT_PATH       = os.path.join(ROOT, "checkpoints", "best_v2_small.pt")
SHADOW_LOG      = os.path.join(ROOT, "shadow_log_v2_small.csv")
STATE_PATH      = os.path.join(ROOT, "shadow_state_v2_small.json")

# Strategy config (matches live_shadow_multi.py)
MINUTES        = list(range(10, 14))
BASE_STAKE     = 10.0
MIN_BET        = 1.0
SLIP_C         = 4
MIN_EDGE_C     = 10
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

N_FEATURES     = 14
WINDOW_MINUTES = 15
DECISION_MIN_FOR_PREFILL = 10   # min decision minute the NN sees


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


def _candle_at(candles, target_ts):
    for c in candles:
        if c["ts"] >= target_ts:
            return c
    return None


def build_features(ticker, open_iso, close_iso, btc_prices):
    """Return (X[15,14] float32, mask[15] bool) or None if data missing.

    Mirrors build_dataset_v2.py exactly.
    """
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())
    btc_t0 = btc_data.lookup(btc_prices, t0)
    if btc_t0 is None: return None
    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles: return None
    c0 = _candle_at(candles, t0)
    if c0 is None: return None
    kalshi_t0 = c0["yes_close"]
    if not (0.01 < kalshi_t0 < 0.99): return None

    X = np.zeros((WINDOW_MINUTES, N_FEATURES), dtype=np.float32)
    mask = np.zeros(WINDOW_MINUTES, dtype=bool)
    hour = open_dt.hour; dow = open_dt.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow / 7)
    btc_history = []
    last_btc = btc_t0
    abs_max  = 0.0
    for m in range(WINDOW_MINUTES):
        t = t0 + m * 60
        btc = btc_data.lookup(btc_prices, t)
        cand = _candle_at(candles, t)
        if btc is None or cand is None: continue
        yc = float(cand["yes_close"])
        if not (0.01 < yc < 0.99): continue
        btc_history.append(btc)
        ret_t0 = (btc / btc_t0) - 1.0
        ret_1m = (btc / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc
        abs_max = max(abs_max, abs(ret_t0))
        ret_5m = (btc / btc_history[-6]) - 1.0 if len(btc_history) >= 6 else 0.0
        yo = float(cand.get("yes_open", yc))
        yh = float(cand.get("yes_high", yc))
        yl = float(cand.get("yes_low",  yc))
        bc = float(cand.get("yes_bid_close", yc))
        ac = float(cand.get("yes_ask_close", yc))
        vol = float(cand.get("volume", 0.0))
        intramin = yc - yo
        rng_norm = (yh - yl) / max(yc, 0.05)
        spread   = max(0.0, min(0.20, ac - bc))
        vol_log  = math.log1p(vol) / 10.0
        X[m, 0]  = ret_t0
        X[m, 1]  = ret_1m
        X[m, 2]  = ret_5m
        X[m, 3]  = abs_max
        X[m, 4]  = yc
        X[m, 5]  = yc - kalshi_t0
        X[m, 6]  = intramin
        X[m, 7]  = rng_norm
        X[m, 8]  = spread
        X[m, 9]  = vol_log
        X[m, 10] = m / float(WINDOW_MINUTES - 1)
        X[m, 11] = hour_sin
        X[m, 12] = hour_cos
        X[m, 13] = dow_sin
        mask[m] = True
    if mask.sum() < 5: return None
    return X, mask


def nn_predict_at(model, mean, std, X_raw, mask, minute):
    X = X_raw.copy(); mk = mask.copy()
    X[minute + 1:, :] = 0.0
    mk[minute + 1:] = False
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logit = model(torch.from_numpy(Xn[None]), torch.from_numpy(mk[None]))
        return float(torch.sigmoid(logit).item())


def simulate_one(X_raw, mask, winner, model, mean, std):
    resolved_yes = (winner == "yes")
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    first_minute = None; first_dir = None
    for m in MINUTES:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret_t = float(X_raw[m, 0])
        kal_t = float(X_raw[m, 4])    # v2 schema: feature 4 = yes_close
        if not (0.01 < kal_t < 0.99) or ret_t == 0: continue
        up = ret_t > 0; pct = abs(ret_t) * 100.0
        p_yes = nn_predict_at(model, mean, std, X_raw, mask, m)
        fair  = p_yes if up else (1.0 - p_yes)
        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal_t + slip)
        no_fill  = min(MAX_FILL, (1.0 - kal_t) + slip)
        if up:
            mispr = fair - yes_fill; fill_use = yes_fill
        else:
            mispr = fair - no_fill;  fill_use = no_fill
        if mispr * 100 < MIN_EDGE_C:
            target = 0.0
        else:
            target = BASE_STAKE * sigmoid_btc(pct) * sigmoid_misp(mispr) * time_decay(m)
        if m <= EARLY_SKIP_M and pct < EARLY_SKIP_P:
            target = 0.0
        if up:
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
        if m >= RH_MIN and legs < MAX_LEGS:
            if up:
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
    if not os.path.exists(WINDOW_LOG_PATH): return []
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
    print(f"v2_small live shadow starting | base_stake=${BASE_STAKE} "
          f"min_edge={MIN_EDGE_C}c RH=T+{RH_MIN} leg_cap={MAX_LEGS}", flush=True)
    ensure_log_header()
    state = load_state()
    print(f"  last_ticker={state['last_ticker']!r}  cum_pnl=${state['cum_pnl']:+.2f}",
          flush=True)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"], dtype=np.float32)
    model = TSWinPredictor(
        n_features=ckpt["n_features"],
        d_model=ckpt.get("d_model", 32),
        n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 2),
        dim_feedforward=ckpt.get("dim_feedforward", 64),
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  model loaded: {CKPT_PATH}", flush=True)

    print("  fetching recent BTC prices...", flush=True)
    from config import CACHE_DIR as _CD
    _today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    _today_cache = os.path.join(os.path.dirname(ROOT), _CD, f"btc_cb_{_today_str}.json")
    try: os.remove(_today_cache)
    except FileNotFoundError: pass
    now = int(time.time())
    btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
    last_btc = time.time()
    print(f"  {len(btc_prices)} BTC minute prices loaded\n", flush=True)

    stop = {"v": False}
    def _sig(*_): stop["v"] = True
    signal.signal(signal.SIGINT, _sig)

    while not stop["v"]:
        try:
            if time.time() - last_btc > 3600:
                from config import CACHE_DIR as _CD2
                _ts = datetime.now(timezone.utc).strftime("%Y%m%d")
                _tc = os.path.join(os.path.dirname(ROOT), _CD2, f"btc_cb_{_ts}.json")
                try: os.remove(_tc)
                except FileNotFoundError: pass
                now = int(time.time())
                btc_prices = btc_data.fetch_btc_prices(now - 30 * 86400, now + 1800)
                last_btc = time.time()

            new_rows = read_new_settled(state)
            for r in new_rows:
                ticker = r["ticker"]
                winner = r.get("market_winner", "")
                if winner not in ("yes", "no"):
                    state["last_ticker"] = ticker; continue
                ts = r["window_ts"]
                try:
                    open_iso  = datetime.fromisoformat(ts.replace("Z","+00:00")).isoformat()
                    close_iso = datetime.fromisoformat(r["close_time"].replace("Z","+00:00")).isoformat()
                except Exception:
                    state["last_ticker"] = ticker; continue

                built = build_features(ticker, open_iso, close_iso, btc_prices)
                if built is None:
                    print(f"  [skip] {ticker}: feature build failed", flush=True)
                    state["last_ticker"] = ticker; continue
                X, mask = built

                d = simulate_one(X, mask, winner, model, mean, std)
                state["cum_pnl"] += d["total_pnl"]

                outcome = ("net_win" if d["total_pnl"] > 0 else "net_loss") \
                          if d["total_wagered"] > 0 else "no_bet"

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
                      f"cum=${state['cum_pnl']:+.2f}", flush=True)
                state["last_ticker"] = ticker
                save_state(state)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}", flush=True)
        for _ in range(int(interval)):
            if stop["v"]: break
            time.sleep(1)

    save_state(state)
    print("\nv2_small shadow stopped.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=30)
    a = p.parse_args()
    run_loop(a.interval)
