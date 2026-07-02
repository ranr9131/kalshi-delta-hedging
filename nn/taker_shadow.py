"""
Forward paper-trading shadow for the VALIDATED NN TAKER strategy.
MULTI-ASSET, SINGLE PROCESS, SELF-FETCHING (no window_log / DH-trader dependency).

Strategy (per asset, validated by the recorded-book replay): at decision minutes
10-13, the FIRST time the asset's NN fair value for the moved direction beats the
REAL touch (cross-spread) by >= MIN_EDGE_C, take ONE taker bet at that touch and
HOLD TO EXPIRY. No hedging. Paper only; places no orders.

FORWARD / out-of-sample: records a start_ts on first run and only acts on windows
that SETTLE after that -> ticks the model has never seen. It polls the Kalshi REST
API for newly-settled markets per series itself (so it needs no other service).

One torch runtime, 4 tiny checkpoints -> ~230MB total (fits a small box).
Logs per asset to nn/taker_shadow_log_<ASSET>.csv ; state nn/taker_shadow_state.json.

Reuses build_features / nn_predict_at / _candle_at from live_shadow_v2_small.
Env: BASE_STAKE (10), MIN_EDGE_C (5), SIZING=flat|dynamic, ASSETS=BTC,ETH,SOL,XRP.
"""
import os, sys, csv, json, math, time, signal
import numpy as np
import torch
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(ROOT))

import kalshi_client
import btc_data, eth_data, sol_data, xrp_data
from model import TSWinPredictor
import live_shadow_v2_small as base

ASSET_CFG = {
    "BTC": ("KXBTC15M", "best_v2_small.pt",     btc_data),
    "ETH": ("KXETH15M", "best_v2_small_eth.pt", eth_data),
    "SOL": ("KXSOL15M", "best_v2_small_sol.pt", sol_data),
    "XRP": ("KXXRP15M", "best_v2_small_xrp.pt", xrp_data),
}

ASSETS     = [a.strip().upper() for a in os.environ.get("ASSETS", "BTC,ETH,SOL,XRP").split(",") if a.strip()]
MINUTES    = list(range(10, 14))
BASE_STAKE = float(os.environ.get("BASE_STAKE", "10"))
MIN_EDGE_C = float(os.environ.get("MIN_EDGE_C", "5"))
SIZING     = os.environ.get("SIZING", "flat")
FEE_KEEP   = 0.93
STATE_PATH = os.path.join(ROOT, "taker_shadow_state.json")
SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0
def f_btc(p):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))
def g_misp(m): return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))
def time_decay(minute): return 0.8 if minute < 10 else 1.2


def log_path(asset): return os.path.join(ROOT, f"taker_shadow_log_{asset}.csv")

def ensure_header(asset):
    p = log_path(asset)
    if os.path.exists(p): return
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow([
            "window_ts", "ticker", "winner", "acted", "minute", "side", "fair",
            "touch_fill", "edge_c", "contracts", "wagered", "won", "pnl", "cum_pnl"])

def load_state():
    if os.path.exists(STATE_PATH): return json.load(open(STATE_PATH))
    return {"start_ts": None, "cum_pnl": {}, "seen": {}}

def save_state(s): json.dump(s, open(STATE_PATH, "w"), indent=2)


def fetch_recent_settled(series, min_close_ts):
    """Poll Kalshi REST for settled markets of `series` closing >= min_close_ts."""
    out = []; cursor = None
    while True:
        params = {"series_ticker": series, "status": "settled",
                  "min_close_ts": int(min_close_ts), "limit": 200}
        if cursor: params["cursor"] = cursor
        data = kalshi_client._get("/markets", params=params)
        b = data.get("markets", []); out.extend(b)
        cursor = data.get("cursor")
        if not cursor or not b: break
        time.sleep(0.2)
    return out


def simulate_taker(ticker, open_iso, close_iso, X, mask, winner, model, mean, std):
    candles = kalshi_client.fetch_candlesticks(ticker, open_iso, close_iso)
    if not candles: return None
    t0 = int(datetime.fromisoformat(open_iso.replace("Z", "+00:00")).timestamp())
    resolved_yes = (winner == "yes")
    for m in MINUTES:
        if not mask[m]: continue
        ret = float(X[m, 0])
        if ret == 0: continue
        up = ret > 0; pct = abs(ret) * 100.0
        p_yes = base.nn_predict_at(model, mean, std, X, mask, m)
        fair = p_yes if up else (1.0 - p_yes)
        cand = base._candle_at(candles, t0 + m * 60)
        if cand is None: continue
        ask = float(cand.get("yes_ask_close", cand["yes_close"]))
        bid = float(cand.get("yes_bid_close", cand["yes_close"]))
        fill = ask if up else (1.0 - bid)
        if not (0.01 < fill < 0.99): continue
        edge = fair - fill
        if edge * 100.0 < MIN_EDGE_C: continue
        wagered = (BASE_STAKE * f_btc(pct) * g_misp(edge) * time_decay(m)
                   if SIZING == "dynamic" else BASE_STAKE)
        contracts = wagered / fill
        win = resolved_yes if up else (not resolved_yes)
        pnl = contracts * ((1.0 - fill) * FEE_KEEP if win else -fill)
        return {"acted": 1, "minute": m, "side": ("yes" if up else "no"),
                "fair": fair, "touch_fill": fill, "edge_c": edge * 100.0,
                "contracts": contracts, "wagered": wagered, "won": int(win), "pnl": pnl}
    return {"acted": 0, "minute": "", "side": "", "fair": "", "touch_fill": "",
            "edge_c": "", "contracts": 0.0, "wagered": 0.0, "won": "", "pnl": 0.0}


def run_loop(interval=60):
    print(f"NN TAKER multi-asset shadow | assets={ASSETS} base=${BASE_STAKE} "
          f"min_edge={MIN_EDGE_C}c sizing={SIZING}", flush=True)
    models = {}; spot = {}
    for a in ASSETS:
        series, ckpt_file, dm = ASSET_CFG[a]
        ck = torch.load(os.path.join(ROOT, "checkpoints", ckpt_file),
                        map_location="cpu", weights_only=False)
        m = TSWinPredictor(n_features=ck["n_features"], d_model=ck["d_model"],
                           n_heads=ck["n_heads"], n_layers=ck["n_layers"],
                           dim_feedforward=ck["dim_feedforward"], dropout=ck["dropout"])
        m.load_state_dict(ck["model_state"]); m.eval()
        models[a] = (m, np.array(ck["feature_mean"], np.float32),
                     np.array(ck["feature_std"], np.float32))
        ensure_header(a)
        print(f"  [{a}] model {ckpt_file}", flush=True)

    def refresh_spot():
        now = int(time.time())
        for a in ASSETS:
            spot[a] = ASSET_CFG[a][2].fetch_btc_prices(now - 30 * 86400, now + 1800)
            print(f"  [{a}] {len(spot[a])} spot minutes", flush=True)
    refresh_spot(); last_fetch = time.time()

    state = load_state()
    if state["start_ts"] is None:
        state["start_ts"] = int(time.time()); save_state(state)
        print(f"  forward-start: only acting on windows settling after "
              f"{datetime.fromtimestamp(state['start_ts'], tz=timezone.utc)}", flush=True)
    state.setdefault("cum_pnl", {}); state.setdefault("seen", {})

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    while not stop["v"]:
        try:
            if time.time() - last_fetch > 3600:
                refresh_spot(); last_fetch = time.time()
            for a in ASSETS:
                series, _, dm = ASSET_CFG[a]
                base.btc_data = dm                      # redirect build_features spot
                model, mean, std = models[a]
                seen = set(state["seen"].get(a, []))
                fresh = fetch_recent_settled(series, time.time() - 3 * 3600)
                for mk in fresh:
                    tk = mk.get("ticker"); res = mk.get("result")
                    if not tk or tk in seen or res not in ("yes", "no"):
                        continue
                    try:
                        close_ts = int(datetime.fromisoformat(
                            mk["close_time"].replace("Z", "+00:00")).timestamp())
                    except Exception:
                        continue
                    seen.add(tk)
                    if close_ts < state["start_ts"]:       # pre-start -> mark seen, skip (forward-only)
                        continue
                    open_iso  = mk["open_time"]; close_iso = mk["close_time"]
                    ts = datetime.fromisoformat(open_iso.replace("Z", "+00:00")).isoformat()
                    built = base.build_features(tk, open_iso, close_iso, spot[a])
                    if built is None:
                        continue
                    X, mask = built
                    d = simulate_taker(tk, open_iso, close_iso, X, mask, res, model, mean, std)
                    if d is None:
                        continue
                    cum = state["cum_pnl"].get(a, 0.0) + d["pnl"]
                    state["cum_pnl"][a] = cum
                    with open(log_path(a), "a", newline="") as f:
                        csv.writer(f).writerow([
                            ts, tk, res, d["acted"], d["minute"], d["side"],
                            f"{d['fair']:.4f}" if d["fair"] != "" else "",
                            f"{d['touch_fill']:.4f}" if d["touch_fill"] != "" else "",
                            f"{d['edge_c']:.2f}" if d["edge_c"] != "" else "",
                            f"{d['contracts']:.2f}", f"{d['wagered']:.2f}",
                            d["won"], f"{d['pnl']:+.4f}", f"{cum:+.4f}"])
                    if d["acted"]:
                        print(f"  [{a}] {ts} {tk} win={res} {d['side']}@{d['touch_fill']:.2f} "
                              f"edge={d['edge_c']:.1f}c P&L=${d['pnl']:+.2f} cum=${cum:+.2f}", flush=True)
                # prune seen to recent (keep last 400 per asset)
                state["seen"][a] = list(seen)[-400:]
            save_state(state)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}", flush=True)
        for _ in range(int(interval)):
            if stop["v"]: break
            time.sleep(1)
    save_state(state)
    print("\nmulti-asset taker shadow stopped.", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--interval", type=int, default=60)
    run_loop(p.parse_args().interval)
