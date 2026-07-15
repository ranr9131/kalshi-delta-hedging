"""
BTC-Kalshi live trader.

Modes (set MODE in .env):
  t+5         — one bet per window at T+5
  dh-target   — delta hedge T+4..T+13, target position sizing
  dh-additive — delta hedge T+4..T+13, additive sizing

Run with PAPER_MODE=true to simulate without placing real orders.
"""

import csv
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from dotenv import dotenv_values

import btc_feed
import eth_feed
import sol_feed
import xrp_feed
import kalshi_auth
import kalshi_feed
import kalshi_trade
import strategy
import asian_pricer

# ── Asset selection (BTC by default, ETH/SOL/XRP via ASSET env) ───────────────
# This module is structured for BTC originally. Multi-asset support added by
# swapping `price_feed` (Coinbase asset-USD WS) and pointing kalshi_trade at
# the right series (KX{asset}15M) via the KALSHI_SERIES env var. The NN v2
# feature builder also uses the matching {asset}_data module.
import os as _os_for_asset
ASSET = _os_for_asset.environ.get("ASSET", "BTC").upper()
_FEEDS = {"BTC": btc_feed, "ETH": eth_feed, "SOL": sol_feed, "XRP": xrp_feed}
_DEFAULT_SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
if ASSET not in _FEEDS:
    raise ValueError(f"Unknown ASSET={ASSET}. Must be one of {list(_FEEDS)}.")
price_feed = _FEEDS[ASSET]
if ASSET != "BTC":
    kalshi_trade.SERIES = _os_for_asset.environ.get("KALSHI_SERIES", _DEFAULT_SERIES[ASSET])

# ── Config ────────────────────────────────────────────────────────────────────
# .env supplies credentials + defaults. os.environ (e.g. from systemd
# Environment= directives) takes precedence so a separate systemd unit can
# override PAPER_MODE, BASE_STAKE, etc. without touching .env.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
env = {**dotenv_values(_env_path), **os.environ}

API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")
PAPER_MODE = env.get("PAPER_MODE", "true").lower() == "true"
BASE_STAKE = float(env.get("BASE_STAKE", "100.0"))
MODE       = env.get("MODE", "dh-target").lower()   # t+5 | dh-target | dh-additive
MIN_BET    = float(env.get("MIN_BET", "5.0"))

# ── Sizing engine ─────────────────────────────────────────────────────────────
# SIZING=sigmoid (default): target = BASE_STAKE * f_btc * g_misprice * td_mult
# SIZING=s6kelly: calibrated logistic P(win) + fractional Kelly. Backtested
# OOS at +79% ROI vs sigmoid's +55% at equal Sharpe (see s6_calibrated.py).
# Model coefficients come from s6_calibration.json (fit on the 90d BTC
# dataset) — BTC-only; do not enable for other assets.
SIZING        = env.get("SIZING", "sigmoid").strip().lower()
# Final-2-min Asian-settlement pricer (settlement = 60s BRTI average; the 2D
# table prices spot and cannot see the banked partial average).
ASIAN_PRICER  = env.get("ASIAN_PRICER", "true").strip().lower() == "true"

# ── Regime throttle ───────────────────────────────────────────────────────────
# Chop persists: backtested on 5,350 windows (32 days), skipping whenever <=4
# of the last 8 SETTLED windows continued raised proxy P&L +14% and cut max
# drawdown 59% (scratchpad/throttle_backtest.py sweep, robust across K=8..24).
REGIME_THROTTLE = env.get("REGIME_THROTTLE", "false").strip().lower() == "true"
REGIME_LOOKBACK = int(env.get("REGIME_LOOKBACK", "8"))
REGIME_MIN_CONT = float(env.get("REGIME_MIN_CONT", "0.55"))

from collections import deque as _rt_deque
_REGIME_HIST = _rt_deque(maxlen=REGIME_LOOKBACK)


def _regime_record(btc_t0, btc_t5, winner):
    """Record one settled window's continuation outcome (T+5 leader held?)."""
    try:
        t0, t5 = float(btc_t0), float(btc_t5)
        if winner in ("yes", "no") and t5 > 0 and t5 != t0:
            _REGIME_HIST.append(1 if ((winner == "yes") == (t5 > t0)) else 0)
    except (TypeError, ValueError):
        pass


def regime_ok() -> tuple[bool, float]:
    """(trade?, trailing continuation rate). Warm-up (deque not full) -> trade."""
    if not REGIME_THROTTLE or len(_REGIME_HIST) < REGIME_LOOKBACK:
        return True, 1.0
    rate = sum(_REGIME_HIST) / len(_REGIME_HIST)
    return rate >= REGIME_MIN_CONT, rate
S6_BANKROLL   = float(env.get("S6_BANKROLL", "1000.0"))
S6_KELLY_FRAC = float(env.get("S6_KELLY_FRAC", "0.25"))
S6_MAX_STAKE  = float(env.get("S6_MAX_STAKE", "150.0"))
S6_EDGE_MIN   = float(env.get("S6_EDGE_MIN", "0.0"))

_S6_MODEL = None
if SIZING == "s6kelly":
    import json as _json
    # S6_CALIBRATION_PATH lets a shadow arm run a candidate model (e.g.
    # s6_calibration_v2.json) while other arms keep the incumbent.
    _s6_path = env.get("S6_CALIBRATION_PATH", "").strip() or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "s6_calibration.json")
    with open(_s6_path) as _f:
        _S6_MODEL = _json.load(_f)


def s6_p_win(fair: float, abs_pct: float, minute_idx: int,
             direction_up: bool = True) -> float:
    """Calibrated P(continuation side wins) from the fitted logistic.
    Predictors must match the fit: logit(fair_2d), abs_pct, minute/14,
    and (v2 models, 4 coefficients) direction_up."""
    c = _S6_MODEL["coef"]
    f = min(1 - 1e-4, max(1e-4, fair))
    z = (_S6_MODEL["intercept"]
         + c[0] * math.log(f / (1 - f))
         + c[1] * abs_pct
         + c[2] * (minute_idx / 14.0))
    if len(c) > 3:
        z += c[3] * (1.0 if direction_up else 0.0)
    return 1.0 / (1.0 + math.exp(-z))

# Hard cap on total dollars wagered within a single 15-min window. The DH loop
# truncates each bet to fit; once the cap is reached no further bets are placed.
# Worst-case window loss ≈ this number. Set to 0 to disable.
#
# Cap policy: if CAP_FRACTION_OF_BALANCE > 0, the cap is computed dynamically
# each window as balance × CAP_FRACTION_OF_BALANCE (lets the cap scale with
# winnings, shrink on losses). Falls back to MAX_WINDOW_WAGERED if balance
# fetch fails or CAP_FRACTION_OF_BALANCE is 0.
MAX_WINDOW_WAGERED      = float(env.get("MAX_WINDOW_WAGERED", "0"))
CAP_FRACTION_OF_BALANCE = float(env.get("CAP_FRACTION_OF_BALANCE", "0.0"))

# Reversal-hedge overlay. If RH_MINUTE is set (e.g. 10), then at each minute
# >= RH_MINUTE during the DH loop, if BTC direction is opposite the side we
# hold meaningful exposure on (>= RH_TRIGGER dollars), buy enough of the
# now-correct side to neutralize the contracts owned on the losing side.
# Unset (or 0) = disabled / baseline behavior.
_rh_raw  = env.get("RH_MINUTE", "").strip()
RH_MINUTE: float | None = float(_rh_raw) if _rh_raw and float(_rh_raw) > 0 else None
RH_TRIGGER = float(env.get("RH_TRIGGER", "10.0"))

# Per-window leg cap. 0 = disabled. See .env for the data motivating this cap.
MAX_LEGS_PER_WINDOW = int(env.get("MAX_LEGS_PER_WINDOW", "0"))

# Side filter — restrict bot to one direction. Live data showed NO bets ran
# -8.2% ROI on 95 windows vs YES bets at +1.6% on 78 windows. Suggests the
# fair-price table is miscalibrated on the NO side. Set "yes_only", "no_only",
# or empty (default) to allow both.
SIDE_FILTER = env.get("SIDE_FILTER", "").strip().lower()
if SIDE_FILTER not in ("", "yes_only", "no_only"):
    print(f"ERROR: SIDE_FILTER must be 'yes_only', 'no_only', or empty. Got: {SIDE_FILTER!r}")
    sys.exit(1)

# ACTIVE_HOURS: comma-separated UTC hours to trade, e.g. "13,14,18,22".
# Empty or unset = trade all 24 hours.
_active_hours_raw = env.get("ACTIVE_HOURS", "").strip()
ACTIVE_HOURS: set[int] | None = (
    {int(h.strip()) for h in _active_hours_raw.split(",") if h.strip()}
    if _active_hours_raw else None
)

raw_pem = env.get("KALSHI_PRIVATE_KEY", "")
if not raw_pem and not PAPER_MODE:
    print("ERROR: KALSHI_PRIVATE_KEY not set in .env. Set PAPER_MODE=true or add the key.")
    sys.exit(1)
PRIVATE_KEY = kalshi_auth.load_private_key(raw_pem) if raw_pem else None

# ── Log paths ─────────────────────────────────────────────────────────────────
# LOG_TAG suffixes the log files (trade_log.<tag>.csv) so parallel paper
# instances don't clobber each other — matches the existing naming convention
# (trade_log.paper-s2.csv etc.).
_dir = os.path.dirname(os.path.abspath(__file__))
LOG_TAG = env.get("LOG_TAG", "").strip()
_log_sfx = f".{LOG_TAG}" if LOG_TAG else ""
TRADE_LOG_PATH  = os.path.join(_dir, f"trade_log{_log_sfx}.csv")
WINDOW_LOG_PATH = os.path.join(_dir, f"window_log{_log_sfx}.csv")

# One row per individual bet (all modes)
TRADE_LOG_FIELDS = [
    "window_ts", "mode", "ticker", "close_time",
    "dh_minute",
    "btc_t0", "btc_now", "btc_price_age_secs", "abs_pct_move",
    "yes_bid", "yes_ask", "spread", "kalshi_yes_mid",
    "direction",
    "yes_target", "no_target",
    "yes_exposure_before", "no_exposure_before",
    "bet_side",
    "mispricing", "f_btc", "g_misprice",
    "stake", "fill_price", "count",
    "order_id", "order_result",
]

# One row per window (totals + settlement outcome)
WINDOW_LOG_FIELDS = [
    "window_ts", "mode", "ticker", "close_time",
    "btc_t0", "btc_t5", "btc_t10",
    "n_yes_bets", "n_no_bets", "total_bets",
    "total_yes_stake", "total_no_stake", "total_wagered",
    "settlement_ts", "market_winner",
    "yes_pnl", "no_pnl", "total_pnl",
    "outcome",
    "cumulative_pnl",
]

WINDOW_MINUTES       = 15
# DH cadence: evaluate every 30s from T+4:00 through T+14:00.
# Extended past T+13 to catch last-second reversals that flip the outcome
# between our last sample and settlement (the "T+13 → T+15 blind spot").
DH_TICK_SECS         = 30
DH_FIRST_TICK_SECS   = 4 * 60
DH_LAST_TICK_SECS    = 14 * 60
DH_OFFSETS_SECS      = list(range(DH_FIRST_TICK_SECS, DH_LAST_TICK_SECS + 1, DH_TICK_SECS))
# DH modes enter at T+4; t+5 mode still waits until T+5
DECISION_OFFSET_SECS = 4 * 60 if MODE.startswith("dh") else 5 * 60
MAX_FILL_PRICE       = float(env.get("MAX_FILL_PRICE", "0.97"))   # skip ENTRY bets whose buffered fill price exceeds this
# Hedges (RH) skip when fill exceeds this. Was 0.995 (allow any fill); lowered
# to 0.80 after observing cascade losses where hedges at $0.85-0.99 had
# minimal insurance value (you pay 95c to receive $1 if right — that's 5c of
# insurance per contract, often less than the spread + fee leakage). Skipping
# expensive hedges means accepting the wrong-side loss, which is bounded.
MAX_HEDGE_FILL_PRICE = 0.80

# Time-decay sizing: shrink early-window bets (signal is mostly noise), grow
# late-window bets (signal has played out). Sim across 6,275 historical
# windows: +$65k P&L, -$224k wagered, +6.5pp ROI, $55k loss-recovery on the
# baseline-losing windows. Disable by setting TIME_DECAY=false in .env.
TIME_DECAY = env.get("TIME_DECAY", "true").strip().lower() not in ("false", "0", "no", "")

# Minimum required edge (in CENTS) for a new entry-direction bet.
# Real Kalshi fees average ~1.75c per contract on a 50c market — any trade
# with edge < 1c is destined to be unprofitable after fees. The sim shows
# adding this filter recovers +5-8pp ROI vs no filter. Set to 0 to disable.
MIN_EDGE_CENTS = float(env.get("MIN_EDGE_CENTS", "1.0"))

# Minimum fill price floor. Refuse to buy a contract for less than this many
# dollars. Cuts the long tail of cheap-side bets where adverse selection is
# worst (e.g. buying YES at 11c against informed sellers). Set to 0 to disable.
MIN_BET_PRICE = float(env.get("MIN_BET_PRICE", "0"))

# Skip DH entry decisions before this window minute (0 = off). Lets an arm bet
# only late-window — e.g. 13.0 restricts to the final ~2 minutes, where the
# Asian-settlement pricer (not the 2D continuation table) drives p_win.
DH_MIN_MINUTE = float(env.get("DH_MIN_MINUTE", "0"))

# Invert mode — swap every YES/NO bet decision. Used to test the inverse-signal
# hypothesis: if the live strategy systematically loses, the inverse should
# win. Strictly PAPER-only by convention (no live code change needed; just set
# PAPER_MODE=true alongside).
INVERT = env.get("INVERT", "false").strip().lower() in ("1", "true", "yes")

# ── NN fair-price source ──────────────────────────────────────────────────────
# When FAIR_PRICE_SOURCE=nn, the trader replaces the 2D-table fair-price lookup
# with NN inference using the multi-minute model. Other strategy logic (sizing,
# RH, leg cap, edge filter) stays identical. NN_MIN_MINUTE constrains the
# trader to only place bets at or after this minute (default 10 — early minutes
# have low ROI per backtest sweep). DAILY_LOSS_CAP auto-pauses trading after a
# bad day (0 = disabled).
FAIR_PRICE_SOURCE = env.get("FAIR_PRICE_SOURCE", "2d").strip().lower()
NN_CHECKPOINT     = env.get("NN_CHECKPOINT", "../nn/checkpoints/best_multi.pt")
NN_MIN_MINUTE     = int(env.get("NN_MIN_MINUTE", "10"))
NN_MAX_MINUTE     = int(env.get("NN_MAX_MINUTE", "13"))
DAILY_LOSS_CAP    = float(env.get("DAILY_LOSS_CAP", "0"))   # dollars; 0 = off

_NN_MODEL = None
_NN_MEAN  = None
_NN_STD   = None
_NN_FEATURES_N = 7
_NN_WINDOW_M   = 15
_NN_SCHEMA = "v1"   # "v1" (7 features) or "v2" (14 features)


def _load_nn():
    """Lazy-load the NN model. Called once on startup if NN mode is active.

    Detects schema from checkpoint: n_features=7 → v1, n_features=14 → v2.
    Reads architecture params (d_model, n_layers, ...) from checkpoint if present.
    """
    global _NN_MODEL, _NN_MEAN, _NN_STD, _NN_FEATURES_N, _NN_SCHEMA
    if _NN_MODEL is not None:
        return
    import torch as _torch
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", "nn"))
    from model import TSWinPredictor          # type: ignore
    ckpt_path = NN_CHECKPOINT
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(_here, ckpt_path))
    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _NN_FEATURES_N = int(ckpt.get("n_features", 7))
    _NN_SCHEMA = "v2" if _NN_FEATURES_N == 14 else "v1"
    _NN_MODEL = TSWinPredictor(
        n_features=_NN_FEATURES_N,
        d_model=ckpt.get("d_model", 32),
        n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 2),
        dim_feedforward=ckpt.get("dim_feedforward", 64),
        dropout=ckpt.get("dropout", 0.1),
    )
    _NN_MODEL.load_state_dict(ckpt["model_state"])
    _NN_MODEL.eval()
    _NN_MEAN = np.array(ckpt["feature_mean"], dtype=np.float32)
    _NN_STD  = np.array(ckpt["feature_std"],  dtype=np.float32)
    return ckpt_path


def _build_nn_features(window_snapshots: dict, btc_t0: float, kalshi_t0: float,
                       hour: int, current_minute: int):
    """Return (X[15,7] float32, mask[15] bool) with data through current_minute."""
    X = np.zeros((_NN_WINDOW_M, _NN_FEATURES_N), dtype=np.float32)
    mask = np.zeros(_NN_WINDOW_M, dtype=bool)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    last_btc = btc_t0
    for m in range(_NN_WINDOW_M):
        if m > current_minute:
            break
        snap = window_snapshots.get(m)
        if snap is None:
            continue
        btc = snap.get("btc")
        kalshi = snap.get("kalshi_yes")
        if btc is None or kalshi is None or not (0.01 < kalshi < 0.99):
            continue
        X[m, 0] = (btc / btc_t0) - 1.0
        X[m, 1] = (btc / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc
        X[m, 2] = kalshi
        X[m, 3] = kalshi - kalshi_t0
        X[m, 4] = m / float(_NN_WINDOW_M - 1)
        X[m, 5] = hour_sin
        X[m, 6] = hour_cos
        mask[m] = True
    return X, mask


def _candle_at(candles, target_ts):
    for c in candles:
        if c["ts"] >= target_ts:
            return c
    return None


# Per-window cache for v2 candles + BTC. Keyed by window open_iso. Reset when
# a new window is seen. Avoids redundant API calls across the 30s decision ticks.
_V2_WINDOW_CACHE: dict = {"open_iso": None, "candles": None, "btc_prices": None,
                          "btc_t0": None, "kalshi_t0": None}


def _build_nn_features_v2(ticker, open_iso, close_iso, current_minute, btc_t0):
    """Return (X[15,14], mask[15]) for v2 schema at decision time.

    Mirrors nn/build_dataset_v2.py exactly. Fetches Kalshi candles + BTC prices
    on first call per window (cached for subsequent ticks).
    """
    # Lazy import — only needed in NN mode
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    import kalshi_client as _kc
    if ASSET == "ETH":
        import eth_data as _bd
    elif ASSET == "SOL":
        import sol_data as _bd
    elif ASSET == "XRP":
        import xrp_data as _bd
    else:
        import btc_data as _bd
    from datetime import datetime as _dt

    open_dt = _dt.fromisoformat(open_iso.replace("Z", "+00:00"))
    t0 = int(open_dt.timestamp())

    # Helper: kalshi_client.fetch_candlesticks has a sticky disk cache that
    # never refreshes once written. For LIVE windows we MUST invalidate it
    # before every fetch, else the NN sees only the first few minutes of
    # candle data forever (fair stays constant across T+10..T+13).
    def _fresh_candles():
        try:
            from config import CACHE_DIR as _CD
            _cache_file = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), _CD, f"candles_{ticker}.json")
            if os.path.exists(_cache_file):
                os.remove(_cache_file)
        except Exception:
            pass
        try:
            return _kc.fetch_candlesticks(ticker, open_iso, close_iso)
        except Exception:
            return None

    # Refresh cache if window changed
    if _V2_WINDOW_CACHE["open_iso"] != open_iso:
        candles = _fresh_candles()
        # Delete today's price cache before fetching so we get fresh data.
        # The per-day cache otherwise returns stale data captured at first call.
        # IMPORTANT: file name must match the asset (btc_cb vs eth_cb).
        try:
            from config import CACHE_DIR as _CD
            _today = _dt.now().strftime("%Y%m%d")
            _prefix = {"ETH": "eth_cb", "SOL": "sol_cb", "XRP": "xrp_cb"}.get(ASSET, "btc_cb")
            _today_cache = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), _CD, f"{_prefix}_{_today}.json")
            if os.path.exists(_today_cache):
                os.remove(_today_cache)
        except Exception:
            pass
        try:
            btc_prices = _bd.fetch_btc_prices(t0 - 60, t0 + 16 * 60)
        except Exception:
            btc_prices = {}
        _V2_WINDOW_CACHE.update({
            "open_iso": open_iso, "candles": candles, "btc_prices": btc_prices,
            "btc_t0": btc_t0, "kalshi_t0": None,
        })
    else:
        # Refresh candles each minute tick (high/low/volume update as minute closes).
        # Use _fresh_candles to bypass the sticky disk cache.
        fresh = _fresh_candles()
        if fresh:
            _V2_WINDOW_CACHE["candles"] = fresh
        # Also refresh per-day price cache mid-window so newer minutes' prices
        # land in lookups for T+11, T+12, T+13 (otherwise they fall back to T+10).
        try:
            from config import CACHE_DIR as _CD
            _today = _dt.now().strftime("%Y%m%d")
            _prefix = {"ETH": "eth_cb", "SOL": "sol_cb", "XRP": "xrp_cb"}.get(ASSET, "btc_cb")
            _today_cache = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), _CD, f"{_prefix}_{_today}.json")
            if os.path.exists(_today_cache):
                os.remove(_today_cache)
            _V2_WINDOW_CACHE["btc_prices"] = _bd.fetch_btc_prices(t0 - 60, t0 + 16 * 60)
        except Exception:
            pass

    candles = _V2_WINDOW_CACHE["candles"]
    btc_prices = _V2_WINDOW_CACHE["btc_prices"]
    if not candles or not btc_prices:
        return None, None

    c0 = _candle_at(candles, t0)
    if c0 is None:
        return None, None
    kalshi_t0 = c0["yes_close"]
    if not (0.01 < kalshi_t0 < 0.99):
        return None, None

    X = np.zeros((_NN_WINDOW_M, 14), dtype=np.float32)
    mask = np.zeros(_NN_WINDOW_M, dtype=bool)
    hour = open_dt.hour
    dow = open_dt.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin  = math.sin(2 * math.pi * dow / 7)
    btc_history = []
    last_btc = btc_t0
    abs_max  = 0.0

    for m in range(_NN_WINDOW_M):
        if m > current_minute:
            break
        t = t0 + m * 60
        btc = _bd.lookup(btc_prices, t)
        cand = _candle_at(candles, t)
        if btc is None or cand is None:
            continue
        yc = float(cand.get("yes_close"))
        if not (0.01 < yc < 0.99):
            continue
        btc_history.append(btc)
        ret_t0 = (btc / btc_t0) - 1.0
        ret_1m = (btc / last_btc) - 1.0 if last_btc else 0.0
        last_btc = btc
        abs_max  = max(abs_max, abs(ret_t0))
        ret_5m   = (btc / btc_history[-6]) - 1.0 if len(btc_history) >= 6 else 0.0
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
        X[m, 10] = m / float(_NN_WINDOW_M - 1)
        X[m, 11] = hour_sin
        X[m, 12] = hour_cos
        X[m, 13] = dow_sin
        mask[m] = True

    return X, mask


def _nn_p_yes(window_snapshots, btc_t0, kalshi_t0, hour, current_minute,
              ticker=None, open_iso=None, close_iso=None):
    """Run NN to get P(YES wins) given data through current_minute.

    Dispatches based on schema: v2 fetches its own candles/btc, v1 uses
    the in-memory window_snapshots dict.
    """
    import torch as _torch
    if _NN_SCHEMA == "v2":
        X, mask = _build_nn_features_v2(ticker, open_iso, close_iso,
                                         current_minute, btc_t0)
        if X is None or mask is None or mask.sum() == 0:
            return 0.5
    else:
        X, mask = _build_nn_features(window_snapshots, btc_t0, kalshi_t0, hour, current_minute)
        if mask.sum() == 0:
            return 0.5
    Xn = ((X - _NN_MEAN) / _NN_STD).astype(np.float32)
    with _torch.no_grad():
        logit = _NN_MODEL(_torch.from_numpy(Xn[None]), _torch.from_numpy(mask[None]))
        return float(_torch.sigmoid(logit).item())


def time_decay_mult(t_min: float) -> float:
    if t_min < 7:
        return 0.4
    if t_min < 10:
        return 0.8
    return 1.2


BTC_RETRY_ATTEMPTS   = 3
BTC_RETRY_DELAY_SECS = 5
MAX_PRICE_AGE_SECS   = 10

# ── 2D fair price table ───────────────────────────────────────────────────────
# Loaded from minute_analysis_2d.csv at startup.
# Key: (minute, bucket_index)  Value: (win_rate, avg_fill, n)
_FAIR_PRICE_2D: dict[tuple[int, int], tuple[float, float, int]] = {}

# FAIR_2D_PATH lets a shadow arm run a candidate table (e.g. the v2 refit)
# while other arms keep the incumbent.
_2D_CSV_PATH = env.get("FAIR_2D_PATH", "").strip() or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "logs", "minute_analysis_2d.csv"
)
_2D_BUCKETS = [
    (0.000, 0.05), (0.050, 0.10), (0.100, 0.20), (0.200, 0.50), (0.500, float("inf")),
]
_2D_BUCKET_LABELS = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]
_2D_MIN_N = 30   # fall back to strategy.FAIR_PRICE for cells with fewer samples

# 1D fallback (used only when a cell has n < _2D_MIN_N)
_FAIR_PRICE_BY_MINUTE_FALLBACK = {
    1: 0.582, 2: 0.617, 3: 0.636, 4: 0.670,
    5: 0.698, 6: 0.728, 7: 0.751, 8: 0.759,
    9: 0.783, 10: 0.798, 11: 0.806, 12: 0.815,
    13: 0.826,
}


def _load_2d_table() -> int:
    label_to_idx = {lbl: i for i, lbl in enumerate(_2D_BUCKET_LABELS)}
    count = 0
    with open(_2D_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            bi = label_to_idx.get(row["bucket"])
            if bi is None:
                continue
            _FAIR_PRICE_2D[(int(row["minute"]), bi)] = (
                float(row["win_rate"]),
                float(row["avg_fill"]),
                int(row["n"]),
            )
            count += 1
    return count


def _get_bucket_idx(abs_pct: float) -> int:
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= abs_pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


def get_fair_price_2d(minute: int, abs_pct_move: float) -> float:
    """2D empirical win rate for (minute, magnitude bucket). Falls back to 1D if cell is sparse."""
    bi    = _get_bucket_idx(abs_pct_move)
    entry = _FAIR_PRICE_2D.get((minute, bi))
    if entry is not None and entry[2] >= _2D_MIN_N:
        return entry[0]
    return _FAIR_PRICE_BY_MINUTE_FALLBACK.get(minute, strategy.FAIR_PRICE)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("trader")

# ── Shutdown flag ─────────────────────────────────────────────────────────────
_shutdown = False

def _handle_sigint(signum, frame):
    global _shutdown
    log.info("Shutdown signal received. Will exit after current window.")
    _shutdown = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── Session P&L tracker ───────────────────────────────────────────────────────
_cumulative_pnl: float = 0.0
_daily_pnl: dict = {}      # {UTC date: pnl} — used for DAILY_LOSS_CAP gate
_daily_pause: bool = False  # set True after cap hit; cleared at next UTC day


# ── Helpers ───────────────────────────────────────────────────────────────────

def window_boundary(dt: datetime) -> datetime:
    """Round a UTC datetime down to the nearest :00/:15/:30/:45 boundary."""
    boundary_minute = (dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES
    return dt.replace(minute=boundary_minute, second=0, microsecond=0)


def elapsed_in_window(dt: datetime) -> float:
    """Seconds elapsed since the most recent :00/:15/:30/:45 UTC boundary."""
    return (dt - window_boundary(dt)).total_seconds()


def get_btc_with_retry() -> float | None:
    for attempt in range(BTC_RETRY_ATTEMPTS):
        price = price_feed.get_price()
        age   = price_feed.get_price_age()
        if price is not None and age < MAX_PRICE_AGE_SECS:
            return price
        reason = "unavailable" if price is None else f"stale ({age:.1f}s old)"
        log.warning(f"BTC price {reason} (attempt {attempt+1}/{BTC_RETRY_ATTEMPTS}), waiting {BTC_RETRY_DELAY_SECS}s...")
        time.sleep(BTC_RETRY_DELAY_SECS)
    return None


def place_order_with_retry(ticker, side, market, stake,
                           max_cost: float | None = None) -> tuple[str | None, str | None, float]:
    """
    Execution policy (2026-07-02): IOC chase ladder with an edge-based price cap.

        IOC at base buffer -> partial accepted -> IOC chase remaining (+2c, +4c)
        -> stop when the limit price would erase the edge (max_cost)

    - Uses V2-native time_in_force=immediate_or_cancel: fills whatever is
      available at the limit instantly and cancels the remainder ATOMICALLY at
      the matching engine. (The 2026-05-20 stacked-order bug came from the old
      V1 expiration_ts=now+3 hack, where "IOC" orders actually rested for 3s and
      chase retries stacked on top of them. True IOC cannot stack: each rung
      only submits after the previous rung's response — with its fill count —
      has returned.)
    - Partial fills count as success: we track filled dollars and only chase
      the remaining stake.
    - max_cost: max acceptable cost per contract for OUR side (edge cap,
      typically p_est - fee - min_edge). Rungs whose limit would exceed it are
      not submitted — maximize FILLED GOOD orders, not raw submissions.

    Returns (order_id, error, actual_stake): actual_stake = dollars actually
    filled (0.0 if nothing filled), keeping exposure/caps consistent with
    reality.
    """
    CHASE_LADDER = (0, 2, 4)   # extra cents over FILL_BUFFER_CENTS per rung
    current_market = market
    filled_total = 0.0
    last_oid = None

    for i, extra in enumerate(CHASE_LADDER):
        remaining = stake - filled_total
        if remaining < 0.25:   # nothing meaningful left to chase
            break

        # Refresh the quote between rungs — the whole point is repricing.
        if i > 0:
            try:
                current_market = kalshi_trade.get_open_market() or current_market
            except Exception:
                pass

        # The limit this rung would submit (mirrors place_order's math).
        buf_c = kalshi_trade.FILL_BUFFER_CENTS + extra
        if side == "yes":
            limit_cost = round(float(current_market["yes_ask_dollars"]) * 100 + buf_c) / 100.0
        else:
            limit_cost = 1.0 - round(float(current_market["yes_bid_dollars"]) * 100 - buf_c) / 100.0

        if max_cost is not None and limit_cost > max_cost:
            log.info(f"  chase stop: rung +{extra}c limit {limit_cost:.2f} > edge cap {max_cost:.2f}")
            break
        if limit_cost >= 0.99:
            break
        if remaining < limit_cost:
            # Residue can't buy a whole contract — place_order would round UP
            # to 1 and overshoot the stake. Treat as fully filled.
            break

        try:
            resp  = kalshi_trade.place_order(PRIVATE_KEY, API_KEY_ID, ticker, side,
                                             current_market, remaining,
                                             extra_buffer_cents=extra, ioc=True)
            order = resp.get("order", {})
            last_oid = order.get("order_id", last_oid)
            fill_count = float(order.get("fill_count") or 0)
            avg = order.get("average_fill_price")
            if fill_count > 0 and avg is not None:
                avg = float(avg)
                per_contract = avg if side == "yes" else (1.0 - avg)
                filled_total += fill_count * per_contract
                log.info(f"  IOC rung +{extra}c: filled {fill_count:g} @ {per_contract:.2f} "
                         f"(cum ${filled_total:.2f} of ${stake:.2f})")
            else:
                log.info(f"  IOC rung +{extra}c: no fill (book moved past {limit_cost:.2f})")
        except Exception as e:
            log.error(f"  IOC rung +{extra}c failed: {e}")
            time.sleep(0.5)

    if filled_total >= 0.01:
        return last_oid, None, filled_total
    return None, "no fill after IOC chase ladder", 0.0


def wait_for_close(close_time_str: str) -> None:
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        remaining = (close_dt - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            log.info(f"Waiting {remaining:.0f}s for market to close at {close_dt.strftime('%H:%M:%S')} UTC...")
            deadline = time.time() + remaining + 3
            while time.time() < deadline and not _shutdown:
                time.sleep(min(5.0, deadline - time.time()))
    except Exception as e:
        log.warning(f"Could not parse close_time '{close_time_str}': {e}. Sleeping 600s.")
        time.sleep(600)


def poll_settlement(ticker: str, timeout_secs: int = 600) -> str | None:
    log.info(f"Polling settlement for {ticker}...")
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            result = kalshi_trade.get_market_result(ticker)
            if result is not None:
                return result
        except Exception as e:
            log.warning(f"Settlement poll error: {e}")
        time.sleep(10)
    log.warning(f"Settlement not confirmed within {timeout_secs}s.")
    return None


def compute_pnl(side: str, fill_price: float, count: float, winner: str | None) -> float:
    """Net P&L after Kalshi's 7% fee on winnings."""
    if winner is None:
        return 0.0
    if side == winner:
        return count * (1.0 - fill_price) * 0.93
    return -(count * fill_price)


def _append_csv(path: str, fields: list, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_bet(row: dict):
    _append_csv(TRADE_LOG_PATH, TRADE_LOG_FIELDS, row)


def log_window(row: dict):
    _regime_record(row.get("btc_t0"), row.get("btc_t5"), row.get("market_winner"))
    _append_csv(WINDOW_LOG_PATH, WINDOW_LOG_FIELDS, row)


# Seed the regime history from this arm's own window log so a restart doesn't
# reset the throttle to warm-up.
if REGIME_THROTTLE and os.path.exists(WINDOW_LOG_PATH):
    try:
        with open(WINDOW_LOG_PATH, newline="") as _rf:
            for _row in list(csv.DictReader(_rf))[-REGIME_LOOKBACK:]:
                _regime_record(_row.get("btc_t0"), _row.get("btc_t5"), _row.get("market_winner"))
    except Exception:
        pass


# ── DH loop ───────────────────────────────────────────────────────────────────

def run_dh_loop(
    window_ts: datetime,
    btc_t0: float,
    ticker: str,
    close_time: str,
    window_cap: float = 0.0,   # dollars; 0 = no cap
) -> tuple[list, list, float, float]:
    """
    Run delta hedging from T+4:00 to T+13:00 on a 30s cadence, placing bets on
    yes and/or no each tick.
    Returns (yes_bets, no_bets, yes_exposure, no_exposure)
    Each bet entry: (stake, fill_price, count, t_min).
    """
    yes_exposure = 0.0
    no_exposure  = 0.0
    yes_bets: list[tuple[float, float, float, float]] = []
    no_bets:  list[tuple[float, float, float, float]] = []
    # Track contracts owned per side (for RH hedge sizing).
    yes_contracts = 0.0
    no_contracts  = 0.0
    n_hedges      = 0

    # Daily loss cap gate — return immediately if paused for the day.
    if _daily_pause:
        log.info("DH loop skipped: daily loss cap active.")
        return yes_bets, no_bets, yes_exposure, no_exposure

    # NN-mode state: per-minute snapshots accumulated across the window.
    # window_snapshots[m] = {"btc": float, "kalshi_yes": float} where m is the
    # integer minute since window open. Used to build the feature tensor for
    # NN inference at each decision tick.
    window_snapshots: dict[int, dict] = {}
    kalshi_t0_local: float | None = None
    if FAIR_PRICE_SOURCE == "nn":
        # Prefill T+0..T+3 by fetching historical Kalshi candles + Coinbase BTC
        # so the NN sees the full history at the first decision tick (T+10).
        try:
            # kalshi_client and btc_data live in the parent directory, not live/
            _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            import kalshi_client as _kc
            if ASSET == "ETH":
                import eth_data as _bd
            else:
                import btc_data as _bd
            open_iso  = window_ts.isoformat()
            close_iso = (window_ts + timedelta(minutes=WINDOW_MINUTES)).isoformat()
            _candles = _kc.fetch_candlesticks(ticker, open_iso, close_iso)
            t0_unix = int(window_ts.timestamp())
            _btc_prices = _bd.fetch_btc_prices(t0_unix - 60, t0_unix + 4 * 60)
            for _m in range(0, NN_MIN_MINUTE):
                _ts = t0_unix + _m * 60
                _btc = _bd.lookup(_btc_prices, _ts)
                _ky  = _kc.get_yes_price_at(_candles, _ts) if _candles else None
                if _m == 0:
                    if _btc is None: _btc = btc_t0   # always have btc_t0 from caller
                    if _ky is not None:
                        kalshi_t0_local = _ky
                if _btc is not None and _ky is not None:
                    window_snapshots[_m] = {"btc": _btc, "kalshi_yes": _ky}
            log.info(f"NN prefill: {len(window_snapshots)} pre-decision snapshots loaded")
        except Exception as _e:
            log.warning(f"NN prefill failed: {_e}; will use partial features")
    # Track WHICH contracts have already been hedged. Without this, the RH
    # block re-fires every tick where conditions hold (direction still
    # opposite the wrong-side exposure that hasn't changed). Tracking the
    # already-hedged amount means RH only fires when wrong-side exposure
    # GROWS beyond what's already covered.
    no_contracts_hedged  = 0.0   # how many NO contracts are already covered by YES hedges
    yes_contracts_hedged = 0.0   # how many YES contracts are already covered by NO hedges

    # Real-time guards: refuse to place orders if our view of "now" diverges from
    # the wall clock. The Kalshi WS can go stale for minutes during network/CPU
    # hiccups, and without these guards the trader sends orders for markets that
    # already settled → Kalshi 404 "market_not_found". See WS staleness incidents.
    WINDOW_END_GRACE_SECS = 30   # stop iterating once we're within this much of window close
    WS_MAX_STALENESS_SECS = 10   # refuse to place orders if Kalshi WS is more stale than this
                                  # (tightened 30→10 on 2026-06-02; ping_interval is 10s so
                                  # legit drift should stay well under this)
    BTC_MAX_STALENESS_SECS = 5   # refuse to place orders if crypto feed is more stale than this
                                  # (tightened 15→5 on 2026-06-02; with 10s pings the feed
                                  # should never be older than 1-2s during normal operation)

    for offset_secs in DH_OFFSETS_SECS:
        t_min      = offset_secs / 60.0           # fractional minute, for logging / dh_minute column
        minute_idx = int(offset_secs // 60)       # integer minute, for 2D fair-price lookup
        target_dt  = window_ts + timedelta(seconds=offset_secs)
        remaining  = (target_dt - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            deadline = time.time() + remaining
            while time.time() < deadline and not _shutdown:
                time.sleep(min(1.0, deadline - time.time()))

        if _shutdown:
            break

        # GUARD 1: if the *real* clock has moved past (window_close - grace), abort
        # the rest of this window's decisions. Otherwise stale WS data tricks the
        # trader into sending orders for an already-settled market.
        elapsed_real = (datetime.now(timezone.utc) - window_ts).total_seconds()
        seconds_left = WINDOW_MINUTES * 60 - elapsed_real
        if seconds_left < WINDOW_END_GRACE_SECS:
            log.warning(
                f"Window almost closed (real elapsed {elapsed_real:.0f}s, "
                f"{seconds_left:.0f}s left) — aborting remaining decisions for {ticker}"
            )
            break

        if DH_MIN_MINUTE and t_min < DH_MIN_MINUTE:
            continue

        btc_now = get_btc_with_retry()
        if btc_now is None:
            log.warning(f"BTC unavailable at T+{t_min:.1f}, skipping interval.")
            continue
        btc_age = price_feed.get_price_age()

        # GUARD 2: BTC/ETH price too stale to trust for decisions.
        if btc_age > BTC_MAX_STALENESS_SECS:
            log.warning(
                f"BTC feed stale ({btc_age:.1f}s > {BTC_MAX_STALENESS_SECS}s) at T+{t_min:.1f}, "
                f"skipping interval."
            )
            continue

        # Use WebSocket prices (real-time) if fresh; fall back to REST on stale feed.
        ws_bid = kalshi_feed.get_bid()
        ws_ask = kalshi_feed.get_ask()
        ws_age = kalshi_feed.get_age()
        if ws_bid is not None and ws_ask is not None and ws_age < 10:
            yes_bid = ws_bid
            yes_ask = ws_ask
            # Synthesize a market dict for place_order (uses yes_bid_dollars / yes_ask_dollars).
            market = {
                "ticker":          ticker,
                "yes_bid_dollars": yes_bid,
                "yes_ask_dollars": yes_ask,
            }
        else:
            # GUARD 3: WS too stale → REST fallback only if WS isn't catastrophically behind.
            # If WS is more than WS_MAX_STALENESS_SECS old, the trader's whole timing
            # model is suspect (we may be acting on outdated window state).
            if ws_age > WS_MAX_STALENESS_SECS:
                log.warning(
                    f"Kalshi WS critically stale ({ws_age:.1f}s > {WS_MAX_STALENESS_SECS}s) "
                    f"at T+{t_min:.1f}. Skipping interval — won't risk 404 on settled market."
                )
                continue
            log.warning(f"Kalshi WS stale ({ws_age:.1f}s) at T+{t_min:.1f}, falling back to REST.")
            try:
                market = kalshi_trade.get_open_market()
            except Exception as e:
                log.warning(f"REST fallback failed at T+{t_min:.1f}: {e}. Skipping interval.")
                continue
            if market is None:
                log.warning(f"No open market at T+{t_min:.1f}. Skipping interval.")
                continue
            # GUARD 4: REST market ticker must match the window's ticker. If they
            # differ, the previous window has already settled and Kalshi handed
            # us the NEW window's market — we'd 404 on the old ticker.
            if market.get("ticker") != ticker:
                log.warning(
                    f"REST returned different ticker ({market.get('ticker')} vs window "
                    f"{ticker}) — previous window settled. Aborting remaining decisions."
                )
                break
            yes_bid = float(market["yes_bid_dollars"])
            yes_ask = float(market["yes_ask_dollars"])

        spread     = round(yes_ask - yes_bid, 4)
        kalshi_mid = (yes_bid + yes_ask) / 2

        abs_pct_move = abs(btc_now - btc_t0) / btc_t0 * 100
        f_btc        = strategy.sigmoid_btc(abs_pct_move)
        direction_up = btc_now > btc_t0

        # NN mode: skip ticks outside the configured minute range; capture the
        # per-minute snapshot so subsequent ticks have full history.
        if FAIR_PRICE_SOURCE == "nn":
            if minute_idx not in window_snapshots:
                # First time we see this integer minute — store snapshot.
                window_snapshots[minute_idx] = {"btc": btc_now, "kalshi_yes": kalshi_mid}
                if minute_idx == 0 and kalshi_t0_local is None:
                    kalshi_t0_local = kalshi_mid
            if minute_idx < NN_MIN_MINUTE or minute_idx > NN_MAX_MINUTE:
                log.info(f"  -> NN: T+{t_min:.1f} outside [{NN_MIN_MINUTE},{NN_MAX_MINUTE}], skip decision")
                continue
            kt0 = kalshi_t0_local if kalshi_t0_local is not None else kalshi_mid
            # v2 schema needs ticker/open_iso/close_iso for candle fetch
            _open_iso  = window_ts.isoformat()
            _close_iso = (window_ts + timedelta(minutes=WINDOW_MINUTES)).isoformat()
            p_yes = _nn_p_yes(window_snapshots, btc_t0, kt0, window_ts.hour, minute_idx,
                              ticker=ticker, open_iso=_open_iso, close_iso=_close_iso)
            # fair = P(directional side wins)
            fair = p_yes if direction_up else (1.0 - p_yes)
        else:
            fair = get_fair_price_2d(minute_idx, abs_pct_move)

        buf = kalshi_trade.FILL_BUFFER_CENTS / 100
        td_mult = time_decay_mult(t_min) if TIME_DECAY else 1.0
        if direction_up:
            mispricing = fair - (yes_ask + buf)   # true edge after buffer cost
            g_misprice = strategy.sigmoid_mispricing(mispricing)
            target_yes = BASE_STAKE * f_btc * g_misprice * td_mult
            target_no  = 0.0
        else:
            mispricing = fair - ((1.0 - yes_bid) + buf)  # P(direction correct) - no_fill cost
            g_misprice = strategy.sigmoid_mispricing(mispricing)
            target_no  = BASE_STAKE * f_btc * g_misprice * td_mult
            target_yes = 0.0

        if SIZING == "s6kelly":
            # Calibrated-probability + fractional-Kelly sizing (overrides the
            # sigmoid targets above; f_btc/g_misprice still logged for compare).
            # cost = what the continuation contract actually costs incl. buffer.
            cost  = (yes_ask + buf) if direction_up else ((1.0 - yes_bid) + buf)
            p_win = s6_p_win(fair, abs_pct_move, minute_idx, direction_up)
            # Final-2-min Asian override: settlement is the 60s average, and
            # once prints start banking, the running partial average is
            # observable state the table/logistic cannot see. The Asian
            # estimator IS a probability, so it bypasses the logistic.
            if ASIAN_PRICER and seconds_left <= 120:
                _close_epoch = (window_ts + timedelta(minutes=WINDOW_MINUTES)).timestamp()
                _p_up_asian = asian_pricer.p_up(btc_t0, _close_epoch)
                if _p_up_asian is not None:
                    _p_asian = _p_up_asian if direction_up else 1.0 - _p_up_asian
                    _ast = asian_pricer.state(_close_epoch)
                    log.info(f"  asian: p_table={p_win:.3f} -> p_asian={_p_asian:.3f} "
                             f"(banked={_ast['banked_frac']:.2f} n={_ast['n_banked']})")
                    p_win = _p_asian
            fee   = 0.07 * cost * (1.0 - cost)          # real Kalshi fee/contract (confirmed vs live fill)
            s6_edge = p_win - cost - fee                # edge NET of fee
            if s6_edge > S6_EDGE_MIN and cost < 0.99:
                kelly  = s6_edge / (1.0 - cost)          # full-Kelly fraction
                target = min(S6_MAX_STAKE, S6_KELLY_FRAC * kelly * S6_BANKROLL)
            else:
                target = 0.0
            target_yes = target if direction_up else 0.0
            target_no  = 0.0 if direction_up else target
            log.info(f"  s6kelly: p_win={p_win:.3f} cost={cost:.3f} fee={fee:.3f} edge={s6_edge:+.3f} target=${target:.2f}")

        # Edge-based execution cap for the IOC chase ladder: never pay more per
        # contract than the price at which the net edge (after fee) would drop
        # below the minimum. "Maximize filled good orders, not raw submissions."
        _p_est = p_win if SIZING == "s6kelly" else fair
        _min_edge_req = S6_EDGE_MIN if SIZING == "s6kelly" else (MIN_EDGE_CENTS / 100.0)
        max_cost_cap = _p_est - 0.07 * _p_est * (1.0 - _p_est) - _min_edge_req

        # Edge filter: skip the bet if expected post-buffer edge is below
        # threshold. Real fees (~1.75c/contract on 50c markets) make sub-1c
        # edges negative-EV. Sim shows +5-8pp ROI from this filter.
        # (s6kelly has its own calibrated edge gate above — skip this one.)
        edge_cents = mispricing * 100
        if edge_cents < MIN_EDGE_CENTS and SIZING != "s6kelly":
            target_yes = 0.0
            target_no  = 0.0

        # Side filter — drop disallowed direction's target.
        if SIDE_FILTER == "yes_only":
            target_no  = 0.0
        elif SIDE_FILTER == "no_only":
            target_yes = 0.0

        # Regime throttle: recent windows show chop -> stand down entirely.
        _r_ok, _r_rate = regime_ok()
        if not _r_ok and (target_yes > 0 or target_no > 0):
            log.info(f"  regime throttle: trailing cont={_r_rate:.2f} < {REGIME_MIN_CONT:.2f} — standing down")
            target_yes = 0.0
            target_no  = 0.0

        if MODE == "dh-target":
            bet_yes = max(0.0, target_yes - yes_exposure)
            bet_no  = max(0.0, target_no  - no_exposure)
        else:  # dh-additive
            bet_yes = target_yes
            bet_no  = target_no

        # Per-window wagered cap: truncate so cumulative wagered (yes + no
        # exposure) never exceeds window_cap. Once at cap, both bets shrink
        # to 0 and fall below MIN_BET below.
        if window_cap > 0:
            remaining = max(0.0, window_cap - (yes_exposure + no_exposure))
            if bet_yes + bet_no > remaining:
                if remaining <= 0:
                    bet_yes = bet_no = 0.0
                else:
                    total_req = bet_yes + bet_no
                    bet_yes = remaining * (bet_yes / total_req)
                    bet_no  = remaining * (bet_no  / total_req)
                log.info(
                    f"  -> CAP: window wagered ${yes_exposure + no_exposure:.2f}/"
                    f"${window_cap:.2f} → bet truncated to "
                    f"yes=${bet_yes:.2f} no=${bet_no:.2f}"
                )

        direction_label = "yes" if direction_up else "no"

        log.info(
            f"DH T+{t_min:.1f}: {direction_label.upper()} | "
            f"cutoff=${btc_t0:,.2f} now=${btc_now:,.2f} ({'+' if direction_up else '-'}{abs_pct_move:.4f}%) | "
            f"bid={yes_bid:.3f}/ask={yes_ask:.3f} mid={kalshi_mid:.3f} | "
            f"fair={fair:.3f} mis={mispricing:+.3f} | "
            f"f={f_btc:.3f} g={g_misprice:.3f} | "
            f"gap_yes=${bet_yes:.2f} gap_no=${bet_no:.2f}"
        )

        base_row = {
            "window_ts":          window_ts.isoformat(),
            "mode":               MODE,
            "ticker":             ticker,
            "close_time":         close_time,
            "dh_minute":          round(t_min, 2),
            "btc_t0":             round(btc_t0, 2),
            "btc_now":            round(btc_now, 2),
            "btc_price_age_secs": round(btc_age, 2),
            "abs_pct_move":       round(abs_pct_move, 6),
            "yes_bid":            round(yes_bid, 4),
            "yes_ask":            round(yes_ask, 4),
            "spread":             spread,
            "kalshi_yes_mid":     round(kalshi_mid, 4),
            "direction":          direction_label,
            "yes_target":         round(target_yes, 4),
            "no_target":          round(target_no, 4),
            "mispricing":         round(mispricing, 6),
            "f_btc":              round(f_btc, 6),
            "g_misprice":         round(g_misprice, 6),
        }

        # Per-window leg cap: skip remaining entry/hedge legs once at limit.
        if MAX_LEGS_PER_WINDOW > 0 and (len(yes_bets) + len(no_bets)) >= MAX_LEGS_PER_WINDOW:
            log.info(f"  -> SKIP: leg cap {MAX_LEGS_PER_WINDOW} reached "
                     f"({len(yes_bets)}Y/{len(no_bets)}N)")
            continue

        # INVERT mode: swap bet sizes so we trade against the NN signal.
        if INVERT and (bet_yes > 0 or bet_no > 0):
            bet_yes, bet_no = bet_no, bet_yes
            log.info(f"  -> INVERT: swapped to yes=${bet_yes:.2f} no=${bet_no:.2f}")

        # MIN_BET_PRICE filter: refuse to fill below the configured price floor
        # (computed against the would-be fill price for each side).
        if MIN_BET_PRICE > 0:
            yes_fill_chk = min(yes_ask + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            no_fill_chk  = min((1.0 - yes_bid) + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            if bet_yes > 0 and yes_fill_chk < MIN_BET_PRICE:
                log.info(f"  -> SKIP YES: fill {yes_fill_chk:.3f} < MIN_BET_PRICE {MIN_BET_PRICE:.3f}")
                bet_yes = 0.0
            if bet_no > 0 and no_fill_chk < MIN_BET_PRICE:
                log.info(f"  -> SKIP NO:  fill {no_fill_chk:.3f} < MIN_BET_PRICE {MIN_BET_PRICE:.3f}")
                bet_no = 0.0

        if bet_yes >= MIN_BET:
            fill  = min(yes_ask + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            if fill > MAX_FILL_PRICE:
                upside_per = max(0.001, 1.0 - fill)
                ratio      = fill / upside_per
                log.info(
                    f"  -> SKIP YES ${bet_yes:.2f}: fill {fill:.3f} > cap {MAX_FILL_PRICE} "
                    f"| reason: risk {fill*100:.1f}c to win {upside_per*100:.1f}c per contract "
                    f"(R:R {ratio:.0f}:1 against)"
                )
                # NOTE: was `continue` — fall through so NO block + RH still evaluate
            else:
                count = max(1, round(bet_yes / fill))
                log.info(f"  -> BET YES ${bet_yes:.2f} @ {fill:.3f} ({fill*100:.1f}c/contract) | {count} contracts")
                actual_stake = bet_yes
                if PAPER_MODE:
                    order_id, order_result = "paper", "paper"
                    log.info("     [PAPER] no order submitted")
                else:
                    order_id, err, actual_stake = place_order_with_retry(ticker, "yes", market, bet_yes, max_cost=max_cost_cap)
                    order_result  = "ok" if order_id else f"error: {err}"
                    if order_id:
                        log.info(f"     YES order placed: {order_id}  (filled ${actual_stake:.2f})")
                    else:
                        log.error(f"     YES order FAILED: {err}")
                if order_id:   # only count bet toward exposure if it actually placed
                    logged_count = max(1, round(actual_stake / fill))
                    yes_bets.append((actual_stake, fill, logged_count, t_min))
                    yes_exposure += actual_stake
                    yes_contracts += actual_stake / fill
                log_bet({**base_row,
                    "yes_exposure_before": round(yes_exposure - (actual_stake if order_id else 0), 4),
                    "no_exposure_before":  round(no_exposure, 4),
                    "bet_side":            "yes",
                    "stake":               round(actual_stake if order_id else bet_yes, 4),
                    "fill_price":          round(fill, 4),
                    "count":               max(1, round(actual_stake / fill)) if order_id else count,
                    "order_id":            order_id or "none",
                    "order_result":        order_result,
                })

        if bet_no >= MIN_BET:
            fill  = min((1.0 - yes_bid) + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            if fill > MAX_FILL_PRICE:
                upside_per = max(0.001, 1.0 - fill)
                ratio      = fill / upside_per
                log.info(
                    f"  -> SKIP NO  ${bet_no:.2f}: fill {fill:.3f} > cap {MAX_FILL_PRICE} "
                    f"| reason: risk {fill*100:.1f}c to win {upside_per*100:.1f}c per contract "
                    f"(R:R {ratio:.0f}:1 against)"
                )
                # NOTE: was `continue` — fall through so RH still evaluates
            else:
                count = max(1, round(bet_no / fill))
                log.info(f"  -> BET NO  ${bet_no:.2f} @ {fill:.3f} ({fill*100:.1f}c/contract) | {count} contracts")
                actual_stake = bet_no
                if PAPER_MODE:
                    order_id, order_result = "paper", "paper"
                    log.info("     [PAPER] no order submitted")
                else:
                    order_id, err, actual_stake = place_order_with_retry(ticker, "no", market, bet_no, max_cost=max_cost_cap)
                    order_result  = "ok" if order_id else f"error: {err}"
                    if order_id:
                        log.info(f"     NO order placed: {order_id}  (filled ${actual_stake:.2f})")
                    else:
                        log.error(f"     NO order FAILED: {err}")
                if order_id:   # only count bet toward exposure if it actually placed
                    logged_count = max(1, round(actual_stake / fill))
                    no_bets.append((actual_stake, fill, logged_count, t_min))
                    no_exposure += actual_stake
                    no_contracts += actual_stake / fill
                log_bet({**base_row,
                    "yes_exposure_before": round(yes_exposure, 4),
                    "no_exposure_before":  round(no_exposure - (actual_stake if order_id else 0), 4),
                    "bet_side":            "no",
                    "stake":               round(actual_stake if order_id else bet_no, 4),
                    "fill_price":          round(fill, 4),
                    "count":               max(1, round(actual_stake / fill)) if order_id else count,
                    "order_id":            order_id or "none",
                    "order_result":        order_result,
                })

        # ── Reversal-Hedge overlay ──────────────────────────────────────────
        # At every minute >= RH_MINUTE, if BTC direction is opposite the side
        # we hold meaningful exposure on, buy enough of the now-correct side
        # to net out the contracts owned on the losing side.
        if RH_MINUTE is None or t_min < RH_MINUTE:
            continue

        hedge_side = None
        # NET-EXPOSURE hedging: hedge only to bring the YES-vs-NO contract
        # balance to ~zero (not to "cover" all wrong-side contracts).
        # Why this matters: under the old "cover all" logic, every direction
        # flip in a choppy window fired another hedge — each leg paid spread,
        # cascading into compounding losses. Under net-exposure logic, once
        # yes_contracts ≈ no_contracts, the position is already neutral and
        # no further hedge fires regardless of how BTC oscillates.
        #
        # RH_TRIGGER here is interpreted as minimum NET imbalance in contracts
        # (was dollars under the old model; close enough at small stake).
        net_yes = yes_contracts - no_contracts   # +ve = net YES, -ve = net NO

        if direction_up and net_yes < -RH_TRIGGER:
            # Net NO exposure with BTC above floor — buy YES to neutralize.
            hedge_side       = "yes"
            hedge_fill       = min(yes_ask + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            cover_contracts  = -net_yes   # contracts of NO not yet offset by YES
            hedge_stake_req  = cover_contracts * hedge_fill
        elif (not direction_up) and net_yes > RH_TRIGGER:
            # Net YES exposure with BTC below floor — buy NO to neutralize.
            hedge_side       = "no"
            hedge_fill       = min((1.0 - yes_bid) + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            cover_contracts  = net_yes
            hedge_stake_req  = cover_contracts * hedge_fill
        else:
            continue

        # Cap to window_cap if configured
        if window_cap > 0:
            remaining = max(0.0, window_cap - (yes_exposure + no_exposure))
            if hedge_stake_req > remaining:
                log.info(
                    f"  -> RH-{hedge_side.upper()} truncated by window cap: "
                    f"${hedge_stake_req:.2f} → ${remaining:.2f}"
                )
                hedge_stake_req = remaining

        if hedge_stake_req < MIN_BET:
            continue
        if hedge_fill > MAX_HEDGE_FILL_PRICE:
            log.info(
                f"  -> SKIP RH-{hedge_side.upper()} ${hedge_stake_req:.2f}: "
                f"fill {hedge_fill:.3f} > hedge cap {MAX_HEDGE_FILL_PRICE}"
            )
            continue

        hedge_count = max(1, round(hedge_stake_req / hedge_fill))
        log.info(
            f"  -> RH-HEDGE {hedge_side.upper()} ${hedge_stake_req:.2f} @ {hedge_fill:.3f} "
            f"({hedge_count} contracts) | covers {cover_contracts:.1f} losing-side contracts"
        )

        actual_stake = hedge_stake_req
        if PAPER_MODE:
            order_id, order_result = "paper", "paper"
            log.info("     [PAPER] no hedge order submitted")
        else:
            order_id, err, actual_stake = place_order_with_retry(
                ticker, hedge_side, market, hedge_stake_req
            )
            order_result = "ok" if order_id else f"error: {err}"
            if order_id:
                log.info(f"     RH-{hedge_side.upper()} order placed: {order_id}  (filled ${actual_stake:.2f})")
            else:
                log.error(f"     RH-{hedge_side.upper()} order FAILED: {err}")

        if order_id:
            logged_count = max(1, round(actual_stake / hedge_fill))
            if hedge_side == "yes":
                yes_bets.append((actual_stake, hedge_fill, logged_count, t_min))
                yes_exposure  += actual_stake
                yes_contracts += actual_stake / hedge_fill
                # Mark NO contracts as hedged so we don't re-fire next tick.
                no_contracts_hedged = no_contracts
            else:
                no_bets.append((actual_stake, hedge_fill, logged_count, t_min))
                no_exposure   += actual_stake
                no_contracts  += actual_stake / hedge_fill
                yes_contracts_hedged = yes_contracts
            n_hedges += 1

        log_bet({**base_row,
            "yes_exposure_before": round(yes_exposure - (actual_stake if (order_id and hedge_side == "yes") else 0), 4),
            "no_exposure_before":  round(no_exposure  - (actual_stake if (order_id and hedge_side == "no")  else 0), 4),
            "bet_side":            f"{hedge_side}-hedge",
            "stake":               round(actual_stake if order_id else hedge_stake_req, 4),
            "fill_price":          round(hedge_fill, 4),
            "count":               max(1, round(actual_stake / hedge_fill)) if order_id else hedge_count,
            "order_id":            order_id or "none",
            "order_result":        order_result,
        })

    return yes_bets, no_bets, yes_exposure, no_exposure


# ── Window execution ──────────────────────────────────────────────────────────

def run_window():
    global _cumulative_pnl, _daily_pause

    now        = datetime.now(timezone.utc)
    window_ts  = window_boundary(now)

    # ── Daily loss cap gate ──
    today = window_ts.date()
    today_pnl = _daily_pnl.get(today, 0.0)
    if DAILY_LOSS_CAP > 0 and today_pnl <= -DAILY_LOSS_CAP:
        if not _daily_pause:
            log.error(f"DAILY LOSS CAP HIT: today P&L=${today_pnl:+.2f} <= -${DAILY_LOSS_CAP:.2f}. "
                      f"Pausing all new bets until next UTC day.")
            _daily_pause = True
        # Still wait out the window to log it but don't bet
    else:
        if _daily_pause:
            # New day with no cap hit yet — clear pause
            log.info(f"Daily pause cleared (new UTC day {today}, prior was {(today - timedelta(days=1))})")
            _daily_pause = False

    elapsed    = (now - window_ts).total_seconds()
    sleep_secs = max(0.0, DECISION_OFFSET_SECS - elapsed)

    entry_min = DECISION_OFFSET_SECS // 60
    log.info(f"Window T+0: {window_ts.strftime('%H:%M:%S')} UTC | T+{entry_min} in {sleep_secs:.0f}s")

    deadline = time.time() + sleep_secs
    while time.time() < deadline and not _shutdown:
        time.sleep(min(1.0, deadline - time.time()))
    if _shutdown:
        return

    # ── Fetch Kalshi market (floor_strike = btc_t0) ───────────────────────────
    try:
        market = kalshi_trade.get_open_market()
    except Exception as e:
        log.error(f"Failed to fetch open market: {e}")
        return
    if market is None:
        log.error("No open KXBTC15M market found. Skipping window.")
        return

    ticker     = market["ticker"]
    close_time = market.get("close_time", "")
    btc_t0     = float(market["floor_strike"])

    # Subscribe WebSocket to this window's ticker for real-time bid/ask.
    kalshi_feed.set_ticker(ticker)

    yes_bid    = float(market["yes_bid_dollars"])
    yes_ask    = float(market["yes_ask_dollars"])
    spread     = round(yes_ask - yes_bid, 4)
    kalshi_mid = (yes_bid + yes_ask) / 2

    btc_entry = get_btc_with_retry()
    if btc_entry is None:
        log.error(f"BTC price unavailable at T+{entry_min}. Skipping window.")
        return
    btc_age_entry = price_feed.get_price_age()

    log.info(
        f"cutoff=${btc_t0:,.2f} | BTC T+{entry_min}=${btc_entry:,.2f} (age={btc_age_entry:.1f}s) | "
        f"Kalshi bid/ask={yes_bid:.3f}/{yes_ask:.3f} spread={spread:.3f} | ticker={ticker}"
    )

    # ── t+5 mode: single bet ─────────────────────────────────────────────────
    if MODE == "t+5":
        side = "yes" if btc_entry > btc_t0 else "no"
        stake, abs_pct_move, mispricing, f_btc, g_misprice = strategy.compute_stake(
            btc_t0, btc_entry, kalshi_mid, side, BASE_STAKE
        )
        buf        = kalshi_trade.FILL_BUFFER_CENTS / 100
        fill_price = (yes_ask + buf) if side == "yes" else ((1.0 - yes_bid) + buf)
        count      = max(1, round(stake / fill_price))

        log.info(
            f"Decision: {side.upper()} | BTC {'+' if btc_entry>btc_t0 else ''}{abs_pct_move:.4f}% | "
            f"mis={mispricing:+.4f} f={f_btc:.4f} g={g_misprice:.4f} | "
            f"stake=${stake:.2f} @ {fill_price:.3f} = {count} contracts"
        )

        if PAPER_MODE:
            order_id, order_result = "paper", "paper"
            log.info("[PAPER] Order not placed.")
        else:
            order_id, err, _actual = place_order_with_retry(ticker, side, market, stake)
            order_result  = "ok" if order_id else f"error: {err}"
            if order_id:
                log.info(f"Order placed: {order_id}")
            else:
                log.error(f"Order failed: {err}")

        wait_for_close(close_time)
        winner    = poll_settlement(ticker)
        settle_ts = datetime.now(timezone.utc)
        pnl       = compute_pnl(side, fill_price, count, winner)
        outcome   = ("win" if side == winner else "loss") if winner else "unknown"
        _cumulative_pnl += pnl

        log.info(
            f"RESULT: {outcome.upper()} | market={winner or '?'} bet={side} | "
            f"pnl={'+' if pnl>=0 else ''}${pnl:.2f} | "
            f"session={'+' if _cumulative_pnl>=0 else ''}${_cumulative_pnl:.2f}"
        )

        log_bet({
            "window_ts":          window_ts.isoformat(),
            "mode":               MODE,
            "ticker":             ticker,
            "close_time":         close_time,
            "dh_minute":          5,
            "btc_t0":             round(btc_t0, 2),
            "btc_now":            round(btc_entry, 2),
            "btc_price_age_secs": round(btc_age_entry, 2),
            "abs_pct_move":       round(abs_pct_move, 6),
            "yes_bid":            round(yes_bid, 4),
            "yes_ask":            round(yes_ask, 4),
            "spread":             spread,
            "kalshi_yes_mid":     round(kalshi_mid, 4),
            "direction":          side,
            "yes_target":         round(stake, 4) if side == "yes" else 0,
            "no_target":          round(stake, 4) if side == "no" else 0,
            "yes_exposure_before": 0,
            "no_exposure_before":  0,
            "bet_side":           side,
            "mispricing":         round(mispricing, 6),
            "f_btc":              round(f_btc, 6),
            "g_misprice":         round(g_misprice, 6),
            "stake":              round(stake, 4),
            "fill_price":         round(fill_price, 4),
            "count":              count,
            "order_id":           order_id or "none",
            "order_result":       order_result,
        })
        log_window({
            "window_ts":       window_ts.isoformat(),
            "mode":            MODE,
            "ticker":          ticker,
            "close_time":      close_time,
            "btc_t0":          round(btc_t0, 2),
            "btc_t5":          round(btc_entry, 2),
            "btc_t10":         "",
            "n_yes_bets":      1 if side == "yes" else 0,
            "n_no_bets":       1 if side == "no" else 0,
            "total_bets":      1,
            "total_yes_stake": round(stake, 4) if side == "yes" else 0,
            "total_no_stake":  round(stake, 4) if side == "no" else 0,
            "total_wagered":   round(stake, 4),
            "settlement_ts":   settle_ts.isoformat(),
            "market_winner":   winner or "unknown",
            "yes_pnl":         round(pnl, 4) if side == "yes" else 0,
            "no_pnl":          round(pnl, 4) if side == "no" else 0,
            "total_pnl":       round(pnl, 4),
            "outcome":         outcome,
            "cumulative_pnl":  round(_cumulative_pnl, 4),
        })
        try:
            nxt = kalshi_trade.get_open_market()
            if nxt and nxt["ticker"] != ticker:
                kalshi_feed.set_ticker(nxt["ticker"])
                log.info(f"Pre-subscribed to next window: {nxt['ticker']}")
        except Exception:
            pass
        return

    # ── dh-target / dh-additive mode: DH loop ────────────────────────────────
    # Determine the per-window wagered cap. If CAP_FRACTION_OF_BALANCE > 0,
    # cap = balance * fraction (recomputed every window). Falls back to the
    # static MAX_WINDOW_WAGERED if balance lookup fails.
    window_cap = MAX_WINDOW_WAGERED
    if CAP_FRACTION_OF_BALANCE > 0 and not PAPER_MODE:
        bal = kalshi_trade.get_balance(PRIVATE_KEY, API_KEY_ID)
        if bal is not None:
            window_cap = bal * CAP_FRACTION_OF_BALANCE
            log.info(f"Window cap: ${window_cap:.2f} (balance ${bal:.2f} × {CAP_FRACTION_OF_BALANCE:.2f})")
        else:
            log.warning(f"Balance fetch failed, falling back to static cap ${window_cap:.2f}")
    elif window_cap > 0:
        log.info(f"Window cap (static): ${window_cap:.2f}")

    yes_bets, no_bets, yes_exp, no_exp = run_dh_loop(window_ts, btc_t0, ticker, close_time, window_cap)

    btc_t10 = get_btc_with_retry()

    wait_for_close(close_time)
    winner    = poll_settlement(ticker)
    settle_ts = datetime.now(timezone.utc)

    yes_pnl_total = sum(compute_pnl("yes", fp, cnt, winner) for _, fp, cnt, _ in yes_bets)
    no_pnl_total  = sum(compute_pnl("no",  fp, cnt, winner) for _, fp, cnt, _ in no_bets)
    total_pnl     = yes_pnl_total + no_pnl_total
    total_wagered = yes_exp + no_exp
    _cumulative_pnl += total_pnl
    _today = window_ts.date()
    _daily_pnl[_today] = _daily_pnl.get(_today, 0.0) + total_pnl
    outcome = "net_win" if total_pnl >= 0 else "net_loss"

    log.info(
        f"RESULT: {outcome.upper()} | market={winner or '?'} | "
        f"yes_pnl={'+' if yes_pnl_total>=0 else ''}${yes_pnl_total:.2f} "
        f"no_pnl={'+' if no_pnl_total>=0 else ''}${no_pnl_total:.2f} | "
        f"total={'+' if total_pnl>=0 else ''}${total_pnl:.2f} | "
        f"session={'+' if _cumulative_pnl>=0 else ''}${_cumulative_pnl:.2f} | "
        f"bets={len(yes_bets)}Y+{len(no_bets)}N wagered=${total_wagered:.2f}"
    )

    log_window({
        "window_ts":       window_ts.isoformat(),
        "mode":            MODE,
        "ticker":          ticker,
        "close_time":      close_time,
        "btc_t0":          round(btc_t0, 2),
        "btc_t5":          round(btc_entry, 2),
        "btc_t10":         round(btc_t10, 2) if btc_t10 else "",
        "n_yes_bets":      len(yes_bets),
        "n_no_bets":       len(no_bets),
        "total_bets":      len(yes_bets) + len(no_bets),
        "total_yes_stake": round(yes_exp, 4),
        "total_no_stake":  round(no_exp, 4),
        "total_wagered":   round(total_wagered, 4),
        "settlement_ts":   settle_ts.isoformat(),
        "market_winner":   winner or "unknown",
        "yes_pnl":         round(yes_pnl_total, 4),
        "no_pnl":          round(no_pnl_total, 4),
        "total_pnl":       round(total_pnl, 4),
        "outcome":         outcome,
        "cumulative_pnl":  round(_cumulative_pnl, 4),
    })

    # Pre-subscribe WebSocket to next window's ticker so it's ready at T+4.
    # The next market opens at T+15; we have ~4 minutes before T+4 of next window.
    try:
        nxt = kalshi_trade.get_open_market()
        if nxt and nxt["ticker"] != ticker:
            kalshi_feed.set_ticker(nxt["ticker"])
            log.info(f"Pre-subscribed to next window: {nxt['ticker']}")
    except Exception:
        pass


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    mode_label = "PAPER" if PAPER_MODE else "LIVE"
    log.info(f"BTC-Kalshi Trader starting | mode={mode_label} | strategy={MODE} | base_stake=${BASE_STAKE:.2f}")
    if RH_MINUTE is not None:
        log.info(f"Reversal-hedge ENABLED | rh_minute=T+{RH_MINUTE} | rh_trigger=${RH_TRIGGER:.2f}")
    else:
        log.info("Reversal-hedge disabled (baseline DH)")
    if MAX_LEGS_PER_WINDOW > 0:
        log.info(f"Per-window leg cap: {MAX_LEGS_PER_WINDOW}")
    if SIDE_FILTER:
        log.info(f"Side filter: {SIDE_FILTER.upper()} (other side will be skipped)")
    if TIME_DECAY:
        log.info("Time-decay sizing ENABLED | × 0.4 (T+4-T+6.5) | × 0.8 (T+7-T+9.5) | × 1.2 (T+10+)")
    else:
        log.info("Time-decay sizing disabled (flat sizing)")
    log.info(f"Edge filter: skip bets with edge < {MIN_EDGE_CENTS:.1f}c (after {kalshi_trade.FILL_BUFFER_CENTS}c buffer)")
    if MIN_BET_PRICE > 0:
        log.info(f"Min bet price floor: ${MIN_BET_PRICE:.3f} (refuse cheaper fills)")
    if DH_MIN_MINUTE > 0:
        log.info(f"DH minute floor: no entries before T+{DH_MIN_MINUTE:.1f} (late-window-only arm)")
    if INVERT:
        log.info("⚠ INVERT MODE: every YES/NO decision is SWAPPED (paper-only test)")

    # NN mode init — load model & log config.
    if FAIR_PRICE_SOURCE == "nn":
        try:
            ckpt_path = _load_nn()
            log.info(f"NN fair-price source ENABLED | checkpoint={ckpt_path}")
            log.info(f"NN minute range: T+{NN_MIN_MINUTE}..T+{NN_MAX_MINUTE}")
        except Exception as e:
            log.error(f"NN load failed: {e}. Falling back to 2D table.")
            globals()["FAIR_PRICE_SOURCE"] = "2d"
    else:
        log.info("Fair-price source: 2D empirical table")

    if DAILY_LOSS_CAP > 0:
        log.info(f"Daily loss cap: -${DAILY_LOSS_CAP:.2f} (auto-pause if exceeded)")
    if CAP_FRACTION_OF_BALANCE > 0:
        log.info(f"Per-window wagered cap: dynamic = balance × {CAP_FRACTION_OF_BALANCE:.2f} "
                 f"(fallback ${MAX_WINDOW_WAGERED:.2f} if balance fetch fails)")
    elif MAX_WINDOW_WAGERED > 0:
        log.info(f"Per-window wagered cap: static ${MAX_WINDOW_WAGERED:.2f}")
    else:
        log.info("Per-window wagered cap: disabled")

    if not os.path.exists(_2D_CSV_PATH):
        log.error(f"2D fair price table not found: {_2D_CSV_PATH}. Run analyze_minutes_2d.py first.")
        sys.exit(1)
    n_cells = _load_2d_table()
    log.info(f"Loaded 2D fair price table: {n_cells} cells from {os.path.basename(_2D_CSV_PATH)}")

    if ACTIVE_HOURS is None:
        log.info("Active hours: all 24h (ACTIVE_HOURS not set)")
    else:
        hours_str = ", ".join(f"{h:02d}:00" for h in sorted(ACTIVE_HOURS))
        log.info(f"Active hours (UTC): {hours_str}")

    price_feed.start()
    log.info("Waiting for first BTC price from Coinbase WebSocket...")
    for _ in range(30):
        if price_feed.get_price() is not None:
            break
        time.sleep(1)
    else:
        log.error("No BTC price received within 30s. Check network and Coinbase WebSocket. Exiting.")
        sys.exit(1)
    log.info(f"BTC feed live: ${price_feed.get_price():,.2f}")
    if ASIAN_PRICER:
        asian_pricer.start(price_feed.get_price)
        log.info("Asian-settlement pricer sampling at 1s (active in final 120s of each window)")

    # Start Kalshi WebSocket feed for real-time bid/ask (REST API lags by 3-5c).
    # Initial ticker will be set when first window opens via set_ticker().
    if PRIVATE_KEY is not None:
        kalshi_feed.start(PRIVATE_KEY, API_KEY_ID)
        log.info("Kalshi WebSocket feed starting...")
    else:
        log.warning("No private key available — Kalshi WebSocket disabled, using REST prices (may be stale).")

    if not PAPER_MODE:
        balance = kalshi_trade.get_balance(PRIVATE_KEY, API_KEY_ID)
        if balance is not None:
            log.info(f"Kalshi balance: ${balance:.2f}")
        else:
            log.warning("Could not fetch Kalshi balance — check credentials.")

    while not _shutdown:
        now        = datetime.now(timezone.utc)
        elapsed    = elapsed_in_window(now)
        window_ts  = window_boundary(now)

        if ACTIVE_HOURS is not None and window_ts.hour not in ACTIVE_HOURS:
            wait = WINDOW_MINUTES * 60 - elapsed + 0.1
            log.info(f"Skipping {window_ts.strftime('%H:%M')} UTC (not in ACTIVE_HOURS), next window in {wait:.0f}s")
            deadline = time.time() + wait
            while time.time() < deadline and not _shutdown:
                time.sleep(min(1.0, deadline - time.time()))
            continue

        if elapsed < DECISION_OFFSET_SECS:
            # Still before entry point of the current window — enter it now
            wait = 0.1
            log.info(f"Entering current window ({elapsed:.0f}s elapsed, T+{DECISION_OFFSET_SECS//60} in {DECISION_OFFSET_SECS - elapsed:.0f}s)")
        else:
            # Wait for the next boundary
            wait = WINDOW_MINUTES * 60 - elapsed + 0.1
            log.info(f"Next window in {wait:.1f}s")

        deadline = time.time() + wait
        while time.time() < deadline and not _shutdown:
            time.sleep(min(1.0, deadline - time.time()))

        if _shutdown:
            break

        try:
            run_window()
        except Exception as e:
            log.error(f"Unhandled error in run_window(): {e}", exc_info=True)

    log.info("Trader shut down cleanly.")


if __name__ == "__main__":
    main()
