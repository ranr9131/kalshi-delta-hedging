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
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import dotenv_values

import btc_feed
import kalshi_auth
import kalshi_feed
import kalshi_trade
import strategy

# ── Config ────────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
env = dotenv_values(_env_path)

API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")
PAPER_MODE = env.get("PAPER_MODE", "true").lower() == "true"
BASE_STAKE = float(env.get("BASE_STAKE", "100.0"))
MODE         = env.get("MODE", "dh-target").lower()   # t+5 | dh-target | dh-additive
MIN_BET      = float(env.get("MIN_BET", "5.0"))
MAX_FILL     = float(env.get("MAX_FILL", "0.97"))     # skip bets where fill price exceeds this (liquidity concern)
BETS_PER_MIN = int(env.get("BETS_PER_MIN", "4"))       # 1=60s, 2=30s, 4=15s intervals
if BETS_PER_MIN not in (1, 2, 4):
    print(f"ERROR: BETS_PER_MIN={BETS_PER_MIN} is invalid. Must be 1, 2, or 4.")
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
_dir = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH  = os.path.join(_dir, "trade_log.csv")
WINDOW_LOG_PATH = os.path.join(_dir, "window_log.csv")

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
DH_INTERVAL_SECS     = 60 // BETS_PER_MIN                       # 60, 30, or 15
DH_START_SECS        = 4 * 60                                    # T+4:00
DH_END_SECS          = 14 * 60 - DH_INTERVAL_SECS               # T+13:00 / T+13:30 / T+13:45
# DH modes enter at T+4; t+5 mode still waits until T+5
DECISION_OFFSET_SECS = DH_START_SECS if MODE.startswith("dh") else 5 * 60

BTC_RETRY_ATTEMPTS   = 3
BTC_RETRY_DELAY_SECS = 5
MAX_PRICE_AGE_SECS   = 10

# ── 2D fair price table ───────────────────────────────────────────────────────
# Loaded from minute_analysis_2d.csv at startup.
# Key: (minute, bucket_index)  Value: (win_rate, avg_fill, n)
_FAIR_PRICE_2D: dict[tuple[int, int], tuple[float, float, int]] = {}

_2D_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "logs", "minute_analysis_2d_15s_kxbtc15m.csv"
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
            _FAIR_PRICE_2D[(int(row["offset_secs"]), bi)] = (
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


def get_fair_price_2d(offset_secs: int, abs_pct_move: float) -> float:
    """2D empirical win rate for (offset_secs, magnitude bucket). Falls back if cell is sparse."""
    bi    = _get_bucket_idx(abs_pct_move)
    entry = _FAIR_PRICE_2D.get((offset_secs, bi))
    if entry is not None and entry[2] >= _2D_MIN_N:
        return entry[0]
    # Try nearest minute boundary in the table
    minute_offset = (offset_secs // 60) * 60
    entry = _FAIR_PRICE_2D.get((minute_offset, bi))
    if entry is not None and entry[2] >= _2D_MIN_N:
        return entry[0]
    return _FAIR_PRICE_BY_MINUTE_FALLBACK.get(offset_secs // 60, strategy.FAIR_PRICE)


# Option-3 f-function: sigmoid of 2D win rate instead of raw magnitude.
# Center=0.65 means f=1.5 at 65% win rate; simulation showed +9.1pp ROI vs magnitude-based f.
_WR_CENTER       = 0.65
_WR_K            = 20.0
_SIGMOID_MAX_MULT = 3.0


def sigmoid_winrate(win_rate: float) -> float:
    import math
    return _SIGMOID_MAX_MULT / (1 + math.exp(-_WR_K * (win_rate - _WR_CENTER)))

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
        price = btc_feed.get_price()
        age   = btc_feed.get_price_age()
        if price is not None and age < MAX_PRICE_AGE_SECS:
            return price
        reason = "unavailable" if price is None else f"stale ({age:.1f}s old)"
        log.warning(f"BTC price {reason} (attempt {attempt+1}/{BTC_RETRY_ATTEMPTS}), waiting {BTC_RETRY_DELAY_SECS}s...")
        time.sleep(BTC_RETRY_DELAY_SECS)
    return None


def place_order_with_retry(ticker, side, market, stake) -> tuple[str | None, str | None]:
    """
    Place an order. If it rests (market moved between fetch and submit),
    cancel it, re-fetch the market, and retry once at the updated price.
    """
    current_market = market
    for attempt in range(2):
        try:
            resp     = kalshi_trade.place_order(PRIVATE_KEY, API_KEY_ID, ticker, side, current_market, stake)
            order    = resp.get("order", {})
            order_id = order.get("order_id", "unknown")
            status   = order.get("status", "")

            if status == "resting":
                log.warning(f"  Order {order_id} is resting (market moved). Cancelling and retrying...")
                kalshi_trade.cancel_order(PRIVATE_KEY, API_KEY_ID, order_id)
                if attempt == 0:
                    try:
                        current_market = kalshi_trade.get_open_market() or current_market
                    except Exception:
                        pass
                    continue
                log.error(f"  Order still resting on attempt 2 — giving up.")
                break

            return order_id, None
        except Exception as e:
            log.error(f"Order attempt {attempt+1} failed: {e}")
            if attempt == 0:
                time.sleep(1)
    return None, "order rested or failed after retry"


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


def poll_settlement(ticker: str, timeout_secs: int = 120) -> str | None:
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
    _append_csv(WINDOW_LOG_PATH, WINDOW_LOG_FIELDS, row)


# ── DH loop ───────────────────────────────────────────────────────────────────

def run_dh_loop(
    window_ts: datetime,
    btc_t0: float,
    ticker: str,
    close_time: str,
) -> tuple[list, list, float, float]:
    """
    Run delta hedging from T+5 to T+10, placing bets on yes and/or no each minute.
    Returns (yes_bets, no_bets, yes_exposure, no_exposure)
    Each bet entry: (stake, fill_price, count, minute).
    """
    yes_exposure = 0.0
    no_exposure  = 0.0
    net_exposure = 0.0   # positive = net YES, negative = net NO (mirrors Kalshi netting)
    yes_bets: list[tuple[float, float, float, int]] = []
    no_bets:  list[tuple[float, float, float, int]] = []

    for offset_secs in range(DH_START_SECS, DH_END_SECS + 1, DH_INTERVAL_SECS):
        target_dt = window_ts + timedelta(seconds=offset_secs)
        remaining = (target_dt - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            deadline = time.time() + remaining
            while time.time() < deadline and not _shutdown:
                time.sleep(min(1.0, deadline - time.time()))

        if _shutdown:
            break

        t_label = f"T+{offset_secs//60}:{offset_secs%60:02d}"

        btc_now = get_btc_with_retry()
        if btc_now is None:
            log.warning(f"BTC unavailable at {t_label}, skipping interval.")
            continue
        btc_age = btc_feed.get_price_age()

        # Use WebSocket prices (real-time) if fresh; fall back to REST on stale feed.
        ws_bid = kalshi_feed.get_bid()
        ws_ask = kalshi_feed.get_ask()
        ws_age = kalshi_feed.get_age()
        if ws_bid is not None and ws_ask is not None and ws_age < 10:
            yes_bid = ws_bid
            yes_ask = ws_ask
        else:
            log.warning(f"Kalshi WS stale ({ws_age:.1f}s) at {t_label}, falling back to REST.")
            try:
                market = kalshi_trade.get_open_market()
            except Exception as e:
                log.warning(f"REST fallback failed at {t_label}: {e}. Skipping interval.")
                continue
            if market is None:
                log.warning(f"No open market at {t_label}. Skipping interval.")
                continue
            yes_bid = float(market["yes_bid_dollars"])
            yes_ask = float(market["yes_ask_dollars"])

        order_market = {"yes_ask_dollars": str(yes_ask), "yes_bid_dollars": str(yes_bid)}
        spread     = round(yes_ask - yes_bid, 4)
        kalshi_mid = (yes_bid + yes_ask) / 2

        abs_pct_move = abs(btc_now - btc_t0) / btc_t0 * 100
        direction_up = btc_now > btc_t0

        fair  = get_fair_price_2d(offset_secs, abs_pct_move)
        f_btc = sigmoid_winrate(fair)

        buf = kalshi_trade.FILL_BUFFER_CENTS / 100
        if direction_up:
            mispricing = fair - (yes_ask + buf)   # true edge after buffer cost
            g_misprice = strategy.sigmoid_mispricing(mispricing)
            target_yes = BASE_STAKE * f_btc * g_misprice
            target_no  = 0.0
        else:
            mispricing = fair - ((1.0 - yes_bid) + buf)  # P(direction correct) - no_fill cost
            g_misprice = strategy.sigmoid_mispricing(mispricing)
            target_no  = BASE_STAKE * f_btc * g_misprice
            target_yes = 0.0

        if MODE == "dh-target":
            # Use net exposure so Kalshi's position netting is accounted for:
            # net_exposure > 0 means we hold YES; < 0 means we hold NO.
            bet_yes = max(0.0, target_yes - max(net_exposure, 0.0))
            bet_no  = max(0.0, target_no  - max(-net_exposure, 0.0))
        else:  # dh-additive
            bet_yes = target_yes
            bet_no  = target_no

        direction_label = "yes" if direction_up else "no"

        log.info(
            f"DH {t_label}: {direction_label.upper()} | "
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
            "dh_minute":          offset_secs,
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

        if bet_yes >= MIN_BET:
            fill = min(yes_ask + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            if fill > MAX_FILL:
                log.info(f"  -> SKIP YES ${bet_yes:.2f} — fill {fill:.3f} > MAX_FILL {MAX_FILL:.2f} (liquidity concern)")
            else:
                count = max(1, round(bet_yes / fill))
                log.info(f"  -> BET YES ${bet_yes:.2f} @ {fill:.3f} ({fill*100:.1f}c/contract) | {count} contracts")
                if PAPER_MODE:
                    order_id, order_result = "paper", "paper"
                    log.info("     [PAPER] no order submitted")
                    placed = True
                else:
                    order_id, err = place_order_with_retry(ticker, "yes", order_market, bet_yes)
                    order_result  = "ok" if order_id else f"error: {err}"
                    if order_id:
                        log.info(f"     YES order placed: {order_id}")
                        placed = True
                    else:
                        log.error(f"     YES order FAILED: {err}")
                        placed = False
                if placed:
                    yes_bets.append((bet_yes, fill, count, minute))
                    yes_exposure += bet_yes
                    net_exposure += bet_yes
                    log_bet({**base_row,
                    "yes_exposure_before": round(yes_exposure - bet_yes, 4),
                    "no_exposure_before":  round(no_exposure, 4),
                    "bet_side":            "yes",
                    "stake":               round(bet_yes, 4),
                    "fill_price":          round(fill, 4),
                    "count":               count,
                    "order_id":            order_id or "none",
                    "order_result":        order_result,
                })
        elif direction_up:
            if target_yes < MIN_BET:
                if mispricing < 0:
                    log.info(f"  -> SKIP YES — no edge: Kalshi ask ({yes_ask:.3f}) > fair ({fair:.3f}), mis={mispricing:+.3f}, target=${target_yes:.2f}")
                else:
                    log.info(f"  -> SKIP YES — BTC move too small for meaningful bet (f={f_btc:.3f} g={g_misprice:.3f}, target=${target_yes:.2f})")
            else:
                log.info(f"  -> SKIP YES — gap ${bet_yes:.2f} < MIN_BET (already hold ${net_exposure:.2f} of ${target_yes:.2f} target)")

        if bet_no >= MIN_BET:
            fill = min((1.0 - yes_bid) + kalshi_trade.FILL_BUFFER_CENTS / 100, 0.99)
            if fill > MAX_FILL:
                log.info(f"  -> SKIP NO  ${bet_no:.2f} — fill {fill:.3f} > MAX_FILL {MAX_FILL:.2f} (liquidity concern)")
            else:
                count = max(1, round(bet_no / fill))
                log.info(f"  -> BET NO  ${bet_no:.2f} @ {fill:.3f} ({fill*100:.1f}c/contract) | {count} contracts")
                if PAPER_MODE:
                    order_id, order_result = "paper", "paper"
                    log.info("     [PAPER] no order submitted")
                    placed = True
                else:
                    order_id, err = place_order_with_retry(ticker, "no", order_market, bet_no)
                    order_result  = "ok" if order_id else f"error: {err}"
                    if order_id:
                        log.info(f"     NO order placed: {order_id}")
                        placed = True
                    else:
                        log.error(f"     NO order FAILED: {err}")
                        placed = False
                if placed:
                    no_bets.append((bet_no, fill, count, minute))
                    no_exposure += bet_no
                    net_exposure -= bet_no
                    log_bet({**base_row,
                        "yes_exposure_before": round(yes_exposure, 4),
                        "no_exposure_before":  round(no_exposure - bet_no, 4),
                        "bet_side":            "no",
                        "stake":               round(bet_no, 4),
                        "fill_price":          round(fill, 4),
                        "count":               count,
                        "order_id":            order_id or "none",
                        "order_result":        order_result,
                    })
        elif not direction_up:
            if target_no < MIN_BET:
                if mispricing < 0:
                    log.info(f"  -> SKIP NO  — no edge: Kalshi no ask ({1-yes_bid:.3f}) > fair ({fair:.3f}), mis={mispricing:+.3f}, target=${target_no:.2f}")
                else:
                    log.info(f"  -> SKIP NO  — BTC move too small for meaningful bet (f={f_btc:.3f} g={g_misprice:.3f}, target=${target_no:.2f})")
            else:
                log.info(f"  -> SKIP NO  — gap ${bet_no:.2f} < MIN_BET (already hold ${-net_exposure:.2f} of ${target_no:.2f} target)")

    return yes_bets, no_bets, yes_exposure, no_exposure


# ── Window execution ──────────────────────────────────────────────────────────

def run_window():
    global _cumulative_pnl

    now        = datetime.now(timezone.utc)
    window_ts  = window_boundary(now)
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
    btc_age_entry = btc_feed.get_price_age()

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
            order_id, err = place_order_with_retry(ticker, side, market, stake)
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
    yes_bets, no_bets, yes_exp, no_exp = run_dh_loop(window_ts, btc_t0, ticker, close_time)

    btc_t10 = get_btc_with_retry()

    wait_for_close(close_time)
    winner    = poll_settlement(ticker)
    settle_ts = datetime.now(timezone.utc)

    yes_pnl_total = sum(compute_pnl("yes", fp, cnt, winner) for _, fp, cnt, _ in yes_bets)
    no_pnl_total  = sum(compute_pnl("no",  fp, cnt, winner) for _, fp, cnt, _ in no_bets)
    total_pnl     = yes_pnl_total + no_pnl_total
    total_wagered = yes_exp + no_exp
    _cumulative_pnl += total_pnl
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

    btc_feed.start()
    log.info("Waiting for first BTC price from Coinbase WebSocket...")
    for _ in range(30):
        if btc_feed.get_price() is not None:
            break
        time.sleep(1)
    else:
        log.error("No BTC price received within 30s. Check network and Coinbase WebSocket. Exiting.")
        sys.exit(1)
    log.info(f"BTC feed live: ${btc_feed.get_price():,.2f}")

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
