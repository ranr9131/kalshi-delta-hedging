"""
Live shadow simulator — runs all reversal-handling strategy variants in parallel
against the live Kalshi KXBTC15M markets WITHOUT placing any real orders.

For each window:
  - Subscribes to Coinbase BTC WS feed (live/btc_feed.py).
  - At T+0 (window open), reads the open market's floor_strike as the cutoff.
  - Every minute T+4..T+13, samples BTC + Kalshi mid/bid/ask, and feeds the
    snapshot to each variant's step function (mirrors simulate_dh.py).
  - After close_time, polls for settlement, computes per-variant P&L, and
    appends one row per (window × variant) to live/shadow_log.csv.

Variants tracked (target-mode sizing, same as live trader):
  baseline       — current production strategy
  rh-12          — reversal hedge from T+12
  rh-11          — reversal hedge from T+11
  rh-10          — reversal hedge from T+10
  ncs-11-0.08    — skip new bets at T+11+ if |move| < 0.08%
  ncs11+rh12     — both overlays composed

Output:
  live/shadow_log.csv — one row per (window × variant) once a window settles

This file never places real orders. No KALSHI_PRIVATE_KEY needed.
"""
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

# Reuse live infra (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_feed
import kalshi_trade
import strategy   # sigmoid_btc, sigmoid_mispricing, FAIR_PRICE

# ── Config ────────────────────────────────────────────────────────────────────
BASE_STAKE = 100.0
MIN_BET    = 5.0
FEE_RATE   = 0.07

DH_MINUTES        = list(range(4, 14))      # T+4..T+13
WINDOW_MINUTES    = 15
ENTRY_OFFSET_SECS = 4 * 60
MAX_PRICE_AGE     = 15
RETRY_DELAY       = 5
RETRY_ATTEMPTS    = 3

# 2D fair price table (built by analyze_minutes_2d.py)
_2D_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "logs", "minute_analysis_2d.csv"
)
_2D_BUCKETS = [(0.000, 0.05), (0.050, 0.10), (0.100, 0.20), (0.200, 0.50), (0.500, float("inf"))]
_2D_LABELS  = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]
_2D_MIN_N   = 30
_FAIR_2D: dict[tuple[int, int], tuple[float, float, int]] = {}

# 1D fallback when a 2D cell is sparse
_FALLBACK_1D = {
    1: 0.582, 2: 0.617, 3: 0.636, 4: 0.670,
    5: 0.698, 6: 0.728, 7: 0.751, 8: 0.759,
    9: 0.783, 10: 0.798, 11: 0.806, 12: 0.815,
    13: 0.826,
}

# Strategy variants tracked in parallel
VARIANTS = [
    ("baseline",      dict()),
    ("rh-12",         dict(rh_minute=12)),
    ("rh-11",         dict(rh_minute=11)),
    ("rh-10",         dict(rh_minute=10)),
    ("ncs-11-0.08",   dict(ncs_minute=11, ncs_threshold=0.08)),
    ("ncs11+rh12",    dict(ncs_minute=11, ncs_threshold=0.08, rh_minute=12)),
]

_DIR             = os.path.dirname(os.path.abspath(__file__))
SHADOW_LOG       = os.path.join(_DIR, "shadow_log.csv")
SHADOW_LOG_FIELDS = [
    "window_ts", "ticker", "close_time", "btc_t0",
    "winner", "settlement_ts",
    "variant",
    "n_yes_bets", "n_no_bets", "n_hedges",
    "yes_stake", "no_stake", "wagered",
    "pnl", "running_pnl",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shadow] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shadow")

_shutdown = False
def _sigint(signum, frame):
    global _shutdown
    log.info("Shutdown signal received. Will exit after current window.")
    _shutdown = True
signal.signal(signal.SIGINT, _sigint)


# ── 2D fair price ─────────────────────────────────────────────────────────────
def load_2d_table() -> int:
    label_to_idx = {lbl: i for i, lbl in enumerate(_2D_LABELS)}
    count = 0
    with open(_2D_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            bi = label_to_idx.get(row["bucket"])
            if bi is None:
                continue
            _FAIR_2D[(int(row["minute"]), bi)] = (
                float(row["win_rate"]), float(row["avg_fill"]), int(row["n"])
            )
            count += 1
    return count


def fair_price(minute: int, abs_pct_move: float) -> float:
    bi = 0
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= abs_pct_move < hi:
            bi = i
            break
    else:
        bi = len(_2D_BUCKETS) - 1
    entry = _FAIR_2D.get((minute, bi))
    if entry is not None and entry[2] >= _2D_MIN_N:
        return entry[0]
    return _FALLBACK_1D.get(minute, strategy.FAIR_PRICE)


# ── Helpers ───────────────────────────────────────────────────────────────────
def window_boundary(dt: datetime) -> datetime:
    boundary_minute = (dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES
    return dt.replace(minute=boundary_minute, second=0, microsecond=0)


def elapsed_in_window(dt: datetime) -> float:
    return (dt - window_boundary(dt)).total_seconds()


def wait_until(target_unix: float):
    while not _shutdown:
        rem = target_unix - time.time()
        if rem <= 0:
            return
        time.sleep(min(1.0, rem))


def get_btc_with_retry() -> float | None:
    for _ in range(RETRY_ATTEMPTS):
        p = btc_feed.get_price()
        a = btc_feed.get_price_age()
        if p is not None and a < MAX_PRICE_AGE:
            return p
        time.sleep(RETRY_DELAY)
    return None


def pnl_bet(side: str, stake: float, fill_price: float, winner: str) -> float:
    contracts = stake / fill_price
    if side == winner:
        return (contracts - stake) * (1 - FEE_RATE)
    return -stake


# ── Per-variant minute step ───────────────────────────────────────────────────
def step_variant(state, snapshot, *,
                 ncs_minute=None, ncs_threshold=0.0,
                 rh_minute=None, rh_trigger=10.0):
    """
    Apply one minute's decision to a variant's state dict (mutates in place).

    snapshot keys: minute, btc_t0, btc_now, abs_pct, direction_up,
                   yes_bid, yes_ask, kalshi_mid
    """
    minute   = snapshot["minute"]
    btc_t0   = snapshot["btc_t0"]
    abs_pct  = snapshot["abs_pct"]
    dir_up   = snapshot["direction_up"]
    yes_bid  = snapshot["yes_bid"]
    yes_ask  = snapshot["yes_ask"]
    kal_yes  = snapshot["kalshi_mid"]
    kal_no   = 1.0 - kal_yes

    f_btc = strategy.sigmoid_btc(abs_pct)
    fair  = fair_price(minute, abs_pct)

    if dir_up:
        mispricing = fair - kal_yes
        g          = strategy.sigmoid_mispricing(mispricing)
        target_yes = BASE_STAKE * f_btc * g
        target_no  = 0.0
    else:
        mispricing = kal_yes - (1.0 - fair)
        g          = strategy.sigmoid_mispricing(mispricing)
        target_yes = 0.0
        target_no  = BASE_STAKE * f_btc * g

    # Near-Cutoff Skip
    if ncs_minute is not None and minute >= ncs_minute and abs_pct < ncs_threshold:
        target_yes = 0.0
        target_no  = 0.0

    # Target-mode gap fill
    gap_yes = max(0.0, target_yes - state["yes_exp"])
    gap_no  = max(0.0, target_no  - state["no_exp"])

    if gap_yes >= MIN_BET:
        fill = yes_ask
        state["yes_bets"].append((gap_yes, fill, minute, "entry"))
        state["yes_exp"]      += gap_yes
        state["yes_contracts"] += gap_yes / fill
    if gap_no >= MIN_BET:
        fill = 1.0 - yes_bid
        state["no_bets"].append((gap_no, fill, minute, "entry"))
        state["no_exp"]      += gap_no
        state["no_contracts"] += gap_no / fill

    # Reversal Hedge
    if rh_minute is not None and minute >= rh_minute:
        if dir_up and state["no_exp"] >= rh_trigger and state["no_contracts"] > 0:
            fill = yes_ask
            hedge = state["no_contracts"] * fill
            if hedge >= MIN_BET:
                state["yes_bets"].append((hedge, fill, minute, "hedge"))
                state["yes_exp"]       += hedge
                state["yes_contracts"] += hedge / fill
                state["n_hedges"]      += 1
        elif (not dir_up) and state["yes_exp"] >= rh_trigger and state["yes_contracts"] > 0:
            fill = 1.0 - yes_bid
            hedge = state["yes_contracts"] * fill
            if hedge >= MIN_BET:
                state["no_bets"].append((hedge, fill, minute, "hedge"))
                state["no_exp"]       += hedge
                state["no_contracts"] += hedge / fill
                state["n_hedges"]      += 1


# ── Window execution ─────────────────────────────────────────────────────────
_running_pnl: dict[str, float] = {label: 0.0 for label, _ in VARIANTS}


def bootstrap_running_pnl() -> int:
    """Recover running totals from shadow_log.csv if it exists."""
    if not os.path.exists(SHADOW_LOG):
        return 0
    rows = list(csv.DictReader(open(SHADOW_LOG)))
    for r in rows:
        label = r["variant"]
        if label in _running_pnl:
            _running_pnl[label] += float(r["pnl"])
    return len(rows)


def init_state():
    return {
        "yes_exp": 0.0, "no_exp": 0.0,
        "yes_contracts": 0.0, "no_contracts": 0.0,
        "yes_bets": [], "no_bets": [],
        "n_hedges": 0,
    }


def run_window():
    now = datetime.now(timezone.utc)
    ws  = window_boundary(now)

    # Wait until T+4
    wait_until(ws.timestamp() + ENTRY_OFFSET_SECS)
    if _shutdown:
        return

    try:
        market = kalshi_trade.get_open_market()
    except Exception as e:
        log.error(f"open-market fetch failed: {e}")
        return
    if market is None:
        log.warning("No open KXBTC15M market — skipping.")
        return

    ticker     = market["ticker"]
    close_time = market.get("close_time", "")
    btc_t0     = float(market["floor_strike"])

    log.info(f"━━ window {ws.strftime('%H:%M')} UTC | {ticker} | cutoff=${btc_t0:,.2f}")

    states = {label: init_state() for label, _ in VARIANTS}
    cfg    = {label: kwargs for label, kwargs in VARIANTS}

    for minute in DH_MINUTES:
        wait_until(ws.timestamp() + minute * 60)
        if _shutdown:
            break

        btc_now = get_btc_with_retry()
        if btc_now is None:
            log.warning(f"T+{minute}: BTC unavailable")
            continue

        try:
            mkt = kalshi_trade.get_open_market()
        except Exception as e:
            log.warning(f"T+{minute}: market refresh failed: {e}")
            continue
        if mkt is None or mkt["ticker"] != ticker:
            continue

        yes_bid = float(mkt["yes_bid_dollars"])
        yes_ask = float(mkt["yes_ask_dollars"])
        kal_mid = (yes_bid + yes_ask) / 2.0
        abs_pct = abs(btc_now - btc_t0) / btc_t0 * 100.0
        dir_up  = btc_now > btc_t0

        snapshot = {
            "minute": minute, "btc_t0": btc_t0, "btc_now": btc_now,
            "abs_pct": abs_pct, "direction_up": dir_up,
            "yes_bid": yes_bid, "yes_ask": yes_ask, "kalshi_mid": kal_mid,
        }

        for label, _ in VARIANTS:
            step_variant(states[label], snapshot, **cfg[label])

        log.info(
            f"  T+{minute:>2} btc=${btc_now:,.2f} ({'+' if dir_up else '-'}{abs_pct:.4f}%) | "
            f"mid={kal_mid:.3f} bid/ask={yes_bid:.3f}/{yes_ask:.3f}"
        )

    # Wait for settlement window to open
    try:
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        wait_until(close_dt.timestamp() + 5)
    except Exception:
        time.sleep(60)

    # Poll for settlement. Settlements typically arrive within 30s but can
    # lag 5+ min in slow periods, so poll up to 10 minutes.
    winner = None
    settle_ts_iso = ""
    deadline = time.time() + 600
    while time.time() < deadline and not _shutdown:
        try:
            r = kalshi_trade.get_market_result(ticker)
            if r is not None:
                winner = r
                settle_ts_iso = datetime.now(timezone.utc).isoformat()
                break
        except Exception:
            pass
        time.sleep(5)

    if winner is None:
        log.warning(f"Settlement not confirmed for {ticker}")
        return

    # Compute P&L per variant and persist
    rows = []
    for label, _ in VARIANTS:
        s = states[label]
        pnl  = sum(pnl_bet("yes", stk, fp, winner) for stk, fp, _m, _t in s["yes_bets"])
        pnl += sum(pnl_bet("no",  stk, fp, winner) for stk, fp, _m, _t in s["no_bets"])
        wagered = s["yes_exp"] + s["no_exp"]
        _running_pnl[label] += pnl
        rows.append({
            "window_ts":     ws.isoformat(),
            "ticker":        ticker,
            "close_time":    close_time,
            "btc_t0":        round(btc_t0, 2),
            "winner":        winner,
            "settlement_ts": settle_ts_iso,
            "variant":       label,
            "n_yes_bets":    len(s["yes_bets"]),
            "n_no_bets":     len(s["no_bets"]),
            "n_hedges":      s["n_hedges"],
            "yes_stake":     round(s["yes_exp"], 4),
            "no_stake":      round(s["no_exp"], 4),
            "wagered":       round(wagered, 4),
            "pnl":           round(pnl, 4),
            "running_pnl":   round(_running_pnl[label], 4),
        })

    _append_rows(SHADOW_LOG, SHADOW_LOG_FIELDS, rows)

    log.info(f"━━ settled winner={winner} | per-variant P&L:")
    for r in rows:
        log.info(
            f"  {r['variant']:<14} pnl=${r['pnl']:>+8.2f}  "
            f"wagered=${r['wagered']:>7.2f}  hedges={r['n_hedges']}  "
            f"running=${r['running_pnl']:>+9.2f}"
        )


def _append_rows(path, fields, rows):
    if not rows:
        return
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log.info(f"Shadow simulator starting | variants: {[v[0] for v in VARIANTS]}")
    if not os.path.exists(_2D_CSV_PATH):
        log.error(f"2D fair price table not found at {_2D_CSV_PATH}")
        sys.exit(1)
    n = load_2d_table()
    log.info(f"Loaded 2D table: {n} cells")

    n_prior = bootstrap_running_pnl()
    if n_prior:
        log.info(f"Bootstrapped running P&L from {n_prior} prior rows in {os.path.basename(SHADOW_LOG)}:")
        for label, _ in VARIANTS:
            log.info(f"  {label:<14} running=${_running_pnl[label]:>+9.2f}")

    btc_feed.start()
    log.info("Waiting for first BTC tick...")
    for _ in range(30):
        if btc_feed.get_price() is not None:
            break
        time.sleep(1)
    else:
        log.error("No BTC price received within 30s.")
        sys.exit(1)
    log.info(f"BTC live: ${btc_feed.get_price():,.2f}")

    while not _shutdown:
        now = datetime.now(timezone.utc)
        el  = elapsed_in_window(now)
        ws  = window_boundary(now)

        if el < ENTRY_OFFSET_SECS:
            wait = 0.5
            log.info(f"Entering current window {ws.strftime('%H:%M')} ({el:.0f}s elapsed)")
        else:
            wait = WINDOW_MINUTES * 60 - el + 0.2
            log.info(f"Next window in {wait:.1f}s ({ws.strftime('%H:%M')} + {WINDOW_MINUTES}m)")

        wait_until(time.time() + wait)
        if _shutdown:
            break

        try:
            run_window()
        except Exception as e:
            log.exception(f"Window error: {e}")

    log.info("Shadow simulator stopped.")


if __name__ == "__main__":
    main()
