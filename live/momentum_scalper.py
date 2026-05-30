"""
Momentum scalper for KXBTC15M — models the Polymarket wallet
0xce25e2…7fdc (13,482 short-duration crypto predictions, +$8.9k cumulative).

Strategy — pure taker, ride the favorite after it forms:
  - Watch BTC tape. When BTC moves ≥ MS_TRIGGER_USD over the last
    MS_LOOKBACK_SECS, flag a directional signal.
  - If the trending-side Kalshi YES sits inside MS_ENTRY_BAND_LOW..HIGH
    (default 60–75¢, matching their observed 64–70¢ entries), TAKE that
    side with a small IOC at top-of-book.
  - Hold to settlement. NO within-window stop-loss — wrong-way bets eat
    a full $1 loss per contract, just like their BTC/ETH DOWN tickets did.
  - Cooldown MS_COOLDOWN_SECS between entries; cap MS_MAX_ENTRIES per
    window and MS_MAX_WAGERED total wagered per window.
  - Skip last MS_STOP_ENTERING_LAST_MINS of the window.

NOT a market maker. NOT delta-hedged. Pure directional momentum take.

PAPER MODE: when MS_PAPER_MODE=true (default), no real orders are placed.
Fills are simulated at the current top-of-book ask (for YES) or 1-bid
(for NO) — taker pricing.

Run:  python3 momentum_scalper.py
"""
import os
import sys
import time
import signal
import logging
from collections import deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import btc_feed
import kalshi_feed
import kalshi_trade
import kalshi_auth

# ── Config ────────────────────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))

PAPER_MODE = env.get("MS_PAPER_MODE", "true").strip().lower() not in ("false", "0", "no")

# Signal: BTC must move ≥ this many dollars in the lookback window
MS_TRIGGER_USD          = float(env.get("MS_TRIGGER_USD", "50"))
MS_LOOKBACK_SECS        = float(env.get("MS_LOOKBACK_SECS", "60"))

# Entry band: only take when the trending side's YES is in this cent range.
# Their observed entries clustered 64–70¢ — we open it a bit wider.
MS_ENTRY_BAND_LOW_C     = int(env.get("MS_ENTRY_BAND_LOW_C", "60"))
MS_ENTRY_BAND_HIGH_C    = int(env.get("MS_ENTRY_BAND_HIGH_C", "75"))

# Per-entry stake — small, consistent with their high-frequency / small-size profile
MS_STAKE_DOLLARS        = float(env.get("MS_STAKE_DOLLARS", "3.0"))

# Cooldown between entries (sec). Prevents stacking on a single signal.
MS_COOLDOWN_SECS        = float(env.get("MS_COOLDOWN_SECS", "30"))

# Per-window caps
MS_MAX_ENTRIES          = int(env.get("MS_MAX_ENTRIES", "4"))
MS_MAX_WAGERED          = float(env.get("MS_MAX_WAGERED", "20"))

# Don't enter the last N minutes of a window (settlement risk)
MS_STOP_ENTERING_LAST_MINS = float(env.get("MS_STOP_ENTERING_LAST_MINS", "2"))

# How often to evaluate (seconds)
MS_REFRESH_SECS         = float(env.get("MS_REFRESH_SECS", "1"))

SERIES = "KXBTC15M"

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
if not API_KEY_ID or not raw_pem:
    print("ERROR: Kalshi credentials missing from .env.")
    sys.exit(1)
PRIVATE_KEY = kalshi_auth.load_private_key(raw_pem)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(_dir, "momentum_scalper.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ms] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ms")


# ── State ─────────────────────────────────────────────────────────────────────
class State:
    ticker:        str | None = None
    close_time:    datetime | None = None
    floor:         float = 0.0

    # Per-window inventory
    yes_contracts: float = 0.0
    no_contracts:  float = 0.0
    total_wagered: float = 0.0
    entries_this_window: int = 0

    # Cumulative across windows
    realized_pnl:  float = 0.0

    # BTC tape for momentum (deque of (ts, price))
    tape:          deque  # type: ignore

    # Throttle
    last_entry_ts: float = 0.0


_state = State()
_state.tape = deque()
_shutdown = False


def _sigint(signum, frame):
    global _shutdown
    log.info("Shutdown signal received.")
    _shutdown = True


# ── Utility ───────────────────────────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc)


def update_tape():
    """Append latest BTC price to tape, trim entries older than lookback."""
    px = btc_feed.get_price()
    if px is None:
        return
    now = time.time()
    _state.tape.append((now, px))
    cutoff = now - MS_LOOKBACK_SECS
    while _state.tape and _state.tape[0][0] < cutoff:
        _state.tape.popleft()


def momentum_signal() -> tuple[str | None, float]:
    """
    Returns (side, move_usd) where:
      side ∈ {'yes', 'no', None}
      move_usd is the BTC move over the lookback window (positive number)
    None means no signal (move too small).
    """
    if len(_state.tape) < 2:
        return None, 0.0
    oldest_px = _state.tape[0][1]
    newest_px = _state.tape[-1][1]
    delta = newest_px - oldest_px
    if abs(delta) < MS_TRIGGER_USD:
        return None, abs(delta)
    return ("yes" if delta > 0 else "no"), abs(delta)


# ── Entry ─────────────────────────────────────────────────────────────────────
def try_enter():
    """If signal + filters pass, take a small position on the trending side."""
    # Per-window caps
    if _state.entries_this_window >= MS_MAX_ENTRIES:
        return
    if _state.total_wagered >= MS_MAX_WAGERED:
        return

    # Cooldown
    if time.time() - _state.last_entry_ts < MS_COOLDOWN_SECS:
        return

    side, move = momentum_signal()
    if side is None:
        return

    # Read top-of-book
    top_bid = kalshi_feed.get_bid()
    top_ask = kalshi_feed.get_ask()
    if top_bid is None or top_ask is None:
        return

    # Trending-side YES price = what we'd PAY for the trending side.
    # If side=yes, we pay the YES ask. If side=no, we pay (1 - YES bid).
    if side == "yes":
        pay_cents = round(top_ask * 100)
    else:
        pay_cents = 100 - round(top_bid * 100)

    if not (MS_ENTRY_BAND_LOW_C <= pay_cents <= MS_ENTRY_BAND_HIGH_C):
        log.info(f"  signal {side.upper()} (BTC move ${move:.0f}) — but trending price "
                 f"{pay_cents}c outside band {MS_ENTRY_BAND_LOW_C}-{MS_ENTRY_BAND_HIGH_C}c. skip.")
        _state.last_entry_ts = time.time()  # cooldown anyway, don't keep re-evaluating
        return

    contracts = max(1, round(MS_STAKE_DOLLARS / (pay_cents / 100.0)))
    cost = contracts * pay_cents / 100.0

    log.info(f"━━ ENTER {side.upper()} | BTC moved ${move:+.0f} in {MS_LOOKBACK_SECS:.0f}s | "
             f"pay {pay_cents}c × {contracts} = ${cost:.2f}")

    if PAPER_MODE:
        if side == "yes":
            _state.yes_contracts += contracts
        else:
            _state.no_contracts  += contracts
        _state.total_wagered += cost
    else:
        try:
            market = kalshi_trade.get_open_market()
            if market is None or market.get("ticker") != _state.ticker:
                log.warning("  market vanished before entry; skip.")
                return
            resp = kalshi_trade.place_order(
                PRIVATE_KEY, API_KEY_ID,
                ticker=_state.ticker, side=side, market=market,
                stake_dollars=MS_STAKE_DOLLARS, ioc=True,
            )
            order = resp.get("order", {}) if isinstance(resp, dict) else {}
            order_id = order.get("order_id")
            if not order_id:
                log.warning(f"  no order_id in response: {resp}")
                return
            # Resolve actual fill
            try:
                filled = kalshi_trade.get_order_filled_stake(PRIVATE_KEY, API_KEY_ID, order_id)
            except Exception as e:
                log.warning(f"  fill check failed: {e}")
                filled = cost
            log.info(f"  filled ${filled:.2f}")
            if side == "yes":
                _state.yes_contracts += filled / (pay_cents / 100.0)
            else:
                _state.no_contracts  += filled / (pay_cents / 100.0)
            _state.total_wagered += filled
        except Exception as e:
            log.error(f"  place_order failed: {e}")
            return

    _state.entries_this_window += 1
    _state.last_entry_ts = time.time()


# ── Settlement ────────────────────────────────────────────────────────────────
def settle_current_window():
    if not _state.ticker:
        return
    log.info(f"Settling {_state.ticker} (Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} wag=${_state.total_wagered:.2f})...")

    deadline = time.time() + 300
    winner = None
    while time.time() < deadline:
        try:
            winner = kalshi_trade.get_market_result(_state.ticker)
            if winner:
                break
        except Exception:
            pass
        time.sleep(10)
    if winner is None:
        log.warning("  no settlement after 5min — skipping P&L")
    else:
        yes_payout = _state.yes_contracts if winner == "yes" else 0
        no_payout  = _state.no_contracts  if winner == "no"  else 0
        window_pnl = yes_payout + no_payout - _state.total_wagered
        _state.realized_pnl += window_pnl
        log.info(
            f"RESULT: winner={winner} | Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} "
            f"wag=${_state.total_wagered:.2f} payout=${yes_payout+no_payout:.2f} "
            f"window_pnl=${window_pnl:+.2f} | cumulative=${_state.realized_pnl:+.2f}"
        )

    _state.yes_contracts = 0.0
    _state.no_contracts  = 0.0
    _state.total_wagered = 0.0
    _state.entries_this_window = 0


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    log.info(f"Momentum scalper starting | mode={'PAPER' if PAPER_MODE else 'LIVE'}")
    log.info(f"  trigger=${MS_TRIGGER_USD:.0f} over {MS_LOOKBACK_SECS:.0f}s | "
             f"entry_band={MS_ENTRY_BAND_LOW_C}-{MS_ENTRY_BAND_HIGH_C}c")
    log.info(f"  stake=${MS_STAKE_DOLLARS:.2f} | cooldown={MS_COOLDOWN_SECS:.0f}s | "
             f"max_entries={MS_MAX_ENTRIES} | max_wagered=${MS_MAX_WAGERED:.2f}")

    btc_feed.start()
    time.sleep(1)
    kalshi_feed.start(PRIVATE_KEY, API_KEY_ID)
    time.sleep(1)

    while not _shutdown:
        try:
            market = kalshi_trade.get_open_market()
        except Exception as e:
            log.warning(f"get_open_market failed: {e}")
            time.sleep(10)
            continue
        if market is None:
            log.info("No open KXBTC15M market. Waiting 30s.")
            time.sleep(30)
            continue

        ticker = market.get("ticker")
        if ticker != _state.ticker:
            if _state.ticker:
                settle_current_window()
            _state.ticker = ticker
            _state.floor  = float(market.get("floor_strike", 0))
            try:
                _state.close_time = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            except Exception:
                _state.close_time = None
            kalshi_feed.set_ticker(ticker)
            _state.tape.clear()
            log.info(f"━━ NEW WINDOW: {ticker} floor=${_state.floor:,.2f} closes={_state.close_time}")

        update_tape()

        # Skip late-window entries (still settle at the end)
        if _state.close_time:
            mins_to_close = (_state.close_time - now_utc()).total_seconds() / 60
            if mins_to_close < MS_STOP_ENTERING_LAST_MINS:
                time.sleep(min(10, max(1, (_state.close_time - now_utc()).total_seconds())))
                continue

        try:
            try_enter()
        except Exception as e:
            log.error(f"try_enter error: {e}")

        time.sleep(MS_REFRESH_SECS)

    log.info(f"Final: realized_pnl=${_state.realized_pnl:+.2f} "
             f"open=Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} "
             f"wagered_this_window=${_state.total_wagered:.2f}")
    log.info("Momentum scalper shut down cleanly.")


if __name__ == "__main__":
    main()
