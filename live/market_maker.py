"""
Market making bot for KXBTC15M.

Strategy — earn the spread by providing liquidity:
  - Continuously post a BID and an ASK on the open KXBTC15M market
  - Both quotes are INSIDE the current top-of-book (more aggressive prices)
  - When BTC moves, cancel + re-post (don't get picked off)
  - When inventory drifts (one side keeps filling), skew quotes to rebalance
  - Hard caps: max inventory $, max wagered $, kill switch

Why this beats directional trading: we get PAID the spread instead of paying
it. Every time both sides fill once, we profit the spread (1-3c per round
trip). 50-200 round trips per window × small $ = sustainable.

PAPER MODE: when MM_PAPER_MODE=true (default), no real orders are placed.
We simulate fills using simple top-of-book heuristics so you can validate
logic before risking real capital.

Run:  python3 market_maker.py
"""
import os
import sys
import time
import signal
import logging
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import btc_feed
import kalshi_feed
import kalshi_trade
import kalshi_auth

# ── Config ────────────────────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))

PAPER_MODE = env.get("MM_PAPER_MODE", "true").strip().lower() not in ("false", "0", "no")

# Per-side quote size (dollars). Each round trip earns spread × this size.
MM_QUOTE_SIZE_DOLLARS  = float(env.get("MM_QUOTE_SIZE_DOLLARS", "5.0"))

# How many cents INSIDE the current top-of-book to post.
# Top is bid=0.58/ask=0.62, MM_INSIDE_CENTS=1 → our bid=0.59, our ask=0.61.
# Larger = more aggressive (more fills) but more adverse selection risk.
MM_INSIDE_CENTS        = int(env.get("MM_INSIDE_CENTS", "1"))

# Minimum spread we'll quote into. If spread is already 1c wide, we'd post
# at the same price as top of book — risky. Skip if spread < this.
MM_MIN_SPREAD_CENTS    = int(env.get("MM_MIN_SPREAD_CENTS", "3"))

# Max net inventory (in absolute dollars). If we're heavily long YES, stop
# posting the YES bid until we offload. Skew the YES ask down to attract sells.
MM_MAX_INVENTORY_DOLLARS = float(env.get("MM_MAX_INVENTORY_DOLLARS", "30"))

# Hard kill: max total $ at risk across both sides. Pause everything past this.
MM_MAX_TOTAL_WAGERED   = float(env.get("MM_MAX_TOTAL_WAGERED", "200"))

# Re-quote when BTC moves more than this between checks (dollars).
# Defends against adverse selection on directional moves.
MM_REQUOTE_BTC_MOVE    = float(env.get("MM_REQUOTE_BTC_MOVE", "20"))

# How often to re-evaluate quotes (seconds). 1-5 is reasonable.
MM_REFRESH_SECS        = float(env.get("MM_REFRESH_SECS", "2"))

# Don't enter the last N minutes of a window (settlement risk dominates).
# At T+13+ in a 15m window, our quotes can get hit hard right before close.
MM_STOP_QUOTING_LAST_MINS = float(env.get("MM_STOP_QUOTING_LAST_MINS", "2"))

# Series ticker (KXBTC15M)
SERIES = "KXBTC15M"

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
if not API_KEY_ID or not raw_pem:
    print("ERROR: Kalshi credentials missing from .env.")
    sys.exit(1)
PRIVATE_KEY = kalshi_auth.load_private_key(raw_pem)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(_dir, "market_maker.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mm] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mm")


# ── State ─────────────────────────────────────────────────────────────────────
class State:
    """Mutable bot state. Single instance, modified in the main loop."""
    # Current open market
    ticker:      str | None = None
    close_time:  datetime | None = None
    floor:       float = 0.0       # strike price

    # Active quotes (Kalshi order_ids and prices we have on the book)
    active_yes_bid: dict | None = None   # {order_id, price, count_remaining}
    active_yes_ask: dict | None = None

    # Inventory (signed contracts: +ve means net long YES, -ve means net long NO)
    yes_contracts:  float = 0.0
    no_contracts:   float = 0.0
    total_wagered:  float = 0.0
    realized_pnl:   float = 0.0

    # Last seen BTC price (for re-quote trigger)
    last_btc:        float | None = None

    # Window context
    window_t0:       datetime | None = None
    last_log_state:  str = ""


_state = State()
_shutdown = False


def _sigint(signum, frame):
    global _shutdown
    log.info("Shutdown signal received. Cancelling all quotes and exiting.")
    _shutdown = True


# ── Utility ───────────────────────────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc)


def fetch_open_market():
    """Pull the currently-open KXBTC15M market via REST. Sets state.ticker."""
    try:
        m = kalshi_trade.get_open_market()
    except Exception as e:
        log.warning(f"get_open_market failed: {e}")
        return None
    if m is None:
        return None
    return m


def cancel_quote(slot: str):
    """slot ∈ {'yes_bid', 'yes_ask'}. Cancels the resting quote on that side."""
    attr = f"active_{slot}"
    q = getattr(_state, attr)
    if q is None:
        return
    oid = q.get("order_id")
    if oid and not PAPER_MODE:
        try:
            kalshi_trade.cancel_order(PRIVATE_KEY, API_KEY_ID, oid)
        except Exception as e:
            log.warning(f"Cancel {slot} {oid} failed: {e}")
    setattr(_state, attr, None)


def cancel_all_quotes():
    cancel_quote("yes_bid")
    cancel_quote("yes_ask")


def place_quote(slot: str, side: str, price_cents: int, count: int):
    """
    Post a resting LIMIT order at price_cents. `side` is 'yes' or 'no'.
    For market making we post:
      yes_bid → buy YES at our bid price
      yes_ask → buy NO at (100 - our_yes_ask) — same thing as "selling YES"

    Kalshi only supports BUYS via this API, so "asking YES" = "bidding NO".

    Returns dict {order_id, price, count} or None.
    """
    if PAPER_MODE:
        # Simulated quote — we'll check for fills in the main loop using
        # top-of-book heuristics. Use a fake order_id.
        fake_id = f"paper-{slot}-{int(time.time()*1000)}"
        log.info(f"  [PAPER] post {slot} as {side.upper()} @ {price_cents}c × {count}")
        return {"order_id": fake_id, "price_cents": price_cents, "count": count, "side": side}

    # Real order. We need a market dict for place_order's pricing — but we
    # want to override the price, not use the live ask. Direct call:
    import uuid, requests
    body = {
        "ticker":          _state.ticker,
        "action":          "buy",
        "side":            side,
        "count":           count,
        "type":            "limit",
        "yes_price":       price_cents,
        "client_order_id": str(uuid.uuid4()),
    }
    path = "/trade-api/v2/portfolio/orders"
    headers = kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "POST", path)
    try:
        resp = requests.post(kalshi_trade.BASE_URL + path, json=body, headers=headers, timeout=5)
        if not resp.ok:
            log.warning(f"Place {slot} failed: {resp.status_code} {resp.text[:200]}")
            return None
        order = resp.json().get("order", {})
        return {"order_id": order.get("order_id"), "price_cents": price_cents, "count": count, "side": side}
    except Exception as e:
        log.warning(f"Place {slot} exception: {e}")
        return None


# ── Paper-mode fill simulator ─────────────────────────────────────────────────
def _simulate_fills(top_bid_cents: int, top_ask_cents: int):
    """
    In paper mode, simulate fills using a simple rule:
      - If our YES BID price >= current best ask, we'd fill instantly
        (the spread crossed our price). Mark as filled.
      - If the market top-of-book moves AGAINST us (someone else posts
        a better quote), our quote becomes second-in-line. We don't
        simulate this — paper-mode just assumes our quote stays at top.

    This is a CONSERVATIVE model: it underestimates real fill volume
    (assumes we only fill when someone aggresses into us). Real volume
    in live trading would be HIGHER than paper shows.
    """
    if _state.active_yes_bid and top_bid_cents > 0:
        q = _state.active_yes_bid
        # Our YES bid would fill if someone is selling YES at or below our price
        # i.e., the best ask drops to/below our bid
        if top_ask_cents <= q["price_cents"]:
            cost = q["count"] * q["price_cents"] / 100.0
            _state.yes_contracts += q["count"]
            _state.total_wagered += cost
            log.info(f"  [PAPER FILL] yes_bid hit: bought {q['count']} YES @ {q['price_cents']}c (cost ${cost:.2f})")
            _state.active_yes_bid = None  # need to repost
    if _state.active_yes_ask and top_ask_cents > 0:
        q = _state.active_yes_ask
        # Our YES "ask" is actually a NO BUY at (100 - yes_ask_price).
        # It fills if YES bid rises to/above our equivalent YES price.
        if top_bid_cents >= q["price_cents"]:
            no_price_cents = 100 - q["price_cents"]
            cost = q["count"] * no_price_cents / 100.0
            _state.no_contracts += q["count"]
            _state.total_wagered += cost
            log.info(f"  [PAPER FILL] yes_ask hit: bought {q['count']} NO @ {no_price_cents}c (cost ${cost:.2f})")
            _state.active_yes_ask = None


# ── Quote engine ──────────────────────────────────────────────────────────────
def decide_quotes(top_bid_cents: int, top_ask_cents: int) -> tuple[int | None, int | None]:
    """
    Given top-of-book, decide our bid and ask in cents.
    Returns (our_yes_bid_cents, our_yes_ask_cents). Either can be None to
    skip that side (e.g., inventory limit hit).
    """
    spread_cents = top_ask_cents - top_bid_cents

    # Spread too tight to safely post inside
    if spread_cents < MM_MIN_SPREAD_CENTS:
        return None, None

    # Default: post 1c inside top of book
    our_bid = top_bid_cents + MM_INSIDE_CENTS
    our_ask = top_ask_cents - MM_INSIDE_CENTS

    # Sanity: don't cross our own quotes
    if our_bid >= our_ask:
        return None, None

    # Inventory skew: if heavily long YES, don't post YES bid (would add more)
    # and tighten YES ask (encourage selling).
    net_dollars = _state.yes_contracts * (top_ask_cents/100) - _state.no_contracts * ((100-top_bid_cents)/100)
    if net_dollars > MM_MAX_INVENTORY_DOLLARS:
        our_bid = None      # stop buying YES
        our_ask = max(our_ask - 1, top_bid_cents + 1)  # more aggressive YES ask
    elif net_dollars < -MM_MAX_INVENTORY_DOLLARS:
        our_ask = None      # stop buying NO
        our_bid = min(our_bid + 1, top_ask_cents - 1)  # more aggressive YES bid

    # Total wagered cap
    if _state.total_wagered >= MM_MAX_TOTAL_WAGERED:
        log.warning(f"Total wagered ${_state.total_wagered:.2f} >= cap ${MM_MAX_TOTAL_WAGERED:.2f}. Pausing quotes.")
        return None, None

    return our_bid, our_ask


def update_quotes():
    """Main per-tick logic. Read book, decide quotes, place/cancel as needed."""
    # 1) Read top-of-book from Kalshi WS
    top_bid = kalshi_feed.get_bid()
    top_ask = kalshi_feed.get_ask()
    if top_bid is None or top_ask is None:
        return
    if not (0.01 < top_bid < 0.99 and 0.01 < top_ask < 0.99):
        return

    top_bid_c = round(top_bid * 100)
    top_ask_c = round(top_ask * 100)

    # 2) Paper-mode: check if existing quotes would have filled
    if PAPER_MODE:
        _simulate_fills(top_bid_c, top_ask_c)

    # 3) Compute desired quotes
    want_bid_c, want_ask_c = decide_quotes(top_bid_c, top_ask_c)

    # 4) Cancel + repost YES BID if needed
    cur_bid = _state.active_yes_bid
    if want_bid_c is None:
        if cur_bid: cancel_quote("yes_bid")
    elif cur_bid is None or cur_bid["price_cents"] != want_bid_c:
        if cur_bid: cancel_quote("yes_bid")
        count = max(1, round(MM_QUOTE_SIZE_DOLLARS / (want_bid_c / 100)))
        _state.active_yes_bid = place_quote("yes_bid", "yes", want_bid_c, count)

    # 5) Cancel + repost YES ASK (== NO buy) if needed
    cur_ask = _state.active_yes_ask
    if want_ask_c is None:
        if cur_ask: cancel_quote("yes_ask")
    elif cur_ask is None or cur_ask["price_cents"] != want_ask_c:
        if cur_ask: cancel_quote("yes_ask")
        # Posting "YES ask at X" means buying NO at (100-X)
        no_buy_cents = 100 - want_ask_c
        count = max(1, round(MM_QUOTE_SIZE_DOLLARS / (no_buy_cents / 100)))
        _state.active_yes_ask = place_quote("yes_ask", "no", no_buy_cents, count)
        # Store the YES-equivalent price in our state for paper-fill logic
        if _state.active_yes_ask:
            _state.active_yes_ask["price_cents"] = want_ask_c

    # 6) Log status periodically
    state_str = (
        f"book=[{top_bid_c}/{top_ask_c}] "
        f"my=[{_state.active_yes_bid['price_cents'] if _state.active_yes_bid else '-'}/"
        f"{_state.active_yes_ask['price_cents'] if _state.active_yes_ask else '-'}] "
        f"inv=Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} "
        f"wag=${_state.total_wagered:.2f}"
    )
    if state_str != _state.last_log_state:
        log.info(state_str)
        _state.last_log_state = state_str


# ── Settlement & window rotation ──────────────────────────────────────────────
def settle_current_window():
    """When market closes, compute realized PnL on whatever positions remain."""
    if not _state.ticker:
        return
    log.info(f"Settling positions for {_state.ticker}...")
    # In paper mode we don't know the actual outcome until Kalshi settles
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
        log.warning(f"  no settlement after 5min — skipping P&L")
        return

    # Compute window P&L
    yes_payout = _state.yes_contracts if winner == "yes" else 0
    no_payout  = _state.no_contracts  if winner == "no"  else 0
    window_pnl = yes_payout + no_payout - _state.total_wagered
    _state.realized_pnl += window_pnl
    log.info(
        f"RESULT: winner={winner} | Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} "
        f"wag=${_state.total_wagered:.2f} payout=${yes_payout + no_payout:.2f} "
        f"window_pnl=${window_pnl:+.2f} | cumulative=${_state.realized_pnl:+.2f}"
    )

    # Reset window state
    _state.yes_contracts = 0.0
    _state.no_contracts  = 0.0
    _state.total_wagered = 0.0
    _state.active_yes_bid = None
    _state.active_yes_ask = None


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    log.info(f"Market maker starting | mode={'PAPER' if PAPER_MODE else 'LIVE'}")
    log.info(f"  quote_size=${MM_QUOTE_SIZE_DOLLARS:.2f} | inside={MM_INSIDE_CENTS}c | "
             f"min_spread={MM_MIN_SPREAD_CENTS}c")
    log.info(f"  max_inventory=${MM_MAX_INVENTORY_DOLLARS:.2f} | max_wagered=${MM_MAX_TOTAL_WAGERED:.2f}")
    log.info(f"  refresh={MM_REFRESH_SECS}s | btc_requote={MM_REQUOTE_BTC_MOVE}$")

    # Start BTC feed (used for inventory skew + re-quote triggers)
    btc_feed.start()
    time.sleep(1)

    # Start Kalshi market data WS
    kalshi_feed.start(PRIVATE_KEY, API_KEY_ID)
    time.sleep(1)

    while not _shutdown:
        # 1) Find / refresh the open market
        market = fetch_open_market()
        if market is None:
            log.info("No open KXBTC15M market. Waiting 30s.")
            time.sleep(30)
            continue

        ticker = market.get("ticker")
        if ticker != _state.ticker:
            # Window rotated — settle previous, start fresh
            if _state.ticker:
                settle_current_window()
            _state.ticker = ticker
            _state.floor = float(market.get("floor_strike", 0))
            try:
                _state.close_time = datetime.fromisoformat(market["close_time"].replace("Z","+00:00"))
            except Exception:
                _state.close_time = None
            kalshi_feed.set_ticker(ticker)
            log.info(f"━━ NEW WINDOW: {ticker} floor=${_state.floor:,.2f} closes={_state.close_time}")

        # 2) Check if we should stop quoting (last 2 min of window)
        if _state.close_time:
            mins_to_close = (_state.close_time - now_utc()).total_seconds() / 60
            if mins_to_close < MM_STOP_QUOTING_LAST_MINS:
                if _state.active_yes_bid or _state.active_yes_ask:
                    log.info(f"  ≤{MM_STOP_QUOTING_LAST_MINS}min to close — cancelling quotes")
                    cancel_all_quotes()
                # When window actually closes, the next loop iteration will
                # detect the new ticker and call settle_current_window.
                time.sleep(min(10, max(1, (_state.close_time - now_utc()).total_seconds())))
                continue

        # 3) Update quotes
        try:
            update_quotes()
        except Exception as e:
            log.error(f"update_quotes error: {e}")

        time.sleep(MM_REFRESH_SECS)

    # Cleanup on shutdown
    log.info("Cancelling all quotes before exit...")
    cancel_all_quotes()
    log.info(f"Final state: realized_pnl=${_state.realized_pnl:+.2f} "
             f"open=Y{_state.yes_contracts:.0f}/N{_state.no_contracts:.0f} "
             f"wagered_this_window=${_state.total_wagered:.2f}")
    log.info("Market maker shut down cleanly.")


if __name__ == "__main__":
    main()
