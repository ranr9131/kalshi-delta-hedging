#!/usr/bin/env python3.11
"""
LIVE market maker — real money, tiny caps. SOL-hourly only by default.

Reuses conv_maker_shadow's brains unchanged (v3 fair value, Asian-window
repricing/withdrawal, one-sided margin quoting, inventory skew) and replaces
the paper fill simulation with real V2 resting orders.

Risk rails (all env-overridable):
  LIVE_MM_SERIES        default KXSOLD      (SOL hourly only)
  LIVE_MM_SIZE          default 10          contracts per quote
  LIVE_MM_MAX_WORST     default 25          $ worst-case loss per market
  LIVE_MM_TOTAL_WORST   default 150         $ worst-case loss across all markets
  LIVE_MM_DAILY_LOSS    default 25          $ realized loss -> halt for UTC day
  LIVE_MM_BALANCE_FLOOR default 60          $ abort if account below this
  kill file: live/STOP_LIVE_MM              cancel-all + exit within one tick

Hard-won API facts honored here (from the LIP pre-flight, 6/10):
  - live order records only populate the fixed-point/_dollars fields
    (kalshi_trade._normalize_v2_order_response handles this);
  - Kalshi 429s order WRITES bursting >1/s -> single global 1/s write bucket
    with backoff;
  - the resting-orders index lags placements 1-2s -> never wholesale-replace
    local state from a remote listing; trust our own order ids.

Startup preflight (also from the LIP lessons): balance check, then one
1-contract bid at 2c placed, status-read, and cancelled. Any failure aborts.
"""
import os
import sys
import csv
import time
import signal
import logging
import threading
from datetime import datetime, timezone

# series must be set BEFORE importing the shadow (it reads env at import)
SERIES = os.environ.get("LIVE_MM_SERIES", "KXSOLD")
os.environ["CONV_MM_SERIES"] = SERIES
os.environ.setdefault("CONV_MM_MIN_SPREAD", "3")

import conv_maker_shadow as mm            # brains: fair, asian, quotes, skew
import kalshi_orderbook
import kalshi_trade
from kalshi_auth import make_auth_headers
import uuid
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))

QUOTE_SIZE    = float(os.environ.get("LIVE_MM_SIZE", "10"))
MAX_WORST     = float(os.environ.get("LIVE_MM_MAX_WORST", "25"))
TOTAL_WORST   = float(os.environ.get("LIVE_MM_TOTAL_WORST", "150"))
DAILY_LOSS    = float(os.environ.get("LIVE_MM_DAILY_LOSS", "25"))
BALANCE_FLOOR = float(os.environ.get("LIVE_MM_BALANCE_FLOOR", "60"))
KILL_FILE     = os.path.join(ROOT, "STOP_LIVE_MM")
TICK_SEC      = 1.0
REQUOTE_DIFF_C = 2        # move order only when desired price differs >= 2c
MIN_REST_SEC  = float(os.environ.get("LIVE_MM_MIN_REST", "8"))
MAX_FAIR_MID_DIVERGE_C = float(os.environ.get("LIVE_MM_MAX_DIVERGE", "20"))
# ^ if |model fair - book mid| exceeds this, DON'T QUOTE (keep safe resting
#   orders). Every prior experiment says huge model-market disagreement means
#   the market is right (conv-taker anti-signal; v3 hourly fair measured 30c
#   low near expiry on 7/14) — a maker must not price off a broken opinion.
# ^ a resting order stays put at least this long unless it's about to cross —
#   chasing the book every tick burns the 1/s write budget, resets queue
#   position, and feeds undercutting wars (observed on first live session)
ORDER_POLL_SEC = 3.0
PNL_CSV = os.path.join(ROOT, "conv_maker_live_pnl.csv")
FILL_CSV = os.path.join(ROOT, "conv_maker_live_fills.csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  LIVE-MM  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("livemm")

PRIVATE_KEY = mm.PRIVATE_KEY
API_KEY_ID = mm.API_KEY_ID
if PRIVATE_KEY is None:
    print("ERROR: no KALSHI_PRIVATE_KEY in .env — refusing to start live maker")
    sys.exit(1)

_running = True


def _sig(_s, _f):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)

# ── 1/s global write bucket ────────────────────────────────────────────────────
_last_write = [0.0]
_write_lock = threading.Lock()


def _write_gate():
    with _write_lock:
        wait = _last_write[0] + 1.0 - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_write[0] = time.time()


def _find_order_by_coid(ticker: str, coid: str) -> dict | None:
    """Look up an order by client_order_id — recovery when the POST's
    response was lost in transit but the exchange accepted the write."""
    path = "/trade-api/v2/portfolio/orders"
    try:
        h = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path)
        r = requests.get(kalshi_trade.BASE_URL + path,
                         params={"ticker": ticker, "limit": 100},
                         headers=h, timeout=10)
        r.raise_for_status()
        for o in r.json().get("orders") or []:
            if o.get("client_order_id") == coid:
                return o
    except Exception:
        pass
    return None


def place_limit(ticker: str, yes_price_c: int, count: float, buy_yes: bool) -> dict:
    """Rest a GTC limit on the single YES book at an EXACT yes-denominated
    price. buy_yes=True -> side bid (buy YES); False -> side ask (buy NO at
    100-p, i.e. sell YES). On a transport error after send, checks whether
    the exchange accepted the order anyway (adopt-or-raise) — a lost response
    must never leave an untracked live order."""
    _write_gate()
    yes_price_c = max(1, min(99, int(yes_price_c)))
    coid = str(uuid.uuid4())
    path = kalshi_trade.V2_ORDERS_PATH
    body = {
        "ticker": ticker,
        "side": "bid" if buy_yes else "ask",
        "count": f"{count:.2f}",
        "price": f"{yes_price_c / 100.0:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": coid,
    }
    headers = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "POST", path)
    try:
        r = requests.post(kalshi_trade.BASE_URL + path, json=body,
                          headers=headers, timeout=10)
    except (requests.Timeout, requests.ConnectionError) as e:
        time.sleep(2.0)
        adopted = _find_order_by_coid(ticker, coid)
        if adopted:
            log.warning(f"POST response lost but order accepted — adopted {coid[:8]}")
            return adopted
        raise RuntimeError(f"order POST failed, not found by coid: {e}")
    if r.status_code == 429:
        log.warning("429 on order write — backing off 5s")
        time.sleep(5)
        raise RuntimeError("rate limited")
    r.raise_for_status()
    rec = kalshi_trade._normalize_v2_order_response(r.json(), count)
    return rec.get("order", rec)   # unwrap the V1-ish {"order": {...}} shape


def cancel(order_id: str) -> bool:
    _write_gate()
    return kalshi_trade.cancel_order(PRIVATE_KEY, API_KEY_ID, order_id)


# ── state ──────────────────────────────────────────────────────────────────────
# _orders[ticker] = {"bid": rec|None, "ask": rec|None}
#   rec = {id, price_c, count, filled}  (filled = contracts seen filled so far)
_orders: dict = {}
# _book[ticker] = {"pos": net YES contracts, "cash_c": +received/-paid cents,
#                  "fees_c": est maker fees}
_book: dict = {}
_daily = {"date": None, "realized_c": 0.0, "halted": False}
_settle_cache: dict = {}


def _bk(t):
    return _book.setdefault(t, {"pos": 0.0, "cash_c": 0.0, "fees_c": 0.0})


def _apply_fill(ticker, rec, newly: float, buy_yes: bool):
    b = _bk(ticker)
    px = rec["price_c"]
    if buy_yes:
        b["pos"] += newly
        b["cash_c"] -= px * newly
    else:
        b["pos"] -= newly
        b["cash_c"] += px * newly
    b["fees_c"] += mm.maker_fee_c(px / 100.0) * newly
    log.info(f"FILL {ticker} {'BUY-YES' if buy_yes else 'SELL-YES'} {newly:.0f} @ {px}c "
             f"-> pos={b['pos']:+.0f}")
    _append(FILL_CSV, ["ts", "ticker", "action", "price_c", "count", "pos_after"],
            {"ts": round(time.time(), 1), "ticker": ticker,
             "action": "buy_yes" if buy_yes else "sell_yes",
             "price_c": px, "count": newly, "pos_after": b["pos"]})


def _append(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def _num(o: dict, *keys, default=0.0) -> float:
    """Read a numeric field that may only exist in fixed-point/_fp/_dollars
    form on raw V2 order records (integer fields come back null)."""
    for k in keys:
        v = o.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def worst_case_c(ticker) -> float:
    """Worst-case additional loss (cents) from current position + resting
    orders in this market, over both settlement outcomes."""
    b = _bk(ticker)
    o = _orders.get(ticker, {})
    def _pnl(outcome, extra_pos=0.0, extra_cash=0.0):
        pos = b["pos"] + extra_pos
        cash = b["cash_c"] + extra_cash
        return cash + pos * (100.0 if outcome else 0.0) - b["fees_c"]
    live_recs = [(s, r) for s, r in (o or {}).items() if r]
    live_recs += [(z["side"], z["rec"]) for z in _zombies if z["ticker"] == ticker]
    worst = 0.0
    for outcome in (0, 1):
        # resting (incl. zombie) orders that would fill against us there
        for side, rec in live_recs:
            rem = rec["count"] - rec["filled"]
            if side == "bid":
                ep, ec = rem, -rec["price_c"] * rem
            else:
                ep, ec = -rem, rec["price_c"] * rem
            worst = min(worst, _pnl(outcome, ep, ec))
        worst = min(worst, _pnl(outcome))
    return -worst  # positive number = $ at risk (in cents)


def total_worst_c() -> float:
    tickers = set(list(_book) + list(_orders)) | {z["ticker"] for z in _zombies}
    return sum(worst_case_c(t) for t in tickers)


_zombies: list = []   # orders released but not yet CONFIRMED terminal


def _release(ticker, side, rec):
    """Cancel an order and CAPTURE any fills that landed since the last poll.
    An order is only forgotten once a TERMINAL state is confirmed; otherwise
    it becomes a zombie that _poll_zombies keeps cancelling/polling and that
    worst_case_c keeps counting. (Cancel-and-forget lost 5 of 6 fills on the
    first live session; an unconfirmed cancel is a live order.)"""
    ok = cancel(rec["id"])
    time.sleep(1.0)
    try:
        st = kalshi_trade.get_order_status(PRIVATE_KEY, API_KEY_ID, rec["id"])
        filled = _num(st, "fill_count", "fill_count_fp")
        if filled > rec["filled"] + 1e-9:
            _apply_fill(ticker, rec, filled - rec["filled"], side == "bid")
            rec["filled"] = filled
        status = (st.get("status") or "").lower()
        remaining = _num(st, "remaining_count", "remaining_count_fp", default=1)
        if status in ("executed", "canceled", "cancelled") or remaining <= 1e-9:
            return                              # confirmed terminal
    except Exception:
        pass
    log.warning(f"release of {rec['id'][:8]} unconfirmed (cancel={ok}) — zombie")
    _zombies.append({"ticker": ticker, "side": side, "rec": rec, "tries": 0})


def _poll_zombies():
    for z in list(_zombies):
        z["tries"] += 1
        rec = z["rec"]
        if z["tries"] % 3 == 0:
            cancel(rec["id"])                   # keep trying to kill it
        try:
            st = kalshi_trade.get_order_status(PRIVATE_KEY, API_KEY_ID, rec["id"])
        except Exception:
            continue
        filled = _num(st, "fill_count", "fill_count_fp")
        if filled > rec["filled"] + 1e-9:
            _apply_fill(z["ticker"], rec, filled - rec["filled"], z["side"] == "bid")
            rec["filled"] = filled
        status = (st.get("status") or "").lower()
        remaining = _num(st, "remaining_count", "remaining_count_fp", default=1)
        if status in ("executed", "canceled", "cancelled") or remaining <= 1e-9:
            _zombies.remove(z)


def reconcile_positions():
    """Seed _book from the exchange's own positions for our series — a restart
    must never forget inventory."""
    path = "/trade-api/v2/portfolio/positions"
    try:
        h = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path)
        r = requests.get(kalshi_trade.BASE_URL + path, params={"limit": 200},
                         headers=h, timeout=10)
        r.raise_for_status()
        items = r.json().get("market_positions") or []
    except Exception as e:
        log.warning(f"position reconcile failed: {e}")
        return
    prefixes = tuple(s.strip() for s in SERIES.split(","))
    for it in items:
        tk = it.get("ticker", "")
        if not tk.startswith(prefixes):
            continue
        pos = _num(it, "position", "position_fp")
        expo_c = _num(it, "market_exposure_dollars", "market_exposure",
                      "market_exposure_fp") * 100.0
        if abs(pos) < 1e-9:
            continue
        if expo_c <= 0:
            raise RuntimeError(f"reconcile: {tk} has pos={pos} but exposure "
                               f"parsed as 0 — field name drift? refusing to "
                               f"start with unpriced inventory")
        b = _bk(tk)
        b["pos"] = pos
        # cash basis such that worst-case loss == exchange-reported exposure
        b["cash_c"] = (abs(pos) * 100.0 - expo_c) if pos < 0 else -expo_c
        log.info(f"reconciled {tk}: pos={pos:+.2f} exposure=${expo_c/100:.2f}")


def _poll_orders():
    """Poll our resting orders for fills; detect via fill_count deltas."""
    for ticker, sides in list(_orders.items()):
        for side, rec in list(sides.items()):
            if not rec:
                continue
            try:
                st = kalshi_trade.get_order_status(PRIVATE_KEY, API_KEY_ID, rec["id"])
            except Exception:
                continue
            filled = _num(st, "fill_count", "fill_count_fp")
            if filled > rec["filled"] + 1e-9:
                _apply_fill(ticker, rec, filled - rec["filled"], side == "bid")
                rec["filled"] = filled
            status = (st.get("status") or "").lower()
            remaining = _num(st, "remaining_count", "remaining_count_fp",
                             default=rec["count"] - filled)
            if status in ("executed", "canceled", "cancelled") or \
                    (remaining <= 1e-9 and filled > 0):
                sides[side] = None


def _cancel_all(reason=""):
    n = 0
    for ticker, sides in list(_orders.items()):
        for side, rec in list(sides.items()):
            if rec:
                _release(ticker, side, rec)
                sides[side] = None
                n += 1
    if n:
        log.info(f"cancelled {n} resting orders {reason}")


def _settle_closed():
    """Settle P&L for markets no longer tracked, add to daily realized."""
    with mm._tracked_lock:
        tracked = set(mm._tracked.keys())
    for ticker in list(_book.keys()):
        if ticker in tracked:
            continue
        b = _book[ticker]
        o = _orders.get(ticker, {})
        if any(o.values()):
            continue                     # still have resting orders; cancel first
        y = mm._settle(ticker, _settle_cache)
        if y is None:
            continue                     # not settled yet — check next tick
        pnl_c = b["cash_c"] + b["pos"] * 100.0 * y - b["fees_c"]
        _daily["realized_c"] += pnl_c
        _save_daily()
        log.info(f"SETTLED {ticker} result={'yes' if y else 'no'} pos={b['pos']:+.0f} "
                 f"pnl=${pnl_c/100:+.2f} | day ${_daily['realized_c']/100:+.2f}")
        _append(PNL_CSV, ["ts", "ticker", "result", "pos", "pnl_usd"],
                {"ts": round(time.time(), 1), "ticker": ticker,
                 "result": "yes" if y else "no", "pos": b["pos"],
                 "pnl_usd": round(pnl_c / 100, 2)})
        del _book[ticker]
        _orders.pop(ticker, None)


DAILY_STATE = os.path.join(ROOT, "conv_maker_live_daily.json")


def _save_daily():
    import json
    try:
        with open(DAILY_STATE, "w") as fh:
            json.dump({"date": str(_daily["date"]), "realized_c": _daily["realized_c"],
                       "halted": _daily["halted"]}, fh)
    except Exception:
        pass


def _load_daily():
    import json
    try:
        with open(DAILY_STATE) as fh:
            d = json.load(fh)
        if d.get("date") == str(datetime.now(timezone.utc).date()):
            _daily.update({"date": datetime.now(timezone.utc).date(),
                           "realized_c": float(d.get("realized_c", 0)),
                           "halted": bool(d.get("halted"))})
            log.info(f"restored daily state: ${_daily['realized_c']/100:+.2f} "
                     f"halted={_daily['halted']}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"daily state load failed: {e}")


def _roll_day():
    today = datetime.now(timezone.utc).date()
    if _daily["date"] != today:
        open_risk = any(abs(b["pos"]) > 1e-9 for b in _book.values()) or \
                    any(any(s.values()) for s in _orders.values()) or _zombies
        if _daily["date"] is not None and open_risk and _daily["realized_c"] < 0:
            # losses straddling midnight must not get a fresh cap — carry
            # the negative realized until we're flat
            log.info(f"UTC day roll with open risk: carrying "
                     f"${_daily['realized_c']/100:+.2f} into {today}")
            _daily["date"] = today
        else:
            _daily.update({"date": today, "realized_c": 0.0, "halted": False})
        _save_daily()


def acquire_singleton_lock() -> bool:
    """Refuse to start if another live maker is running (flock on a pidfile).
    Two instances double exposure and race each other's orders — happened
    once with nohup on 7/14."""
    import fcntl
    global _lock_fh
    _lock_fh = open(os.path.join(ROOT, ".conv_maker_live.lock"), "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(str(os.getpid())); _lock_fh.flush()
        return True
    except OSError:
        return False


def cancel_orphan_orders():
    """Cancel ALL resting orders in our series at startup — a crashed prior
    instance leaves orders the new instance can't see or risk-count."""
    path = "/trade-api/v2/portfolio/orders"
    try:
        h = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path)
        r = requests.get(kalshi_trade.BASE_URL + path,
                         params={"status": "resting", "limit": 200},
                         headers=h, timeout=10)
        r.raise_for_status()
        orders = r.json().get("orders") or []
    except Exception as e:
        log.error(f"orphan sweep failed to list orders: {e} — refusing to start")
        raise
    prefixes = tuple(s.strip() for s in SERIES.split(","))
    n = 0
    for o in orders:
        if (o.get("ticker") or "").startswith(prefixes):
            if cancel(o["order_id"]):
                n += 1
    log.info(f"orphan sweep: cancelled {n} resting orders in {prefixes}")


PREFLIGHT_STAMP = os.path.join(ROOT, ".conv_maker_live.preflight")
PREFLIGHT_TTL = 48 * 3600


def preflight() -> bool:
    bal = kalshi_trade.get_balance(PRIVATE_KEY, API_KEY_ID)
    log.info(f"balance: ${bal}")
    if bal is None or bal < BALANCE_FLOOR:
        log.error(f"balance below floor (${BALANCE_FLOOR}) — abort")
        return False
    # the order-plumbing test costs ~1c when it crosses a bot's 1c ask on a
    # dead strike — don't re-prove plumbing that passed recently
    try:
        if time.time() - os.path.getmtime(PREFLIGHT_STAMP) < PREFLIGHT_TTL:
            log.info("preflight: order test passed recently — skipping")
            return True
    except OSError:
        pass
    ticker = None
    cands = mm.refresh_candidates()
    if cands:
        ticker = sorted(cands)[0]
    else:
        # nothing inside the 20-min tracking window — any open market in the
        # series works for an order-plumbing test
        try:
            r = mm._get("/trade-api/v2/markets",
                        {"series_ticker": SERIES.split(",")[0].strip(),
                         "status": "open", "limit": 20})
            ms = r.json().get("markets", []) if r.ok else []
            if not ms:
                log.warning(f"preflight fetch: HTTP {r.status_code}, {len(ms)} markets")
            # pick one where a 1c bid RESTS (ask well above it) instead of
            # crossing a deep-OTM 1c ask like the first attempt did
            for m in ms:
                ask = float(m.get("yes_ask_dollars") or 0)
                # a 1c bid can only cross a 1c ask — anything else rests
                if ask == 0 or ask >= 0.02:
                    ticker = m["ticker"]
                    break
            if not ticker and ms:
                # every listed market has a 1c ask (bots blanket the OTM
                # ladder) — accept a 1c instant fill; preflight treats an
                # executed test order as a pass too
                ticker = ms[0]["ticker"]
        except Exception as e:
            log.error(f"preflight market fetch failed: {e}")
    if not ticker:
        log.error("no open market found for preflight — abort")
        return False
    log.info(f"preflight: 1-contract bid @1c on {ticker}")
    try:
        rec = place_limit(ticker, 1, 1, buy_yes=True)
    except Exception as e:
        log.error(f"preflight place failed: {e}")
        return False
    oid = rec.get("order_id") or rec.get("id")
    if not oid:
        log.error(f"preflight: no order_id in response: {rec}")
        return False
    try:
        # the orders index lags placements 1-2s (LIP lesson) — retry the read
        st = None
        for attempt in range(4):
            time.sleep(1.5)
            try:
                st = kalshi_trade.get_order_status(PRIVATE_KEY, API_KEY_ID, oid)
                break
            except Exception as e:
                log.info(f"preflight status read retry {attempt+1}: {e}")
        if st is None:
            log.error("preflight: order never appeared in index")
            return False
        log.info(f"preflight raw status keys: { {k: v for k, v in st.items() if 'count' in k or k == 'status'} }")
        rem = _num(st, "remaining_count", "remaining_count_fp", default=-1)
        filled = _num(st, "fill_count", "fill_count_fp")
        log.info(f"preflight status: {st.get('status')} remaining={rem} filled={filled}")
        # resting(rem=1) proves place+read; an instant 1c fill (bots rest 1c
        # asks on dead strikes) proves place+read+fill — both are a pass
        if rem == 1 or filled == 1:
            with open(PREFLIGHT_STAMP, "w") as fh:
                fh.write(str(time.time()))
            return True
        log.error("preflight: count fields parse mismatch — check fp scaling")
        return False
    finally:
        ok = cancel(oid)
        log.info(f"preflight cancel: {ok}")


def main():
    log.info(f"LIVE maker starting | series={SERIES} size={QUOTE_SIZE} "
             f"caps: mkt=${MAX_WORST} total=${TOTAL_WORST} daily=${DAILY_LOSS} "
             f"kill={KILL_FILE}")
    if not acquire_singleton_lock():
        log.error("another conv_maker_live instance holds the lock — exiting")
        sys.exit(1)
    if not preflight():
        sys.exit(1)
    cancel_orphan_orders()
    reconcile_positions()
    _load_daily()

    cands = mm.refresh_candidates()
    with mm._tracked_lock:
        mm._tracked.update(cands)
    mm._ensure_feeds()
    kalshi_orderbook.start(PRIVATE_KEY, API_KEY_ID, list(cands.keys()))
    threading.Thread(target=mm._refresh_thread, daemon=True).start()
    time.sleep(3)

    last_poll = 0.0
    try:
      while _running:
        t0 = time.time()
        _roll_day()
        if os.path.exists(KILL_FILE):
            log.warning("kill file present — cancelling all and exiting")
            _cancel_all("(kill file)")
            break
        if time.time() - last_poll >= ORDER_POLL_SEC:
            _poll_orders()
            _poll_zombies()
            _settle_closed()
            last_poll = time.time()
        if _daily["realized_c"] <= -DAILY_LOSS * 100 and not _daily["halted"]:
            log.warning(f"DAILY LOSS CAP hit (${_daily['realized_c']/100:+.2f}) — "
                        f"cancelling all, halting until UTC midnight")
            _cancel_all("(daily cap)")
            _daily["halted"] = True
        if _daily["halted"]:
            time.sleep(5)
            continue

        with mm._tracked_lock:
            items = list(mm._tracked.items())
        for ticker, meta in items:
            sides = _orders.setdefault(ticker, {"bid": None, "ask": None})
            sp = mm._spot(meta["asset"])
            ttc = (meta["close_dt"] - datetime.now(timezone.utc)).total_seconds()
            book = kalshi_orderbook.get_book(ticker)
            bb = book.yes_bid() if book else None
            ba = book.yes_ask() if book else None
            if ttc <= 5 or not sp or bb is None or ba is None:
                for s in ("bid", "ask"):
                    if sides[s]:
                        _release(ticker, s, sides[s]); sides[s] = None
                continue
            bb_c, ba_c = round(bb * 100), round(ba * 100)
            mid = (bb_c + ba_c) / 200.0
            gates_ok = (ba_c - bb_c >= mm.MIN_SPREAD_C and mm.MID_LO <= mid <= mm.MID_HI)
            fair_c, withdraw_all = None, False
            if gates_ok:
                if ttc <= mm.ASIAN_ONLY_SEC:
                    fair_c = mm._asian_fair_c(meta)
                    if fair_c is None:
                        withdraw_all = True   # final window, unpriceable: go dark
                else:
                    fair_c = mm._fair_c(meta, sp, ttc)
                # divergence stand-down only where mid is CREDIBLE (tight-ish
                # book) — the mid of a 10/90 book carries no information and
                # must not block quoting the empty books we exist to serve
                if (fair_c is not None and ba_c - bb_c <= 15
                        and abs(fair_c - mid * 100) > MAX_FAIR_MID_DIVERGE_C):
                    fair_c = None
            if fair_c is None:
                # gate flicker (spread bouncing 2c<->3c) must NOT churn quotes:
                # keep safe resting orders, pull only ones about to cross —
                # unless the asian pricer refused (withdraw_all)
                for s in ("bid", "ask"):
                    rec = sides[s]
                    if not rec:
                        continue
                    crossing = (s == "bid" and rec["price_c"] >= ba_c) or \
                               (s == "ask" and rec["price_c"] <= bb_c)
                    if withdraw_all or crossing:
                        _release(ticker, s, rec); sides[s] = None
                continue

            wb, wa = mm.desired_quotes(fair_c, bb_c, ba_c, _bk(ticker)["pos"])
            # never cross the live book
            if wb is not None and wb >= ba_c:
                wb = None
            if wa is not None and wa <= bb_c:
                wa = None
            for side, want, buy_yes in (("bid", wb, True), ("ask", wa, False)):
                cur = sides[side]
                if want is None:
                    if cur:
                        _release(ticker, side, cur); sides[side] = None
                    continue
                if cur:
                    would_cross = (side == "bid" and cur["price_c"] >= ba_c) or \
                                  (side == "ask" and cur["price_c"] <= bb_c)
                    age = time.time() - cur.get("placed_ts", 0)
                    if not would_cross and (abs(cur["price_c"] - want) < REQUOTE_DIFF_C
                                            or age < MIN_REST_SEC):
                        continue
                    _release(ticker, side, cur); sides[side] = None
                # room check AFTER any release, so replacements are gated too
                room_mkt = MAX_WORST * 100 - worst_case_c(ticker)
                room_tot = TOTAL_WORST * 100 - total_worst_c()
                add_c = want * QUOTE_SIZE if buy_yes else (100 - want) * QUOTE_SIZE
                if add_c > room_mkt or add_c > room_tot:
                    continue            # would exceed worst-case caps
                try:
                    rec = place_limit(ticker, int(want), QUOTE_SIZE, buy_yes)
                except Exception as e:
                    log.warning(f"place failed {ticker} {side}@{want}: {e}")
                    continue
                oid = rec.get("order_id") or rec.get("id")
                filled_now = _num(rec, "fill_count", "fill_count_fp")
                sides[side] = {"id": oid, "price_c": int(want),
                               "count": QUOTE_SIZE, "filled": 0.0,
                               "placed_ts": time.time()}
                if filled_now:
                    # instant (taker-at-placement) fills execute at the book's
                    # price, not necessarily our limit — book them accurately
                    afp = _num(rec, "average_fill_price", "average_fill_price_fp")
                    tmp = dict(sides[side])
                    if afp > 0:
                        tmp["price_c"] = round(afp * 100)
                    _apply_fill(ticker, tmp, filled_now, buy_yes)
                    sides[side]["filled"] = filled_now
                log.info(f"QUOTE {ticker} {side}={want}c x{QUOTE_SIZE:.0f} "
                         f"(book {bb_c}/{ba_c} fair {fair_c:.1f} ttc {ttc:.0f}s)")

        time.sleep(max(0.0, TICK_SEC - (time.time() - t0)))
    except Exception:
        import traceback
        log.error("CRASH in main loop:\n" + traceback.format_exc())
        raise
    finally:
        # NEVER exit with quotes resting — crash, kill, or clean shutdown
        _cancel_all("(shutdown/crash)")
        _poll_zombies()
        log.info("live maker exited")


if __name__ == "__main__":
    main()
