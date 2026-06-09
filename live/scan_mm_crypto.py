"""
Crypto market-making opportunity scanner for Kalshi.

Step 1 of the low-volume-coin MM exploration (see STRATEGY_LOG.md §"MM on thin
coin markets"). Read-only — places no orders.

What it does
------------
Pages through every open Kalshi market once (the bulk /markets list already
returns the `*_dollars` / `*_fp` quote fields), keeps the crypto series, and
for each two-sided market computes a **fee-aware, volume-weighted MM score**.

The core lesson from the first manual probe: the *widest* spreads (e.g. KXXRP
brackets at 33-34c) have ZERO volume — they're traps, not opportunities. So we
score spread AND flow together, and a market with no recent volume scores zero.

MM economics modelled here (maker round-trip, both legs resting):
  gross capture / contract = spread                       ($ = ask - bid)
  maker fee  / side        = 0.25 * 0.07 * P * (1-P)      (≈25% of taker fee)
  net capture / round-trip = spread - 2 * maker_fee(mid)
  optimistic $/day         = net_capture * (volume_24h * CAPTURE_FRACTION)

CAPTURE_FRACTION is the share of daily flow we assume our resting quote earns.
This is the OPTIMISTIC bound — it assumes zero adverse selection (every fill is
uninformed). Step 2 (the passive-fill shadow logger) measures the real number.
Treat the $/day column as a ceiling, not a forecast.

Run:  python3 scan_mm_crypto.py
      python3 scan_mm_crypto.py --coin XRP        # focus one coin
      python3 scan_mm_crypto.py --min-vol 50      # require more flow
"""
import os
import sys
import math
import time
import argparse

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import dotenv_values
import kalshi_auth

_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))
API_KEY_ID  = env.get("KALSHI_API_KEY_ID", "")
raw_pem     = env.get("KALSHI_PRIVATE_KEY", "")
PRIVATE_KEY = kalshi_auth.load_private_key(raw_pem) if raw_pem else None

BASE_URL = "https://api.elections.kalshi.com"

# Coin symbols we treat as "crypto". Matched against the series prefix of each
# ticker (the part before the first '-'), e.g. KXXRPD -> XRP, KXBTC15M -> BTC.
COIN_SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE", "ADA", "TON",
    "LTC", "AVAX", "LINK", "DOT", "MATIC", "TRX", "SHIB", "PEPE", "BCH",
    "KAS", "SUI", "APT", "ARB", "OP", "NEAR", "ATOM", "FIL", "ETC",
]

# Maker fee ≈ 25% of the taker fee. Taker = 0.07 * C * (1-C) per contract.
# VERIFY against your real fills before sizing on this — the official PDF is
# behind a bot-wall and this is from secondary sources (June 2026).
MAKER_FEE_FRACTION = 0.25
TAKER_FEE_RATE     = 0.07

# Optimistic share of daily flow a resting quote captures (zero adverse
# selection assumed). The $/day column is a CEILING, not a forecast.
CAPTURE_FRACTION = 0.30


def maker_fee_per_contract(price_dollars: float) -> float:
    """Maker fee in $ for one contract at the given price (0..1)."""
    p = max(0.0, min(1.0, price_dollars))
    return MAKER_FEE_FRACTION * TAKER_FEE_RATE * p * (1.0 - p)


def coin_of(ticker: str) -> str | None:
    """Return the coin symbol for a ticker, or None if not crypto."""
    prefix = ticker.split("-")[0].upper()       # e.g. KXXRPD
    body = prefix[2:] if prefix.startswith("KX") else prefix
    # longest match first so ETH doesn't shadow ETC, BTC before generic, etc.
    for sym in sorted(COIN_SYMBOLS, key=len, reverse=True):
        if body.startswith(sym):
            return sym
    return None


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# IMPORTANT: the bulk /markets list returns quote fields as "0.0000" — live
# quotes only populate when the query is filtered by series_ticker. So we
# enumerate crypto series and page through each one individually.
SERIES_SUFFIXES = ["", "D", "15M", "H", "W", "M", "Y", "HOURLY", "DAILY", "MAX", "RANGE"]


def _get(path: str, params: dict):
    headers = (kalshi_auth.make_auth_headers(PRIVATE_KEY, API_KEY_ID, "GET", path)
               if PRIVATE_KEY else None)
    return requests.get(BASE_URL + path, params=params, headers=headers, timeout=20)


def discover_crypto_series() -> list[str]:
    """Probe coin × suffix combos, keep series that have ≥1 open market."""
    found = []
    for sym in COIN_SYMBOLS:
        for suf in SERIES_SUFFIXES:
            st = f"KX{sym}{suf}"
            try:
                r = _get("/trade-api/v2/markets",
                         {"series_ticker": st, "status": "open", "limit": 1})
                if r.status_code == 200 and r.json().get("markets"):
                    found.append(st)
            except Exception:
                pass
            time.sleep(0.02)
    return found


def fetch_crypto_markets(series_list: list[str]) -> list:
    out = []
    for st in series_list:
        cursor, page = "", 0
        while page < 20:
            params = {"series_ticker": st, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = _get("/trade-api/v2/markets", params)
                resp.raise_for_status()
            except Exception as e:
                print(f"{st} page {page} error: {e}")
                break
            data = resp.json()
            markets = data.get("markets", [])
            out.extend(markets)
            cursor = data.get("cursor", "")
            page += 1
            if not cursor or not markets:
                break
            time.sleep(0.03)
    return out


def evaluate(m: dict, min_vol: float):
    """Return (score, info) for a market, or (0, None) if not an MM candidate."""
    coin = coin_of(m.get("ticker", ""))
    if coin is None:
        return 0.0, None

    bid = _f(m.get("yes_bid_dollars"))
    ask = _f(m.get("yes_ask_dollars"))
    if bid <= 0 or ask <= 0 or ask <= bid:
        return 0.0, None

    bid_c, ask_c = round(bid * 100), round(ask * 100)
    spread_c = ask_c - bid_c
    if spread_c < 2:                       # need ≥2c of room to quote inside
        return 0.0, None
    if bid_c < 3 or ask_c > 97:            # extreme prices: little spread upside
        return 0.0, None

    vol24    = _f(m.get("volume_24h_fp"))
    bid_sz   = _f(m.get("yes_bid_size_fp"))
    ask_sz   = _f(m.get("yes_ask_size_fp"))
    if bid_sz < 1 or ask_sz < 1:           # need a real two-sided book
        return 0.0, None
    if vol24 < min_vol:                    # NO FLOW = trap, regardless of spread
        return 0.0, None

    mid = (bid + ask) / 2.0
    rt_fee_c   = 2.0 * maker_fee_per_contract(mid) * 100.0   # round-trip, cents
    net_cap_c  = spread_c - rt_fee_c                          # net per RT contract
    if net_cap_c <= 0:
        return 0.0, None

    daily_rt    = vol24 * CAPTURE_FRACTION
    opt_usd_day = (net_cap_c / 100.0) * daily_rt              # optimistic ceiling

    # Score favours net capture AND flow (sqrt-damped so one huge market doesn't
    # dominate). This is what separates real candidates from wide-but-dead books.
    score = net_cap_c * math.sqrt(max(vol24, 1.0))

    return score, {
        "coin": coin, "ticker": m.get("ticker"), "title": (m.get("title") or "")[:48],
        "bid": bid_c, "ask": ask_c, "spread": spread_c,
        "rt_fee": rt_fee_c, "net_cap": net_cap_c, "opt_usd_day": opt_usd_day,
        "vol24": vol24, "bid_sz": bid_sz, "ask_sz": ask_sz,
        "close": (m.get("close_time") or "")[:16],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default=None, help="filter to one coin, e.g. XRP")
    ap.add_argument("--min-vol", type=float, default=20.0,
                    help="min 24h volume to count as having flow (default 20)")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    print("Discovering crypto series...")
    series_list = discover_crypto_series()
    print(f"  {len(series_list)} crypto series: {', '.join(series_list)}")
    print("Fetching open markets per series (authenticated)...")
    markets = fetch_crypto_markets(series_list)
    print(f"  {len(markets)} open crypto markets\n")

    scored = []
    for m in markets:
        s, info = evaluate(m, args.min_vol)
        if s > 0 and (args.coin is None or info["coin"] == args.coin.upper()):
            scored.append((s, info))
    scored.sort(reverse=True, key=lambda x: x[0])

    coin_label = args.coin.upper() if args.coin else "all crypto"
    print(f"{len(scored)} MM candidates ({coin_label}, spread≥2c, vol24≥{args.min_vol:.0f}, "
          f"net-of-fee>0)\n")

    hdr = (f"{'#':>3}  {'coin':<5} {'spr':>4} {'bid/ask':>8} {'rtFee':>6} "
           f"{'net':>5} {'vol24':>7} {'bsz':>6} {'asz':>6} {'~$/day':>7}  {'close':<16} ticker")
    print(hdr); print("-" * len(hdr))
    for i, (s, x) in enumerate(scored[:args.top]):
        print(f"{i+1:>3}  {x['coin']:<5} {x['spread']:>3}c "
              f"{x['bid']:>3}/{x['ask']:<3} {x['rt_fee']:>5.2f}c "
              f"{x['net_cap']:>4.1f}c {x['vol24']:>7.0f} {x['bid_sz']:>6.0f} {x['ask_sz']:>6.0f} "
              f"{x['opt_usd_day']:>6.1f}  {x['close']:<16} {x['ticker'][:34]}")

    # Per-coin / per-series rollup.
    print("\nBy series prefix (candidates only):")
    by_series: dict[str, list] = {}
    for s, x in scored:
        by_series.setdefault(x["ticker"].split("-")[0], []).append(x)
    rows = []
    for pref, lst in by_series.items():
        rows.append((
            sum(y["opt_usd_day"] for y in lst), pref, len(lst),
            sum(y["spread"] for y in lst) / len(lst),
            sum(y["vol24"] for y in lst),
        ))
    rows.sort(reverse=True)
    print(f"{'series':<14} {'#mkts':>6} {'avgSpread':>10} {'sumVol24':>10} {'sum~$/day':>10}")
    print("-" * 54)
    for usd, pref, n, avgsp, vol in rows:
        print(f"{pref:<14} {n:>6} {avgsp:>9.1f}c {vol:>10,.0f} {usd:>10.1f}")

    print(f"\nNote: ~$/day assumes capturing {CAPTURE_FRACTION:.0%} of flow with ZERO "
          f"adverse selection.\nIt is a CEILING. Step 2 (passive-fill shadow logger) "
          f"measures the real, post-adverse-selection number.")


if __name__ == "__main__":
    main()
