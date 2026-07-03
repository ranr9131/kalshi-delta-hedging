"""
Latency probe — measure the real order path from THIS host to Kalshi (V2 API).

Run this ON the AWS box: it measures that host's network + matching-engine
round-trip, which is what the live strategy actually experiences.

NOTE: Kalshi killed the V1 create-order endpoint (410 deprecated_v1_order_endpoint,
~May 2026). This probe uses the V2 endpoint:
    POST /trade-api/v2/portfolio/events/orders
V2 semantics: single YES book. side="bid" buys yes; buying NO = ask at 1-price.
count/price are fixed-point STRINGS. IOC/FOK are first-class time_in_force values.
The response carries average_fill_price and average_fee_paid — ground truth for
both slippage AND the true fee multiplier.

Modes:
  SAFE (default) — 1-contract 1c bid with time_in_force=fill_or_kill. It can
  never fill, so Kalshi rejects it (409 fill_or_kill_insufficient_resting_volume)
  after a full trip through auth + risk + matching engine. No position, no fee,
  no cleanup. Also runs a GTC-place/cancel pair each K probes to time cancels.

  --fill (REAL MONEY, opt-in) — 1-contract marketable IOC bid that DOES execute.
  Reports ack latency plus actual fill price and actual fee straight from the
  response. Leaves a 1-contract position; use sparingly.

Usage (on AWS):
    python latency_probe.py                 # 20 safe probes
    python latency_probe.py --n 50
    python latency_probe.py --fill --n 5    # 5 REAL 1-contract fills
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone

import requests
from dotenv import dotenv_values

import kalshi_auth
import kalshi_trade
from kalshi_auth import make_auth_headers

BASE_URL = "https://api.elections.kalshi.com"
V2_ORDERS_PATH = "/trade-api/v2/portfolio/events/orders"

_dir = os.path.dirname(os.path.abspath(__file__))
env = dotenv_values(os.path.join(_dir, ".env"))
PRIVATE_KEY = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
API_KEY_ID = env.get("KALSHI_API_KEY_ID", "")

SESSION = requests.Session()   # keep-alive: reflects steady-state (warm) latency


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def report(name, vals):
    if not vals:
        print(f"  {name:<28} (no samples)")
        return
    print(f"  {name:<28} min {min(vals):6.1f}  med {pct(vals,0.5):6.1f}  "
          f"p90 {pct(vals,0.9):6.1f}  p99 {pct(vals,0.99):6.1f}  max {max(vals):6.1f}  (ms, n={len(vals)})")


def v2_order(ticker, side, count, price, tif):
    """POST a V2 order. Returns (status_code, parsed_json_or_text)."""
    body = {
        "ticker": ticker, "side": side,
        "count": f"{count:.2f}", "price": f"{price:.4f}",
        "time_in_force": tif,
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": str(uuid.uuid4()),
    }
    headers = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "POST", V2_ORDERS_PATH)
    resp = SESSION.post(BASE_URL + V2_ORDERS_PATH, json=body, headers=headers, timeout=10)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def v2_cancel(order_id):
    path = f"{V2_ORDERS_PATH}/{order_id}"
    headers = make_auth_headers(PRIVATE_KEY, API_KEY_ID, "DELETE", path)
    resp = SESSION.delete(BASE_URL + path, headers=headers, timeout=10)
    return resp.status_code


def safe_probe(n, interval):
    print(f"\nSAFE latency probe: {n} x 1-contract 1c FOK bids (never fill, auto-killed)\n")
    m = kalshi_trade.get_open_market()
    if not m:
        print("No open market — cannot probe.")
        return
    ticker = m["ticker"]
    print(f"Market: {ticker}\n")

    # Warm the TLS/TCP connection so sample #1 isn't a cold handshake.
    try:
        SESSION.get(f"{BASE_URL}/trade-api/v2/markets",
                    params={"series_ticker": kalshi_trade.SERIES, "status": "open"}, timeout=10)
    except Exception:
        pass

    fok_ms, place_ms, cancel_ms, get_ms = [], [], [], []
    engine_skew_ms = []
    for i in range(n):
        if i and i % 20 == 0:
            mm = kalshi_trade.get_open_market()
            if mm:
                ticker = mm["ticker"]

        # 1) FOK probe: one full matching-engine round trip, self-cleaning.
        try:
            t0 = time.time()
            code, resp = v2_order(ticker, "bid", 1, 0.01, "fill_or_kill")
            dt = (time.time() - t0) * 1000
            ok_reject = (code == 409 and isinstance(resp, dict)
                         and resp.get("error", {}).get("code") == "fill_or_kill_insufficient_resting_volume")
            if ok_reject:
                fok_ms.append(dt)
            else:
                print(f"  ! probe {i}: unexpected {code}: {str(resp)[:150]}")
        except Exception as e:
            print(f"  probe {i} FOK error: {e}")

        # 2) Every 5th probe: GTC place + cancel to time the cancel path too.
        if i % 5 == 0:
            try:
                t1 = time.time()
                code, resp = v2_order(ticker, "bid", 1, 0.01, "good_till_canceled")
                dt1 = (time.time() - t1) * 1000
                oid = resp.get("order", {}).get("order_id") if isinstance(resp, dict) else None
                oid = oid or (resp.get("order_id") if isinstance(resp, dict) else None)
                if code in (200, 201) and oid:
                    place_ms.append(dt1)
                    ts_ms = None
                    if isinstance(resp, dict):
                        ts_ms = resp.get("ts_ms") or resp.get("order", {}).get("ts_ms")
                    if ts_ms:
                        # engine timestamp vs our midpoint estimate of arrival
                        engine_skew_ms.append(float(ts_ms) - (t1 * 1000 + dt1 / 2))
                    t2 = time.time()
                    ccode = v2_cancel(oid)
                    dt2 = (time.time() - t2) * 1000
                    if ccode in (200, 204):
                        cancel_ms.append(dt2)
                    else:
                        print(f"  ! probe {i}: cancel status {ccode} for {oid} — CHECK OPEN ORDERS")
                else:
                    print(f"  ! probe {i}: GTC place {code}: {str(resp)[:150]}")
            except Exception as e:
                print(f"  probe {i} GTC error: {e}")

        # 3) Plain read for comparison.
        t3 = time.time()
        try:
            SESSION.get(f"{BASE_URL}/trade-api/v2/markets",
                        params={"series_ticker": kalshi_trade.SERIES, "status": "open"}, timeout=10)
            get_ms.append((time.time() - t3) * 1000)
        except Exception:
            pass
        time.sleep(interval)

    print("Results:")
    report("FOK order round-trip", fok_ms)
    report("GTC place", place_ms)
    report("cancel", cancel_ms)
    report("GET markets (read)", get_ms)
    if fok_ms:
        print(f"\n  Actionable: your decision->exchange-matched latency is ~{pct(fok_ms,0.5):.0f} ms median "
              f"(p99 ~{pct(fok_ms,0.99):.0f} ms).")


def fill_probe(n, interval):
    print(f"\n*** REAL-MONEY fill probe: {n} marketable 1-contract IOC bids ***")
    print("Each leaves a 1-contract YES position (settles at expiry) and pays the real fee.\n")
    ack_ms = []
    fills = []
    for i in range(n):
        m = kalshi_trade.get_open_market()
        if not m:
            print("No open market; stopping.")
            break
        ticker = m["ticker"]
        ask = float(m["yes_ask_dollars"])
        limit = min(0.99, ask + kalshi_trade.FILL_BUFFER_CENTS / 100)
        try:
            t0 = time.time()
            code, resp = v2_order(ticker, "bid", 1, limit, "immediate_or_cancel")
            dt = (time.time() - t0) * 1000
            ack_ms.append(dt)
            if isinstance(resp, dict):
                o = resp.get("order", resp)
                afp = o.get("average_fill_price")
                fee = o.get("average_fee_paid")
                fc = o.get("fill_count")
                fills.append({"ask_seen": ask, "limit": limit, "avg_fill": afp,
                              "fee": fee, "fill_count": fc, "ms": round(dt, 1)})
                print(f"  fill {i}: {dt:6.1f} ms | ask seen {ask:.2f} limit {limit:.2f} "
                      f"-> filled {fc} @ {afp} fee {fee}")
            else:
                print(f"  fill {i}: {code}: {str(resp)[:150]}")
        except Exception as e:
            print(f"  fill {i} error: {e}")
        time.sleep(interval)

    print("\nResults:")
    report("IOC submit -> fill ack", ack_ms)
    if fills:
        print("\n  Fill detail (compare avg_fill vs ask_seen for slippage; fee is the REAL fee):")
        for f in fills:
            print(f"    {json.dumps(f)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--fill", action="store_true", help="REAL MONEY: marketable 1-contract IOC fills")
    args = ap.parse_args()

    print(f"Latency probe (V2 API) @ {datetime.now(timezone.utc).isoformat()}")
    if args.fill:
        fill_probe(args.n, args.interval)
    else:
        safe_probe(args.n, args.interval)


if __name__ == "__main__":
    main()
