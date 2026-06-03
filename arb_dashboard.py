"""Kalshi ↔ Polymarket arbitrage dashboard.

Pulls open markets from both, fuzzy-matches by question text, and shows
any pairs where the prices imply an arbitrage opportunity.

Polymarket: public API, no auth.
Kalshi: uses live/.env credentials (we already have these).

Usage: python arb_dashboard.py [--min-edge 0.02] [--min-volume 1000]
"""
import os, sys, time, base64, json, math, re
from difflib import SequenceMatcher
from datetime import datetime, timezone
import requests
from dotenv import dotenv_values

# Args
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.02"))   # 2% default
MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "500"))  # min daily volume
TOP_N = int(os.environ.get("TOP_N", "30"))

KALSHI_BASE = "https://api.elections.kalshi.com"
POLY_BASE   = "https://gamma-api.polymarket.com"

ENV_PATH = "/home/ec2-user/kalshi-delta-hedging/live/.env"
if not os.path.exists(ENV_PATH):
    ENV_PATH = "/Users/leolee/kalshi-delta-hedging/live/.env"
env = dotenv_values(ENV_PATH)


# ── Kalshi auth ──────────────────────────────────────────────────────────────
def kalshi_sig(method, path):
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    key_pem = env["KALSHI_PRIVATE_KEY"].replace("\\n", "\n").encode()
    ts = str(int(time.time() * 1000))
    msg = (ts + method + path).encode()
    k = serialization.load_pem_private_key(key_pem, password=None)
    s = k.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.DIGEST_LENGTH),
               hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": env["KALSHI_API_KEY_ID"],
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(s).decode()}


# ── Pull all open Kalshi markets ─────────────────────────────────────────────
def fetch_kalshi_markets():
    print("Pulling Kalshi open markets...", flush=True)
    all_mkts = []
    cursor = None
    path = "/trade-api/v2/markets"
    for page in range(50):
        params = {"status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}{path}", headers=kalshi_sig("GET", path),
                         params=params, timeout=15)
        if not r.ok:
            print(f"  err {r.status_code}: {r.text[:200]}")
            break
        d = r.json()
        batch = d.get("markets", [])
        if not batch:
            break
        all_mkts.extend(batch)
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)
    print(f"  total: {len(all_mkts)} Kalshi markets", flush=True)
    return all_mkts


# ── Pull all active Polymarket markets ───────────────────────────────────────
def fetch_polymarket():
    print("Pulling Polymarket markets...", flush=True)
    all_mkts = []
    offset = 0
    while True:
        params = {"closed": "false", "active": "true", "limit": 100, "offset": offset}
        r = requests.get(f"{POLY_BASE}/markets", params=params, timeout=15)
        if not r.ok:
            print(f"  err {r.status_code}")
            break
        d = r.json()
        if not d:
            break
        all_mkts.extend(d)
        if len(d) < 100:
            break
        offset += 100
        time.sleep(0.3)
        if offset > 5000:
            break
    print(f"  total: {len(all_mkts)} Polymarket markets", flush=True)
    return all_mkts


# ── Normalize question text for matching ─────────────────────────────────────
def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop common words that add noise
    stopwords = {"will", "the", "by", "in", "of", "a", "an", "to", "be", "is",
                 "at", "on", "for", "with", "do", "does", "does", "have", "has"}
    words = [w for w in text.split() if w not in stopwords and len(w) > 1]
    return " ".join(words)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ── Extract YES price from Kalshi market ─────────────────────────────────────
def _f(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def kalshi_yes_price(mk):
    # Kalshi v2 uses *_dollars fields (sometimes returned as strings)
    yes_ask = _f(mk.get("yes_ask_dollars"))
    yes_bid = _f(mk.get("yes_bid_dollars"))
    last    = _f(mk.get("last_price_dollars"))
    if yes_ask and yes_bid and yes_ask > 0 and yes_bid > 0:
        return (yes_ask + yes_bid) / 2.0
    if last and last > 0:
        return last
    if yes_ask and yes_ask > 0:
        return yes_ask
    return None


# ── Extract YES price from Polymarket market ────────────────────────────────
def poly_yes_price(mk):
    outcomes = mk.get("outcomes")
    prices = mk.get("outcomePrices")
    if not outcomes or not prices:
        return None
    try:
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
    except Exception:
        return None
    if not isinstance(outcomes, list) or not isinstance(prices, list):
        return None
    for o, p in zip(outcomes, prices):
        if str(o).strip().lower() in ("yes", "true"):
            try:
                return float(p)
            except Exception:
                return None
    # Fall back to first outcome price
    try:
        return float(prices[0])
    except Exception:
        return None


# ── Find matches and compute arb opportunities ───────────────────────────────
def find_arbs(kalshi_mkts, poly_mkts):
    # Filter to markets with prices + meaningful volume
    k_pool = []
    for mk in kalshi_mkts:
        py = kalshi_yes_price(mk)
        if py is None or py <= 0.02 or py >= 0.98:
            continue
        q = mk.get("title", "") or mk.get("subtitle", "") or mk.get("ticker", "")
        vol = mk.get("volume_24h_fp", 0) or mk.get("volume_fp", 0) or 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0
        k_pool.append({"q": q, "n": normalize(q), "yes": py, "ticker": mk.get("ticker"),
                       "vol": vol, "category": mk.get("category", "")})
    p_pool = []
    for mk in poly_mkts:
        py = poly_yes_price(mk)
        if py is None or py <= 0.02 or py >= 0.98:
            continue
        q = mk.get("question", "")
        try:
            vol24 = float(mk.get("volume24hr") or mk.get("volumeNum") or 0)
        except Exception:
            vol24 = 0
        p_pool.append({"q": q, "n": normalize(q), "yes": py,
                       "id": mk.get("id"), "vol": vol24})

    print(f"  Kalshi pool (priced, mid): {len(k_pool)}", flush=True)
    print(f"  Polymarket pool (priced, mid): {len(p_pool)}", flush=True)

    # Find matches
    matches = []
    for kp in k_pool:
        best_sim = 0
        best_pp = None
        for pp in p_pool:
            s = similarity(kp["n"], pp["n"])
            if s > best_sim:
                best_sim = s
                best_pp = pp
        if best_pp and best_sim >= 0.5:
            divergence = kp["yes"] - best_pp["yes"]
            matches.append({
                "sim": best_sim, "div": divergence,
                "k_q": kp["q"], "k_yes": kp["yes"], "k_vol": kp["vol"], "k_ticker": kp["ticker"],
                "p_q": best_pp["q"], "p_yes": best_pp["yes"], "p_vol": best_pp["vol"],
            })
    return matches


def main():
    print(f"Filters: MIN_EDGE={MIN_EDGE:.0%}, MIN_VOLUME=${MIN_VOLUME}, TOP_N={TOP_N}\n")
    kalshi_mkts = fetch_kalshi_markets()
    poly_mkts = fetch_polymarket()
    print()
    matches = find_arbs(kalshi_mkts, poly_mkts)
    # Filter and sort
    edge_matches = [m for m in matches if abs(m["div"]) >= MIN_EDGE
                    and (m["k_vol"] >= MIN_VOLUME or m["p_vol"] >= MIN_VOLUME)]
    edge_matches.sort(key=lambda m: abs(m["div"]), reverse=True)

    print(f"\n=== Matched market pairs: {len(matches)} total ===")
    print(f"=== With edge >= {MIN_EDGE:.0%} and meaningful volume: {len(edge_matches)} ===\n")

    for i, m in enumerate(edge_matches[:TOP_N]):
        print(f"#{i+1}  similarity={m['sim']:.2f}  divergence={m['div']:+.3f}")
        print(f"  Kalshi    YES={m['k_yes']:.3f}  vol24=${m['k_vol']:>9.0f}  [{m['k_ticker']}]")
        print(f"            \"{m['k_q'][:90]}\"")
        print(f"  Polymarket YES={m['p_yes']:.3f}  vol24=${m['p_vol']:>9.0f}")
        print(f"            \"{m['p_q'][:90]}\"")
        # Suggest the arb
        if m["div"] > 0:
            print(f"  → Buy NO on Kalshi at {1-m['k_yes']:.3f}, Buy YES on Poly at {m['p_yes']:.3f}")
            cost = (1 - m["k_yes"]) + m["p_yes"]
            print(f"  → Cost: ${cost:.3f}, Payout: $1.00, Profit: ${1-cost:+.3f} per pair")
        else:
            print(f"  → Buy YES on Kalshi at {m['k_yes']:.3f}, Buy NO on Poly at {1-m['p_yes']:.3f}")
            cost = m["k_yes"] + (1 - m["p_yes"])
            print(f"  → Cost: ${cost:.3f}, Payout: $1.00, Profit: ${1-cost:+.3f} per pair")
        print()


if __name__ == "__main__":
    main()
