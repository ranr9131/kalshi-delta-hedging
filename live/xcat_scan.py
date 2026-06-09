"""
Cross-venue arb *probability* scanner: Kalshi <-> Polymarket, ALL categories.

Goal isn't "find one arb right now" — it's to answer WHERE a real cross-venue
lock is structurally possible, and rank categories by that.

A divergence between two venues is only a real lock if BOTH legs resolve on the
SAME source/criteria. That's the trap crypto fell into (Kalshi=CF Benchmarks
60s-avg, Poly=Chainlink spot -> "arb" = basis risk, never a lock). So for every
matched pair we tag a RESOLUTION-IDENTITY risk, not just a price gap.

Output: for each category bucket —
  - # matched pairs, liquidity on both sides
  - divergence distribution (median / p90)
  - resolution-identity verdict (SAME / DIFFERENT / UNKNOWN)
  - the structural arb verdict for the bucket

Pure observation, no orders.
"""
from __future__ import annotations
import os, time, base64, json, re, statistics
from collections import defaultdict
from difflib import SequenceMatcher
import requests
from dotenv import dotenv_values

KALSHI_BASE = "https://api.elections.kalshi.com"
POLY_BASE   = "https://gamma-api.polymarket.com"

ENV_PATH = "/Users/leolee/kalshi-delta-hedging/live/.env"
if not os.path.exists(ENV_PATH):
    ENV_PATH = "/home/ec2-user/kalshi-delta-hedging/live/.env"
env = dotenv_values(ENV_PATH)


def kalshi_sig(method, path):
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    key_pem = env["KALSHI_PRIVATE_KEY"].replace("\\n", "\n").encode()
    ts = str(int(time.time() * 1000))
    k = serialization.load_pem_private_key(key_pem, password=None)
    s = k.sign((ts + method + path).encode(),
               padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                           salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": env["KALSHI_API_KEY_ID"],
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(s).decode()}


def _f(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def fetch_kalshi():
    """Drive off the EVENTS endpoint (carries real `category`), with nested
    markets. The markets endpoint is 98% multi-game-parlay (MVE) junk that
    floods pagination before any liquid market — events are clean."""
    out, cursor, path = [], None, "/trade-api/v2/events"
    for _ in range(120):
        p = {"status": "open", "limit": 200, "with_nested_markets": "true"}
        if cursor:
            p["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}{path}", headers=kalshi_sig("GET", path),
                         params=p, timeout=25)
        if not r.ok:
            break
        d = r.json()
        evs = d.get("events", [])
        if not evs:
            break
        for e in evs:
            cat = e.get("category", "")
            etitle = e.get("title", "")
            for m in (e.get("markets") or []):
                m["_cat"] = cat
                m["_etitle"] = etitle
                out.append(m)
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(0.12)
    return out


def fetch_poly():
    out, off = [], 0
    while off <= 8000:
        r = requests.get(f"{POLY_BASE}/markets",
                         params={"closed": "false", "active": "true",
                                 "limit": 100, "offset": off}, timeout=20)
        if not r.ok:
            break
        d = r.json()
        if not d:
            break
        out += d
        if len(d) < 100:
            break
        off += 100
        time.sleep(0.2)
    return out


STOP = {"will", "the", "by", "in", "of", "a", "an", "to", "be", "is", "at",
        "on", "for", "with", "do", "does", "have", "has", "than", "more", "or"}


def toks(t):
    t = re.sub(r"[^\w\s]", " ", (t or "").lower())
    return {w for w in t.split() if w not in STOP and len(w) > 1}


def nums(t):
    """Significant numbers in a question (strikes, %, $, counts). Used to reject
    pairs that share words but differ on the threshold — the #1 false-match
    source ('Above 10,000 jobs' vs 'Bitcoin dip to $10,000')."""
    out = set()
    for m in re.findall(r"\d[\d,\.]*", (t or "")):
        v = m.replace(",", "")
        try:
            out.add(round(float(v), 2))
        except Exception:
            pass
    return out


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ── resolution-identity classifier ──────────────────────────────────────────
# Tag whether a category's two venues plausibly resolve on the SAME source.
def res_identity(cat: str, kq: str) -> str:
    c = (cat or "").lower()
    q = (kq or "").lower()
    if any(w in c + q for w in ("bitcoin", "ethereum", "crypto", "btc", "eth",
                                "solana", "xrp", "dogecoin", "price of")):
        return "DIFFERENT"   # CF Benchmarks vs Chainlink — proven basis risk
    if any(w in c + q for w in ("temperature", "weather", "snow", "rain", "hurricane")):
        return "UNKNOWN"     # station/source may differ
    if any(w in c + q for w in ("election", "president", "senate", "governor",
                                "primary", "nominee", "win the", "mayor")):
        return "SAME"        # objective outcome, identical for both
    if any(w in c + q for w in ("cpi", "inflation", "fed", "rate", "gdp",
                                "jobs", "unemployment", "recession")):
        return "SAME"        # same govt/agency release
    if any(w in c + q for w in ("super bowl", "champion", "playoff", "world cup",
                                "finals", "mvp")):
        return "SAME"        # objective sports outcome (but converges fast)
    return "UNKNOWN"


# bucket using Kalshi's real event category (+ crypto split out of Financials)
def bucket(cat, q):
    qq = (q or "").lower()
    if any(k in qq for k in ("bitcoin", "ethereum", "solana", "btc", "ether",
                             "xrp", "dogecoin", "crypto")):
        return "Crypto"
    return cat or "Other"


def main():
    print("pulling venues...", flush=True)
    km = fetch_kalshi()
    pm = fetch_poly()
    print(f"  kalshi={len(km)}  poly={len(pm)}", flush=True)

    # build poly pool
    P = []
    for m in pm:
        ocs, prs = m.get("outcomes"), m.get("outcomePrices")
        try:
            if isinstance(ocs, str): ocs = json.loads(ocs)
            if isinstance(prs, str): prs = json.loads(prs)
        except Exception:
            continue
        if not ocs or not prs:
            continue
        yes = None
        for o, pr in zip(ocs, prs):
            if str(o).strip().lower() in ("yes", "true"):
                yes = _f(pr)
        if yes is None:
            yes = _f(prs[0])
        if yes is None or yes <= 0.02 or yes >= 0.98:
            continue
        q = m.get("question", "")
        try:
            cl = json.loads(m.get("clobTokenIds") or "[]")
        except Exception:
            cl = []
        # token order matches outcomes order; yes-token = the one paired w/ "Yes"
        yes_tok = None
        if isinstance(ocs, list) and len(ocs) == len(cl):
            for o, tk in zip(ocs, cl):
                if str(o).strip().lower() in ("yes", "true"):
                    yes_tok = tk
        P.append({"q": q, "t": toks(q), "n": nums(q), "yes": yes,
                  "vol": _f(m.get("volume24hr")) or _f(m.get("volumeNum")) or 0,
                  "spread": _f(m.get("spread")) or 0, "yes_tok": yes_tok})

    # match each kalshi mkt to best poly mkt
    rows = []
    for m in km:
        yk = None
        ya, yb = _f(m.get("yes_ask_dollars")), _f(m.get("yes_bid_dollars"))
        if ya and yb and ya > 0 and yb > 0:
            yk = (ya + yb) / 2
        elif _f(m.get("last_price_dollars")):
            yk = _f(m.get("last_price_dollars"))
        if yk is None or yk <= 0.02 or yk >= 0.98:
            continue
        # meaningful text = event title + the outcome's sub-title
        q = (m.get("_etitle", "") + " " + (m.get("yes_sub_title") or "")).strip()
        kt = toks(q)
        if not kt:
            continue
        vol = _f(m.get("volume_24h_fp")) or 0
        liq = _f(m.get("liquidity_dollars")) or 0
        vol = max(vol, liq)   # liquidity standing in book counts as 2-sided depth
        cat = m.get("_cat", "")
        kn = nums(q)
        best, bs = None, 0
        for pp in P:
            inter = kt & pp["t"]
            if len(inter) < 2:          # require >=2 shared content words
                continue
            # numeric-consistency gate: if BOTH carry numbers and they're
            # disjoint, it's a threshold mismatch, not the same question.
            if kn and pp["n"] and not (kn & pp["n"]):
                continue
            jacc = len(inter) / max(1, len(kt | pp["t"]))
            s = sim(" ".join(sorted(kt)), " ".join(sorted(pp["t"])))
            score = 0.5 * s + 0.5 * jacc
            if score > bs:
                bs, best = score, pp
        if best and bs >= 0.5:
            rows.append({
                "cat": bucket(cat, q), "kcat": cat, "score": bs,
                "div": yk - best["yes"], "kq": q, "pq": best["q"],
                "kyes": yk, "pyes": best["yes"], "kvol": vol, "pvol": best["vol"],
                "kt": m.get("ticker", ""), "ptok": best.get("yes_tok"),
                "res": res_identity(cat, q),
            })

    print(f"  matched pairs (>=2 shared words, score>=.35): {len(rows)}\n", flush=True)

    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)

    print(f"{'category':12} {'n':>3} {'res':>9} {'medDiv':>7} {'p90Div':>7} "
          f"{'bothLiq':>7}  verdict")
    print("-" * 78)
    order = sorted(by.items(), key=lambda kv: -len(kv[1]))
    for cat, rs in order:
        divs = [abs(r["div"]) for r in rs]
        med = statistics.median(divs)
        p90 = sorted(divs)[int(0.9 * (len(divs) - 1))]
        both = sum(1 for r in rs if r["kvol"] > 200 and r["pvol"] > 200)
        res = max(set(r["res"] for r in rs), key=lambda x: sum(r["res"] == x for r in rs))
        if res == "DIFFERENT":
            v = "FAKE (diff index)"
        elif both == 0:
            v = "no 2-sided liq"
        elif res == "SAME" and p90 > 0.05:
            v = "** CANDIDATE **"
        elif res == "SAME":
            v = "same src, tight"
        else:
            v = "check res source"
        print(f"{cat[:12]:12} {len(rs):>3} {res:>9} {med:>7.3f} {p90:>7.3f} "
              f"{both:>7}  {v}")

    # HIGH-CONFIDENCE pairs only: tight text match (score>=.6) so the questions
    # are plausibly the SAME event, same-source resolution, liquid on both sides.
    cand = [r for r in rows if r["res"] == "SAME" and r["kvol"] > 200
            and r["pvol"] > 200 and abs(r["div"]) >= 0.04 and r["score"] >= 0.60]
    cand.sort(key=lambda r: -abs(r["div"]))
    print(f"\n=== high-confidence (score>=.6) same-source 2-sided divergences >=4c: "
          f"{len(cand)} ===")
    for r in cand[:30]:
        side = "NO@K + YES@P" if r["div"] > 0 else "YES@K + NO@P"
        cost = (1 - r["kyes"]) + r["pyes"] if r["div"] > 0 else r["kyes"] + (1 - r["pyes"])
        print(f"\n[{r['cat']}] div={r['div']:+.3f} -> {side}  lockcost=${cost:.3f} "
              f"edge={(1-cost)*100:+.1f}c  (score {r['score']:.2f})")
        print(f"  K {r['kyes']:.2f} v${r['kvol']:.0f}  {r['kq'][:78]}")
        print(f"  P {r['pyes']:.2f} v${r['pvol']:.0f}  {r['pq'][:78]}")

    executable_check(cand)


import math


def kalshi_fee(p):
    if p is None:
        return 0.0
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def kalshi_book(ticker):
    """(yes_ask, no_ask) in dollars from the live REST market."""
    p = f"/trade-api/v2/markets/{ticker}"
    try:
        r = requests.get(f"{KALSHI_BASE}{p}", headers=kalshi_sig("GET", p), timeout=10)
        mk = r.json().get("market", {})
        return _f(mk.get("yes_ask_dollars")), _f(mk.get("no_ask_dollars"))
    except Exception:
        return None, None


def poly_book(token):
    """(yes_ask, no_ask). asks descending -> best = last. NO ask = 1 - best YES bid."""
    try:
        r = requests.get("https://clob.polymarket.com/book",
                         params={"token_id": token}, timeout=10)
        b = r.json()
        asks, bids = b.get("asks") or [], b.get("bids") or []
        ya = _f(asks[-1]["price"]) if asks else None
        yb = _f(bids[-1]["price"]) if bids else None
        na = (1 - yb) if yb is not None else None
        return ya, na
    except Exception:
        return None, None


def executable_check(cand, n=12):
    """For the cleanest mid-gap pairs, fetch REAL asks on both venues and compute
    the true lock cost (buy the cheaper side of each leg, incl. Kalshi fee).
    This is the 'mid divergence != executable edge' test."""
    print(f"\n=== EXECUTABLE check (real asks + Kalshi fee), top {n} clean pairs ===")
    print("  to lock $1: pick the side implied by the mid gap, pay ask on each venue\n")
    for r in cand[:n]:
        kya, kna = kalshi_book(r["kt"])
        pya, pna = (poly_book(r["ptok"]) if r["ptok"] else (None, None))
        if r["div"] > 0:        # K rich -> sell K (buy NO@K) + buy YES@P
            k_leg, k_raw = ("NO@K", kna)
            p_leg, p_raw = ("YES@P", pya)
        else:                   # P rich -> buy YES@K + buy NO@P
            k_leg, k_raw = ("YES@K", kya)
            p_leg, p_raw = ("NO@P", pna)
        if k_raw is None or p_raw is None:
            print(f"  [{r['cat']}] {r['kq'][:42]:42} | no book ({r['kt']})")
            continue
        cost = k_raw + kalshi_fee(k_raw) + p_raw
        edge = (1 - cost) * 100
        flag = "  <-- LOCK" if edge > 0 else ""
        print(f"  [{r['cat']:9}] midgap={r['div']*100:+5.1f}c | "
              f"{k_leg} {k_raw:.2f}(+f{kalshi_fee(k_raw):.2f}) {p_leg} {p_raw:.2f} "
              f"-> cost ${cost:.3f} edge {edge:+5.1f}c{flag}")
        print(f"             {r['kq'][:70]}")


if __name__ == "__main__":
    main()
