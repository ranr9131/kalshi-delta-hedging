"""
Faithful backtest of the LIVE trader.py config:
  - dh-target sizing: stake = BASE_STAKE × f(BTC) × g(mispricing) × time_decay
  - 2D fair-price table (built from TRAIN ONLY — no look-ahead)
  - leg cap 2, RH disabled, MIN_EDGE_CENTS=1, MAX_FILL_PRICE=0.97
  - tests side ∈ {both, yes_only, no_only} on the held-out 20%

Replicates constants from live/trader.py + live/strategy.py exactly.

Timing: no look-ahead. BTC price at decision time t = close of the candle
ending at t (btc_data.lookup keys candles by start, so we use lookup(t-60)).
The SAME convention is used to build the fair table, so table and sim are
consistent.

Run:  python3 simulate_yesonly.py
"""
import os
import math
from datetime import datetime, timezone

import kalshi_client
import btc_data
from config import DATA_DAYS, FEE_RATE

# ── Live constants (mirror live/strategy.py + live/trader.py) ────────────────
SIGMOID_CENTER, SIGMOID_K, SIGMOID_MAX_MULT = 0.10, 20.0, 3.0
MISPRICING_K,   MISPRICING_MAX              = 8.0, 2.0
FAIR_PRICE_CONST = 0.698

BASE_STAKE      = 10.0   # live .env
MIN_BET         = 1.0
MIN_EDGE_CENTS  = 1.0
MAX_FILL_PRICE  = 0.97
FILL_BUFFER     = 0.03   # 3c
MAX_LEGS        = 2
WINDOW_CAP      = 30.0   # static fallback
SPREAD_HALF     = 0.01   # model: ask = close + 1c, bid = close - 1c (conservative)

TABLE_DAYS      = 30     # rebuild fair table on most-recent N days of TRAIN

BUCKETS = [(0.0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50),(0.50,float("inf"))]
MIN_N   = 30
FALLBACK_1D = {1:0.582,2:0.617,3:0.636,4:0.670,5:0.698,6:0.728,7:0.751,
               8:0.759,9:0.783,10:0.798,11:0.806,12:0.815,13:0.826}

DH_MINUTES = list(range(4, 15))   # T+4 .. T+14


def sigmoid_btc(abs_pct):
    return SIGMOID_MAX_MULT / (1.0 + math.exp(-SIGMOID_K * (abs_pct - SIGMOID_CENTER)))

def sigmoid_mis(mis):
    return MISPRICING_MAX / (1.0 + math.exp(-MISPRICING_K * mis))

def time_decay(t_min):
    if t_min < 7:  return 0.4
    if t_min < 10: return 0.8
    return 1.2

def bucket_idx(abs_pct):
    for i,(lo,hi) in enumerate(BUCKETS):
        if lo <= abs_pct < hi:
            return i
    return len(BUCKETS)-1

def btc_close_at(prices, t):
    return btc_data.lookup(prices, t - 60)


# ── Fair-price table built from a set of markets (no look-ahead) ─────────────
def build_table(markets, btc_prices):
    stats = {m: [{"n":0,"correct":0} for _ in BUCKETS] for m in range(1,15)}
    for mk in markets:
        oi, ci, res = mk.get("open_time",""), mk.get("close_time",""), mk.get("result","")
        if res not in ("yes","no") or not oi:
            continue
        try:
            t0 = int(datetime.fromisoformat(oi.replace("Z","+00:00")).timestamp())
        except Exception:
            continue
        btc_t0 = btc_close_at(btc_prices, t0)
        if btc_t0 is None:
            continue
        candles = kalshi_client.fetch_candlesticks(mk["ticker"], oi, ci)
        if not candles:
            continue
        resolved_yes = res == "yes"
        for minute in range(1,15):
            t = t0 + minute*60
            btc_t = btc_close_at(btc_prices, t)
            yc = kalshi_client.get_yes_price_at(candles, t)
            if btc_t is None or yc is None or not (0.01 < yc < 0.99):
                continue
            abs_pct = abs(btc_t - btc_t0)/btc_t0*100
            up = btc_t > btc_t0
            bi = bucket_idx(abs_pct)
            s = stats[minute][bi]
            s["n"] += 1
            s["correct"] += int(up == resolved_yes)
    table = {}
    for m in range(1,15):
        for bi in range(len(BUCKETS)):
            s = stats[m][bi]
            if s["n"] >= MIN_N:
                table[(m,bi)] = s["correct"]/s["n"]
    return table

def fair_price(table, minute, abs_pct):
    v = table.get((minute, bucket_idx(abs_pct)))
    if v is not None:
        return v
    return FALLBACK_1D.get(minute, FAIR_PRICE_CONST)


# ── Per-window sim (faithful dh-target replica) ──────────────────────────────
def sim_window(market, btc_prices, table, side_filter):
    oi, ci, res = market.get("open_time",""), market.get("close_time",""), market.get("result","")
    if res not in ("yes","no") or not oi:
        return None
    try:
        t0 = int(datetime.fromisoformat(oi.replace("Z","+00:00")).timestamp())
    except Exception:
        return None
    btc_t0 = btc_close_at(btc_prices, t0)
    if btc_t0 is None:
        return None
    candles = kalshi_client.fetch_candlesticks(market["ticker"], oi, ci)
    if not candles:
        return None

    yes_exp = no_exp = 0.0
    yes_contracts = no_contracts = 0.0
    legs = 0
    sides = []

    for minute in DH_MINUTES:
        if legs >= MAX_LEGS:
            break
        t = t0 + minute*60
        btc_t = btc_close_at(btc_prices, t)
        yc = kalshi_client.get_yes_price_at(candles, t)
        if btc_t is None or yc is None or not (0.01 < yc < 0.99):
            continue
        yes_ask = min(yc + SPREAD_HALF, 0.99)
        yes_bid = max(yc - SPREAD_HALF, 0.01)

        abs_pct = abs(btc_t - btc_t0)/btc_t0*100
        f = sigmoid_btc(abs_pct)
        up = btc_t > btc_t0
        fair = fair_price(table, minute, abs_pct)
        td = time_decay(minute)

        if up:
            mis = fair - (yes_ask + FILL_BUFFER)
            g = sigmoid_mis(mis)
            tgt_yes, tgt_no = BASE_STAKE*f*g*td, 0.0
        else:
            mis = fair - ((1.0 - yes_bid) + FILL_BUFFER)
            g = sigmoid_mis(mis)
            tgt_no, tgt_yes = BASE_STAKE*f*g*td, 0.0

        if mis*100 < MIN_EDGE_CENTS:
            tgt_yes = tgt_no = 0.0

        if side_filter == "yes_only": tgt_no = 0.0
        elif side_filter == "no_only": tgt_yes = 0.0

        bet_yes = max(0.0, tgt_yes - yes_exp)
        bet_no  = max(0.0, tgt_no  - no_exp)

        # window cap
        rem = max(0.0, WINDOW_CAP - (yes_exp + no_exp))
        if bet_yes + bet_no > rem:
            if rem <= 0:
                bet_yes = bet_no = 0.0
            else:
                tot = bet_yes + bet_no
                bet_yes, bet_no = rem*bet_yes/tot, rem*bet_no/tot

        if bet_yes >= MIN_BET:
            fill = min(yes_ask + FILL_BUFFER, 0.99)
            if fill <= MAX_FILL_PRICE:
                yes_exp += bet_yes
                yes_contracts += bet_yes / fill
                legs += 1
                sides.append("yes")
        if legs < MAX_LEGS and bet_no >= MIN_BET:
            fill = min((1.0 - yes_bid) + FILL_BUFFER, 0.99)
            if fill <= MAX_FILL_PRICE:
                no_exp += bet_no
                no_contracts += bet_no / fill
                legs += 1
                sides.append("no")

    if legs == 0:
        return None
    wagered = yes_exp + no_exp
    gross = (yes_contracts if res=="yes" else 0.0) + (no_contracts if res=="no" else 0.0)
    pnl_pre = gross - wagered
    fee = FEE_RATE * pnl_pre if pnl_pre > 0 else 0.0
    pnl = pnl_pre - fee
    return {"wagered":wagered, "pnl":pnl, "legs":legs, "sides":sides, "result":res}


def summarize(label, rows):
    rows = [r for r in rows if r]
    if not rows:
        print(f"  {label:<14} no windows"); return
    n = len(rows)
    wag = sum(r["wagered"] for r in rows)
    pnl = sum(r["pnl"] for r in rows)
    win = sum(1 for r in rows if r["pnl"] > 0)
    tix = sum(r["legs"] for r in rows)
    roi = pnl/wag*100 if wag else 0
    print(f"  {label:<14} n={n:>4}  wag=${wag:>8.0f}  pnl=${pnl:>+8.2f}  roi={roi:>+6.2f}%  "
          f"win={win/n*100:>4.0f}%  tix={tix}")


def main():
    print("Loading markets…")
    markets = kalshi_client.fetch_settled_markets(days=DATA_DAYS)
    dated = []
    for m in markets:
        try:
            dt = datetime.fromisoformat(m["open_time"].replace("Z","+00:00"))
            dated.append((dt,m))
        except Exception:
            pass
    dated.sort(key=lambda x:x[0])
    sm = [m for _,m in dated]
    ts = [int(d.timestamp()) for d,_ in dated]
    print("Loading BTC prices…")
    btc_prices = btc_data.fetch_btc_prices(min(ts)-600, max(ts)+1800)

    split = int(len(sm)*0.8)
    train, test = sm[:split], sm[split:]
    print(f"Train: {len(train)} ({dated[0][0].date()} → {dated[split-1][0].date()})")
    print(f"Test:  {len(test)} ({dated[split][0].date()} → {dated[-1][0].date()})")

    # Build table from the most-recent TABLE_DAYS of TRAIN only (mirrors the
    # live 30-day rolling table, but strictly excludes test windows).
    train_dated = dated[:split]
    max_train_ts = int(train_dated[-1][0].timestamp())
    cutoff = max_train_ts - TABLE_DAYS*86400
    table_markets = [m for d,m in train_dated if int(d.timestamp()) >= cutoff]
    print(f"\nBuilding fair table from last {TABLE_DAYS}d of TRAIN "
          f"({len(table_markets)} windows, no test leakage)…")
    table = build_table(table_markets, btc_prices)
    print(f"  {len(table)} cells")

    print("\nBacktesting each side filter on TRAIN and TEST…")
    for sf in ["both", "yes_only", "no_only"]:
        tr = [sim_window(m, btc_prices, table, sf) for m in train]
        te = [sim_window(m, btc_prices, table, sf) for m in test]
        print(f"\n── SIDE_FILTER = {sf} ──")
        summarize("TRAIN", tr)
        summarize("TEST",  te)


if __name__ == "__main__":
    main()
