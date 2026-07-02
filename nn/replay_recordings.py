"""
EC2 RECORDING-REPLAY for the NN models -- real-fill validation (memory-efficient,
multi-asset, flat + dynamic sizing).

Reconstructs the live Kalshi order book + trade tape from recorder.py output and
replays the NN decisions against it. One streaming pass captures ALL configured
assets; each is then simulated as a TAKER under two sizing schemes:
  - flat:    fixed --target-contracts per signal
  - dynamic: live sizing  target$ = BASE * f_btc(move) * g_misp(edge) * decay
and (optionally) as a conservative back-of-queue MAKER.

  python3 nn/replay_recordings.py --assets BTC,ETH,SOL,XRP --mode both

Real Kalshi WS schema: orderbook_delta has price_dollars/delta_fp/side (dollars
as strings); snapshots carry no levels (book built from deltas); ticker gives
yes_bid_dollars/yes_ask_dollars/price_dollars/volume_fp.
"""
from __future__ import annotations
import os, sys, gzip, json, glob, math, argparse, re, time, pickle
from datetime import datetime, timezone, timedelta

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, REPO)

from model import TSWinPredictor
import kalshi_client
import btc_data
import build_dataset_v2 as bd

MINUTES   = list(range(10, 14))
WINDOW_MINUTES = 15
MIN_EDGE_C = 10
MIN_BET    = 5.0
FEE_KEEP   = 0.93

ASSET = {
    "BTC": ("KXBTC15M", "BTC-USD", "best_v2_small.pt"),
    "ETH": ("KXETH15M", "ETH-USD", "best_v2_small_eth.pt"),
    "SOL": ("KXSOL15M", "SOL-USD", "best_v2_small_sol.pt"),
    "XRP": ("KXXRP15M", "XRP-USD", "best_v2_small_xrp.pt"),
}

# live sizing constants (match sharpe_eval.py / live engine)
BASE_STAKE = 100.0
SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0
def f_btc(pct):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (pct - SIG_C)))
def g_misp(m):   return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))
def time_decay(minute):
    if minute < 7:  return 0.4
    if minute < 10: return 0.8
    return 1.2


# ── recording IO (time-ordered: oldest part -> newest) ──────────────────────

def _open_any(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def _file_order_key(path):
    m = re.search(r"\.part-(\d+)", os.path.basename(path))
    return (0, int(m.group(1))) if m else (1, 0)

def iter_stream(rec_dir, stream, days=None):
    day_dirs = sorted(d for d in glob.glob(os.path.join(rec_dir, "*")) if os.path.isdir(d))
    if days:
        day_dirs = day_dirs[-days:]
    for d in day_dirs:
        for f in sorted(glob.glob(os.path.join(d, f"{stream}.jsonl*")), key=_file_order_key):
            try:
                with _open_any(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try: yield json.loads(line)
                            except Exception: pass
            except Exception as e:
                print(f"  [warn] read {f}: {e}", flush=True)


# ── order book (cents-keyed, float sizes; built from deltas) ────────────────

def _cents(p): return int(round(float(p) * 100))

class Book:
    __slots__ = ("yes", "no")
    def __init__(self): self.yes = {}; self.no = {}
    def snapshot(self, msg):
        y, n = msg.get("yes"), msg.get("no")
        if y is None and n is None: return
        self.yes = {_cents(p): float(s) for p, s in (y or []) if float(s) > 0}
        self.no  = {_cents(p): float(s) for p, s in (n or []) if float(s) > 0}
    def delta(self, msg):
        side = self.yes if msg.get("side") == "yes" else self.no
        p = _cents(msg["price_dollars"]); side[p] = side.get(p, 0.0) + float(msg["delta_fp"])
        if side[p] <= 0: side.pop(p, None)
    def best_yes_bid(self): return max(self.yes) if self.yes else None
    def best_no_bid(self):  return max(self.no) if self.no else None
    def best_yes_ask(self):
        nb = self.best_no_bid(); return (100 - nb) if nb is not None else None
    def yes_mid(self):
        b, a = self.best_yes_bid(), self.best_yes_ask()
        return None if (b is None or a is None) else (b + a) / 200.0
    def _walk(self, levels, want_contracts=None, want_dollars=None):
        """Walk levels (price_cents desc); price paid = (100-cents)/100.
        Stop at contract count or dollar budget. Returns (avg_px, contracts)."""
        cost = 0.0; got = 0.0
        for q in sorted(levels, reverse=True):
            px = (100 - q) / 100.0
            if px <= 0: continue
            avail = levels[q]
            if want_contracts is not None:
                take = min(avail, want_contracts - got)
            else:
                take = min(avail, (want_dollars - cost) / px)
            if take <= 0: break
            cost += take * px; got += take
            if want_contracts is not None and got >= want_contracts - 1e-9: break
            if want_dollars is not None and cost >= want_dollars - 1e-9: break
        return ((cost / got) if got > 0 else None), got
    def buy_yes(self, contracts=None, dollars=None): return self._walk(self.no, contracts, dollars)
    def buy_no(self,  contracts=None, dollars=None): return self._walk(self.yes, contracts, dollars)
    def copy_levels(self): return (dict(self.yes), dict(self.no))

def book_from(levels):
    b = Book(); b.yes, b.no = levels[0], levels[1]; return b


GATE_S = 120  # seconds after decision to look for a confirming real trade

def traded_through(tape, t, up, side_touch, gate_s=GATE_S, slack=0.01):
    """True if a REAL trade printed at/through your fill price within gate_s of
    the decision -> evidence the touch was actually transactable (not a stale
    quote). For a YES buy, a print at yes-price <= touch confirms; for NO buy,
    convert the print to no-price (1 - yes_price)."""
    for tt, yes_tp, dv in tape:
        if dv <= 0 or tt < t or tt > t + gate_s:
            continue
        side_px = yes_tp if up else (1.0 - yes_tp)
        if side_px <= side_touch + slack:
            return True
    return False


# ── settled markets, per series (fetch_settled_markets is BTC-only) ──────────

def fetch_settled_multi(series_list, days=14):
    markets = {}
    cdir = os.path.join(REPO, "data", "cache")
    for series in series_list:
        cache = os.path.join(cdir, f"markets_{series}_{days}d.json")
        if os.path.exists(cache):
            batch = json.load(open(cache))
        else:
            cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
            batch = []; cursor = None
            print(f"  fetching settled {series} ...", flush=True)
            while True:
                params = {"series_ticker": series, "status": "settled",
                          "min_close_ts": cutoff, "limit": 200}
                if cursor: params["cursor"] = cursor
                data = kalshi_client._get("/markets", params=params)
                b = data.get("markets", []); batch.extend(b)
                cursor = data.get("cursor")
                if not cursor or not b: break
                time.sleep(0.3)
            json.dump(batch, open(cache, "w"))
        for m in batch:
            markets[m["ticker"]] = m
        print(f"  {series}: {len(batch)} settled markets", flush=True)
    return markets


# ── streaming reduce: one pass, all series, compact per-ticker records ───────

def stream_reduce(rec_dir, series_set, products, days):
    print(f"Streaming recordings (series={sorted(series_set)}) ...", flush=True)
    live = {}; n = 0
    def rec(tk):
        r = live.get(tk)
        if r is None:
            r = {"book": Book(), "per_min": {}, "book_at": {}, "tape": [],
                 "last_vol": None, "cur_mts": None}
            live[tk] = r
        return r
    for msg in iter_stream(rec_dir, "kalshi", days):
        typ = msg.get("type")
        if typ not in ("orderbook_snapshot", "orderbook_delta", "ticker"): continue
        body = msg.get("msg", {}); tk = body.get("market_ticker")
        if not tk: continue
        pref = tk.split("-", 1)[0]
        if pref not in series_set: continue
        t = msg.get("_t", 0) / 1000.0
        r = rec(tk); book = r["book"]
        if typ == "orderbook_snapshot": book.snapshot(body)
        elif typ == "orderbook_delta":
            try: book.delta(body)
            except Exception: pass
        else:
            pr, vol = body.get("price_dollars"), body.get("volume_fp")
            if pr is not None and vol is not None:
                vol = float(vol)
                dv = 0.0 if r["last_vol"] is None else max(0.0, vol - r["last_vol"])
                r["last_vol"] = vol
                if dv > 0: r["tape"].append((t, float(pr), dv))
        mid = book.yes_mid()
        if mid is None: continue
        mts = int(t // 60) * 60 + 60
        if r["cur_mts"] is not None and mts != r["cur_mts"]:
            r["book_at"][r["cur_mts"]] = book.copy_levels()
        r["cur_mts"] = mts
        c = r["per_min"].get(mts)
        if c is None:
            c = {"o": mid, "h": mid, "l": mid, "c": mid, "v": 0.0}; r["per_min"][mts] = c
        c["h"] = max(c["h"], mid); c["l"] = min(c["l"], mid); c["c"] = mid
        yb, ya = book.best_yes_bid(), book.best_yes_ask()
        c["bid"] = (yb / 100.0) if yb is not None else mid
        c["ask"] = (ya / 100.0) if ya is not None else mid
        n += 1
        if n % 2_000_000 == 0:
            print(f"  ... {n:,} msgs, {len(live)} tickers", flush=True)
    out = {}
    for tk, r in live.items():
        if r["cur_mts"] is not None:
            r["book_at"][r["cur_mts"]] = r["book"].copy_levels()
        for t, _p, dv in r["tape"]:
            mts = int(t // 60) * 60 + 60
            if mts in r["per_min"]: r["per_min"][mts]["v"] += dv
        candles = [{"ts": mts, "yes_open": c["o"], "yes_close": c["c"], "yes_high": c["h"],
                    "yes_low": c["l"], "yes_mean": c["c"], "yes_bid_close": c.get("bid", c["c"]),
                    "yes_ask_close": c.get("ask", c["c"]), "volume": c["v"]}
                   for mts, c in sorted(r["per_min"].items())]
        out[tk] = (candles, r["tape"], r["book_at"])
    print(f"  {n:,} msgs across {len(out)} tickers", flush=True)
    spot = {p: {} for p in products}
    for msg in iter_stream(rec_dir, "crypto", days):
        p = msg.get("product_id")
        if p not in spot: continue
        try:
            t = msg["_t"] / 1000.0; spot[p][int(t // 60) * 60] = float(msg["price"])
        except Exception: pass
    for p in products:
        print(f"  {p}: {len(spot[p])} spot minutes", flush=True)
    return out, spot


# ── model ───────────────────────────────────────────────────────────────────

def load_model(ckpt_file):
    ck = torch.load(os.path.join(ROOT, "checkpoints", ckpt_file),
                    map_location="cpu", weights_only=False)
    mean = np.array(ck["feature_mean"], dtype=np.float32)
    std  = np.array(ck["feature_std"], dtype=np.float32)
    m = TSWinPredictor(n_features=ck["n_features"], d_model=ck["d_model"],
                       n_heads=ck["n_heads"], n_layers=ck["n_layers"],
                       dim_feedforward=ck["dim_feedforward"], dropout=ck["dropout"])
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, mean, std

def predict_at(model, mean, std, X_raw, mask, minute):
    X = X_raw.copy(); mk = mask.copy()
    X[minute+1:, :] = 0.0; mk[minute+1:] = False
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.from_numpy(Xn[None]),
                                         torch.from_numpy(mk[None]))).item())


# ── per-asset simulation ────────────────────────────────────────────────────

def simulate_asset(asset, reduced, spot_min, markets, target_contracts,
                   maker_edge_c, do_maker):
    series, _, ckpt = ASSET[asset]
    model, mean, std = load_model(ckpt)
    _cand = {}
    kalshi_client.fetch_candlesticks = lambda tk, o, c: _cand.get(tk, [])
    btc_data.lookup = lambda s, t: s.get(int(t // 60) * 60)

    rows = []; n_eval = 0
    sig = {"flat": 0, "tt": 0, "dynamic": 0, "maker": 0}
    fil = {"flat": 0, "tt": 0, "dynamic": 0, "maker": 0}
    for tk, (candles, tape, book_at) in reduced.items():
        if not tk.startswith(series): continue
        mk = markets.get(tk)
        if not mk or mk.get("result") not in ("yes", "no") or len(candles) < 6:
            continue
        _cand[tk] = candles
        built = bd.build_window(mk, spot_min)
        if built is None: continue
        X, mask, label, t0 = built
        won_yes = (label == 1.0); n_eval += 1
        for m in MINUTES:
            if not mask[m]: continue
            p_yes = predict_at(model, mean, std, X, mask, m)
            kal = float(X[m, 4]); ret = float(X[m, 0])
            if not (0.01 < kal < 0.99) or ret == 0: continue
            up = ret > 0; pct = abs(ret) * 100
            fair = p_yes if up else (1 - p_yes)
            t = t0 + m * 60; mts = int(t // 60) * 60 + 60
            levels = book_at.get(mts) or book_at.get(mts + 60) or book_at.get(mts - 60)
            win = won_yes if up else (not won_yes)

            if levels is not None:
                book = book_from(levels)
                top_px, _ = (book.buy_yes(contracts=1) if up else book.buy_no(contracts=1))
                if top_px is not None:
                    edge = fair - top_px
                    if edge * 100 >= MIN_EDGE_C:
                        # FLAT sizing
                        fpx, got = (book.buy_yes(contracts=target_contracts) if up
                                    else book.buy_no(contracts=target_contracts))
                        if fpx is not None and got > 0:
                            sig["flat"] += 1; fil["flat"] += 1
                            pnl = got * ((1 - fpx) * FEE_KEEP if win else -fpx)
                            rows.append(dict(asset=asset, ticker=tk, minute=m, kind="flat",
                                side=("yes" if up else "no"), fair=round(fair, 4),
                                fill_px=round(fpx, 4), contracts=round(got, 2),
                                edge_c=round(edge*100, 2), won=int(win), pnl=round(pnl, 4)))
                            # TRADE-THROUGH-GATED: only count if a real trade printed
                            # at/through the touch within GATE_S (honest fill).
                            sig["tt"] += 1
                            if traded_through(tape, t, up, top_px):
                                fil["tt"] += 1
                                rows.append(dict(asset=asset, ticker=tk, minute=m, kind="tt",
                                    side=("yes" if up else "no"), fair=round(fair, 4),
                                    fill_px=round(fpx, 4), contracts=round(got, 2),
                                    edge_c=round(edge*100, 2), won=int(win), pnl=round(pnl, 4)))
                        # DYNAMIC sizing (live formula -> dollar budget walk)
                        target_d = BASE_STAKE * f_btc(pct) * g_misp(edge) * time_decay(m)
                        if target_d >= MIN_BET:
                            fpx, got = (book.buy_yes(dollars=target_d) if up
                                        else book.buy_no(dollars=target_d))
                            if fpx is not None and got > 0:
                                sig["dynamic"] += 1; fil["dynamic"] += 1
                                pnl = got * ((1 - fpx) * FEE_KEEP if win else -fpx)
                                rows.append(dict(asset=asset, ticker=tk, minute=m, kind="dynamic",
                                    side=("yes" if up else "no"), fair=round(fair, 4),
                                    fill_px=round(fpx, 4), contracts=round(got, 2),
                                    edge_c=round(edge*100, 2), won=int(win), pnl=round(pnl, 4)))

            if do_maker:
                side = "yes" if up else "no"
                rest_px = max(0.01, min(0.99, fair - maker_edge_c / 100.0))
                hit = any(tt >= t and dv > 0 and (ypx if up else 1 - ypx) <= rest_px
                          for tt, ypx, dv in tape)
                sig["maker"] += 1
                if hit:
                    fil["maker"] += 1
                    pnl = target_contracts * ((1 - rest_px) * FEE_KEEP if win else -rest_px)
                    rows.append(dict(asset=asset, ticker=tk, minute=m, kind="maker", side=side,
                        fair=round(fair, 4), fill_px=round(rest_px, 4),
                        contracts=target_contracts, edge_c=round(maker_edge_c, 2),
                        won=int(win), pnl=round(pnl, 4)))
    return rows, n_eval, sig, fil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir",
                    default=os.environ.get("REC_ROOT", os.path.join(REPO, "recordings")))
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--target-contracts", type=int, default=20)
    ap.add_argument("--mode", default="both", choices=["taker", "both"])
    ap.add_argument("--maker-edge-c", type=float, default=2.0)
    ap.add_argument("--min-edge-c", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "replay_fills.csv"))
    args = ap.parse_args()
    global MIN_EDGE_C
    MIN_EDGE_C = args.min_edge_c
    print(f"(min edge to fire: {MIN_EDGE_C}c)")

    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    series_set = {ASSET[a][0] for a in assets}
    products = [ASSET[a][1] for a in assets]
    rec_dir = os.path.abspath(args.recordings_dir)
    if not os.path.isdir(rec_dir):
        print(f"ERROR: recordings dir not found: {rec_dir}"); sys.exit(1)

    cache_pkl = os.path.join(ROOT, "_reduced_cache.pkl")
    if os.environ.get("REPLAY_USE_CACHE") == "1" and os.path.exists(cache_pkl):
        print(f"Loading cached reduced index {cache_pkl}", flush=True)
        with open(cache_pkl, "rb") as f:
            reduced, spot = pickle.load(f)
    else:
        reduced, spot = stream_reduce(rec_dir, series_set, products, args.days)
        try:
            with open(cache_pkl, "wb") as f:
                pickle.dump((reduced, spot), f, protocol=4)
            print(f"  cached reduced index -> {cache_pkl}", flush=True)
        except Exception as e:
            print(f"  [warn] cache failed: {e}", flush=True)
    if not reduced:
        print("No matching tickers."); return

    print("Fetching settled markets (per series) ...", flush=True)
    markets = fetch_settled_multi(series_set, days=14)
    print(f"  {len(markets)} settled markets total\n")

    all_rows = []
    do_maker = args.mode == "both"
    print(f"{'asset':<5} {'sizing':<8} {'fills':>6} {'fill%':>6} {'win%':>6} "
          f"{'ROI%':>8} {'PnL$':>10} {'avg$/fill':>9}")
    print("-" * 66)
    for a in assets:
        _, product, _ = ASSET[a]
        rows, n_eval, sig, fil = simulate_asset(
            a, reduced, spot[product], markets, args.target_contracts,
            args.maker_edge_c, do_maker)
        all_rows += rows
        for kind in (["flat", "tt", "dynamic", "maker"] if do_maker else ["flat", "tt", "dynamic"]):
            r = [x for x in rows if x["kind"] == kind]
            if not r:
                print(f"{a:<5} {kind:<8} {'0':>6}  (no fills, n_eval={n_eval})"); continue
            pnl = sum(x["pnl"] for x in r); cost = sum(x["contracts"]*x["fill_px"] for x in r)
            win = np.mean([x["won"] for x in r]) * 100
            frate = (fil[kind] / sig[kind] * 100) if sig[kind] else 0
            print(f"{a:<5} {kind:<8} {len(r):>6} {frate:>5.0f}% {win:>5.1f}% "
                  f"{pnl/cost*100:>+7.1f}% {pnl:>+10.2f} {cost/len(r):>9.2f}")

    import csv
    with open(args.out, "w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} fills -> {args.out}")


if __name__ == "__main__":
    main()
