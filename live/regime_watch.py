"""
Calibration-based crypto regime watcher.

Computes the things that actually predict whether the strategy makes money:
  - Edge realization rate: realized_pnl / sum(predicted_edge_dollars)
                           1.0 = model's claimed edge is fully showing up
                           0.0 = no edge being captured
                          <0   = model is anti-predictive (we LOSE on edges)
  - Brier score:          avg (predicted_p - actual_outcome)^2
                          0.25 = random (50/50 guesser)
                          <0.20 = well calibrated
                          >0.30 = badly broken
  - Per-asset go/no-go    so we can trade only the assets where the model works
  - Stability (recent 2h vs full 6h) — has the regime been steady?

Supplementary (forward-looking, no settled data needed):
  - Realized vol vs normal, cross-asset correlation, 3h trend

Verdict is driven primarily by calibration (the empirical truth).  Market-
conditions block is shown as context but does not override the data.

Usage:
  python3.11 regime_watch.py             # one-shot status
  python3.11 regime_watch.py --loop      # runs forever, every 5 min
"""
from __future__ import annotations
import os, sys, time, csv, math, requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict


BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, "regime.txt")
SETTLEMENTS_PATH = os.path.join(BASE, "settlements.csv")

# Source CSVs.  Paper sources are the cleanest signal (always-on, no guards);
# live sources confirm the same edge shows up under real fills/fees.
SOURCES = [
    ("V2-paper", os.path.join(BASE, "snipes_v2.csv")),
    ("V1-paper", os.path.join(BASE, "snipes.csv")),
    ("LEO-live", os.path.join(BASE, "snipes_leo.csv")),
    ("FRI-live", os.path.join(BASE, "snipes_friend.csv")),
]

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE"]
LOOKBACK_HOURS = 6
RECENT_HOURS = 2
MIN_FIRES_FOR_ASSET = 6
MIN_FIRES_FOR_OVERALL = 20

# --- Supplementary market-conditions block ---
COINBASE = "https://api.exchange.coinbase.com"
COINBASE_PRODUCTS = [("BTC", "BTC-USD"), ("ETH", "ETH-USD"), ("SOL", "SOL-USD"),
                     ("XRP", "XRP-USD"), ("HYPE", "HYPE-USD")]
NORMAL_HOURLY_VOL_PCT = {"BTC": 0.6, "ETH": 0.8, "SOL": 1.2, "XRP": 1.0, "HYPE": 1.5}


# ----- calibration helpers -----

def load_settlements():
    """ticker -> 'yes'/'no'"""
    out = {}
    if not os.path.exists(SETTLEMENTS_PATH):
        return out
    try:
        with open(SETTLEMENTS_PATH) as f:
            for r in csv.DictReader(f):
                t = (r.get("ticker") or "").strip()
                res = (r.get("result") or "").strip().lower()
                if t and res in ("yes", "no"):
                    out[t] = res
    except Exception:
        pass
    return out


def load_fires(path, settlements):
    """Return list of SETTLED fire dicts.  Joins ticker against settlements
    and computes pnl from fill / qty / side / result."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    ticker = (r.get("ticker") or "").strip()
                    # First try the inline result column; fall back to settlements
                    inline = (r.get("result") or "").strip().lower()
                    if inline in ("yes", "no"):
                        result = inline
                    else:
                        result = settlements.get(ticker)
                        if result not in ("yes", "no"):
                            continue
                    side = (r.get("side") or "").strip().lower()
                    qty = float(r.get("qty") or 0)
                    fill = float(r.get("fill_cents_est") or 0)
                    # Settled PnL: contracts settle at 100 if our side wins, 0 if it loses
                    won = (side == result)
                    pnl = (qty * (100.0 - fill) / 100.0) if won \
                        else (-qty * fill / 100.0)
                    # Prefer stored settled_pnl if present (more accurate, accounts for fees)
                    try:
                        stored = (r.get("settled_pnl") or "").strip()
                        if stored:
                            pnl = float(stored)
                    except Exception:
                        pass
                    out.append({
                        "ts":          r.get("ts_iso", ""),
                        "asset":       (r.get("asset") or "").strip().upper(),
                        "side":        side,
                        "fair_p":      float(r.get("fair_p") or 0),
                        "edge_cents":  float(r.get("edge_cents") or 0),
                        "qty":         qty,
                        "fill_cents":  fill,
                        "result":      result,
                        "settled_pnl": pnl,
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return out


def within(ts_iso: str, hours: float) -> bool:
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t) <= timedelta(hours=hours)
    except Exception:
        return False


def brier_score(fires):
    """Avg squared error between predicted probability of OUR side and actual."""
    if not fires:
        return None
    s = 0.0
    for f in fires:
        # We bought our side; the question is "did our side win?"
        # P(our side wins) = fair_p if YES, (1 - fair_p) if NO
        p = f["fair_p"] if f["side"] == "yes" else (1.0 - f["fair_p"])
        won = 1.0 if f["side"] == f["result"] else 0.0
        s += (p - won) ** 2
    return s / len(fires)


def edge_realization(fires):
    """sum(realized_pnl) / sum(predicted_edge_dollars).  1.0 = edge fully real."""
    if not fires:
        return None, 0.0, 0.0
    realized = sum(f["settled_pnl"] for f in fires)
    predicted = sum((f["edge_cents"] / 100.0) * f["qty"] for f in fires)
    if abs(predicted) < 1e-6:
        return None, realized, predicted
    return realized / predicted, realized, predicted


def win_rate(fires):
    if not fires:
        return None
    return sum(1 for f in fires if f["side"] == f["result"]) / len(fires)


def asset_verdict(brier, er, pnl, n):
    """Verdict per-asset.  Edge realization is the dominant signal (direct
    financial outcome).  Brier is informational and can flag broken cases."""
    if n < MIN_FIRES_FOR_ASSET:
        return f"insufficient ({n})"
    if er is None:
        return "no edge data"
    # Edge realization >= 0.5 AND positive money AND brier not catastrophic
    if er >= 0.5 and pnl > 0 and (brier is None or brier <= 0.35):
        return "TRADE"
    # Some edge realization, money positive, brier ok
    if er >= 0.2 and pnl > 0 and (brier is None or brier <= 0.40):
        return "careful"
    return "SKIP"


# ----- market-conditions helpers (supplementary) -----

def fetch_1m_closes(product, hours):
    end = int(time.time() * 1000)
    start = end - hours * 3600 * 1000
    start_iso = datetime.fromtimestamp(start / 1000, tz=timezone.utc).isoformat()
    end_iso   = datetime.fromtimestamp(end / 1000, tz=timezone.utc).isoformat()
    try:
        r = requests.get(f"{COINBASE}/products/{product}/candles",
                         params={"granularity": 60, "start": start_iso, "end": end_iso},
                         timeout=10)
        rows = r.json()
        rows.sort(key=lambda x: x[0])
        return [float(row[4]) for row in rows]  # close at index 4
    except Exception:
        return []


def realized_vol_pct_per_hour(closes):
    if len(closes) < 30:
        return 0.0
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    return math.sqrt(var) * math.sqrt(60) * 100  # %/hour


def market_conditions():
    """Returns dict with avg_vol_ratio, avg_trend, correlation, per-asset rows."""
    rows = []
    vol_ratios, trends = [], []
    for label, product in COINBASE_PRODUCTS:
        closes = fetch_1m_closes(product, hours=3)
        if not closes:
            rows.append((label, None, None, None))
            continue
        rv = realized_vol_pct_per_hour(closes)
        ratio = rv / NORMAL_HOURLY_VOL_PCT.get(label, 1.0)
        chg_3h = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0
        rows.append((label, chg_3h, rv, ratio))
        vol_ratios.append(ratio)
        trends.append(chg_3h)
    if not vol_ratios:
        return None
    avg_vol = sum(vol_ratios) / len(vol_ratios)
    avg_trend = sum(trends) / len(trends)
    same_dir = sum(1 for t in trends if (t > 0) == (avg_trend > 0)) / len(trends)
    return {"rows": rows, "avg_vol": avg_vol, "avg_trend": avg_trend, "correlation": same_dir}


# ----- analysis -----

def analyse():
    L = []
    now = datetime.now(timezone.utc)
    L.append(f"=== regime check  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC ===")
    L.append(f"(lookback: {LOOKBACK_HOURS}h of SETTLED fires; calibration-based)")
    L.append("")

    # ---- Section 1: per-source calibration ----
    settlements = load_settlements()
    by_source = {}
    all_fires = []
    for label, path in SOURCES:
        fires = [f for f in load_fires(path, settlements) if within(f["ts"], LOOKBACK_HOURS)]
        by_source[label] = fires
        all_fires.extend(fires)

    L.append("  source       n  edge_real  brier  win%   pnl$")
    for label, _ in SOURCES:
        fires = by_source[label]
        n = len(fires)
        if n == 0:
            L.append(f"  {label:<10} {n:>3}      --      --     --      --")
            continue
        er, realized, _ = edge_realization(fires)
        b = brier_score(fires)
        wr = win_rate(fires)
        er_s = f"{er:+.2f}" if er is not None else "  --"
        b_s  = f"{b:.3f}"   if b  is not None else "  --"
        wr_s = f"{wr*100:.0f}%" if wr is not None else "--"
        L.append(f"  {label:<10} {n:>3}    {er_s:>5}  {b_s:>5}   {wr_s:>3}  {realized:+7.2f}")
    L.append("")

    # ---- Section 2: per-asset (combined across sources) ----
    by_asset = defaultdict(list)
    for f in all_fires:
        by_asset[f["asset"]].append(f)

    L.append("  asset    n  edge_real  brier  win%   pnl$   verdict")
    for asset in ASSETS:
        fires = by_asset.get(asset, [])
        n = len(fires)
        er, realized, _ = edge_realization(fires) if n else (None, 0.0, 0.0)
        b = brier_score(fires)
        wr = win_rate(fires)
        v = asset_verdict(b, er, realized, n)
        er_s = f"{er:+.2f}" if er is not None else "  --"
        b_s  = f"{b:.3f}"   if b  is not None else "  --"
        wr_s = f"{wr*100:.0f}%" if wr is not None else "--"
        L.append(f"  {asset:<5}  {n:>3}    {er_s:>5}  {b_s:>5}   {wr_s:>3}  {realized:+7.2f}   {v}")
    L.append("")

    # ---- Section 3: overall ----
    n_all = len(all_fires)
    er_all, realized_all, predicted_all = edge_realization(all_fires) if n_all else (None, 0, 0)
    b_all = brier_score(all_fires)
    wr_all = win_rate(all_fires)

    L.append("  -- overall --")
    L.append(f"  n_settled:        {n_all}")
    er_all_s = f"{er_all:+.2f}" if er_all is not None else "--"
    b_all_s  = f"{b_all:.3f}"   if b_all  is not None else "--"
    wr_all_s = f"{wr_all*100:.0f}%" if wr_all is not None else "--"
    L.append(f"  edge realization: {er_all_s}   "
             "(1.0 = edge fully real; 0 = no edge; <0 = anti-edge)")
    L.append(f"  brier score:      {b_all_s}   "
             "(<0.20 well calibrated; 0.25 = random; >0.30 broken)")
    L.append(f"  win rate:         {wr_all_s}")
    L.append(f"  realized P&L:     ${realized_all:+.2f}  "
             f"(predicted edge ${predicted_all:+.2f})")

    # Recent 2h check for stability
    recent = [f for f in all_fires if within(f["ts"], RECENT_HOURS)]
    if len(recent) >= MIN_FIRES_FOR_ASSET:
        er_r, _, _ = edge_realization(recent)
        b_r = brier_score(recent)
        er_r_s = f"{er_r:+.2f}" if er_r is not None else "--"
        b_r_s  = f"{b_r:.3f}"   if b_r  is not None else "--"
        L.append(f"  recent {RECENT_HOURS}h:        edge_real={er_r_s}, brier={b_r_s}, n={len(recent)}")
    L.append("")

    # ---- Section 4: verdict (driven by edge realization) ----
    # Edge realization = realized $ / predicted edge $.  This is the direct
    # financial measure of whether the strategy works in current conditions.
    # Brier alone can mislead (confident-and-mostly-right > timid-and-noisy on $$
    # but loses on brier).  So edge realization drives the verdict.
    n_trade  = sum(1 for a in ASSETS if asset_verdict(brier_score(by_asset.get(a, [])),
                                                      edge_realization(by_asset.get(a, []))[0],
                                                      sum(f["settled_pnl"] for f in by_asset.get(a, [])),
                                                      len(by_asset.get(a, []))) == "TRADE")
    if n_all < MIN_FIRES_FOR_OVERALL:
        verdict = "NO SIGNAL -- too few settled fires to judge calibration"
    elif er_all is None or realized_all <= 0:
        verdict = "RED -- losing money in current regime; stay off"
    elif er_all >= 0.5 and realized_all > 0 and n_trade >= 2:
        verdict = f"GREEN -- edge realized ({n_trade} assets pass); safe to run"
    elif er_all >= 0.2 and realized_all > 0:
        verdict = f"YELLOW -- edge marginal ({n_trade} TRADE-rated); run those only, half stake"
    else:
        verdict = "RED -- edge realization too weak; stay off"
    L.append(f"  VERDICT: {verdict}")
    L.append("")

    # ---- Section 5: supplementary market conditions (forward indicator) ----
    L.append("  -- market conditions (forward-looking, supplementary) --")
    try:
        mc = market_conditions()
    except Exception as e:
        mc = None
        L.append(f"  market conditions fetch failed: {e}")
    if mc:
        for label, chg_3h, rv, ratio in mc["rows"]:
            if chg_3h is None:
                L.append(f"  {label}: no data")
            else:
                L.append(f"  {label}: 3h={chg_3h:+.2f}%  vol={rv:.2f}%/h ({ratio:.1f}x normal)")
        L.append(f"  avg vol vs normal: {mc['avg_vol']:.1f}x")
        L.append(f"  avg 3h trend:      {mc['avg_trend']:+.2f}%")
        L.append(f"  correlation:       {mc['correlation']*100:.0f}% same direction")

    return "\n".join(L)


def main():
    if "--loop" in sys.argv:
        while True:
            try:
                out = analyse()
                with open(OUT_PATH, "w") as f:
                    f.write(out + "\n")
                print(out, flush=True)
            except Exception as e:
                print(f"err: {e}", flush=True)
            time.sleep(300)
    else:
        out = analyse()
        with open(OUT_PATH, "w") as f:
            f.write(out + "\n")
        print(out)


if __name__ == "__main__":
    main()
