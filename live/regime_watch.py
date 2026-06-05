"""
Crypto regime watcher.  Every 5 min, computes realized vol + directional
bias across BTC/ETH/SOL/XRP/HYPE and writes a clear status to regime.txt.

Usage:
  python3.11 regime_watch.py             # one-shot status
  python3.11 regime_watch.py --loop      # runs forever, updating every 5 min
"""
from __future__ import annotations
import os, sys, time, math, requests
from datetime import datetime, timezone


COINBASE = "https://api.exchange.coinbase.com"
ASSETS = [("BTC","BTC-USD"), ("ETH","ETH-USD"), ("SOL","SOL-USD"),
          ("XRP","XRP-USD"), ("HYPE","HYPE-USD")]

# Per-asset "normal" hourly vol (rough — calibrated to last 30d of typical regimes)
NORMAL_HOURLY_VOL_PCT = {"BTC": 0.6, "ETH": 0.8, "SOL": 1.2, "XRP": 1.0, "HYPE": 1.5}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regime.txt")


def fetch_1m_candles(product: str, hours: int) -> list:
    """Returns [[t_ms, open, high, low, close, vol], ...] ascending."""
    end = int(time.time() * 1000)
    start = end - hours * 3600 * 1000
    start_iso = datetime.fromtimestamp(start/1000, tz=timezone.utc).isoformat()
    end_iso   = datetime.fromtimestamp(end/1000, tz=timezone.utc).isoformat()
    try:
        r = requests.get(f"{COINBASE}/products/{product}/candles",
                         params={"granularity":60,"start":start_iso,"end":end_iso}, timeout=10)
        rows = r.json()
        # Coinbase returns descending [t_sec, low, high, open, close, vol]
        rows.sort(key=lambda x: x[0])
        return [[int(r[0])*1000, float(r[3]), float(r[2]), float(r[1]),
                 float(r[4]), float(r[5])] for r in rows]
    except Exception:
        return []


def realized_vol_pct_per_hour(closes: list) -> float:
    """Stdev of 1-min log returns scaled to %/hour."""
    if len(closes) < 30: return 0.0
    rets = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0 and closes[i] > 0:
            rets.append(math.log(closes[i] / closes[i-1]))
    if len(rets) < 2: return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r-mean)**2 for r in rets) / max(1, len(rets)-1)
    stdev_per_min = math.sqrt(var)
    return stdev_per_min * math.sqrt(60) * 100  # → %/hour


def analyse():
    lines = []
    now = datetime.now(timezone.utc)
    lines.append(f"=== regime check  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC ===\n")

    overall = {"vol_ratio": [], "trend": []}
    for label, product in ASSETS:
        candles = fetch_1m_candles(product, hours=3)
        if not candles:
            lines.append(f"  {label}: no data")
            continue
        closes = [c[4] for c in candles]
        rv = realized_vol_pct_per_hour(closes)
        normal = NORMAL_HOURLY_VOL_PCT.get(label, 1.0)
        ratio = rv / normal if normal else 0
        change_3h = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0
        change_1h = (closes[-1] - closes[max(0, len(closes)-60)]) / closes[max(0, len(closes)-60)] * 100
        overall["vol_ratio"].append(ratio)
        overall["trend"].append(change_3h)
        lines.append(f"  {label}: 1h={change_1h:+.2f}%  3h={change_3h:+.2f}%  "
                     f"vol={rv:.2f}%/h ({ratio:.1f}× normal)")

    if not overall["vol_ratio"]:
        return "no data"

    avg_vol_ratio = sum(overall["vol_ratio"]) / len(overall["vol_ratio"])
    avg_trend = sum(overall["trend"]) / len(overall["trend"])
    same_dir = sum(1 for t in overall["trend"] if (t > 0) == (avg_trend > 0))
    correlation = same_dir / len(overall["trend"])

    lines.append("")
    lines.append(f"  ─── aggregate ───")
    lines.append(f"  avg vol vs normal: {avg_vol_ratio:.1f}×")
    lines.append(f"  avg 3h trend:      {avg_trend:+.2f}%")
    lines.append(f"  correlation:       {correlation*100:.0f}% of assets moving same direction")
    lines.append("")

    # Verdict
    if avg_vol_ratio < 1.3 and abs(avg_trend) < 1.5 and correlation < 0.8:
        verdict = "🟢 NORMAL — safe to restart sniper"
    elif avg_vol_ratio < 1.8 and abs(avg_trend) < 2.5:
        verdict = "🟡 ELEVATED — risky but recoverable, your call"
    else:
        verdict = "🔴 HOSTILE — same regime that bled us, keep sniper OFF"
    lines.append(f"  VERDICT: {verdict}")

    return "\n".join(lines)


def main():
    if "--loop" in sys.argv:
        while True:
            try:
                out = analyse()
                with open(OUT_PATH, "w") as f:
                    f.write(out + "\n")
                print(out)
            except Exception as e:
                print(f"err: {e}")
            time.sleep(300)
    else:
        out = analyse()
        with open(OUT_PATH, "w") as f:
            f.write(out + "\n")
        print(out)


if __name__ == "__main__":
    main()
