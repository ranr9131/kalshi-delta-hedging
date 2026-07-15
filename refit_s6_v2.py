"""
S6 model refit v2 — walk-forward validated, with a direction feature.

Discipline:
  - TRAIN  = markets older than the last 14 days
  - HOLDOUT = the most recent 14 days (never touched during fitting)
  - The 2D table is rebuilt FROM TRAIN ONLY, then the logistic is fit on train
    ticks using that table. Holdout metrics therefore measure the whole
    pipeline out-of-sample, exactly as it would have been deployed 2 weeks ago.
  - Old model (live/s6_calibration.json + data/logs/minute_analysis_2d.csv,
    both fit on Feb-May data) is scored on the same holdout for comparison.

New predictor: direction_up (0/1). Recent 5.3k-window measurement says
continuation is direction-symmetric (~70% both ways), so its coefficient
should land near zero — it's insurance, not alpha.

Outputs (only after validation passes):
  - live/s6_calibration_v2.json      (4-coef logistic + fit metadata)
  - data/logs/minute_analysis_2d_v2.csv  (production table, ALL 90d)

Run: python refit_s6_v2.py
"""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

import btc_data
import kalshi_client

HOLDOUT_DAYS = 14
EPS = 1e-4
BUCKETS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 99.0)]
BUCKET_NAMES = ["0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+"]
MIN_N = 30
FALLBACK = 0.65


def bucket_idx(abs_pct):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= abs_pct < hi:
            return i
    return len(BUCKETS) - 1


def logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def collect_ticks(markets, btc_prices):
    """One row per (market, minute 1..14): [minute, abs_pct, dir_up, won]."""
    rows = []
    for m in markets:
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        resolved_yes = m["result"] == "yes"
        b0 = btc_data.lookup(btc_prices, t0)
        if b0 is None:
            continue
        for minute in range(1, 15):
            bt = btc_data.lookup(btc_prices, t0 + minute * 60)
            if bt is None or bt == b0:
                continue
            abs_pct = abs(bt - b0) / b0 * 100
            up = bt > b0
            won = 1 if (resolved_yes == up) else 0
            rows.append((minute, abs_pct, 1 if up else 0, won))
    return rows


def build_table(ticks):
    """(minute, bucket) -> win rate, honoring the MIN_N rule."""
    agg = {}
    for minute, abs_pct, up, won in ticks:
        k = (minute, bucket_idx(abs_pct))
        n, w = agg.get(k, (0, 0))
        agg[k] = (n + 1, w + won)
    return {k: w / n for k, (n, w) in agg.items() if n >= MIN_N}, \
           {k: n for k, (n, _) in agg.items()}


def table_fair(table, minute, abs_pct):
    return table.get((minute, bucket_idx(abs_pct)), FALLBACK)


def features(ticks, table, with_direction):
    X, y = [], []
    for minute, abs_pct, up, won in ticks:
        f = table_fair(table, minute, abs_pct)
        row = [logit(f), abs_pct, minute / 14.0]
        if with_direction:
            row.append(float(up))
        X.append(row)
        y.append(won)
    return np.array(X), np.array(y)


def old_model_preds(ticks):
    """Score the CURRENTLY DEPLOYED model (old table + old 3-coef logistic)."""
    import csv as _csv
    old_table = {}
    bucket_map = {n: i for i, n in enumerate(BUCKET_NAMES)}
    with open("data/logs/minute_analysis_2d.csv") as f:
        for r in _csv.reader(f):
            try:
                minute, bi, n, wr = int(r[0]), bucket_map.get(r[1]), int(r[2]), float(r[3])
                if bi is not None and n >= MIN_N:
                    old_table[(minute, bi)] = wr
            except (ValueError, IndexError):
                continue
    cal = json.load(open("live/s6_calibration.json"))
    c, b = cal["coef"], cal["intercept"]
    preds = []
    for minute, abs_pct, up, won in ticks:
        f = table_fair(old_table, minute, abs_pct)
        z = b + c[0] * logit(f) + c[1] * abs_pct + c[2] * (minute / 14.0)
        preds.append(1 / (1 + math.exp(-z)))
    return np.array(preds)


def reliability(preds, y, label):
    print(f"\n  {label}  (n={len(y):,}, Brier={np.mean((preds - y) ** 2):.4f})")
    print(f"  {'bucket':<12}{'n':>8}{'pred':>8}{'actual':>8}{'gap':>7}")
    edges = np.linspace(0.5, 1.0, 11)
    for i in range(10):
        m = (preds >= edges[i]) & (preds < edges[i + 1])
        if m.sum() < 50:
            continue
        gap = y[m].mean() - preds[m].mean()
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}   {m.sum():>8}{preds[m].mean():>8.3f}{y[m].mean():>8.3f}{gap:>+7.3f}")


def main():
    markets = kalshi_client.fetch_settled_markets(days=90)
    markets = [m for m in markets if m.get("result") in ("yes", "no") and m.get("open_time")]
    markets.sort(key=lambda m: m["open_time"])
    tmax = datetime.fromisoformat(markets[-1]["open_time"].replace("Z", "+00:00"))
    cut = tmax - timedelta(days=HOLDOUT_DAYS)
    train_m = [m for m in markets if datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")) < cut]
    hold_m = [m for m in markets if datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")) >= cut]
    print(f"markets: {len(markets)} | train {train_m[0]['open_time'][:10]}..{train_m[-1]['open_time'][:10]} "
          f"({len(train_m)}) | holdout {hold_m[0]['open_time'][:10]}.. ({len(hold_m)})")

    ts = [int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()) for m in markets]
    btc = btc_data.fetch_btc_prices(min(ts) - 600, max(ts) + 1800)
    print(f"btc points: {len(btc):,}")

    train_ticks = collect_ticks(train_m, btc)
    hold_ticks = collect_ticks(hold_m, btc)
    print(f"ticks: train {len(train_ticks):,} | holdout {len(hold_ticks):,}")

    # ── Walk-forward fit: table + logistic from TRAIN only ──
    table_tr, counts = build_table(train_ticks)
    print(f"train table cells (n>={MIN_N}): {len(table_tr)}")

    Xtr, ytr = features(train_ticks, table_tr, with_direction=True)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
    print("coef [logit(fair), abs_pct, minute/14, dir_up]:",
          [round(v, 4) for v in clf.coef_[0]], "b0:", round(clf.intercept_[0], 4))

    # ── Holdout comparison: old deployed model vs new ──
    Xh, yh = features(hold_ticks, table_tr, with_direction=True)
    new_preds = clf.predict_proba(Xh)[:, 1]
    old_preds = old_model_preds(hold_ticks)
    print("\n=== HOLDOUT (last 14 days, never seen by either fit) ===")
    reliability(old_preds, yh, "OLD (deployed: Feb-May fit)")
    reliability(new_preds, yh, "NEW (walk-forward: Apr-Jun fit + direction)")

    improve = np.mean((old_preds - yh) ** 2) - np.mean((new_preds - yh) ** 2)
    print(f"\nBrier improvement (old - new): {improve:+.5f}  ({'NEW wins' if improve > 0 else 'OLD wins'})")

    # ── Production artifacts: refit on ALL data ──
    table_all, counts_all = build_table(train_ticks + hold_ticks)
    Xa, ya = features(train_ticks + hold_ticks, table_all, with_direction=True)
    clf_all = LogisticRegression(C=1.0, max_iter=1000).fit(Xa, ya)

    with open("data/logs/minute_analysis_2d_v2.csv", "w") as f:
        f.write("minute,bucket,n,win_rate,avg_fill\n")
        for (minute, bi), wr in sorted(table_all.items()):
            f.write(f"{minute},{BUCKET_NAMES[bi]},{counts_all[(minute, bi)]},{wr:.6f},{wr:.6f}\n")

    out = {
        "model": "logistic",
        "predictors": ["logit_fair_2d", "abs_pct", "minute_frac14", "direction_up"],
        "coef": clf_all.coef_[0].tolist(),
        "intercept": float(clf_all.intercept_[0]),
        "n_samples": int(len(ya)),
        "base_rate": float(ya.mean()),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "data_window": f"{markets[0]['open_time'][:10]}..{markets[-1]['open_time'][:10]}",
        "table_file": "minute_analysis_2d_v2.csv",
        "holdout_brier_new": float(np.mean((new_preds - yh) ** 2)),
        "holdout_brier_old": float(np.mean((old_preds - yh) ** 2)),
    }
    with open("live/s6_calibration_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote live/s6_calibration_v2.json + data/logs/minute_analysis_2d_v2.csv")
    print("production coef:", [round(v, 4) for v in out["coef"]], "b0:", round(out["intercept"], 4))


if __name__ == "__main__":
    main()
