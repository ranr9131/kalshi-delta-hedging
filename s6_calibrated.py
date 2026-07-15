"""
S6 — calibrated-combine variant of the S2 momentum-DH strategy.

Research (MATH_STRATEGY_RESEARCH_3) flagged that S2 multiplies two sigmoids
(f_btc . g_misprice), which double-counts correlated signals and is uncalibrated,
breaking Kelly. This replaces that with:

  1. LOGISTIC COMBINE + CALIBRATION: a logistic regression fit ONLY on train
     samples that maps market-structure signals -> a calibrated P(continuation
     wins). Predictors are structural only (logit(table-fair), move magnitude,
     minute) -- price is deliberately excluded so the probability is
     market-independent and can be compared against the quoted price to get edge.

  2. FRACTIONAL KELLY sizing: for a binary at price c with calibrated win-prob p,
     the full-Kelly bankroll fraction is (p - c) / (1 - c). We stake
     KELLY_FRAC * that * BANKROLL, capped, and scale into it like S2.

Fit on train, apply frozen to test. Compared head-to-head vs S2 refit-defensive.

Run:  python s6_calibrated.py
"""

import math
import numpy as np
from sklearn.linear_model import LogisticRegression

import strategy_lab as sl
import btc_data
import kalshi_client

EPS = 1e-4


def _logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


# ── Sample collection: walk a market's ticks, emit (predictors, win-label) ──
def collect_samples(markets, btc_prices):
    """Mirror sl.simulate_market's tick loop, but record labeled training rows.

    Label = 1 if the CONTINUATION side (bet with the current BTC move) settles
    a winner. One row per valid tick T+1..T+14.
    """
    X, y = [], []
    for m in markets:
        open_dt = sl.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
        t0 = int(open_dt.timestamp())
        resolved_yes = m["result"] == "yes"
        btc_t0 = btc_data.lookup(btc_prices, t0)
        if btc_t0 is None:
            continue
        candles = kalshi_client.fetch_candlesticks(m["ticker"], m["open_time"], m["close_time"])
        if not candles:
            continue
        kt0 = candles[0].get("yes_open")
        if kt0 is None or not (0.01 < kt0 < 0.99):
            continue
        for minute in range(1, 15):
            t = t0 + minute * 60
            btc_t = btc_data.lookup(btc_prices, t)
            kalshi_yes = kalshi_client.get_yes_price_at(candles, t)
            if btc_t is None or kalshi_yes is None:
                continue
            if not (0.01 < kalshi_yes < 0.99):
                continue
            abs_pct = abs(btc_t - btc_t0) / btc_t0 * 100
            direction_up = btc_t > btc_t0
            fair = sl.get_fair(minute, abs_pct)
            X.append([_logit(fair), abs_pct, minute / 14.0])
            # continuation wins when the settled outcome agrees with the move
            y.append(1 if (resolved_yes == direction_up) else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def fit_model(markets, btc_prices):
    X, y = collect_samples(markets, btc_prices)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X, y)
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    print(f"  fit on {len(y):,} samples | base win-rate {y.mean():.3f}")
    print(f"  coef: logit(fair)={coef[0]:+.3f}  abs_pct={coef[1]:+.3f}  minute={coef[2]:+.3f}  b0={intercept:+.3f}")
    return {"coef": coef.tolist(), "intercept": float(intercept)}


def _p_win(fair, abs_pct, minute, model):
    c = model["coef"]
    z = model["intercept"] + c[0] * _logit(fair) + c[1] * abs_pct + c[2] * (minute / 14.0)
    return 1.0 / (1.0 + math.exp(-z))


# ── S6 strategy fn: calibrated prob -> fractional-Kelly stake, scale into it ──
def S6_calibrated(f, p):
    model = p["model"]
    minute = f["minute"]
    abs_pct = f["abs_pct"]
    fair = sl.get_fair(minute, abs_pct)
    p_win = _p_win(fair, abs_pct, minute, model)

    up = f["direction_up"]
    # cost of continuation contract at the EXECUTABLE price (ask side)
    price = f.get("yes_fill", f["kalshi_yes"]) if up else \
        f.get("no_fill", 1 - f["kalshi_yes"])
    edge = p_win - price
    if edge <= p["edge_min"] or price >= 1 - EPS:
        return None, 0
    kelly = edge / (1 - price)                       # full-Kelly bankroll fraction
    target = min(p["max_stake"], p["kelly_frac"] * kelly * p["bankroll"])

    if up:
        gap = max(0.0, target - f["yes_exposure"])
        return ("yes", gap) if gap >= sl.MIN_BET else (None, 0)
    else:
        gap = max(0.0, target - f["no_exposure"])
        return ("no", gap) if gap >= sl.MIN_BET else (None, 0)


# ── Reliability / calibration check on a set of markets ──
def reliability(markets, btc_prices, model, nbins=10):
    X, y = collect_samples(markets, btc_prices)
    preds = np.array([_p_win(0, 0, 0, model)] * 0)  # placeholder
    # recompute p from raw predictors stored in X: [logit(fair), abs_pct, minute/14]
    c = model["coef"]; b = model["intercept"]
    z = b + c[0] * X[:, 0] + c[1] * X[:, 1] + c[2] * X[:, 2]
    preds = 1.0 / (1.0 + np.exp(-z))
    print(f"  {'pred bucket':<14}{'n':>7}{'avg pred':>10}{'actual':>9}")
    edges = np.linspace(preds.min(), preds.max(), nbins + 1)
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi if i < nbins - 1 else preds <= hi)
        if mask.sum() < 20:
            continue
        print(f"  {lo:.2f}-{hi:.2f}     {mask.sum():>7}{preds[mask].mean():>10.3f}{y[mask].mean():>9.3f}")


def main():
    markets, btc = sl.load_data()
    n_train = int(len(markets) * sl.TRAIN_FRAC)
    train_mkts, test_mkts = markets[:n_train], markets[n_train:]
    print(f"\nTrain: {train_mkts[0]['open_time'][:10]} -> {train_mkts[-1]['open_time'][:10]} ({len(train_mkts)})")
    print(f"Test:  {test_mkts[0]['open_time'][:10]} -> {test_mkts[-1]['open_time'][:10]} ({len(test_mkts)})\n")

    print("Fitting calibrated model on TRAIN only...")
    model = fit_model(train_mkts, btc)

    print("\nCalibration on TEST (predicted vs actual win-rate):")
    reliability(test_mkts, btc, model)

    base = {"model": model, "bankroll": 1000.0, "max_stake": 150.0, "edge_min": 0.0}
    variants = [
        ({**base, "kelly_frac": 0.25}, "S6 calibrated 1/4-Kelly"),
        ({**base, "kelly_frac": 0.50}, "S6 calibrated 1/2-Kelly"),
        ({**base, "kelly_frac": 0.25, "edge_min": 0.03}, "S6 1/4-Kelly edge>=0.03"),
    ]

    print("\n" + "=" * 78)
    print("  S6 (calibrated combine + fractional Kelly)  vs  S2 refit-defensive")
    print("=" * 78)
    hdr = f"  {'variant':<34} {'split':<6} {'n':>5} {'pnl':>9} {'roi%':>7} {'win%':>6} {'sharpe':>7} {'worst':>8}"
    print(hdr)

    def show(fn, params, label):
        for split, mkts in (("train", train_mkts), ("TEST", test_mkts)):
            r = sl.backtest(mkts, btc, fn, params, label)
            if r:
                print(f"  {label:<34} {split:<6} {r['n_mkts']:>5} {r['total_pnl']:>+9.0f} "
                      f"{r['roi_pct']:>+7.2f} {r['win_rate']*100:>6.1f} {r['sharpe']:>+7.3f} {r['worst']:>+8.0f}")
        print()

    for params, label in variants:
        show(S6_calibrated, params, label)

    s2 = {"stake": sl.BASE_STAKE, "k": 30, "center": 0.05, "max_mult": 3.0, "mk": 12, "mm": 2.5}
    show(sl.S2_momentum_dh, s2, "S2 refit-defensive (ref)")


if __name__ == "__main__":
    main()
