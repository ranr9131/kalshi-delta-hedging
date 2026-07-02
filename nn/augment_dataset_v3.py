"""
Augment dataset_v2.npz with two physics features from fair_price_model_v3:

   14: v3_fair       — v3 fair P(YES) at this minute (trailing realized vol,
                       calibrated, 15m family)
   15: v3_minus_mid  — v3_fair - kalshi_yes_close (model-vs-market gap)

Output: nn/data/dataset_v2p.npz (X widened 14 -> 16, other arrays unchanged).

v3 is scale-invariant in log space, so fair_p is computed from the stored
cumulative return (spot = 1 + ret_t0, floor_strike = 1.0) — exactly consistent
with how the dataset was built. Trailing sigma comes from the cached Coinbase
minute closes (past-only 60-minute window ending at the decision minute).
"""

import os
import sys
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "live"))

import btc_data                      # noqa: E402
from fair_price_model_v3 import fair_p, calibration_active  # noqa: E402

IN_PATH  = os.path.join(ROOT, "data", "dataset_v2.npz")
OUT_PATH = os.path.join(ROOT, "data", "dataset_v2p.npz")

WINDOW_MINUTES = 15
VOL_SLOTS      = 60    # trailing minutes for realized sigma
VOL_MIN_N      = 30    # min 1-min returns required, else global fallback


def build_sigma_series(ts_min, ts_max):
    """Per-minute trailing realized sigma (std of 1-min log returns over the
    previous VOL_SLOTS minutes, past-only). Returns (grid0, sig_array)."""
    start = (ts_min - (VOL_SLOTS + 5) * 60) // 60 * 60
    end   = ts_max + WINDOW_MINUTES * 60 + 60
    prices = btc_data.fetch_btc_prices(start, end)

    n = (end - start) // 60 + 1
    p = np.full(n, np.nan)
    for k, v in prices.items():
        i = (int(k) - start) // 60
        if 0 <= i < n:
            p[i] = v

    logp = np.log(p)
    ret = np.full(n, np.nan)
    ret[1:] = logp[1:] - logp[:-1]          # valid only if both minutes present

    valid = np.isfinite(ret)
    r0 = np.where(valid, ret, 0.0)
    c_n  = np.concatenate([[0], np.cumsum(valid)])
    c_s  = np.concatenate([[0.0], np.cumsum(r0)])
    c_s2 = np.concatenate([[0.0], np.cumsum(r0 * r0)])

    sig = np.full(n, np.nan)
    for i in range(VOL_SLOTS, n):
        lo = i - VOL_SLOTS + 1
        cnt = c_n[i + 1] - c_n[lo]
        if cnt < VOL_MIN_N:
            continue
        s  = c_s[i + 1] - c_s[lo]
        s2 = c_s2[i + 1] - c_s2[lo]
        var = max(s2 / cnt - (s / cnt) ** 2, 0.0)
        sig[i] = np.sqrt(var)

    fallback = float(np.nanmedian(sig))
    print(f"  sigma series: {np.isfinite(sig).sum()}/{n} minutes, "
          f"median={fallback*1e4:.2f} bps/min")
    return start, sig, fallback


def main():
    z = np.load(IN_PATH, allow_pickle=True)
    X, mask, y, ts, tk = z["X"], z["mask"], z["y"], z["ts"], z["tickers"]
    n, T, F = X.shape
    print(f"Input: {n} windows x {T} min x {F} features | "
          f"calibration_active={calibration_active()}")

    print("Building trailing-vol series from cached Coinbase minutes…", flush=True)
    grid0, sig, sig_fallback = build_sigma_series(int(ts.min()), int(ts.max()))

    Xp = np.zeros((n, T, F + 2), dtype=np.float32)
    Xp[:, :, :F] = X

    n_cells = 0
    n_fallback_sig = 0
    for i in range(n):
        t0 = int(ts[i])
        for m in range(T):
            if not mask[i, m]:
                continue
            t = t0 + 60 * m
            slot = (t - grid0) // 60
            s = sig[slot] if 0 <= slot < len(sig) else np.nan
            if not np.isfinite(s) or s <= 0:
                s = sig_fallback
                n_fallback_sig += 1
            spot = 1.0 + float(X[i, m, 0])
            v3 = fair_p(spot, float(WINDOW_MINUTES - m), "BTC",
                        floor_strike=1.0, strike_type="greater_or_equal",
                        sigma_per_min=float(s), ts_epoch=float(t),
                        calibrated=True)
            Xp[i, m, F]     = v3
            Xp[i, m, F + 1] = v3 - float(X[i, m, 4])
            n_cells += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    print(f"\nComputed v3 fair for {n_cells} (window, minute) cells "
          f"({n_fallback_sig} used fallback sigma).")

    # Sanity: Brier at decision minute 10 — v3 vs market mid vs always-0.5
    sel = mask[:, 10]
    v3_10  = Xp[sel, 10, F]
    mid_10 = X[sel, 10, 4]
    lab    = y[sel]
    print(f"\nSanity @ m=10 (n={sel.sum()}):")
    print(f"  Brier v3   : {np.mean((v3_10 - lab) ** 2):.4f}")
    print(f"  Brier mid  : {np.mean((mid_10 - lab) ** 2):.4f}")
    print(f"  Brier 0.5  : {np.mean((0.5 - lab) ** 2):.4f}")
    print(f"  corr(v3, mid): {np.corrcoef(v3_10, mid_10)[0,1]:.3f}")

    np.savez_compressed(OUT_PATH, X=Xp, mask=mask, y=y, ts=ts, tickers=tk)
    print(f"\nSaved: {OUT_PATH}  X={Xp.shape}")


if __name__ == "__main__":
    main()
