"""
Next-step analysis for NN14:
  (1) Sharpe on the LIVE shadow log (real fills) -- shadow_log.csv
  (3) Max drawdown on the backtest daily series -- sharpe_eval_trades.csv

Run: python3 sharpe_live_dd.py
"""
import os, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_CSV  = os.path.join(ROOT, "shadow_log.csv")
BT_CSV    = os.path.join(ROOT, "sharpe_eval_trades.csv")


def max_drawdown(daily_pnl):
    """Return (max_dd_abs, peak_to_trough_dates_idx) on cumulative PnL curve."""
    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak                 # <= 0
    trough = int(np.argmin(dd))
    peak_i = int(np.argmax(cum[:trough + 1])) if trough > 0 else 0
    return -dd.min(), peak_i, trough, cum


def sharpe_block(daily, pnl_col, label):
    print(f"\n===== {label} =====")
    print(f"Trading days: {len(daily)}")
    tot = daily[pnl_col].sum()
    print(f"Total PnL: ${tot:+,.2f}")
    print(f"Daily PnL: mean=${daily[pnl_col].mean():+.2f} "
          f"std=${daily[pnl_col].std():.2f} "
          f"min=${daily[pnl_col].min():+.0f} max=${daily[pnl_col].max():+.0f}")

    dd, pi, ti, cum = max_drawdown(daily[pnl_col].values)
    dates = daily["date"].values
    print(f"Max drawdown: ${dd:,.2f}  "
          f"(peak {dates[pi]} ${cum[pi]:,.0f} -> trough {dates[ti]} ${cum[ti]:,.0f})")
    if cum[pi] > 0:
        print(f"Max drawdown as % of peak equity: {dd / cum[pi] * 100:.1f}%")

    print("--- Sharpe (annualized sqrt(365), rf=0) ---")
    for cap_label, cap in [("$10k", 10_000), ("$25k", 25_000),
                           ("$50k", 50_000), ("$100k", 100_000)]:
        r = daily[pnl_col] / cap
        if r.std() == 0:
            continue
        sh = r.mean() / r.std() * math.sqrt(365)
        print(f"  {cap_label}: ann_ret={r.mean()*365*100:+.1f}%  "
              f"ann_vol={r.std()*math.sqrt(365)*100:.1f}%  "
              f"Sharpe={sh:.2f}  MaxDD={dd/cap*100:.1f}% of cap")


def main():
    # ---------- (1) LIVE shadow Sharpe ----------
    live = pd.read_csv(LIVE_CSV, parse_dates=["window_ts"])
    live["date"] = live["window_ts"].dt.date
    acted = live[live["total_bets"] > 0].copy()
    print(f"LIVE shadow_log.csv: {len(live)} windows, "
          f"{len(acted)} acted, {live['window_ts'].min()} -> {live['window_ts'].max()}")
    print(f"Acted win rate: {(acted['total_pnl'] > 0).mean()*100:.1f}%")
    tot_wag = acted["total_wagered"].sum()
    tot_pnl = acted["total_pnl"].sum()
    print(f"Total wagered ${tot_wag:,.2f}  total pnl ${tot_pnl:+,.2f}  "
          f"ROI on wagered {tot_pnl/tot_wag*100:+.2f}%")

    daily_live = (live.groupby("date")
                  .agg(pnl=("total_pnl", "sum"))
                  .reset_index())
    sharpe_block(daily_live, "pnl", "LIVE SHADOW (real fills, all days)")

    # Also restrict to days with at least one bet (drop dead days)
    active_dates = set(acted["date"])
    daily_live_active = daily_live[daily_live["date"].isin(active_dates)].reset_index(drop=True)
    sharpe_block(daily_live_active, "pnl", "LIVE SHADOW (active days only)")

    # ---------- (3) Backtest max drawdown ----------
    bt = pd.read_csv(BT_CSV)
    bt["date"] = pd.to_datetime(bt["ts"], unit="s", utc=True).dt.date
    daily_bt = (bt.groupby("date")
                .agg(pnl=("pnl", "sum"), wagered=("wagered", "sum"))
                .reset_index())
    sharpe_block(daily_bt, "pnl", "BACKTEST (NN14 walkforward, for drawdown)")


if __name__ == "__main__":
    main()
