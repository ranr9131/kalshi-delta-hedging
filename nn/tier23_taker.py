"""
Tier 2 (taker execution realism) + Tier 3 (adverse selection) on the LIVE
shadow taker logs -- no real money, real order-book-derived fills.

Inputs (live/):
  trade_log.csv  -- one row per shadow taker order: predicted mispricing,
                    actual fill_price, touch (yes_bid/yes_ask), stake, contracts
  window_log.csv -- per-window realized outcome (market_winner, settlement)

Outputs: prints execution + adverse-selection diagnostics.
"""
import os
import numpy as np
import pandas as pd

LIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live")
FEE = 0.07  # Kalshi fee on gross winnings (approx)


def main():
    tl = pd.read_csv(os.path.join(LIVE, "trade_log.csv"))
    wl = pd.read_csv(os.path.join(LIVE, "window_log.csv"))

    # keep only filled orders
    tl = tl[tl["order_result"] == "ok"].copy()
    print(f"trade_log filled orders: {len(tl)}")
    print(f"modes: {tl['mode'].value_counts().to_dict()}")
    print(f"bet_side: {tl['bet_side'].value_counts().to_dict()}")

    # join realized winner
    w = wl[["window_ts", "ticker", "market_winner"]].drop_duplicates(
        subset=["window_ts", "ticker"])
    df = tl.merge(w, on=["window_ts", "ticker"], how="inner")
    print(f"joined to outcomes: {len(df)} (dropped {len(tl)-len(df)} w/o outcome)\n")

    # contracts: prefer 'count', else stake/fill_price
    df["contracts"] = df["count"].where(df["count"] > 0,
                                        df["stake"] / df["fill_price"].clip(lower=1e-6))
    df["won"] = (df["bet_side"] == df["market_winner"]).astype(int)

    # ---- realized per-$ economics using ACTUAL fill price ----
    # buy 'side' at fill_price, get $1 if won else $0
    df["gross_per_contract"] = np.where(df["won"] == 1, 1 - df["fill_price"], -df["fill_price"])
    df["gross_pnl"] = df["gross_per_contract"] * df["contracts"]
    # fee on winnings only
    df["fee"] = np.where(df["won"] == 1, FEE * (1 - df["fill_price"]) * df["contracts"], 0.0)
    df["net_pnl"] = df["gross_pnl"] - df["fee"]
    df["cost"] = df["fill_price"] * df["contracts"]

    # ---- TIER 2: execution realism (real fill vs touch vs mid+4c assumption) ----
    # touch for the side actually taken
    yes_ask, yes_bid = df["yes_ask"], df["yes_bid"]
    df["side_touch"] = np.where(df["bet_side"] == "yes", yes_ask, 1 - yes_bid)   # best available ask for side
    df["side_mid"]   = np.where(df["bet_side"] == "yes", df["kalshi_yes_mid"], 1 - df["kalshi_yes_mid"])
    df["slip_vs_touch"] = df["fill_price"] - df["side_touch"]      # >0 = paid worse than touch
    df["slip_vs_mid"]   = df["fill_price"] - df["side_mid"]        # real cost vs mid
    backtest_assumed = df["side_mid"] + 0.04                       # the sim's mid+4c
    df["real_minus_assumed"] = df["fill_price"] - backtest_assumed # >0 = sim too optimistic

    print("===== TIER 2: TAKER EXECUTION REALISM =====")
    print(f"Avg fill price: {df['fill_price'].mean():.4f}")
    print(f"Slippage vs touch (fill - best ask):  mean={df['slip_vs_touch'].mean()*100:+.2f}c  "
          f"median={df['slip_vs_touch'].median()*100:+.2f}c")
    print(f"Real cost vs mid (fill - mid):         mean={df['slip_vs_mid'].mean()*100:+.2f}c  "
          f"median={df['slip_vs_mid'].median()*100:+.2f}c")
    print(f"Real fill vs backtest's (mid+4c):      mean={df['real_minus_assumed'].mean()*100:+.2f}c  "
          f"(>0 => backtest was optimistic)")
    print(f"Avg spread at decision: {df['spread'].mean()*100:.2f}c")

    # ---- TIER 3: adverse selection ----
    # predicted edge: mispricing column = model fair_for_side - price (verify via calibration)
    df["pred_edge"] = df["mispricing"]
    df["pred_fair"] = (df["fill_price"] + df["pred_edge"]).clip(0, 1)   # implied predicted win prob
    df["realized_edge"] = df["won"] - df["fill_price"]                  # actual edge captured per $1

    print("\n===== TIER 3: ADVERSE SELECTION =====")
    print(f"Trades: {len(df)}  |  realized win rate: {df['won'].mean()*100:.1f}%")
    print(f"Predicted fair (avg win prob model expected): {df['pred_fair'].mean():.3f}")
    print(f"  -> if calibrated, realized win rate ~= predicted fair")
    print(f"Predicted edge (mispricing):  mean={df['pred_edge'].mean()*100:+.2f}c")
    print(f"Realized edge (won - fill):   mean={df['realized_edge'].mean()*100:+.2f}c")
    gap = df['pred_edge'].mean() - df['realized_edge'].mean()
    print(f"  EDGE DECAY (predicted - realized): {gap*100:+.2f}c per $1  "
          f"({'edge survives' if df['realized_edge'].mean()>0 else 'EDGE GONE after fills'})")

    # Brier of predicted fair vs outcome on FIRED trades
    brier = ((df["pred_fair"] - df["won"]) ** 2).mean()
    print(f"Brier on fired trades (pred_fair vs win): {brier:.3f}")

    # bucket by predicted edge size -> does bigger 'edge' actually pay?
    print("\nRealized edge by predicted-edge bucket:")
    df["edge_bucket"] = pd.cut(df["pred_edge"], [-1, 0, 0.05, 0.10, 0.20, 1],
                               labels=["<0", "0-5c", "5-10c", "10-20c", ">20c"])
    g = df.groupby("edge_bucket", observed=True).agg(
        n=("won", "size"), pred_edge_c=("pred_edge", lambda s: s.mean()*100),
        realized_edge_c=("realized_edge", lambda s: s.mean()*100),
        win=("won", lambda s: s.mean()*100), net=("net_pnl", "sum"))
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    # bucket by |move| -> adverse selection worst on fast moves?
    print("\nRealized edge by abs price move at decision:")
    df["mv_bucket"] = pd.cut(df["abs_pct_move"], [0, 0.03, 0.06, 0.12, 10],
                             labels=["<3bp", "3-6bp", "6-12bp", ">12bp"])
    g2 = df.groupby("mv_bucket", observed=True).agg(
        n=("won", "size"), realized_edge_c=("realized_edge", lambda s: s.mean()*100),
        win=("won", lambda s: s.mean()*100), net=("net_pnl", "sum"))
    print(g2.to_string(float_format=lambda x: f"{x:.2f}"))

    # ---- bottom line P&L with real fills ----
    print("\n===== BOTTOM LINE (real fills) =====")
    print(f"Total cost (capital cycled): ${df['cost'].sum():,.0f}")
    print(f"Gross PnL: ${df['gross_pnl'].sum():+,.2f}  ROI {df['gross_pnl'].sum()/df['cost'].sum()*100:+.1f}%")
    print(f"Net PnL (after {int(FEE*100)}% fee): ${df['net_pnl'].sum():+,.2f}  "
          f"ROI {df['net_pnl'].sum()/df['cost'].sum()*100:+.1f}%")

    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier23_taker_trades.csv"), index=False)
    print(f"\nWrote per-trade detail -> nn/tier23_taker_trades.csv")


if __name__ == "__main__":
    main()
