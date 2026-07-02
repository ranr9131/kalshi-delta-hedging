"""
Tier 2 (maker) -- conservative market-making fill replay on REAL recorded book.

Input: live/mm_shadow_snapshots.csv
  ts, ticker, asset, spot, strike, mins_left, bb_c, ba_c, bb_sz, ba_sz,
  spread_c, my_bid_c, my_ask_c, fair_c, quote_age_s   (prices in CENTS, 0-100)

Conservative fill model (back-of-queue, no optimism):
  - I rest a BID at my_bid_c (buy YES). It fills ONLY if a later snapshot of the
    SAME ticker shows best ask <= my_bid_c  (market actually traded down through
    my resting price -> someone hit a price at/below me).
  - I rest an ASK at my_ask_c (sell YES). It fills ONLY if best bid >= my_ask_c.
  - On fill I hold to settlement. PnL per YES contract:
        bought YES @ b  -> outcome - b
        sold   YES @ a  -> a - outcome
  - Only count one fill per side per ticker (first time it triggers).

Outcomes resolved from fv/corpus*.csv + live/window_log.csv (ticker -> 0/1).
"""
import os
import numpy as np
import pandas as pd

LIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live")
SNAP = os.path.join(LIVE, "mm_shadow_snapshots.csv")
FEE = 0.07


def load_outcomes():
    m = {}
    for f in ["fv/corpus.csv", "fv/corpus_15m_extra.csv"]:
        p = os.path.join(LIVE, f)
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=["ticker", "outcome"])
            d = d.dropna(subset=["outcome"]).drop_duplicates("ticker")
            m.update(dict(zip(d["ticker"], d["outcome"].astype(int))))
    wl = os.path.join(LIVE, "window_log.csv")
    if os.path.exists(wl):
        d = pd.read_csv(wl, usecols=["ticker", "market_winner"]).dropna()
        for t, w in zip(d["ticker"], d["market_winner"]):
            m.setdefault(t, 1 if str(w).lower() == "yes" else 0)
    return m


def main():
    outc = load_outcomes()
    print(f"outcome map: {len(outc)} tickers")

    cols = ["ts", "ticker", "asset", "mins_left", "bb_c", "ba_c",
            "my_bid_c", "my_ask_c", "fair_c"]
    df = pd.read_csv(SNAP, usecols=cols)
    print(f"snapshots: {len(df):,}")

    # focus on short-dated (15-min-style) markets and rows where we actually quoted
    df = df[(df["mins_left"] <= 20) & (df["mins_left"] >= 0)]
    df = df[(df["my_bid_c"] > 0) | (df["my_ask_c"] > 0)]
    df = df.dropna(subset=["bb_c", "ba_c"])
    df["outcome"] = df["ticker"].map(outc)
    df = df.dropna(subset=["outcome"])
    df["outcome"] = df["outcome"].astype(int)
    print(f"after 15m + quoted + resolved filter: {len(df):,} rows, "
          f"{df['ticker'].nunique():,} tickers, assets={df['asset'].value_counts().to_dict()}")
    if len(df) == 0:
        print("No usable rows."); return

    df = df.sort_values(["ticker", "ts"])
    fills = []
    for tk, g in df.groupby("ticker", sort=False):
        out = int(g["outcome"].iloc[0])
        ba = g["ba_c"].values; bb = g["bb_c"].values
        mybid = g["my_bid_c"].values; myask = g["my_ask_c"].values
        fair = g["fair_c"].values
        bid_done = ask_done = False
        for i in range(len(g)):
            # resting BID fills if later best ask trades down to <= my bid
            if (not bid_done) and mybid[i] > 0:
                future_ask = ba[i + 1:]
                if future_ask.size and np.nanmin(future_ask) <= mybid[i]:
                    b = mybid[i] / 100.0
                    fills.append(("bid", tk, b, out, out - b, fair[i] / 100.0))
                    bid_done = True
            # resting ASK fills if later best bid trades up to >= my ask
            if (not ask_done) and myask[i] > 0:
                future_bid = bb[i + 1:]
                if future_bid.size and np.nanmax(future_bid) >= myask[i]:
                    a = myask[i] / 100.0
                    fills.append(("ask", tk, a, out, a - out, fair[i] / 100.0))
                    ask_done = True
            if bid_done and ask_done:
                break

    f = pd.DataFrame(fills, columns=["side", "ticker", "price", "outcome", "pnl", "fair"])
    if len(f) == 0:
        print("No maker fills triggered."); return
    # net of fee on winning legs
    f["won"] = f["pnl"] > 0
    f["fee"] = np.where(f["won"], FEE * f["pnl"].abs(), 0.0)
    f["net_pnl"] = f["pnl"] - f["fee"]
    # predicted edge of the quote vs fair at quote time
    f["pred_edge"] = np.where(f["side"] == "bid", f["fair"] - f["price"],
                              f["price"] - f["fair"])

    print("\n===== TIER 2: CONSERVATIVE MAKER FILLS (real book, back-of-queue) =====")
    print(f"Total maker fills: {len(f)}  (bid {sum(f.side=='bid')} / ask {sum(f.side=='ask')})")
    print(f"Fill win rate (held to settlement): {f['won'].mean()*100:.1f}%")
    print(f"Avg predicted edge at quote (fair-price): {f['pred_edge'].mean()*100:+.2f}c")
    print(f"Avg REALIZED pnl per contract (gross):    {f['pnl'].mean()*100:+.2f}c")
    print(f"Avg REALIZED pnl per contract (net fee):  {f['net_pnl'].mean()*100:+.2f}c")
    print(f"Edge decay (predicted - realized gross):  {(f['pred_edge'].mean()-f['pnl'].mean())*100:+.2f}c")
    print(f"\nTotal gross PnL (1 contract/fill): ${f['pnl'].sum():+,.2f}")
    print(f"Total net PnL:                     ${f['net_pnl'].sum():+,.2f}")

    print("\nBy asset:")
    g = f.groupby(f["ticker"].str.split("-").str[0]).agg(
        n=("pnl", "size"), win=("won", lambda s: s.mean()*100),
        pred_c=("pred_edge", lambda s: s.mean()*100),
        realized_c=("pnl", lambda s: s.mean()*100),
        net_total=("net_pnl", "sum"))
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    f.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier2_maker_fills.csv"), index=False)
    print(f"\nWrote -> nn/tier2_maker_fills.csv")


if __name__ == "__main__":
    main()
