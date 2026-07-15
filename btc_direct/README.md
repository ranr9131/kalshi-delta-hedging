# btc_direct — direct BTC trading research

Research sandbox for trading BTC itself (spot/perps) rather than Kalshi binaries.

## Ground rules

1. **Import-only.** This folder may import from the parent repo (feeds, data
   loaders, vol estimation) but must NEVER modify files outside `btc_direct/`.
   The parent repo contains production code running real money on the AWS box.
2. **Nothing here trades.** Research and backtests only until explicitly
   promoted — at which point this folder graduates to its own repo and its own
   process, separate from the Kalshi trader.
3. **Validation discipline carries over from `research/`:** walk-forward with
   frozen parameters, cost-inclusive P&L from day one, calibration checks,
   and a graveyard for killed ideas. No backtest result counts until it
   survives out-of-sample data and realistic costs.

## Starting hypothesis (inherited from the Kalshi project)

15-minute intra-window continuation is real as a *directional* signal
(~70% of windows that have moved by minute M settle on the same side).
The Kalshi edge monetized this through mispriced binaries, which amplify
small-move information and cap losses. Question 1 for this project: does the
signal survive **magnitude weighting** (P&L ∝ move size, not hit rate) and
**~5bps round-trip costs** when expressed as a direct BTC position?

`continuation_backtest.py` answers question 1. If the answer is no (expected),
the graveyard gets its first entry and any future work here needs a different
signal, not a different venue for this one.

## Data

Uses the parent repo's cached 1-minute Coinbase candles
(`data/cache/btc_cb_YYYYMMDD.json`, ~Feb 22 2026 onward, refreshed by the
Kalshi refit pipeline).
