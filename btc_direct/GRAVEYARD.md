# Graveyard — killed ideas

## 2. Turn-of-the-candle anomaly (killed 2026-07-06)

**Idea:** deep-research candidate #1 (PMC10015199, verified 3-0): BTC 1-min
returns concentrate at minutes 0/15/30/45 (+0.58 bp/min through Aug 2022).

**Test:** `turn_of_candle_backtest.py`, 132 days of 2026 Coinbase 1-min data
(12,527 turn-minutes vs 175,379 others).

**Result:** effect has decayed to zero post-publication: turn minutes
+0.013 bp/min (t=0.21) vs others −0.004 bp/min. Spread +0.017 bp/min ≈ 1/34th
of the published figure and statistically indistinguishable from noise —
before even discussing the 5 bp cost hurdle.

**Conclusion:** textbook post-publication anomaly decay. Do not revisit
without evidence it re-emerged.

## 1. Magnitude-weighted continuation (killed 2026-07-04)

**Idea:** the Kalshi project's validated signal — ~70% of windows that have
moved by minute M settle on the same side — expressed as a direct BTC
position: enter in the move's direction at minute M, exit at window close.

**Test:** `continuation_backtest.py`, 132 days of 1-min Coinbase closes
(~12,500 windows), sweep of entry minute ∈ {2,4,6,8,10} × entry-move filter
∈ {0,1,2,5,10 bps}, 5 bps round-trip cost. No fitted parameters.

**Result:** dead on arrival, and not because of costs.
- Hit rate 48–50% across every cell (vs 70% on Kalshi) — because the binary
  pays when price merely *stays* on the strike's side, while a direct position
  needs it to *keep moving*. Post-entry drift is a coin flip: the continuation
  edge lives entirely in "already banked" distance, which a binary monetizes
  and a linear position cannot.
- Gross expectancy ≈ 0 bps/trade (−0.5 to +0.9 in the extremes, no structure).
- Net after 5 bps: ≈ −5 bps/trade, i.e. −$200 to −$500/day per $10k notional.

**Conclusion:** the Kalshi edge does not port to direct BTC. Any future work
in this folder needs a genuinely new signal; the venue was never the problem.
