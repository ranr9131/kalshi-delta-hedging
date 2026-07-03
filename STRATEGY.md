# S6-Kelly: BTC 15-Minute Binary Strategy

*Last updated: 2026-07-03. Live (small-stakes) since 2026-07-02 20:02 UTC.*

## What it trades

Kalshi `KXBTC15M` markets — binary contracts on "will BTC be **up** over this
15-minute window?", one market every 15 minutes (~96/day). A contract costs its
price in dollars (e.g. YES @ $0.64) and pays $1.00 if right, $0 if wrong.
Settlement is a 60-second average of the CF Benchmarks BRTI index at window
close.

## The edge

**Kalshi systematically underprices intra-window continuation.** When BTC has
moved (say, down 0.05% by minute 5), the probability the move persists to the
close is higher than the market's price implies. Validated by walk-forward
test (train on the earlier ⅔ of ~90 days of data, test frozen on the later ⅓):
**≈ +4¢/contract net edge out-of-sample**, consistent cell-by-cell
(`research/MATH_STRATEGY_RESEARCH_5_WALKFORWARD.md`).

The edge is largest near even-money prices, where market makers quote widest
and retail anchors to 50¢.

## The pipeline (each 30-second decision tick)

```
                    BTC spot (Coinbase WS)        Kalshi bid/ask (WS)
                            │                            │
                            ▼                            ▼
 1. FAIR VALUE   2D table lookup: (minute, |move| bucket) → historical win rate
                            │
                            ▼
 2. CALIBRATED   logistic model: p_win = σ( a·logit(fair) + b·|move| + c·minute + d )
    PROBABILITY  fit on 79k historical ticks; verified calibrated out-of-sample
                            │
                            ▼
 3. EDGE GATE    edge = p_win − cost − fee
                 cost = ask + 1¢ buffer   fee = 0.07·cost·(1−cost)  [exchange-verified]
                 edge ≤ min → bet $0. No exceptions.
                            │
                            ▼
 4. KELLY SIZE   kelly = edge / (1 − cost)
                 stake = min(MAX_STAKE, ¼ · kelly · BANKROLL)   → bigger edge, bigger bet
                            │
                            ▼
 5. EXECUTION    IOC chase ladder (V2 API):
                 IOC at base buffer → partial fills count → chase remainder
                 (+2¢, +4¢) → STOP at edge cap (never pay past where edge dies)
```

Direction is always **with** the current move (continuation): BTC up → buy YES,
BTC down → buy NO. Buying NO at price p is submitted as an ask on the single
YES book at 1−p (V2 API semantics).

## Fees (verified against real fills)

`fee = 0.07 · C · P · (1−P)` per contract, charged on **every** fill (win or
lose), rounded sub-cent. Peaks at P=0.50 (~1.75¢/contract), → 0 near the
extremes. The old "7% of winnings" model in earlier code was wrong (and
pessimistic). Confirmed 2026-07-02: 1 contract @ 0.51 → fee $0.0175 exactly.

## Execution economics (measured, not assumed)

| Quantity | Measured value | How |
|---|---|---|
| Order round-trip (AWS us-east-1 → Kalshi) | 27 ms median, 90 ms p99 | latency probe, resting-limit round-trips |
| Book walk at ≤200 contracts | **0.0¢** (book is deep) | passive depth logger, thousands of snapshots |
| Half-spread | ~0.5¢ | same |
| Fill buffer | **1¢** (was 5¢) | 5¢ was sized for ~300ms latency that no longer exists; it erased most edges (hurdle ~7¢ vs ~4¢ gross edge) |

The buffer cut is the single biggest fix: identical model with a 5¢ buffer
finds ~¼ the positive-edge ticks of the 1¢ version.

## Risk controls

- **¼-Kelly** (not full) — model probabilities are imperfect; overbetting a
  wrong edge is ruinous, underbetting only costs growth.
- **Per-window wagered cap** — hard truncation of total exposure per market.
- **Per-window leg cap** — max 2 entries per window.
- **Daily loss cap** — auto-pauses trading for the day when hit.
- **Edge cap on execution** — the chase ladder never pays a price at which
  the net edge would fall below minimum.
- **Reversal hedge** — optional late-window hedge if the position moves against.

## Current deployment (AWS, `kalshi-trader` box)

| Arm | Mode | Buffer | Purpose |
|---|---|---|---|
| `paper-s6kelly` | paper | 5¢ | control: old friction assumptions |
| `paper-s6kelly-1c` | paper | 1¢ | buffer A/B treatment |
| `live-s6kelly` | **real money** | 1¢ | half-scale live validation |

Launch (env-driven, no code changes per arm):

```bash
LOG_TAG=live-s6kelly SIZING=s6kelly FILL_BUFFER_CENTS=1 PAPER_MODE=false \
S6_BANKROLL=500 S6_MAX_STAKE=30 MAX_WINDOW_WAGERED=15 DAILY_LOSS_CAP=20 \
python3.11 -u trader.py
```

Alongside: `book_logger.py` (full order-book depth snapshots) and
`recorder.py` (every WS message, ~850MB/day) building the corpus for an
event-driven replay backtest.

## Key files

| File | Role |
|---|---|
| `live/trader.py` | main loop: sizing engines (`SIZING=sigmoid\|s6kelly`), IOC chase execution, guardrails |
| `live/s6_calibration.json` | fitted logistic coefficients (79k samples, 90d) |
| `live/kalshi_trade.py` | V2 order API (create/cancel/fills), fee-verified |
| `live/kalshi_feed.py` | WS feed: top-of-book + full depth + `expected_fill(side, n)` |
| `data/logs/minute_analysis_2d.csv` | 2D fair-value table (runtime dependency) |
| `s6_calibrated.py` | research prototype + backtest of the calibrated/Kelly approach |
| `slippage_report.py`, `live/depth_slippage.py`, `live/latency_probe.py` | execution measurement tools |
| `research/MATH_STRATEGY_RESEARCH*.md` | the underlying math research (pricing, fees, settlement, validation) |

## Honest caveats

1. **The probability model is fit on Feb–May 2026 data.** Regime drift is the
   main threat to the edge; refit on recent data is planned, not done.
2. **Live sample is small.** One strongly-trending session of live results;
   the strategy's worst weather is chop (whipsaw windows lose full stakes at
   Kelly size). Judge after 100+ live trades including bad days.
3. **Settlement nuance unmodeled.** The contract is an Asian-style digital
   (60s average), not European — the current pricer is instantaneous-spot. The
   discrepancy matters most in the final minute (research report 3 has the
   correct estimator; not yet implemented).
4. **Backtest ≠ live.** The predecessor sigmoid strategy showed +55% ROI in
   backtest and −16% in live paper before the frictions were found and fixed.
   The replay backtest (from the recorded corpus) is the planned guard against
   repeating that.
