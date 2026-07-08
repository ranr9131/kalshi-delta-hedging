# Iteration 6 — Operational Capstone: True P&L Under the Corrected Fee

*Reconstructs the saved paper strategies' P&L under Kalshi's actual taker fee
`ceil(0.07·C·P·(1−P))` (charged on every fill) vs the code's current model
(`count·(1−fill)·0.93`, i.e. 7% of winnings on winners only). Analysis only — no
changes made to `live/trader.py`. Script: `scratchpad/truepnl.py`.*

## Result (small samples — directional, not definitive)

```
strat        bets   cur_pnl  corr_pnl    delta   cur_fee  corr_fee
s2             42    -42.85    -43.07    -0.22     8.81     9.03
s3              3    +53.34    +55.29    +1.95     4.01     2.06
s6-latarb      18     +1.23     +4.75    +3.53    16.65    13.12
s7-velconf      6     +5.10     +5.56    +0.46     1.87     1.41
```
(Many `s6` bets dropped for missing winner — its `window_log` covers only 12 windows.)

## What it shows

1. **The fee bug's error changes sign with price.** Code model = 7% of *winnings* (winners
   only) ∝ `(1−fill)`. Real fee = `0.07·P·(1−P)` (every fill, both sides). So:
   - **High-confidence bets (fill 0.7–0.9):** the code **over-charges** itself — real fee is
     ~half (s3: 2.06 vs 4.01; s6: 13.12 vs 16.65). These strategies bet at high fill prices,
     so correcting the fee **improves** their reconstructed P&L (delta > 0).
   - **Near-coin-flip bets (fill ≈ 0.5):** real fee is **higher** than the code assumes
     (s2, which bets lower-confidence, gets slightly *worse*: delta −0.22).
2. **Direction of the bias is strategy-dependent**, so you cannot apply a flat correction —
   the fee must be recomputed per fill at `0.07·C·fill·(1−fill)`, rounded up to the cent.
3. **Magnitude is modest here** (cents to a few dollars on tiny samples) but **systematic**;
   over thousands of live bets it compounds and it shifts the relative ranking of strategies
   that bet at different price points.

## Recommended code change (when you choose to implement)
In `live/trader.py`, replace `compute_pnl`:
```python
# current
def compute_pnl(side, fill_price, count, winner):
    if winner is None: return 0.0
    if side == winner: return count * (1.0 - fill_price) * 0.93
    return -(count * fill_price)

# corrected (Kalshi taker fee charged on the trade, both outcomes)
def compute_pnl(side, fill_price, count, winner):
    if winner is None: return 0.0
    fee = math.ceil(0.07 * count * fill_price * (1.0 - fill_price) * 100) / 100.0
    payoff = count if side == winner else 0.0
    return payoff - count * fill_price - fee
```
**Caveat (verify first):** the 0.07 multiplier for the *crypto* series is unconfirmed —
pull the real fee off a live API fill before trusting it (it may be a "premium" category
above 0.07). Also confirm maker vs taker (maker is ~¼ or 0).

---

## This closes the research arc — summary of all six iterations

| # | Report | Core result |
|---|---|---|
| 1 | `..._RESEARCH.md` | Fair value ≈ `Φ(cushion/σ√τ)`; edge = gap to market; Kelly with fees; validation methodology |
| 2 | `..._2.md` | Settlement = 60s BRTI average; fee = `0.07·P·(1−P)`; execution tax; latency-arb (retail usually prey) |
| 3 | `..._3.md` | **Asian** digital (not European); optimal-stopping timing; market-making pivot; calibration fixes |
| 4 | `..._4_EMPIRICAL.md` | On real data: **+4¢ net edge** vs market; fat-tail + Asian deviations confirmed |
| 5 | `..._5_WALKFORWARD.md` | **Walk-forward PASS: +4.08¢ out-of-sample.** Edge is real, not overfit. Basis ~4 bps |
| 6 | `..._6_FEE_PNL.md` | Fee bug error changes sign with price; corrected `compute_pnl` |

**Bottom line:** the strategy's edge is **empirically validated out-of-sample (~4¢/contract
net)**. The remaining risks to live P&L are **execution slippage and the fee model — not the
edge's existence.** The highest-value next steps are operational, not research:
1. Fix `compute_pnl` (above) + verify the crypto fee multiplier from a live fill.
2. Build the last-minute **Asian pricer** `Φ((μ_A−K)/den)` for offset ≥ 14:00.
3. Replace `f_btc·f_vel·g_misprice` with a **calibrated logistic** combination → trustworthy
   Kelly sizing.
4. Measure **live fill slippage** vs paper `avg_fill` as live data accumulates — the binding
   uncertainty now.
