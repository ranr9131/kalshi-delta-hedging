# Iteration 5 — Walk-Forward Validation: The Edge Survives Out-of-Sample

*Iteration 4 found a ~4¢ net edge but flagged it as in-sample and possibly overfit,
and named the walk-forward test as "the single decisive next experiment." This
iteration runs it, on the repo's own offline data (6,423 settled markets,
2026-02-22 → 2026-05-02; per-minute Coinbase prices; per-ticker candle caches).*

> Script: `scratchpad/walkforward.py`. Fully reproducible offline from
> `data/cache/markets_90d.json` + `candles_*.json` + `btc_cb_*.json`.

---

## Result: PASS — the edge is out-of-sample real

Rebuilt the win-rate table on the **earlier 2/3 of dates** (train: 50,923 obs through
2026-04-08), froze it, and measured realized net edge on the **later 1/3** (test:
26,285 obs from 2026-04-09 on). Net edge per contract = `win_rate − fill − 0.07·c·(1−c)`.

| Measure | Net edge | Obs |
|---|---|---|
| **All test cells** (n-weighted) | **+3.92¢** | 26,151 |
| **Train-gated** (only cells the train table flagged +EV) | **+4.08¢** | 24,950 (95% of test) |

**The in-sample ~4¢ edge reproduced on held-out *future* data at ~4¢.** It is not an
artifact of fitting the table to its own sample.

### Cell-by-cell stability (train vs test net edge)
```
min  bucket        train_win train_fill train_net | test_win test_fill test_net
 5   0.05-0.10%      0.694     0.630     +0.048   |  0.727    0.666    +0.046
 5   0.10-0.20%      0.761     0.707     +0.040   |  0.805    0.741    +0.051
 5   0.20-0.50%      0.882     0.788     +0.083   |  0.921    0.820    +0.091
 9   0.05-0.10%      0.760     0.702     +0.043   |  0.806    0.735    +0.057
 9   0.10-0.20%      0.872     0.799     +0.062   |  0.909    0.842    +0.058
13   0.05-0.10%      0.878     0.776     +0.090   |  0.890    0.782    +0.097
13   0.10-0.20%      0.937     0.830     +0.097   |  0.930    0.866    +0.056
```
Train and test net edge agree cell-by-cell to ~1–4¢ — strong out-of-sample stability,
not just a favorable aggregate. The market's underpricing of continuation is a
**persistent, structural** feature across the 70-day span, not a one-regime fluke.

---

## Basis: Coinbase vs Kalshi's BRTI settlement (iteration 2's open question)

Using the markets' actual `floor_strike` (Kalshi's BRTI floor) and `expiration_value`
(the BRTI 60-second settlement print), both vs Coinbase:

| Basis | median | p5 | p95 |
|---|---|---|---|
| Coinbase(open) − Kalshi floor (BRTI) | **−0.04%** | −0.12% | +0.11% |
| Settlement(BRTI 60s avg) − Coinbase(close) | **+0.04%** | −0.11% | +0.12% |

(Means are corrupted by a handful of bad `floor_strike` records — use the medians.)

**Reading:** Coinbase tracks Kalshi's BRTI settlement to within **~0.04% (4 bps)
median**, but the 5–95% range is **±0.11%** — comparable to a full move-bucket width.
So the Coinbase→BRTI basis is **small on average but occasionally large enough to flip
a bucket or a marginal direction call.** It is a real second-order error in the current
Coinbase-based model; for marginal (near-strike) signals, modeling BRTI directly (or a
multi-venue median) would remove a noise source. Not first-order vs the ~4¢ edge, but
worth correcting for the smallest-move cells where it can dominate.

---

## What this resolves, and what remains

**Resolved:**
- ✅ The ~4¢ net edge is **out-of-sample real** (walk-forward), not overfit. This is
  the strongest evidence short of live trading.
- ✅ The Coinbase→BRTI basis is **quantified**: ~4 bps median, ±11 bps tails.

**Still open (the gap between this and bankable live P&L):**
1. **Execution realism.** This uses the candle-implied fill (`avg_fill` in the
   direction of the move). Live, you cross a real bid/ask with queue position and
   slippage on a thin book. The live `latarb` sample (3 windows) is too small to
   measure the haircut. **Compare paper `avg_fill` to actual live fills as live data
   accumulates** — this is now the binding uncertainty, not the edge's existence.
2. **Correct fee in the live accounting.** The edge above uses `0.07·c·(1−c)`; the
   code's `compute_pnl` still uses "7% of winnings." Fixing it changes the per-trade
   economics (and is mildly favorable here, since `0.07·c·(1−c)` < a winnings-fee at
   high c).
3. **Capacity.** ±$1M position limit per strike is generous, but the thin book means
   real fills well below the limit; the edge is capacity-constrained by liquidity, not
   by the rule.
4. **Decay.** The edge is the market underpricing continuation; it can compress as
   Kalshi's BTC books mature / faster makers arrive. Monitor the net-edge metric on a
   rolling basis (it's one cheap query on new settlements).

---

## Bottom line of the whole research arc
- The strategy's **core thesis is empirically validated**: betting in the direction of
  an intra-window BTC move has a **persistent, out-of-sample ~4¢/contract net edge**
  because Kalshi underprices continuation.
- The **shape** of the win-rate surface is the option model (`N(d₂)`), with the
  **fat-tail and Asian-settlement deviations** confirmed in-data — so the table is
  sound, and the two model upgrades (fat-tail cap at large moves — already implicit in
  `sigmoid_winrate`; Asian pricer in the last minute) are refinements, not foundations.
- The **risks to live P&L are now execution and fees, not the edge's existence.** The
  highest-value remaining work is operational: fix the fee model, measure live fill
  slippage, and (optionally) build the last-minute Asian pricer and the calibration
  pipeline to size the validated edge with proper fractional Kelly.
