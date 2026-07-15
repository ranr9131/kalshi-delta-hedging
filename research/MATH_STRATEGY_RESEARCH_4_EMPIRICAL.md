# Iteration 4 — Empirical Test on Your Own Data: Is the Edge Real or Mechanical?

*This iteration stops researching and **tests** the central claim of iterations 1–3
against your actual 2D table (`data/logs/minute_analysis_2d_15s_kxbtc15m.csv`),
built from ~90 days / 224,628 conditional observations. Two tests: (1) does the
empirical win rate beat the market price after the corrected fee? (2) is the
win-rate structure just the option model `N(d₂)` rediscovered? The answers are
concrete — and one of them partly **contradicts** the pessimistic prior.*

> Script: `scratchpad/edge_test.py`. Reproducible from the committed CSV.

---

## Headline results

| Test | Result |
|---|---|
| **Gross edge vs market price** (`win_rate − avg_fill`), sample-weighted | **+5.36¢ per $1 contract** |
| **Net edge after corrected fee** (`− 0.07·c·(1−c)`) | **+4.02¢** |
| **Fraction of samples in cells with positive net edge** | **94.3%** |
| **Best-fit single σ for `win_rate ≈ N(d₂)`** | **34% annualized** |
| **Mean \|win_rate − N(d₂)\|** (sample-weighted) | **0.040** |

**Two findings that matter:**
1. **The market underprices the continuation.** Across 224k observations the
   empirical win rate sits **~4 cents (net) above the price you actually paid** —
   this is a *real* edge versus the market, **not** purely mechanical. This is the
   most important result in the whole project and it **softens** iteration 1/3's
   worry that "it's all mechanical and already priced." In *this* sample, it wasn't
   fully priced.
2. **But the *shape* is mechanical, with exploitable deviations.** A single-σ option
   model fits the win rates to within ~4 points on average — so the table is largely
   `N(d₂)` rediscovered. The deviations are the interesting part: **`N(d₂)`
   systematically *overshoots* at large moves and in the last minute** — exactly the
   fat-tail / averaging effects predicted in iterations 2–3.

---

## Test 1 — Does the win rate beat the market price? (the real edge test)

Each table cell carries `win_rate` (empirical P(win)) and `avg_fill` (the historical
contract price ≈ market-implied probability). The tradeable edge per contract is
`win_rate − avg_fill − fee`, with the **corrected** per-contract fee `0.07·c·(1−c)`.
A representative slice:

```
offset  bucket        n     win   fill   gross   fee    net
  240   0.05-0.10%  1652   0.664  0.615  +0.049  0.017  +0.032
  240   0.20-0.50%   539   0.863  0.780  +0.082  0.012  +0.070
  450   0.10-0.20%  1531   0.855  0.794  +0.061  0.011  +0.049
  660   0.05-0.10%  1351   0.828  0.766  +0.061  0.013  +0.049
  840   0.10-0.20%   291   0.873  0.748  +0.125  0.013  +0.111   <- last minute
  840   0.00-0.01%   255   0.471  0.466  +0.005  0.017  -0.013   <- tiny move, no edge
```

**The win rate exceeds the fill price in almost every populated cell**, by 2–13
cents gross. Sample-weighted: **+5.36¢ gross, +4.02¢ net**, with **94.3% of all
observations** in positive-net-edge cells. The edge is largest for **bigger moves**
(market underprices strong continuation most) and **vanishes for the smallest moves**
(0.00–0.01% cells are coin-flips with no edge after fee — correctly, your code already
skips these via the magnitude floor).

**Interpretation:** in this sample, Kalshi's price did **not** fully reflect how
predictive a move is — the market underprices continuation. That is precisely the
*slow-reprice / favorite-longshot* edge that iteration 2's latency cluster flagged as
"the one plausibly-real source." Here it shows up directly as `win_rate > price`.

### ⚠️ The caveats that decide whether this is real money
This is the optimistic read; here is why it is **not yet bankable**:
1. **In-sample.** The table was *built* on these same observations, so ~4¢ is an
   upper bound inflated by fit. **Walk-forward (train/test split) is mandatory** before
   believing it — iteration 1's validation checklist (DSR, PBO, purged CV) still applies.
2. **`avg_fill` may be optimistic.** If it's the displayed price rather than a
   queue-realistic executable fill, live slippage + the bid/ask you actually cross
   eats into the 4¢. Your live `latarb` log (only 3 windows) is too small to confirm;
   the paper logs assume favorable fills.
3. **Overlapping windows.** 224k observations are *not* independent — effective n is
   far smaller, and a few trending regimes supply most of the edge.
4. **The fee correction matters here.** I used `0.07·c·(1−c)` (peaks ~1.7¢ at c=0.5),
   not "7% of winnings." Re-running with your code's current fee model would mis-state
   net edge — another reason to fix `compute_pnl`.

Net: **the edge is real *in-sample and gross of execution frictions*. The open
question is how much survives out-of-sample and live** — which is exactly the
experiment to run next.

---

## Test 2 — Is the structure mechanical (= N(d₂))?

Fitting a single annualized σ so that `N( move / (σ√τ_remaining) )` best matches the
win rates gives **σ ≈ 34%** and a sample-weighted mean error of just **0.040**. So the
table's *shape* — win rate rising with move size and with time elapsed — **is largely
the option model**. The genuinely informative part is **where it deviates**:

```
offset  bucket        win   N(d2)   diff
  240   0.20-0.50%   0.863  0.988  -0.125   <- N(d2) wildly overconfident (fat tails)
  240   0.10-0.20%   0.755  0.833  -0.077
  660   0.20-0.50%   0.967  1.000  -0.033
  840   0.05-0.10%   0.775  0.945  -0.170   <- last minute: model badly overshoots
  840   0.20-0.50%   0.840  1.000  -0.160
  450   0.01-0.05%   0.632  0.592  +0.040   <- small moves slightly MORE predictive
```

Two systematic, **exploitable** deviations — both predicted by earlier iterations:

1. **Large moves: `N(d₂)` overshoots by 5–13 points.** The Gaussian says a 0.2–0.5%
   move is ~99% locked; reality is 86–97%. This is the **fat-tail / jump effect**
   (iteration 2, cluster 8): a comfortable lead gets reversed by a jump more often
   than Gaussian allows. **Your `sigmoid_winrate` saturates exactly here** — so the
   empirical table is *right* to cap below the Gaussian, and a naive `N(d₂)` pricer
   would over-bet these. Good news: your table already encodes this.

2. **Last minute (T+14 / offset 840): the model breaks down and win rates *drop*.**
   `N(d₂)` says 0.94 for a small move; actual is 0.78, and win rates at T+14 are
   *lower* than at T+11. A European model with 60s of full variance is the wrong tool
   here — this is the **Asian-settlement regime** (iteration 3): the outcome is the
   60-second *average*, small late moves are within settlement noise and don't predict
   the average cleanly. **This confirms the European fair value is wrong in the final
   minute** and should be replaced by the running-partial-average estimator. (Note:
   small n at 840 — treat as indicative, not conclusive.)

---

## What this iteration changes about the conclusions

- **Iteration 1/3 were too pessimistic about "it's all mechanical."** The shape is
  mechanical, but the **market price sits below the win rate** — there *is* a real,
  ~4¢ (in-sample, net) edge from the market underpricing continuation. The skeptical
  framing ("edge only where you beat N(d₂) *and* the price") is right; the data says
  you **do** beat the price across most cells.
- **The fat-tail and Asian-settlement predictions are confirmed in your own data**:
  `N(d₂)` overshoots at large moves and in the last minute, by exactly the sign and
  rough magnitude predicted.
- **The single most valuable next step is no longer research — it's a clean
  walk-forward test** of that 4¢ edge with realistic fills and the corrected fee.

---

## Recommended next experiments (concrete, on this repo)
1. **Walk-forward the edge:** rebuild the 2D table on the first 60 days, freeze it,
   measure `win_rate − avg_fill − fee` on the last 30 days. If the ~4¢ survives → real,
   bankable edge. If it collapses → it was overfit. (Highest priority.)
2. **Reconstruct true P&L** of the saved strategies (s2/s3/s6/s7 window logs) under the
   **corrected fee** `0.07·c·(1−c)` and re-rank them.
3. **Build the Asian last-minute pricer** and re-bucket the offset-840 cells with it;
   check whether the win-rate "breakdown" at T+14 is explained by the
   European→Asian correction.
4. **Compare paper vs live fills** (`avg_fill` paper vs the 3 live windows + any new
   live data) to estimate the real execution haircut on the 4¢.
