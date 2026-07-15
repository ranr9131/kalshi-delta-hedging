# The Mathematics of Kalshi BTC 15-Minute Binaries — Research Synthesis

*Deep-research synthesis, iteration 1. Covers risk-neutral pricing, volatility
estimation, return statistics / edge validation, and Kelly sizing. Every major
claim is sourced inline; sources are collected at the end of each section.*

> **Market recap.** Kalshi's `KXBTC15M` markets ask: *will BTC be ≥ a fixed floor
> strike at the end of a 15-minute window?* The floor is BTC spot at window open
> (the contract opens **at-the-money**). Payoff is $1 on a win, $0 otherwise.
> The traded price ≈ implied probability. Kalshi takes **7% of net winnings**.
> Our trader (`live/trader.py`) reads a 2-D empirical win-rate table indexed by
> `(seconds elapsed, |% move from floor|)`, sizes bets with sigmoids of win-rate
> and of mispricing vs Kalshi's quote, and delta-hedges from T+4 to T+14.

---

## TL;DR — the five things that matter

1. **Your 2-D table is an empirical estimate of one closed-form number.** The
   risk-neutral fair price of "BTC > floor" mid-window is
   **`Φ( ln(Sₜ/K) / (σ·√τ) )`** — current cushion ÷ remaining-horizon vol, where
   `τ = (15−t)/525600` years. That single z-score is the theoretical backbone of
   the entire table and explains the time-decay steepening you observe.

2. **The uncomfortable truth (Cluster 3):** most of the table's apparent edge is
   **mechanical in-the-moneyness** — `N(d₂)` rediscovered empirically. A far-ITM,
   little-time-left cell wins ~95% of the time *by construction*, and any rational
   counterparty prices that. **Mechanical ≠ alpha.** Real, tradeable edge exists
   **only where empirical win rate > max( N(d₂), market price + fees )** — i.e.
   where **Kalshi's quote is stale/wrong**, not merely where BTC moved a lot.

3. **The one plausibly-real edge is latency / stale quotes.** Because you *cannot
   continuously hedge BTC on Kalshi*, Black-Scholes no-arbitrage does **not** pin
   the market price. Your P&L is the gap between Kalshi's posted price and the true
   conditional frequency. The part of that gap most likely to be real is the venue
   quote (or its settlement oracle) lagging true spot — a **microstructure/latency
   edge**, capacity- and speed-limited.

4. **The 7% fee raises your break-even more than you think,** and worst on cheap
   contracts. At c = 0.70 you need **true q ≥ 0.715**, not 0.70. Bets with
   c < q < q_be look like edge but are **negative-EV after the fee** — they must be
   filtered out entirely.

5. **Size the *aggregate* window position with deep-fractional Kelly (¼ or less).**
   All your delta-hedge legs settle on the *same* outcome → correlation ≈ 1 → they
   do **not** diversify. Sizing each leg at full Kelly = ~N× overbet. Your
   `dh-target` mode already caps the aggregate correctly; `dh-additive` does **not**.

---

## Cluster 1 — Risk-neutral pricing of the 15-min binary

### The core formula
The contract is a **European cash-or-nothing digital call**. Its no-arbitrage price:

```
V = e^(−rT) · N(d₂)
d₁ = [ ln(S₀/K) + (r + σ²/2)·T ] / (σ·√T)
d₂ = d₁ − σ·√T
```

`N(d₂)` **is** the risk-neutral probability of finishing in the money. Over a
15-min window the discount factor `e^(−rT) ≈ 0.999999`, so **fair price ≈ N(d₂)**
to four decimals.

### Parameter scale (memorize this)
With 24/7 minutes, `T = 15/525600 = 2.854e−5 yr`, `√T = 5.342e−3`. The only quantity
that matters for a short-dated ATM digital is the **per-window 1σ move**:

| σ (annual) | σ·√T (per-window 1σ) |
|---|---|
| 50% | 0.267% |
| 60% | 0.321% |
| 80% | 0.427% |

So over 15 minutes BTC moves on the order of **±0.3% (1σ)**. Everything keys off this.

### Risk-neutral vs real-world probability — **irrelevant here**
The physical probability swaps drift `r → μ`. The gap is
`d₂ᴾ − d₂ = (μ−r)·√T / σ`. Even with an aggressive μ = 50%/yr, σ = 0.60, the shift
is ~0.004 in d₂ ≈ **0.16 cents**. The drift term is **~0.45% of the diffusion term**
(`μT/σ√T ≈ 0.0045`). **Conclusion:** at 15 min you can treat risk-neutral and
real-world probabilities as identical. *Caveat:* if short-horizon momentum/
microstructure produces a transient effective drift, that's an empirical effect the
GBM formula can't see — and exactly what your table might capture.

### The ATM contract is just under 0.50
With K = S₀ and r ≈ 0, `d₂ ≈ −σ√T/2`, so
`V_ATM ≈ N(−σ√T/2) ≈ ½ − σ√T/(2√(2π))`. The "above-floor" call is **slightly below
50¢** (median of a lognormal < its mean — the −σ²/2 vol drag):

| σ | ATM call value | below 50¢ by |
|---|---|---|
| 50% | 0.49947 | 0.05¢ |
| 60% | 0.49936 | 0.06¢ |
| 80% | 0.49915 | 0.09¢ |

Real but **below the 1¢ tick and the spread/fee** — not directly harvestable.

### European ≠ one-touch (reflection principle)
Kalshi settles **at expiry** (European). Do **not** price off "will BTC touch the
floor." For driftless BM the reflection principle gives
`P(max ≥ b) = 2·P(W_T ≥ b)` — the **touch** probability is ~**2× the finish**
probability near the barrier. A naive one-touch intuition overstates the relevant
probability by ~2×. The European structure is actually a *robustness asset*:
intra-window wicks and gaps are irrelevant — **only the terminal print counts**.

### The mid-window conditional probability (your table, in closed form)
By the Markov property, with `τ = (15−t)/525600` remaining and cushion
`x = ln(Sₜ/K)`:

```
Q(win | Sₜ) = N(d₂(τ)),   d₂(τ) = [ ln(Sₜ/K) + (r − σ²/2)·τ ] / (σ·√τ)
            ≈ N( ln(Sₜ/K) / (σ·√τ) )          # drift/drag negligible
```

Worked surface (σ = 0.60):

| elapsed t | τ (min) | cushion | z = x/(σ√τ) | Q(win) |
|---|---|---|---|---|
| 0 | 15 | 0.00% | −0.002 | 0.499 |
| 5 | 10 | +0.10% | +0.38 | 0.649 |
| 7.5 | 7.5 | +0.20% | +0.88 | 0.811 |
| 10 | 5 | +0.30% | +1.62 | 0.947 |
| 14 | 1 | −0.05% | −0.60 | 0.273 |

Note the **time-decay sharpening**: the same +0.10% cushion is worth 0.649 with 10
min left but 0.757 with 3 min left — `σ√τ` shrinks, the surface steepens toward a
step function (digital gamma blows up near expiry).

**Where the closed form fails and your table should win:** constant-σ Gaussian
**underprices the tails** — a +0.30% lead is *not* 94.7% safe if a jump can occur;
and static σ makes the formula overconfident in calm states and wrong after a vol
shift. An empirically-fit table automatically encodes fat tails + vol clustering,
which is why it should beat `N(d₂)` **in the tails and the final 1–2 minutes**.

*Sources:* [QuantPie digital BS](https://www.quantpie.co.uk/bsm_bin_c_formula/bs_bin_c_summary.php) ·
[MathWorks cashbybls](https://www.mathworks.com/help/fininst/cashbybls.html) ·
[MIT 18.600 L36 (risk-neutral / N(d₂))](https://math.mit.edu/~sheffield/2018600/Lecture36.pdf) ·
[Reflection principle (Almost Sure)](https://almostsuremath.com/2023/04/18/the-maximum-of-brownian-motion-and-the-reflection-principle/) ·
[MathWorks touchbybls](https://www.mathworks.com/help/fininst/touchbybls.html) ·
[Drift & the risk-free rate (Gadidov)](https://www.hindawi.com/journals/jps/2011/595741/)

---

## Cluster 2 — Volatility estimation for the next 15 minutes

You need σ to compute the closed-form fair price above. From second-resolution
Coinbase data:

### Realized-vol estimators (efficiency vs close-to-close)
| Estimator | Inputs | Drift-robust | Gap-robust | Efficiency |
|---|---|---|---|---|
| Close-to-close | C | – | – | 1× |
| Parkinson | H,L | no | no | ~5× |
| Garman-Klass | OHLC | no | no | ~7.4× |
| Rogers-Satchell | OHLC | **yes** | no | ~6–8× |
| **Yang-Zhang** | OHLC | **yes** | **yes** | **~14×** |

Parkinson: `σ²_P = 1/(4 ln2) · mean( ln(Hᵢ/Lᵢ)² )`. Yang-Zhang combines overnight +
open-close + Rogers-Satchell and is the best OHLC estimator. **But** with
second-resolution ticks you have something better than any range estimator:
**realized variance `RV = Σ rⱼ²`** computed directly.

### Scaling and the microstructure trap
- **σ₁₅ₘ = σ₁ₘ · √15.** Annualize crypto with **√35040** (15-min blocks/yr) or
  **√525600 ≈ 725** (minutes/yr) — never √252.
- **Do NOT sample at 1 second.** Bid-ask bounce inflates RV — the "volatility
  signature plot" explodes as the interval shrinks. Microstructure bias is
  negligible only above ~20s; the standard robust choice is **5-minute sampling**.
  With second data, use 5-min RV, or noise-robust estimators (two-scale RV /
  realized kernel / pre-averaging).
- **√t scaling fails** under autocorrelation/mean-reversion (biases the estimate)
  and under vol clustering (a calm-minute estimate doesn't persist).

### Intraday seasonality — σ is *not* constant across the day
Crypto has no open/close but shows clear time-of-day structure: surges at US and
Asia session onsets, **8-hour perpetual funding stamps (00/08/16 UTC)**, CME BTC
futures open, and **scheduled macro releases (CPI/FOMC/NFP, ~12:30/14:00/18:00
UTC)**. A 15-min window straddling a CPI print has multiples of the vol of a 04:00
UTC Sunday window. **This validates your `ACTIVE_HOURS` filter** and argues for a
time-of-day seasonal multiplier `s(τ)` on σ, plus an economic-calendar guard.

### Forecasting the next 15 min
- **EWMA / RiskMetrics:** `σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·r²ₜ₋₁`. Deploy this first
  (one parameter, instantly reactive). Tune λ to your bar size (λ≈0.97 on 1-min).
- **HAR-RV (Corsi):** OLS on daily/weekly/monthly (here: 15-min / 1–2h / 1-day) RV;
  best accuracy, hard to beat, captures clustering cheaply.
- **GARCH(1,1):** adds mean reversion but heaviest to maintain on streaming data.

### Recommended pipeline
1. Build 5-min (or 1-min) returns from Coinbase seconds → trailing **realized
   variance** (5-min sampling to dodge noise).
2. Forecast the level with **EWMA** (now) and/or **HAR-RV** (accuracy).
3. Scale to horizon: `σ₁₅ₘ = σ₁ₘ·√15`.
4. Multiply by **time-of-day factor s(τ)**; bump for scheduled events in-window.
5. **Jump guard:** if a jump is detected, switch to jump-robust RV (bipower /
   median RV) so one spike doesn't poison σ.

*Sources:* [Parkinson](https://metricgate.com/docs/parkinson-volatility-estimator/) ·
[Yang-Zhang](https://portfolioslab.com/tools/yang-zhang) ·
[Microstructure noise & optimal sampling](https://www.researchgate.net/publication/4817469_Microstructure_noise_realized_volatility_and_optimal_sampling) ·
[Periodicity in crypto vol/liquidity (arXiv 2109.12142)](https://arxiv.org/pdf/2109.12142) ·
[Corsi HAR-RV](https://academic.oup.com/jfec/article-pdf/7/2/174/2543795/nbp001.pdf)

---

## Cluster 3 — Is the edge real, or mechanical? (the linchpin)

### The mechanical / tautological component
"Currently ITM by x% with τ left ⇒ finishes ITM" is **largely mechanical**: if the
cushion is large relative to the vol achievable in the remaining time, it finishes
there *by construction*. This is exactly `N(d₂) ≈ N( m/(σ√τ) )` with moneyness
`m = (S−K)/K`. A high empirical win rate for "deep ITM, little time left" is
**expected and real — but it is not alpha.** Any rational market prices it near
`N(d₂)`; you pay ~that probability and win ~that often → **zero edge before fees,
negative after.** Your table's two axes (|% move|, minutes elapsed) are monotone
proxies for moneyness and remaining time — **the table is re-deriving N(d₂)**.

### The only valid edge test
```
Edge = p_empirical − max( N(d₂)[fat-tail adjusted],  π_market + fees )
```
You have tradeable edge **only if** empirical win rate exceeds *both* the
option-model probability (at actual τ and σ) **and** Kalshi's price + all fees.
Where `p_empirical ≈ N(d₂) ≈ π_market`, the cell is a tautology. **The interesting
residual is where `p_empirical > π_market + fees` — i.e. the market price is stale.**

### Conditioning / overfitting artifacts to neutralize
- **Regression to the mean:** conditioning on an interim extreme over-samples
  transient excursions that partly revert.
- **Overlapping windows:** 90 days of 15-min windows are **not independent**;
  trending regimes supply most "continuation" observations. Effective sample size
  ≪ nominal.
- **Look-ahead in vol:** a high-win-rate cell may just be flagging low-σ√τ periods.

### Return-statistics reality check
- 1–15 min BTC is **approximately a martingale**; signed-return autocorrelation
  decays fast. The persistent memory is in **squared/absolute** returns (vol
  clustering), **not** signed returns.
- At short lags, traded-price returns show **negative** autocorrelation from
  **bid-ask bounce** (Roll) — this *works against* a momentum story. Use
  **mid-quotes, not trades**, and the heteroskedasticity-robust **variance-ratio**
  test `Z*(q)` to check for genuine continuation.
- BTC returns are **fat-tailed with a power-law ~inverse-square tail** (heavier than
  equities' inverse-cube) → **jump-diffusion**, not Gaussian. Plain `N(d₂)`
  **overstates** the safety of marginal-ITM cells (a jump can flip them).

### Validation checklist (run before trusting any cell)
- [ ] Re-benchmark every cell against **N(d₂)** at actual τ and σ; keep only cells
      where `p_empirical − N(d₂)` is materially > 0. Use a **fat-tail-adjusted** ITM
      probability, not Gaussian.
- [ ] **Subtract the market:** require `p_empirical > π_market + fees + slippage`.
- [ ] Use **mid-quotes**; run heteroskedasticity-robust **variance-ratio** on signed
      returns.
- [ ] **Walk-forward only** (time-ordered train/test); freeze table + rule on train,
      report test-only. Never random CV.
- [ ] **Effective sample size:** down-weight overlapping windows. To detect a 3%
      edge over 50% you need **≈2,200 independent bets**; 2% needs **≈4,900**
      (`n ≈ 1.96/δ²`).
- [ ] Win-rate test vs `p₀ = max(N(d₂), π_market+fees)`, **not 0.5**; report Wilson CI.
- [ ] **Multiple-testing haircut:** count every cell/param tried as a trial; demand
      **t > 3** (Harvey-Liu), not t > 2.
- [ ] **Deflated Sharpe Ratio** (Bailey & López de Prado) with your N trials and
      skew/kurtosis; require **DSR > 0.95**.
- [ ] **PBO via CSCV** over the full grid; require **low PBO** (< 0.1–0.2).
- [ ] **Paper-trade forward** with real latency/fees; confirm the residual is the
      stale-price/latency component.

### Verdict
- **Most likely illusory:** mechanical in-the-moneyness (dominant), overfit table,
  short-lag "momentum," Gaussian-benchmark error.
- **Plausibly real:** **stale / slow venue quotes (or settlement oracle) vs true
  spot** — a latency/microstructure edge, capacity- and speed-limited, that can
  evaporate as Kalshi tightens. *This is the part most likely to transfer.*

*Sources:* [Lo & MacKinlay variance ratio](https://finance.martinsewell.com/stylized-facts/dependence/LoMacKinlay1988.pdf) ·
[Stylized facts of HF Bitcoin (arXiv 2402.11930)](https://arxiv.org/html/2402.11930v2) ·
[Bitcoin inverse-square tail (arXiv 1905.03211)](https://arxiv.org/pdf/1905.03211) ·
[Roll / bid-ask bounce](https://finance.martinsewell.com/stylized-facts/dependence/) ·
[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) ·
[Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) ·
[Harvey-Liu-Zhu (t>3)](https://www.nber.org/system/files/working_papers/w20592/w20592.pdf)

---

## Cluster 4 — Optimal bet sizing (Kelly) with the 7% fee + correlated legs

### Baseline Kelly for a prediction-market contract
Buy at price c (pays $1). Net odds `b = (1−c)/c`. With estimated true win prob q:
```
f* = (q − c) / (1 − c)        # = edge / odds
```
Positive iff `q > c`; **break-even at q = c** (no fee).

### Fee-adjusted break-even and Kelly
Kalshi's 7% fee hits only the **winning payoff**: net win = `0.93·(1−c)`. Effective
odds `b' = 0.93·b`. Break-even rises to:
```
q_be = c / (1 − 0.07·(1 − c))
```

| price c | break-even q (no fee) | break-even q (7% fee) | extra edge demanded |
|---|---|---|---|
| 0.55 | 0.550 | **0.568** | +1.79 pts |
| 0.70 | 0.700 | **0.715** | +1.50 pts |
| 0.85 | 0.850 | **0.859** | +0.90 pts |

The fee bites **hardest on cheap contracts**. Fee-adjusted Kelly:
```
f*_fee = q − (1−q)·c / (0.93·(1−c))
```
At c=0.70, q=0.75: no-fee f* = 0.167 → fee f* = **0.123** (a ~26% cut, *on top of*
disqualifying marginal bets where `c < q < q_be`).

### Fractional Kelly — bet ¼ or less
Growth at a fraction λ of full Kelly: `g(λf*)/g(f*) = 2λ − λ²`.
- Half-Kelly keeps **75%** of growth at **¼ the variance**.
- Quarter-Kelly keeps **44%** of growth at **1/16 the variance**.
Overbetting is punished convexly and asymmetrically: at **2f\*** growth is **zero**.
Since the win-prob table is noisy (you'll sometimes overestimate q), **deep
fractional Kelly (¼ or less) is mandatory**, not optional. A cheap robustification:
shrink q toward c before sizing, `q̃ = c + κ·(q − c)`, κ < 1.

### Correlated legs — the biggest sizing risk
All delta-hedge legs T+4→T+14 settle on the **same** BTC outcome → correlation ≈ 1
→ **zero diversification**; they are economically *one* position. Vector Kelly
`f* = Σ⁻¹(μ − r1)` collapses to a single scalar bet on the **aggregate** exposure.
**Sizing each leg at its own full Kelly = ~N× overbet** — straight into the
negative-growth region.
- ✅ **`dh-target` mode already does this correctly:** it targets `target_yes` and
  bets only `max(0, target − exposure)` → the *aggregate* is capped.
- ⚠️ **`dh-additive` mode does NOT:** it adds full size each interval → the
  dangerous mode. Prefer `dh-target`.

### Drawdown control
`P(ever drawdown to fraction x) = x^(2/λ − 1)`:

| strategy | P(ever ≤ 50%) | P(ever ≤ 25%) |
|---|---|---|
| full Kelly | **50%** | 25% |
| half Kelly | 12.5% | 1.6% |
| quarter Kelly | **0.78%** | ~0.006% |

Full Kelly has a ~50% chance of a 50% drawdown. Quarter-Kelly: <1%.

### Concrete sizing recommendation
```
stake_window = λ · f*_fee · W,   λ = 0.25
f*_fee = q − (1−q)·c / (0.93·(1−c))
```
subject to: (a) trade only if `q ≥ q_be + margin`; (b) hard cap `Σ active-window
stake ≤ 0.20·W`; (c) the **sum** of intra-window legs ≤ the single window Kelly
budget; (d) shrink q toward c for table noise.

*Sources:* [Kelly criterion (Wikipedia)](https://en.wikipedia.org/wiki/Kelly_criterion) ·
[Thorp, Portfolio Choice & the Kelly Criterion](https://gwern.net/doc/statistics/decision/1975-thorp.pdf) ·
[MacLean-Thorp-Ziemba, Good & Bad Properties of Kelly](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf) ·
[Kelly for prediction markets (arXiv 2412.14144)](https://arxiv.org/html/2412.14144v1)

---

## How this maps onto the current code (`live/trader.py`)

| Current code | Research says | Suggested change |
|---|---|---|
| `get_fair_price_2d()` empirical table | Table ≈ `N(ln(Sₜ/K)/(σ√τ))`; mostly mechanical | Add a **closed-form fair price** computed live from cushion + forecast σ; trade the **gap to Kalshi's quote**, not the gap to the table |
| `mispricing = fair − (yes_ask + buf)` | Fee raises break-even to `q_be = c/(1−0.07(1−c))` | Gate on **fee-adjusted break-even**, not raw ask; require margin |
| `sigmoid_winrate` × `sigmoid_mispricing` × `BASE_STAKE` | Heuristic, not growth-optimal; noisy q ⇒ overbet risk | Move toward **fee-adjusted ¼-Kelly on the aggregate window**; keep sigmoids only as a smoother |
| `dh-target` (gap sizing) | Correct: caps aggregate correlated exposure | **Prefer this mode**; treat `dh-additive` as deprecated/risky |
| `ACTIVE_HOURS` filter | Validated by intraday vol seasonality | Extend with a **σ seasonal multiplier** + **macro-calendar guard** (CPI/FOMC/NFP, funding stamps) |
| 2-D table built on 90 days | Overlapping windows ⇒ small effective n; overfit risk | Run **walk-forward + DSR + PBO**; benchmark cells vs `N(d₂)` and vs `π_market + fees` before trusting |
| Direction = sign of move | Bid-ask bounce gives negative short-lag autocorr | Use **mid-quotes / median tick** for direction (already partly done via `USE_MED_DIRECTION`) |

### The single most important next experiment
For each historical bet, compute the **theoretical `N(d₂)`** (using cushion at bet
time, remaining τ, and a forecast σ) and compare three numbers per cell:
`p_empirical` vs `N(d₂)` vs `π_market + fees`. **Edge lives only where
`p_empirical` beats both.** If your win rate just tracks `N(d₂)`, the table is
mechanical and the real money is in the **stale-quote / latency** residual — which
should be isolated and measured directly.

---

## Open questions for the next research iteration
1. **Kalshi's settlement oracle & quote latency** — how is the floor strike and
   settlement price sourced, how fast do quotes update vs Coinbase spot, and is
   there a measurable, persistent lag to arbitrage? (This is the *one plausibly-real
   edge* — worth its own deep dive.)
2. **Fat-tail-adjusted digital pricing** — replace Gaussian `N(d₂)` with a
   jump-diffusion (Merton) or Student-t / variance-gamma digital so marginal-ITM
   cells aren't overstated.
3. **Adverse selection / order-flow** — who is on the other side, and does filling
   at the ask systematically pick you off when fast traders have already moved?
4. **Empirical execution** of the N(d₂)-vs-table-vs-market diagnostic on the repo's
   own `window_log.*` / `tick_log.*` data.
