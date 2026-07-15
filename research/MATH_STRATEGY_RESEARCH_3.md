# Kalshi BTC Binaries — Research Synthesis, Iteration 3: Advanced Math (Asian Pricing, Timing, Market-Making, Calibration)

*Deep-research synthesis, iteration 3. Iteration 1 = the core math (pricing, vol,
edge validation, Kelly). Iteration 2 = market structure (settlement, fees, execution,
latency). This iteration = the advanced math that follows from iteration 2's
structural findings — and, unusually, **three of the four clusters each land on a
specific named construct in `live/trader.py`.** Sources cited inline.*

> Read after `MATH_STRATEGY_RESEARCH.md` and `_2.md`. The throughline of all three
> reports: *edge = gap between Kalshi's price and the true conditional frequency.*
> This iteration makes "the true conditional frequency" **computable** (Asian
> digital), tells you **when** to act on it (optimal stopping), how to **capture**
> the spread instead of paying it (market-making), and how to make your probability
> **trustworthy enough to size on** (calibration).

---

## TL;DR — four findings, each tied to your code

1. **Your market is an Asian digital, not a European one — and the mispricing this
   creates is large and in the regime you trade.** Settlement is a 60-second average,
   so the correct fair value is a digital on an *arithmetic average*. Worked example:
   30s into the settlement minute, running average +0.10% above strike — your current
   instantaneous-spot model says **P ≈ 0.68**, the correct Asian model says **P ≈
   0.94**. A **26-point error**, right where your T+14 loop operates. There's a clean
   second-by-second formula to fix it.

2. **Your `f_btc × f_vel × g_misprice` sigmoid-multiplication is naive-Bayes
   double-counting, and is almost certainly uncalibrated.** Those signals are all
   functions of the same BTC path → multiplying them counts shared evidence 3× →
   systematic over-confidence → **broken Kelly sizing** (Kelly is a function of the
   probability; miscalibration mis-sizes every bet). Fix: shrink → combine (logistic)
   → calibrate (Platt/isotonic) → verify.

3. **Replace the hard `_2D_MIN_N = 30` fallback with empirical-Bayes shrinkage toward
   `N(d₂)`.** A cell with n=29 discarded and n=31 fully trusted is a discontinuity on
   a noisy statistic. The Beta-Binomial posterior mean
   `p_shrunk = w·p_cell + (1−w)·N(d₂)`, `w = n/(n+κ)`, is strictly better
   (James-Stein) and uses your option theory as the prior.

4. **The delta-hedging spray (T+4…T+14) is challenged on optimal-stopping grounds**,
   and there's a credible taker→maker pivot. Both are debatable design changes, laid
   out with the math for and against below.

---

## Cluster 9 — The market is an ASIAN digital (the key reframing)

Settlement = simple arithmetic average of 60 one-per-second BRTI prints. The payoff is
`$1 if (60-print average) ≥ K` — a **cash-or-nothing digital on an arithmetic
average**, *not* a European digital on terminal spot. Three consequences:

### Averaging crushes volatility — the "1/3 rule"
The variance of a time-average is `σ²T/3`, not `σ²T`, so the average's effective vol is
**`σ_eff = σ/√3 ≈ 0.577σ`** (discrete-60 factor: 0.582σ). The average can't be pushed
as far as spot — a terminal spike moves the 60-print mean by only 1/60 of its spot
move. **Your European model overstates how much room the price has to flip the
outcome.**

### The partially-observed average — the live formula
During the settlement minute, `m` of 60 prints are already realized constants. The win
condition reduces to a *fresh* Asian on the remaining `r = 60−m` prints with an adjusted
strike. Over 60s with tiny drift the average is ~normal, giving a clean live estimator:

```
P(A ≥ K | prints so far) ≈ Φ( (μ_A − K) / den )
  μ_A = (S_m + r·S_now) / 60                       # locked-in sum + projected remainder
  den = (σ_per_sec / 60)·√( r(r+1)(2r+1)/6 ),  r = 60 − m
  if r == 0:  P = 1{ S_m/60 ≥ K }
```
As `r → 0` the denominator → 0 and P snaps to 0/1 — the settlement cone pinches shut,
automatically. (Kemna-Vorst geometric closed form `σ_G = σ/√3` and Turnbull-Wakeman
moment-matching give the same answer with full machinery; over seconds the normal
approx is fine.)

### The worked example (why this matters)
30 of 60s elapsed, realized partial average +0.10% above K, BTC-realistic per-second vol:

| Model | P(yes) |
|---|---|
| **European (current code)** — grants spot 30s of *full* variance | **0.676** |
| **Asian (correct)** — half the average banked + 1/n + ⅓ variance cut | **0.939** |

**A 26-point gap, in the exact final-minute regime your `dh-target` loop trades.**
Trading the European number systematically underpays for "yes" / overpays for "no" late
in the window.

### Caveats
Jumps in the final seconds (move `A` by only r/60× but fatten the tail vs Φ);
per-second BRTI prints are smoothed/autocorrelated (calibrate σ_per_sec from the
*1-second print series itself*, not faster ticks); use the exact discrete factor
`r(r+1)(2r+1)/6`, not continuous σ²T/3, when few prints remain.

> **Code action:** in the final minute (offset ≥ 14:00), replace/augment
> `get_fair_price_2d` with the running-partial-average estimator above. This is the
> single most concrete, theoretically-grounded edge found in the whole project.

*Sources:* [Asian option (Wikipedia)](https://en.wikipedia.org/wiki/Asian_option) ·
[Privault, Asian Options (NTU)](https://personal.ntu.edu.sg/nprivault/MA5182/asian-options.pdf) ·
[Kemna-Vorst (MATLAB asianbykv)](https://www.mathworks.com/help/fininst/asianbykv.html) ·
[Turnbull-Wakeman moment matching](https://www.mathworks.com/help/fininst/asianbytw.html) ·
[FinPricing — in-progress adjusted strike](https://www.finpricing.com/lib/EqAsian.html)

---

## Cluster 10 — Optimal stopping: when to bet (challenges the spray)

Placing a bet inside the window = **exercising an American option on your information
edge.** The optimal policy is a Bellman/Snell free boundary: bet when edge
`e_t = p_t − q_t` clears a **time-dependent hurdle**.

### The two forces
- **Information gain (favors waiting):** `p_t` resolves toward {0,1} as the settlement
  locks in; waiting buys certainty. Late-window edge is small but its *realized Sharpe*
  `e_t/√(p_t(1−p_t))` can be high because `p_t(1−p_t)→0`.
- **Opportunity decay (favors acting):** the market also learns, `q_t → p_t`, so the
  harvestable edge shrinks. Waiting costs the fee and lets the price correct.

### The practical hurdle rule
```
BET at first t where  e_t = p_t − q_t  ≥  h_t = c + (λ/2)·p_t(1−p_t) + β·(∂q/∂t)⁺
```
The uncertainty term `(λ/2)·p_t(1−p_t)` **rises mid-window** (max at p≈0.5) and **falls
near expiry**, producing an interior optimal entry. `c` = the *true* round-trip fee +
half-spread.

### The challenge to delta-hedging
The memo's pointed conclusion: the **T+4…T+14 spray is "the worst of both worlds"** —
too slow for the early stale-price (latency) edge, too early for late-window certainty,
and it pays the per-trade fee **~11×** on bets that are **~perfectly correlated** (all
on the same settlement average), so **maximal fee drag, near-zero diversification.**
Scaling-in only beats a single optimally-timed bet when tranches are *independently*
+EV and impact is convex — but your tranches are correlated, so splitting buys almost
no variance reduction while paying k×c in fees.

**Recommendation:** pick *one* regime by your actual capability —
- **Fast (sub-second):** early stale-price trade, act when spot-implied `p_t` diverges
  from the lagging Kalshi quote (memo cites a third-party "3–7s lag" claim — **verify
  on your own `tick_log`**, it decides which regime is yours).
- **Not fast:** late near-lock trade (~T+11–13), small edge but high realized Sharpe.
— and prefer a **single shot** at the first hurdle crossing; scale in only under
genuine impact or genuinely new information.

### The counter-argument (for balance)
Delta-hedging-to-a-target has a real rationale: it *builds conviction as the move
develops* and your `dh-target` mode already caps aggregate exposure (mitigating the
overbet). The optimal-stopping critique is strongest on **fee drag**, weakest on the
"you should be able to pick the single best moment" assumption (you can't, ex-ante).
A reasonable middle path: **fewer, larger, fee-gated entries** (e.g. 2–3 conviction
steps, each independently clearing `e_t ≥ c`) rather than 11 uniform ones.

*Sources:* [Optimal stopping (Wikipedia)](https://en.wikipedia.org/wiki/Optimal_stopping) ·
[Snell envelope / American options (ESAIM)](https://www.esaim-ps.org/articles/ps/pdf/2002/01/psVol6-1.pdf) ·
[Almgren-Chriss execution](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/) ·
[Polymarket price convergence (arXiv 2606.04217)](https://arxiv.org/html/2606.04217v1)

---

## Cluster 11 — Optimal market making: the taker→maker pivot

Prior research showed you're a constant *taker* paying the peak fee at the peak-fee
price. Maker fees are ~0 and *earn* the spread. Can you make KXBTC15M?

### The framework (Avellaneda-Stoikov / GLFT)
- **Reservation price:** `r(s,q,t) = s − q·γσ²(T−t)` — skew quotes against inventory q.
- **Optimal spread:** `δ_a + δ_b = γσ²(T−t) + (2/γ)·ln(1 + γ/k)` (inventory/risk term +
  microstructure term).
- **GLFT practical form:** *fair value ± half-spread, shifted by −skew·inventory, pulled
  to ∞ on jumps.*

### The key adaptation: your mid is COMPUTABLE, not observed
Textbook A-S assumes the mid is a latent random walk you observe passively. Here the mid
is `V_t = P(BTC settles above K)` — **a known function of a feed you can stream** (the
Asian digital from Cluster 9). That collapses OMM's hardest problem (estimating fair
value) and turns the whole game into **relative latency**: whoever recomputes `V_t` and
cancel-replaces first eats the other.

### The honest verdict + the synthesis
On a computable-fair-value market, the structural edge may **invert** from making to
**taking stale quotes** (be the sniper, not the snipee). The defensible design is a
**dual-mode engine** on one fair-value computation:
- **Make** when spreads wide, vol low, no jump flag, inventory ≈ 0 — capture spread +
  save the ~1.75¢ taker fee.
- **Take** a resting quote when your fresh `V_t` says it's stale beyond (fee + buffer).
- **Pull both quotes** on a jump (`|Δln Ref| > κσ√Δt`) or high VPIN — losses concentrate
  there.

**Go/No-Go gate (critical):** before risking capital, paper-measure **how often your
`V_t` is fresher than the resting quotes you face.** Usually fresher → maker/sniper edge
is real. Book usually fresher than you → **don't make**; revert to selective taking.

*Sources:* [Avellaneda-Stoikov 2008](https://ideas.repec.org/a/taf/quantf/v8y2008i3p217-224.html) ·
[Guéant-Lehalle-Fernandez-Tapia (arXiv 1105.3115)](https://arxiv.org/pdf/1105.3115) ·
[GLFT practical (hftbacktest)](https://hftbacktest.readthedocs.io/en/py-v2.0.0/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html) ·
[Quoting under adverse selection (arXiv 2508.20225)](https://arxiv.org/pdf/2508.20225) ·
[VPIN](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf)

---

## Cluster 12 — Signal combination & calibration (fixes the sizing chain)

Your code computes `target = BASE_STAKE · f_btc · f_vel · g_misprice` — a product of
sigmoids of (win-rate, velocity, mispricing). Two things must hold for that to be sound,
and neither does:

### Problem 1: multiplying sigmoids = naive Bayes = double-counting
Multiplying signal-wise sigmoids implicitly assumes the signals carry **conditionally
independent** evidence. But `f_btc` (win-rate from move magnitude), `f_vel` (velocity),
and the move underlying `g_misprice` are all functions of the **same BTC path** → their
shared component is counted 3× → **systematic over-confidence** (probabilities pushed
toward 0/1). The principled fix combines in **log-odds space** and *learns* the weights:
```
logit(p) = β₀ + Σ βⱼ·xⱼ,   x = [logit(s_tab), s_mom, s_mis, logit(s_opt), τ, m]
```
ℓ₂-regularized **logistic regression** (ridge stabilizes the collinear features) — or a
small monotone-constrained GBM if interactions earn it. This automatically discounts
redundant signals; naive multiplication can't.

### Problem 2: the output is uncalibrated → breaks Kelly
Kelly's `f* = (p−q)/(1−q)` is a function of the *probability*. If `p̂` is miscalibrated
by Δ, every bet is mis-sized by Δ/(1−q), and because growth `g(f)` is concave,
**over-betting is penalized asymmetrically and can drive g < 0 with a real edge.**
*Miscalibration breaks Kelly.* Fix: a held-out **Platt scaling** (now, few data) →
**isotonic** (once ≳1000 settlements) recalibration layer on the *combined* output.
Grade with proper scores: **Brier** (+ its Reliability↓/Resolution↑ decomposition) and
**log-loss** (maps 1:1 to Kelly growth). **Must beat the Kalshi-quote-implied
probability out-of-sample** (skill score) or there's no tradable edge.

### Problem 3: the hard n<30 fallback → use empirical-Bayes shrinkage
Replace `_2D_MIN_N = 30` (discontinuous) with the Beta-Binomial posterior mean, shrinking
each cell toward the **option-theoretic `N(d₂)` prior** in log-odds:
```
logit(p_shrunk) = w·logit(p_cell) + (1−w)·logit(N(d₂)),   w = n/(n+κ)
```
`w` rises smoothly with sample size; sparse cells fall back to a *theoretically sensible*
surface, rich cells express empirical deviations (where your real edge lives). Maintain
online with discounted conjugate updates `(α,β) ← (γα + y, γβ + (1−y))` to track regime
drift. James-Stein guarantees lower aggregate MSE.

### Bonus: adaptive sizing as a bandit
Since edges decay, layer **Thompson sampling** over cells — sample `p̃ ~ Beta(α,β)` per
cell and act on it. Wide posteriors (under-explored cells) get re-probed automatically;
tight ones get exploited. Nearly free given the conjugate Beta you already maintain.
*Caveat:* exploration spends real money — cap exploratory stakes.

### The recommended pipeline
```
shrink (EB toward N(d₂)) → combine (regularized logistic) → calibrate (Platt→isotonic)
  → verify (log-loss/Brier/skill-vs-quote, purged walk-forward) → size (¼–½ Kelly on the
  calibrated p, Thompson over cells)
```

*Sources:* [Brier decomposition (Bröcker 2009)](https://arxiv.org/pdf/0806.0813) ·
[Platt vs isotonic (Niculescu-Mizil & Caruana)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) ·
[Naive Bayes vs logistic (Mitchell)](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) ·
[James-Stein / empirical Bayes (Efron-Hastie CASI)](https://efron.ckirby.su.domains/other/CASI_Chap7_Nov2014.pdf) ·
[Kelly with probability uncertainty (arXiv 1701.02814)](https://arxiv.org/pdf/1701.02814)

---

## Consolidated code-action list (iteration 3)

| Priority | Finding | Concrete change in `live/trader.py` |
|---|---|---|
| **P0** | Market is an Asian digital; final-minute fair value is materially mispriced by the European model | Add the running-partial-average estimator `Φ((μ_A−K)/den)` for the settlement minute; trade the gap vs Kalshi |
| **P0** | `f_btc·f_vel·g_misprice` double-counts correlated signals + uncalibrated | Replace with logistic-regression combination + Platt/isotonic calibration on a held-out fold; verify with log-loss/Brier vs quote |
| **P1** | Hard `_2D_MIN_N = 30` fallback | Empirical-Bayes Beta-Binomial shrinkage toward `N(d₂)`; online discounted updates |
| **P1** | Delta-hedge spray pays ~11× fee on correlated bets | Move to fewer, larger, fee-gated entries (each clears `e_t ≥ c`); pick one timing regime by measured latency |
| **P2** | Taker→maker pivot possible | Prototype the dual-mode (make/take/pull) engine on the Asian `V_t`; run the go/no-go freshness gate in paper first |
| **P2** | Sizing must be conservative under noise | ¼–½ Kelly on the *calibrated* probability; Thompson sampling over cells with capped exploration |

---

## Where the three iterations converge
All three reports point at the same target from different angles:

- **The edge is the gap** between Kalshi's price and the true conditional frequency
  (iteration 1).
- **The true frequency is an Asian-digital `V_t`** computable second-by-second from a
  BRTI feed (iterations 2–3).
- **The durable way to harvest it** is a fast `V_t` engine used **dual-mode** (take
  stale quotes / make when safe / pull on jumps), sized with **calibrated fractional
  Kelly**, entered at a **fee-aware hurdle**, in the **regime your latency actually
  supports** (iterations 2–3).
- **Everything else is tax**: the fee curve (worst at 0.50), the spread, adverse
  selection, and fragmentation drag — all measurable on your own logs (Roll, Kyle λ,
  toxicity), all to be subtracted *before* believing any backtest.

## Open questions for iteration 4 (more empirical than mathematical)
1. **Validate the Asian `V_t` model on the repo's own settled windows** — quantify its
   edge over the European model in the final 2 minutes, on real `tick_log`/`window_log`
   data.
2. **Measure the Coinbase→BRTI basis and the Kalshi quote-reprice latency** from
   existing logs — settles the early-vs-late regime question.
3. **Reconstruct true P&L with the correct fee formula** and re-rank the saved
   strategies (s2/s3/s6/s7) under honest costs.
4. Possibly shift the loop from *research* to *implementation*: prototype the Asian
   pricer + calibration pipeline as actual Python in the repo.
