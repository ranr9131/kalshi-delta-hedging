# Kalshi BTC Binaries — Research Synthesis, Iteration 2: Microstructure, Settlement, Fees & Latency

*Deep-research synthesis, iteration 2. Where iteration 1 covered the pure
mathematics (pricing, vol, edge validation, Kelly), this iteration covers the
**market structure** that determines whether any mathematical edge survives contact
with reality: how Kalshi actually settles and charges, where the execution tax
hides, whether latency arbitrage is capturable, and how fat tails bite. Sources are
cited inline.*

> Read alongside `MATH_STRATEGY_RESEARCH.md` (iteration 1). That report's central
> conclusion — *real edge lives only in the gap between Kalshi's price and the true
> conditional frequency* — is sharpened here into **two concrete, structural edges**
> (the settlement-average cone; being a maker not a taker) and **two structural
> taxes** (the fee curve; adverse selection) that the current code does not model.

---

## ⚠️ TL;DR — the four findings that should change the code

1. **The fee is almost certainly mis-modeled.** Three independent sources converge:
   Kalshi's fee is **`fee = ceil(0.07 · C · P · (1−P))` per contract, charged on the
   trade (taker)** — *not* 7% of winnings. It's **maximized at P = 0.50** (≈1.75¢/
   contract) and shrinks to ~0 at the wings. Your `compute_pnl()` uses
   `count·(1−fill)·0.93` (7% of profit on winners only) and the README says "7% on
   gross winnings." **These are the wrong shape.** Re-derive break-even and re-run
   every backtest with the real per-contract fee. *(Caveat: the crypto-specific
   multiplier may exceed 0.07 — verify from a real fill before hardcoding.)*

2. **Settlement is a 60-second average of CF Benchmarks BRTI — not Coinbase, not a
   single closing print.** (Verbatim from Kalshi's filed
   [BTC contract spec](https://kalshi-public-docs.s3.amazonaws.com/contract_terms/BTC.pdf).)
   Two consequences: (a) your Coinbase feed has **basis risk** vs the settlement
   index, and (b) **with t seconds left, (60−t)/60 of the settlement is already
   locked in** — the achievable settlement lives in a *shrinking cone*. This is the
   cleanest, most defensible edge found in the entire project, and your
   instantaneous-spot model ignores it.

3. **You are a constant *taker*, paying the worst fee at the worst price.** Your
   delta-hedging crosses the spread many times per window, at prices near 0.50 where
   the fee peaks. Round-trip cost ≈ spread (4% on a 2¢/50¢ contract) + fee (~7% at
   50¢) ≈ **11% of notional, compounding multiplicatively** over k bets while edge
   adds linearly. **Maker fees are ~0%.** The single highest-leverage structural fix
   is to **post (maker) instead of take** wherever the signal isn't time-critical.

4. **Latency arbitrage is real but you're usually the prey, not the sniper.** A
   non-colocated websocket trader loses every race to a colocated pro. It's
   capturable *only* in a narrow regime: counterparties slower than your sub-second
   loop, on sharp BTC jumps, as the aggressive taker, away from 50¢ — and it decays
   the moment a faster firm notices. **Never rest naked maker orders on a BTC jump**
   or you become the stale quote being picked off.

---

## Cluster 5 — Kalshi settlement & contract mechanics (verbatim spec)

From the filed [BTC.pdf contract terms](https://kalshi-public-docs.s3.amazonaws.com/contract_terms/BTC.pdf)
(HIGH confidence — Kalshi-official):

- **Settlement index = CF Benchmarks BRTI** (Bitcoin **Real-Time** Index — per-second,
  order-book-driven, consolidated across regulated USD venues). *Not* the BRR/BRRNY
  fixing; *not* Coinbase. Confirmed by
  [CF Benchmarks](https://www.cfbenchmarks.com/blog/kalshi-leads-surging-crypto-event-contract-market-powered-by-cf-benchmarks)
  and [Kalshi Help](https://help.kalshi.com/en/articles/13823838-crypto-markets).
- **Settlement value = simple arithmetic average of the 60 one-per-second BRTI prints
  in the final minute.** Not a single closing tick, not a median.
- **Payout $1.00, tick $0.01, position limit $1,000,000 per strike per member.**
- **Edge case:** if no data at expiration, market resolves **NO**.
- **Settlement guaranteed only T+1** (usually seconds in practice, but contractually
  next-day; can be delayed by review).

**Open questions to verify empirically (do not hardcode):**
- Exact floor-strike print source/timestamp at window open (third-party says ≈ BRTI
  at open; not verbatim in spec).
- Whether the 60s settle is a *plain* mean or a 20%-trimmed mean (spec says "simple
  average"; one source claims trimming "for certain markets" — the contract text
  governs, so likely plain mean, but confirm by scraping a settled window).

### The two edges this unlocks
**(A) The settlement-average cone — the headline edge.** Because settlement is a
60-second mean, the win probability late in the window is *far more computable than
"where is spot now."* Let the running partial average over the elapsed part of the
final minute be `Ā_elapsed`. With t seconds left in the final minute, the settlement
is `S = ((60−t)·Ā_elapsed + (sum of remaining t prints)) / 60`. The remaining t
prints can only move `S` within a band of width ∝ `t/60 · (achievable BRTI move)`.
**The settlement variance collapses linearly in remaining time, faster than a naive
spot model assumes.** If Kalshi's quote is still pricing off instantaneous spot while
you compute the partial mean, you have a model edge — *especially* in the final 60–90
seconds, which is exactly where your delta-hedge loop (T+14) operates.

> **Code implication:** your `get_fair_price_2d` keys off instantaneous % move from
> floor. In the final minute, replace/augment it with a **running-BRTI-partial-average
> model**: fair = P(60s-average ends above floor | prints so far). This is a genuinely
> different (and better-specified) quantity than your current one near the close.

**(B) Basis risk: model BRTI, not Coinbase.** Your `btc_t0` (Kalshi floor strike) is a
BRTI value; your `btc_now` (Coinbase WebSocket) is a single venue. You are comparing
two different price series. The basis is usually small but systematic and noisy at the
tick level — precisely the scale your signal operates on. **Fix:** subscribe to a
BRTI/CF-Benchmarks-style consolidated feed (or at minimum a multi-venue median), and
measure the Coinbase→BRTI basis distribution before trusting tight mispricing signals.

*Sources:* [Kalshi BTC contract spec](https://kalshi-public-docs.s3.amazonaws.com/contract_terms/BTC.pdf) ·
[CF Benchmarks / Kalshi](https://www.cfbenchmarks.com/blog/kalshi-leads-surging-crypto-event-contract-market-powered-by-cf-benchmarks) ·
[Kalshi Help — Crypto Markets](https://help.kalshi.com/en/articles/13823838-crypto-markets) ·
[Kalshi Market Maker Program](https://help.kalshi.com/markets/market-maker-program)

---

## Cluster 6 — The fee (re-derived) and the execution tax

### The fee formula (three sources agree)
```
fee per contract = ceil_to_cent( 0.07 · P · (1 − P) )      # P in dollars
total            = ceil_to_cent( 0.07 · C · P · (1 − P) )  # C contracts
```
- **Taker** pays this; **maker historically ~0%** (verify for crypto series).
- **Peaks at P = 0.50** → ≈1.75¢/contract (3.5% of a 50¢ contract per side);
  **→ 0 at the wings** (a 0.85 contract costs ≈0.89¢, half as much).
- **Rounded UP per fill** → tiny clips pay an effective rate far above 7%; size for
  fee efficiency.
- **Crypto multiplier UNCONFIRMED** — third parties claim crypto may be a "premium"
  category above 0.07; the official fee-schedule PDF was rate-limited and unverified.
  **Pull the live fee off a real API fill before sizing break-even-sensitive bets.**

Sources: [Kalshi Help — Fees](https://help.kalshi.com/en/articles/13823805-fees) ·
[Maker/Taker Math on Kalshi](https://whirligigbear.substack.com/p/makertaker-math-on-kalshi) ·
[marketmath.io](https://marketmath.io/platforms/kalshi).

### Why this matters: the wrong-shape fee in the code
Your `compute_pnl()`:
```python
if side == winner:  return count * (1.0 - fill_price) * 0.93   # 7% of winnings
else:               return -(count * fill_price)
```
The real fee is charged **on the trade, both winners and losers, ∝ P(1−P)** — it does
*not* scale with `(1−fill)` and is not levied only on winners. At P=0.50 the real fee
(1.75¢) is **larger** than a 7%-of-winnings model on a cheap-cushion winning contract;
at the wings it's **smaller**. Net: your EV gate and your historical P&L reconstruction
are both biased, and the bias flips sign across the price range. **Action:** replace
the fee model with `ceil(0.07·C·P·(1−P))` charged at fill time, re-derive break-even
(below), and re-run `simulate*.py` / backtests.

### Fee-corrected break-even
A taker buying at price c, holding to settlement, with per-contract fee `φ(c) =
0.07·c·(1−c)`, breaks even when `q·(1) − c − φ(c) = 0` ⇒
```
q_be = c + 0.07·c·(1−c)        # ≈ c + 1.75% at c=0.5, +1.05% at c=0.7, +0.9% at c=0.85
```
(Compare iteration 1's winnings-fee model `q_be = c/(1−0.07(1−c))`; both raise the bar,
but this per-contract form is the correct one if the fee is on the trade. The two
agree to ~first order near the middle but differ at the wings — which model is right
depends on the empirical fee, so **verify**.)

### The execution tax is probably your dominant P&L term
Round-trip cost per bet ≈ `spread + 2·fee + adverse-selection drift`. Worked at
P=0.50, 2¢ spread: **4% (spread) + ~7% (2× taker fee) ≈ 11% of notional**, before
adverse selection. With k correlated bets per window, capital survival scales like
`(1−c)^k` — **~44% drag over 5 round-trips** — while your statistical edge adds
linearly. **Most backtests fill at mid with no fees and die live for exactly this
reason.**

**Measure the tax on your own logs (all computable from `trade_log`/`tick_log`):**
- **Roll's effective spread:** `s = 2·√(−cov(Δp_t, Δp_{t−1}))` from the trade-price
  tape — reveals what you *actually* pay (incl. walked levels). Positive cov ⇒
  trending/toxic regime (estimator undefined = warning flag).
- **Kyle's λ (price impact):** regress signed price change on signed size from your
  fills; λ ∝ σ/√liquidity, large and convex in size on a thin book. Don't trust the
  closed form — trust your regression slope.
- **Toxicity gate:** signed order-flow imbalance + trade-sign autocorrelation over the
  last N trades; both spike when you're being picked off. Tag each fill with the
  *subsequent* mid move — if your buys are followed by marks-down, you're the adversely
  selected party.

*Sources:* [Kyle 1985](https://frds.io/measures/kyle_lambda/) ·
[Glosten-Milgrom 1985](https://cs.gmu.edu/~sanmay/papers/das-qf-rev3.pdf) ·
[Roll 1984](https://www.bauer.uh.edu/rsusmel/phd/roll1984.pdf) ·
[VPIN / flow toxicity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695041)
([critique](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731)) ·
[Kalshi Fee Schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)

### Execution fixes (highest leverage first)
1. **Post the non-urgent / hedge leg as a maker** (¼–0 fee + earn the spread). Tradeoff
   is fill risk + you become the adversely-selected quote — so only when the signal
   isn't decaying fast.
2. **Avoid trading pinned at 0.50** (max fee *and* max pick-off sensitivity). Prefer
   exposure away from the middle when the signal allows.
3. **Gate out toxic moments** (hot imbalance / positive trade-sign autocorr / fast
   unreflected BTC move).
4. **Size small** — λ is convex; split clips below the level that walks the book.
5. **Hedge BTC delta on a deep venue** (spot/perp), not by round-tripping the thin
   Kalshi book; net offsetting positions across windows before sending orders.

---

## Cluster 7 — Latency arbitrage: real mechanism, honest retail verdict

**The mechanism (Budish-Cramton-Shim):** a BTC binary's fair value is a deterministic
function of spot, so a sharp BTC move is the BCS "correlated-asset jump" and a lagging
Kalshi quote is the stale order. The sniping rent ≈ `(J − s/2)` per event, and
**competition doesn't compete it away — it just collapses the *window*** (≈100ms in
2005 → <10ms by 2011). Per-snipe profit is roughly constant over a decade; only speed
thresholds rose.

**The retail reality (Aquilina-Budish-O'Neill):** races are won by **~5–10 microseconds**;
**top ~6 firms take >80%** of races. A websocket retail trader runs **tens–hundreds of
ms** — thousands of times too slow to beat a colocated pro. **Against a colo maker you
are the victim, not the sniper.**

**The Betfair in-play analogy (directly transferable):** live event, continuously
updating true probability, documented latency edge ("courtsiding"), and the standard
venue defense is to **delay/disadvantage the aggressive taker** (Betfair's 1–8s in-play
delay, with a passive-bet carve-out that *protects makers*). If Kalshi tightens
KXBTC15M microstructure, expect it to hurt the taker-snipe play specifically.

**When the retail snipe IS plausibly capturable (narrow):**
- Counterparties **slower than your sub-second loop** (KXBTC15M is thin and may not yet
  be colo-dominated — you're trying to be the *least slow*, not win a µs race).
- **Only on sharp BTC jumps** where `J` clears spread + fee.
- **As the aggressive taker** against stale resting size, **never as a resting maker**
  during a jump (then you're the stale quote).
- **Away from 50¢** to cut the fee — but note max probability-sensitivity to BTC is *at*
  50¢ (max gamma), a genuine tension.
- Treat it as **decaying, capacity-constrained, regime-dependent**. Measure your real
  end-to-end loop latency vs how fast Kalshi actually reprices on a BTC move *before*
  risking capital.

> **Reconciliation with the settlement cone:** naive "BTC moved, fade the lag" latency
> arb is *bounded by the 60s averaging* — a spot jump moves fair value **less than 1:1**
> late in the window. The cleaner version of the edge is to **price the running 60s
> partial-average** and trade when the book over-reacts to an instantaneous spike the
> average can no longer fully absorb. The settlement structure turns a fragile speed
> race into a more durable *model* edge.

*Sources:* [BCS — HFT Arms Race (QJE 2015)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2388265) ·
[Aquilina-Budish-O'Neill (QJE 2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3636323) ·
[Betfair in-play delay](https://www.betangel.com/betfair-inplay-delay/) ·
[passive-bet delay](https://www.betangel.com/betfair-passive-bet-delay/) ·
[retail latency arb reality](https://cryptoweekly.co/crypto-latency-arbitrage-retail/)

---

## Cluster 8 — Fat-tail / jump pricing: when it matters

Two-regime answer (full memo: jump-diffusion + smile-skew + Student-t):
- **Near the money (most of your bets):** the fat-tail correction to `N(d₂)` is only
  **~1–3 probability points — the same order as the spread, so it's swamped by
  execution costs.** Don't over-engineer pricing here.
- **Deep ITM/OTM near expiry:** **first-order.** Gaussian mis-prices the tail by 5–20×
  in relative terms — it says 0.1% loss, truth is ~1–2% (Student-t₃₋₄ or a single
  jump). **A +0.3% cushion with 5 min left is *not* the ~95% your table's top cells
  imply** — exactly where `sigmoid_winrate` saturates and where overbetting hurts most.
- **Tool:** the **Barndorff-Nielsen–Shephard jump test** (`max(RV − BV, 0)`, bipower
  variation) is trivial on your tick data and flags jump-fragile windows live.
- **Recommendation:** body → `N(d₂)` + a smile-skew correction `−(ν/S)·∂Σ/∂K` (BTC has
  pronounced downside skew); tail → Merton/Kou or a Student-t CDF swap. Hunt the
  mispriced tail, not the middle.

*Sources:* [Merton jump-diffusion](https://quant-next.com/the-merton-jump-diffusion-model/) ·
[Kou double-exponential](http://www.columbia.edu/~sk75/MagSci02.pdf) ·
[Breeden-Litzenberger / smile→RND](https://www.srabbani.com/volatility.pdf) ·
[BNS jump test](https://public.econ.duke.edu/~get/browse/courses/883/Spr15/COURSE-MATERIALS/Z_Papers/BNSJFEC2006.pdf)

---

## Updated code-action list (supersedes/extends iteration 1's table)

| Priority | Finding | Concrete change in `live/` |
|---|---|---|
| **P0** | Fee is `ceil(0.07·C·P·(1−P))` per contract, not 7% of winnings | Fix `compute_pnl()`; add a real fee function; **re-run all backtests**; verify the crypto multiplier from a live fill |
| **P0** | Settlement = 60s BRTI average | In the final minute, price the **running partial-average cone**, not instantaneous spot; this is a new, better fair-value near the close |
| **P1** | You pay basis (Coinbase) vs settlement (BRTI) | Add a BRTI/multi-venue feed; measure & correct the Coinbase→BRTI basis |
| **P1** | Constant-taker tax dominates P&L | Measure Roll spread + empirical λ + summed fees on your logs; compare to gross edge before any signal tuning |
| **P1** | Maker fee ~0 vs taker fee peak at 0.50 | Post the hedge/non-urgent leg as maker; avoid trading pinned at 0.50 |
| **P2** | Latency arb = mostly victim | Never rest naked maker orders during a BTC jump; only take stale quotes on sharp moves; measure your true reprice latency |
| **P2** | Toxic-flow pick-off | Add a signed-imbalance / trade-sign-autocorr execution gate; tag fills with subsequent mid move |
| **P3** | Fat tails first-order only deep-ITM near expiry | Don't trust top `sigmoid_winrate` cells as ~95% safe; add BNS jump flag; deep-fractional Kelly there |

### The single most important next experiment (unchanged, now sharper)
Reconstruct, per historical bet: **(a)** the true Kalshi fee, **(b)** the BRTI-based
settlement (not Coinbase), **(c)** the running 60s-average fair value near the close,
and compare `p_empirical` vs `N(d₂)` vs `π_market + true_fee`. Edge survives only where
empirical beats *both* — and the most likely place it does is the **settlement-cone +
stale-quote** residual in the final 1–2 minutes, on sharp moves, as a taker.

---

## Open questions for iteration 3
1. **Build & validate the running-BRTI-partial-average pricer** against settled
   windows in the repo's data — quantify how much better it is than instantaneous spot
   in the final 2 minutes.
2. **Empirically measure the Coinbase→BRTI basis** and Kalshi quote-reprice latency
   from the existing `tick_log` / WebSocket logs.
3. **Maker-strategy design:** can a passive two-sided quote with a fast BRTI re-quote
   loop and a jump kill-switch capture spread on KXBTC15M without being adversely
   selected? (This flips the whole strategy from taker to maker.)
4. **Optimal-stopping / dynamic-programming** formulation of *when* in the window to
   bet, given the shrinking settlement cone and fee curve (an interesting math problem
   in its own right).
