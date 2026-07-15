# Alternative Strategies for Kalshi 15-Minute Crypto Markets — Final Research Report

**Prepared for:** solo operator, Kalshi crypto binaries (KXBTC15M / KXETH15M / KXSOL15M / KXXRP15M + hourly/daily ladders)
**Baseline:** profitable momentum-continuation taker strategy (S6), +4.08c/contract walk-forward OOS, ~$250 live bankroll, 27ms order RTT from AWS us-east-1
**Date:** 2026-06-30
**Method:** 21 candidate strategies across 7 families, each adversarially verified against five gates (fees, latency, capacity, evidence quality, ops cost). 3 survived. 18 are in the graveyard with kill reasons documented so we never re-litigate them.

---

## 1. Executive Summary

**The single dominant finding of this research cycle: every surviving opportunity is the same underlying physics — Kalshi settles on a 60-second average of CF BRTI (Asian-style digital), most participants (including our own current pricer) price the last tick (European-style), and the exploitable expression is TAKER-side, in the final 1–2 minutes, on BTC only.** Every maker expression, every cross-venue expression, and every "new signal" expression of anything died in verification. The fee structure actively favors the surviving trades: 0.07·P·(1−P) is 0.33c at P=0.95 vs 1.75c at P=0.50, and hold-to-settlement means one fee leg only (settlement is fee-free — verified against our own fills and [research/MATH_STRATEGY_RESEARCH_5_WALKFORWARD.md](research/MATH_STRATEGY_RESEARCH_5_WALKFORWARD.md)).

### Ranked shortlist

| Rank | Opportunity | Adj. confidence | Realistic net edge | Gate before capital |
|---|---|---|---|---|
| 1 | **Locked-average settlement snipe** (deterministic endgame) | 0.45 | +2–4c/contract on fills, low end of $2–15/day, BTC-only | Replay must show ≥3 qualifying stale asks/day persisting >100ms, zero mislocks over ~1,900 settlements |
| 2 | **Final-minute Asian repricer** (probabilistic generalization of #1) | 0.40 | +3–8c/contract claimed; treat as unmeasured until replay; realistic $5–25/day ceiling | Estimator must beat 2D table's Brier in minute 15 OOS; sigma calibrated for BRTI print autocorrelation |
| 3 | **Final-60s Asian-settlement snipe** (near-duplicate of #1/#2; merge, don't build separately) | 0.40 | Realistic ~$1–5/day standalone | Same replay; measure proxy-vs-settlement basis from ground truth |
| 4 | *(salvage, not a strategy)* **Fix the Asian-settlement bug in the S6 pricer** | n/a — pure upgrade | Removes a known systematic bias from every late-window S6 trade | None — it's a bug fix flagged in [memory/kalshi-math-research.md](memory/kalshi-math-research.md) |
| 5 | *(cheap experiment)* **Maker-attempt-with-taker-fallback hybrid on S6 signals** | 0.28 (as pitched: killed; as hybrid: worth a probe) | +0–1c/contract incremental if fill-conditional mark-outs are benign | Shadow/live mark-out gate, NOT backtest; hard no-quote zone in final 2 min |

Items 1–3 are one build, not three: a single real-time partial-average BRTI-proxy pricer with two trigger modes (deterministic lock, probabilistic divergence). Item 4 is mandatory step zero regardless of everything else. Item 5 is optional and only after 1–4.

**Honest expected value:** the modal outcome of the decisive (and nearly free) replay test is that a professional MM already prices the Asian rule on BTC and the residual is ~$1–5/day of retail stale-quote scraps. That is still worth building because (a) the test costs one afternoon against data we already have, (b) the pricer it requires is the same artifact that fixes our known S6 bug, and (c) if the market genuinely hasn't priced it, the per-contract gaps are the largest of anything surveyed (theoretical 11–30c at 0.5–1σ displacement).

---

## 2. Surviving Candidates

### 2.1 Locked-Average Settlement Snipe (rank 1, confidence 0.45)

**Mechanism.** Settlement = simple average of 60 once-per-second CF BRTI prints in the final minute ([help.kalshi.com/en/articles/13823838](https://help.kalshi.com/en/articles/13823838); official contract terms PDF, kalshi-public-docs S3, confirms "simple average… for the minute (60 seconds) prior" and that last trading time equals expiration — the settlement minute **is** tradeable). With r seconds left, (60−r)/60 of the average is a banked constant. When the banked partial average sits far enough from the strike that no achievable move of the remaining prints can flip the outcome, P(win) ≈ 1 by arithmetic — yet asks rest at 90–97c because retail (and naive bots) price the live ticker, which can still cross the strike. Grounding math: at 40% annualized vol with 15s left, remaining prints contribute ~0.4bps std to the average — a 2bps banked gap is a 5σ lock; flipping a 3bps gap with 10s left needs an 18bps move in 10s.

**Why the fee gate clears decisively.** One taker fill, held to settlement: 0.07·0.95·0.05 = **0.33c/contract** at P=0.95 (fee formula and sub-cent rounding verified against our own real fill). The fee curve is structurally cheapest exactly where this edge lives.

**Why latency plausibly clears.** The prey's mispricing is *model*-stale (wrong pricing model), not *time*-stale — retail limit orders rest for seconds, so 27ms beats them trivially. The only race is against other lock-aware snipers; locks deepen continuously and asks repost, so it's not strictly winner-take-all. This is measurable offline (ask persistence after lock onset) from the recorder's exchange+local timestamps.

**Realistic edge at our scale.** +2–4c/contract net on qualifying fills. Verification haircut the original estimate hard: effectively **BTC-only** (ETH/SOL/XRP final-minute books too thin per our own 15–35% IOC failure data), stale-ask size at final-20s extreme prices is plausibly 5–20 lots not 10–100, and the basis-tail-widened lock threshold cuts qualifying windows. Expect the **low end of $2–15/day**. Hard low ceiling on scaling — this pays for itself at $250, not at $25k.

**Validation before money (all on existing infra):**
1. Replay ~5 days × 96 windows of the recorder corpus (~1,900 BTC settlements). Per second of each final minute: reconstruct the proxy partial average from our spot feeds, flag "locked" states under a **jump-aware bound** (require gap > worst-case-move·(r/60), not 3σ diffusion; hard r<15s cutoff), record best resting ask + size at that instant, mark to actual settlement.
2. **Zero-loss check**: did any "locked" call ever lose? Breakeven mislock rate is ~4%; target <1%. Note 1,900 settlements only bounds mislock to ~0.2% — acceptable but keep monitoring live.
3. **Ask-persistence check**: do locked-but-cheap asks survive >100ms after lock onset, or are faster bots already sweeping? This is the go/no-go — if persistence ≈ 0, the seat is taken and we kill it without spending a dollar.
4. **Basis gate**: our Coinbase/Binance proxy vs true BRTI has ~4bps median, ±11bps tail basis — and **Binance is not a BRTI constituent** (constituents: Coinbase, Kraken, Bitstamp, Gemini, itBit, LMAX, Bullish, Crypto.com — [CME CF methodology](https://docs.cfbenchmarks.com/CME%20CF%20Real%20Time%20Indices%20Methodology.pdf)). Skip any signal inside the ~11bps-equivalent tail. Critically, cascades produce the jump AND the basis blowout simultaneously — the correlated-tail failure mode — hence the jump-aware bound is non-negotiable.
5. Shadow arm in the existing per-arm harness; confirm final-15s fill rates and order-rejection behavior at window close; then 1-lot live probes.

**Implementation sketch.** New module alongside `live/trader.py`: consumes existing `btc_multi_feed.py` spot stream, maintains a rolling 60-print proxy average per open window, emits lock events; execution reuses the V2 IOC path (no chase needed — we take resting asks). Recorder already captures everything needed for the replay.

**Key risks.** (1) Publicity decay — the quirk is documented on Kalshi's own help center, on kalshibacktest.com/predictionmarketspicks.com, and the Polymarket analog (~$40M extracted, then killed by dynamic fees — [financemagnates.com](https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/)) makes Kalshi the obvious port for displaced bots. (2) Correlated jump+basis tail as above. (3) Venue risk: Polymarket removed its 500ms taker delay in Feb 2026 to kill a related edge; Kalshi could restrict late-window trading. (4) Final-second order rejections.

### 2.2 Final-Minute Asian Repricer (rank 2, confidence 0.40)

**Mechanism.** Generalizes #1 to probabilistic states: correct fair value during the settlement minute is Φ((μ_A−K)/den), where μ_A blends the banked partial average with expected remaining prints and the denominator collapses as the average accretes (time-average variance = σ²T/3; Kemna–Vorst / Turnbull–Wakeman / Privault — all verified real and correctly applied). Spot-anchored pricers reprice 1:1 off the ticker when 30–50% of the settlement decision is already banked. Buy whenever |Asian fair − executable price| > fee + basis buffer + minimum edge, hold to settlement.

**Verification finding that matters:** our internal worked example (European 0.68 vs Asian 0.94) is internally inconsistent — it implies ~241% annualized vol. Recomputed at realistic 45% vol, the gap **direction survives and widens** (Asian ~0.976 vs European ~0.127 when spot re-crosses the strike against a banked offset), but the qualifying **states are rarer** than the pitch's 20–50 trades/day. Also: our 224k-obs table ([research/MATH_STRATEGY_RESEARCH_4_EMPIRICAL.md](research/MATH_STRATEGY_RESEARCH_4_EMPIRICAL.md)) proves our *own model* deviates in the Asian direction — it does **not** prove the *market price* does. Gap frequency at executable prices is the load-bearing unmeasured number; the deep 1c-spread BTC book suggests a professional MM who has read the contract terms. Treat a null result on BTC as the modal outcome (~50–60%).

**Realistic edge.** +3–8c/contract on qualifying ticks after 0.3–1.6c fee; frequency unknown until replay. If the MM already prices it, residual lives only in retail stale quotes — a fraction of theory. Honest range $0–25/day.

**Validation before money.**
1. Same replay harness as #1, scoring the full estimator: per second, compute Asian fair from the proxy partial average, bucket (fair − best executable ask), mark to settlement over ~1,900 windows.
2. **Calibrate σ_per_sec from replayed data**: BRTI prints are order-book mids, positively autocorrelated — the iid σ²t/3 denominator understates variance. Uncalibrated, this estimator overstates P exactly like our failed NN (82% claimed / 55% realized). This is the single biggest model risk.
3. **Gate:** estimator must beat the 2D table's Brier in minute 15 out-of-sample (reuse the S6 Brier/reliability tooling) before any sizing.
4. Gate every trade on banked offset >> basis error (±11bps tails).
5. Paper shadow arm alongside live-s6; remember final-minute entries cannot be exited — every model error rides to settlement, so fractional Kelly is load-bearing.

**Implementation sketch.** Same pricer module as #1 — the lock snipe is just this estimator evaluated at the P≈1 boundary. One codebase, two trigger thresholds. Slippage assumption (2.4c chase, n=11) should improve since we take resting quotes rather than chase.

**Key risks.** Model miscalibration (autocorrelation), basis (first-order here, worse than in #1 because probabilistic states are more basis-sensitive), adverse selection on final-minute resting quotes (a 70c ask at second 45 may know something), competent BTC MM already pricing the rule, no-exit-at-full-loss on errors.

### 2.3 Final-60s Asian-Settlement Snipe (rank 3, confidence 0.40) — merge into #1/#2

This candidate (from the latency family) survived on the same verified mechanism: buy 75–90c quotes whose implied prob ignores the locked running average. Its independent verification confirmed: fees pass clearly (0.3–1.1c single leg), latency is a model-edge not a speed-edge, settlement mechanics confirmed via Kalshi help center, ops trivial. Its haircuts also replicate: expect ~1–3 gated events/day at 5–50 contracts (**~$1–5/day**), not 5–20 events at 50–200, because (a) the quirk is public knowledge among bot builders ([protos.com](https://protos.com/polymarket-ends-trading-loophole-for-bitcoin-quants) documents quants already bridging Polymarket 5-min and Kalshi 15-min), (b) the proxy-error gate binds hardest exactly when triggers fire, (c) final-minute resting depth << mid-window depth (the "zero walk to 200 contracts" figure was measured across whole windows and does not apply to the final 45s). Its cited Polymarket win-rate stats (BenjaminCup, 5-min markets) settle on point-in-time Chainlink prints and **do not transfer** to averaging settlement — averaging specifically dampens final-10s moves.

**Action:** do not build separately. Fold its trigger (75–90c band, wider than #1's deep-lock band) and its haircutted expectations into the combined pricer as a third threshold tier. Its distinctive contribution to the plan: measure proxy-vs-actual-settlement error directly from settlement ground truth we already hold — the cheapest possible bound on the basis problem all three trades share.

### 2.4 Mandatory step zero: fix the Asian-settlement bug in the S6 pricer

Not a strategy — a flagged, unfixed bug ([memory/kalshi-math-research.md](memory/kalshi-math-research.md)). Every verifier that touched the final minute flagged it, and one killed candidate (embedded-IV engine) showed European inversion produces huge *spurious* late-window signals purely from settlement mechanics. Fixing `get_fair_price_2d` (or gating it out in the final ~2 minutes) is a pure upgrade to the profitable incumbent, is prerequisite to trusting any p_win in candidates 1–3, and shares 100% of its code with the new pricer.

---

## 3. Graveyard

Eighteen candidates died in verification. Do not re-litigate without new evidence of the specific type noted.

### Market-making (all three dead)

- **Maker-ize the S6 edge (passive NO-side quoting)** — adj. 0.28. Killed by: 1c spread + deep book = no price to improve, so you join behind 200+ contracts and fill exactly when the level sweeps (conditional edge gone); unfilled-signal opportunity cost (forfeited +4.08c) needs ~35–50%+ fill rate that was asserted, never measured; 27ms cancel loop loses every pick-off race vs colocated MMs (5–30c loss per pick-off vs 1–2.5c per-fill gain); Stanford maker-profit finding (Bartlett & O'Hara, [SSRN 6615739](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6615739)) is longshot bias in predominantly-NO event markets — doesn't transfer to ~50/50 crypto binaries; queue-replay backtests are structurally optimistic. *Salvage kept as shortlist #5: maker-attempt-with-fast-taker-fallback, gated on live mark-outs only, no-quote zone final 2 min.*
- **Dual-mode A-S/GLFT fair-value MM on KXBTC15M** — adj. 0.12. Zero-maker-fee assumption likely false (help center + 2026 guides: maker ≈ 25% of taker, which alone kills the +0.2–0.8c honest edge); Binance-Tokyo→us-east-1 jump-pull path ~95–110ms at *median*, races decided at tails; back-of-queue behind SIG (Kalshi's designated MM since 2024, [businesswire.com](https://www.businesswire.com/news/home/20240403664852/en/)) = structurally toxic fills; one 20-lot jump-through near 50c erases days at a $3–10/day ceiling; unhedgeable inventory on a 15-min digital.
- **Final-two-minutes Asian-settlement maker** — adj. 0.12. Right math, wrong side of the book: resting quotes in the most race-prone 90s earn 2–5c clean but lose 20–50c to correct-model snipers with no exit; needs >~90% clean-fill rate nobody measured; "zero maker fee" premise false; both Polymarket citations describe point-in-time settlement and the $40M figure describes the *predators* (taker bots). *Its taker inversion is exactly shortlist #1–3.*

### Cross-venue arb (all three dead)

- **Kalshi↔Polymarket complementary-pair box** — adj. 0.12. Pre-fee PM profits ($40M; $313→$414k bot) were single-venue latency sniping at ~50/50, killed by PM's purpose-built dynamic taker fee (peak ~1.8c at 50c, [docs.polymarket.com](https://docs.polymarket.com/market-makers/maker-rebates)); post-fee crossing frequency at skew is pure conjecture; legging risk (one −40c leg-out erases 40+ boxes) breaks "near-riskless" at a $250 bankroll; Chainlink-vs-BRTI oracle tail makes the box ~97% not 100%; Polymarket-US API access unresolved. *If ever revisited: 2 weeks of free PM CLOB recording first, and express it maker-side on PM.*
- **Deribit/IBIT vol-surface pricing of Kalshi hourly/daily ladders** — adj. 0.12. Core premise factually wrong: SIG prices these ladders off the same surfaces with a production vol desk; a >3c gap vs a home-built 0DTE→1h interpolation is more likely our model error; digital vega math shows you'd need 8–12 vol points of disagreement — the same size as irreducible interpolation noise (the NN failure mode again); BSIC "analog" was zero-fee Polymarket *weeklies* whose authors abandoned Kalshi for illiquidity; 5–15s REST oracle vs millisecond re-quoting = adverse selection.
- **Final-minute Kalshi-Asian vs PM-point relative value** — adj. 0.13. Hedged leg admitted dead to PM taker fees, collapsing to a naked Kalshi bet (= shortlist #2, worse-specified); our BRTI proxy misses 4+ of 8 constituents and its error exceeds remaining settlement σ exactly when the gap is largest; open-source repos already trade the pair. *Its kernel — bound proxy basis with real BRTI data, fix the pricer bug — was absorbed into the surviving plan.*

### Latency / lead-lag (two of three dead)

- **Spot-jump stale-quote snipe (mid-P, large-move gate)** — adj. 0.20. Kalshi's 1.75c at P=0.5 (3.5% of premium) exceeds the 3.15% fee Polymarket calibrated to kill this exact trade; hold-to-settle converts it into a directional bet on a model with confirmed optimistic fat-tail bias; replay fills are an upper bound (can't see cancel races); only Kalshi-specific lag evidence is a bot repo README; probable alpha overlap with S6. *Its own verifier pointed to the late-window high-P variant = shortlist #1–3.*
- **Alt-binary staleness snipe (ETH/SOL/XRP)** — adj. 0.08. Internally contradictory: thinness claimed as both the source of staleness and (admitted) source of unfillability; fee ≥ claimed gross edge at mid-P; cited academic support (Sifat & Mohamad 2019, hourly/daily) argues *against* intraday exploitability; $1–5/day ceiling, XRP admitted untradeable.

### Vol structure (all three dead)

- **Final-minute vol crush (buy ITM at 0.70–0.90)** — adj. 0.12. **Sign error in its own evidence**: the 224k-obs table shows final-minute realized win rates *below* the European model (0.775 vs 0.945), i.e., evidence against buying ITM there; Burgi/Deng/Whelan ([karlwhelan.com/Papers/Kalshi.pdf](https://www.karlwhelan.com/Papers/Kalshi.pdf)) samples only ≥24h contracts — structurally excludes 15-min crypto; entry at 0.85 needs <2pt calibration where own table shows 12–17pt errors.
- **Embedded-IV vs forecast-RV engine** — adj. 0.20. σ_imp inversion is tick-quantization noise where it trades (1 tick ≈ 10–20% of σ_imp at P=0.55–0.60, exactly the fee peak); European inversion is systematically wrong late (Asian settlement); BTC VRP evidence is 27-day Deribit, ~2600x horizon mismatch; GWU taker numbers misquoted (takers −32%); exit leg doubles fees, killing the advertised kill-switch; 2-week shadow promotion test underpowered ~10x. *Salvage: one-day corpus study — win-rate vs IV-RV-gap deciles restricted to P>0.70, τ>3min; if monotone, add forecast-RV as one conditioning dim to the 2D table.*
- **Dead-hours short-vol overlay (weekend deep-ITM)** — adj. 0.12. Mean-vs-tail confusion: digitals at 82–92c price *tail crossing probability*, and weekends have lower mean vol but Kaiko-documented 2–3x fatter conditional tails — this is S5 in a seasonality costume; the "no time-of-day adjustment" claim sits inside our own 0.040 calibration residual; mechanism requires retail present in hours defined by retail absence; unattended weekend short-tail book for a solo operator.

### Settlement microstructure (one of three dead — the other two survived)

- **BRTI-replica lead-lag inside the averaging window (fade book overreaction)** — adj. 0.12. Round-trip fee (~3–3.5c near mid) consumes the 1–4c edge, and the single-fee version removes the exit exactly when a second jump can flip the banked-average claim; 2026 final-minute counterparties are bots, not 1:1-ticker retail; all profitability evidence is Chainlink point-settlement (different mechanism); we hold 2 of 8 constituent feeds. *The (60−s)/60 attenuation arithmetic is real and lives on inside shortlist #2's pricer.*

### Behavioral flow (all three dead)

- **Popular-side premium fade** — adj. 0.25. GWU WP 2026-001 ([gwu.edu PDF](https://www2.gwu.edu/~forcpgm/2026-001.pdf)) *explicitly excludes* sub-24h crypto markets; its taker/maker gap is fee-mechanical; its favorite-longshot direction says the cheap faded side is the historically losing leg; crypto 0–1h calibration slope ≈ 0.99–1.05 ([arXiv 2602.19520](https://arxiv.org/abs/2602.19520)) leaves no unconditional room; taker version is S6 reworded. *Salvage: free corpus study of signed calibration residual by candle-context bucket, netted against S6 overlap.*
- **Late-window favorite harvesting (FLB + TWAP)** — adj. 0.15. Same GWU inapplicability; slope-0.99 kills the FLB leg, leaving only the TWAP effect priced off our known-broken model; 1:13–1:100 payoff shape vs our tail-calibration track record; entries at T-120/90s precede the averaging window (spec error). *Its backtest design was absorbed into shortlist #2's validation.*
- **Retail-session conditioning (weekend/overnight Kelly boost)** — adj. 0.22. Kaiko weekend data is *spot* liquidity, not Kalshi calibration; 24/7 arb bots don't thin out on weekends; detecting a 1c residual difference needs ~19,000 windows/bucket (~6 months of corpus, not "several weeks"); XRP Jan-2026 incident ($233k weekend BRTI push) is manipulation-risk evidence *against* concentrating late-window exposure off-hours. *Salvage: hour-of-week bucket study as free analysis with vol controls; no capital decisions on it.*

### Multi-market structure (all three dead)

- **15M vs hourly KXBTCD cross-tenor box** — adj. 0.15. Settlement identity is real (verified from live API), but the only observed 11c gap failed its own simultaneous control (ghost arb); strike-grid premise wrong ($500 spacing, not $100), cutting aligned hours ~4–6/day; a missed second leg puts ~35% of bankroll into a max-gamma digital. *The 2-week joint-capture measurement (add ~6 NTM KXBTCD strikes to the recorder) is cheap and correctly designed — optional background task.*
- **Range-vs-ladder replication scanner** — adj. 0.04. Anchor observation was impossible ($100-wide range on a verified $250 grid); live measurement showed **zero pre-fee executable gap** at peak activity, books enforced to the penny by one MM; ranges are daily/weekly only (no 15m/hourly tenor); ~2.9c three-leg fee hurdle; SOL has no range series.
- **Intra-ladder monotonicity sniper** — adj. 0.05. Own 188-strike live scan found zero violations at a 0c threshold vs a ~3c+ fee-adjusted hurdle; the transient lives inside one MM's ms-scale sequential re-quote — 27ms×2 legs arrives after it's fixed; SIG + Jump confirmed active; Polymarket base rate ([arXiv:2508.03474](https://arxiv.org/abs/2508.03474)) exists only because that venue is fee-free.

### ML features (all three dead)

- **Binance-lead features (perp flow + Binance-vs-BRTI gap)** — adj. 0.30, highest of the dead. Killed as an event trade by geofencing (fstream.binance.com returns 451 from US cloud IPs — must test from the prod box), commoditization/adverse selection, and misapplied contemporaneous-R² evidence. *Salvage worth doing: the calibration-feature variant (trailing 1s/5s/30s offshore-perp return + signed taker volume as logistic inputs, no racing) tested offline for marginal walk-forward Brier over S6 features — after confirming feed reachability and fixing the Asian bug first so late-window improvements aren't confounded.*
- **Kalshi own-book OFI / large-trade flow** — adj. 0.20. Evidence from venues with genuine private information (elections, NFL); in 15-min crypto the only information is the public spot tape we already read faster; conditioning on the cause of a deviation whose effect (price vs p_win) is already a feature adds ~0; veto arrives after MM re-quotes. *Low-priority offline redundancy regression only.*
- **Liquidation-cascade conditioning (forceOrder bursts)** — adj. 0.13. Certifying a 3pp lift needs ~1,300+ events vs ~50–300 obtainable; feed throttled to largest liquidation per symbol per 1000ms (crude velocity proxy — already a feature in `live/trader.py`); calm-market depth stats invert during cascades; continuation becomes the popular side retail overprices — mirrored S5. *Optional: Tardis backfill as exploratory research with honest CIs.*

---

## 4. Portfolio Integration with the Incumbent Momentum Strategy

### Correlation and overlap

- **Time separation is the key property.** S6 trades mid-window continuation; the surviving trio trades the final 60–90 seconds on a settlement-arithmetic signal. Directional overlap exists (a locked-up window often follows an up-move S6 already bought), so at the portfolio level treat any final-minute add-on in a window where S6 holds the same side as a **size increase on the same bet**, not a new bet — cap combined per-window exposure under one Kelly budget. Simplest rule: skip the snipe in windows where S6 is already positioned same-side above half its cap; take it freely on the opposite side or in untraded windows (where it is genuinely diversifying, since lock events fire regardless of whether continuation set up).
- **Anti-correlation of failure modes is partial, not full.** S6 loses on reversals; the lock snipe loses on late jumps + basis blowout. Both cluster in cascade regimes. Keep a global cascade kill-switch (spot velocity threshold) that halts *both* arms — this also addresses the correlated jump+basis tail that verification flagged as the snipe's worst case.

### Shared infrastructure (the real reason to build this)

One new artifact serves everything: a **real-time partial-average BRTI-proxy pricer**.

| Component | Status | Serves |
|---|---|---|
| WS recorder (~850MB/day, exchange+local ts) | exists | replay validation for all three survivors |
| Coinbase/Binance spot feeds (`btc_multi_feed.py`, `binance_feed.py`) | exists | proxy average input |
| Partial-average Asian pricer | **build (~days)** | shortlist #1–3 triggers **and** the S6 bug fix (#4) |
| V2 IOC execution + fills/fees ground truth | exists | snipe execution (take resting asks — no chase) |
| Per-arm shadow harness + Brier tooling | exists | all validation gates |
| Jump-aware bound + basis gate | build (small) | snipe safety; doubles as S6 late-window filter |

Total incremental build: roughly one week of work, most of which we owe ourselves anyway as the bug fix.

### Capital split

At $250, capital is not the binding constraint (S6 runs tens of contracts/window; a 200-lot lock snipe at 95c is ~$190 held for <60s). Proposed split once the replay gates pass:

- **S6 (incumbent): unchanged sizing.** It is the proven earner; nothing in this cycle justifies diverting from it.
- **Lock snipe (#1): up to ~40% of bankroll per event** at deep-lock thresholds only (post-zero-loss-check), because holding period is <60s and true P ≈ 1 under the jump-aware bound — capital recycles into the next window.
- **Probabilistic repricer (#2/#3): fractional Kelly at ¼ of the S6 fraction** until the shadow arm shows ≥2 weeks of calibrated Brier — this is the arm with NN-style miscalibration risk and no exit.
- **Everything in the graveyard: $0.** Salvage items are analyst time only (each is a ≤1-day corpus study with a pre-registered kill threshold).

### Sequencing

1. **Week 1:** Build the partial-average pricer; fix/gate the S6 Asian bug; run the ~1,900-settlement replay (lock frequency, ask persistence, zero-loss check, proxy-basis bound from settlement ground truth).
2. **Decision point:** if locked-but-cheap asks persisting >100ms occur <~3/day on BTC, the seat is taken — keep the bug fix, shelve the snipe, spend salvage-study time instead.
3. **Weeks 2–3:** Shadow-arm #1 (and #2 if it beat the 2D table's minute-15 Brier OOS); 1-lot live probes.
4. **Only after 1–3:** optional maker-hybrid probe (#5) and the Binance-feature offline study, in that order of expected value per hour of attention — which, for a solo operator running a profitable book, is the scarcest asset this report allocates.