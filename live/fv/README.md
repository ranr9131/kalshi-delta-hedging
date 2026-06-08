# Pin-point fair-value model for Kalshi crypto (v3)

Goal: the most accurate possible fair value for Kalshi crypto markets, scored by
out-of-sample **Brier / log-loss against true CF Benchmarks settlement**. Pure
accuracy — not assumed to be tradeable (see the selection-effect caveat below).

## What Kalshi actually settles on (verified vs the live API)

Every crypto market — 15-min up/down **and** hourly threshold/range — settles on
the SAME engine: the **60-second simple average of the CF Benchmarks Real-Time
Index** (BRTI for BTC, per-asset RTI otherwise) over the final minute before
close. The API's `expiration_value` *is* that realized 60s-average.

The 15-min up/down market is not special: it carries a `floor_strike` equal to
the prior window's settle (the locked start-of-window 60s-avg), so it reduces to
the same shape as a threshold market:

> P( settle 60s-avg index  {>= , > , < , between}  strike(s) )

## Why v2 was wrong, and what v3 fixes

v2 (`fair_price_model_v2.py`) priced `P(Coinbase spot >= strike)` with a Gaussian
log-normal and variance `σ²·τ`. Five structural errors, all worst near expiry
where you actually trade:

1. **Terminal object is a 60-second average, not a spot snapshot.** Under a
   diffusion with per-min vol σ, the variance of the average log-price `Ȳ` over
   the final 1-min window is closed-form:
   - `τ ≥ 1 min`: `Var[Ȳ] = σ²·(τ − 2/3)`
   - `τ < 1 min`: `Var[Ȳ] = σ²·τ³/3`  (→ σ²/3 at τ=1; continuous)

   Averaging shaves 2/3 of a minute off the effective horizon. v2's `σ²·τ` is
   overconfident near expiry.

2. **Settlement index ≠ Coinbase spot.** Settlement is multi-venue BRTI. We add
   a small mean basis `μ_b` and an additive index-basis variance `σ_b²`. `σ_b` is
   the residual uncertainty at τ→0 (Coinbase ≠ settlement index, and the model
   sees the *endpoint* not the realized average) — it stops the model collapsing
   to 0/1 too fast.

3. **Fat tails.** Standardized Student-t innovations (calibrated dof `ν`).

4. **Martingale drift.** `−σ²/2` log-drift over the effective horizon; corrects
   the long-horizon YES over-prediction v2 showed empirically.

5. **Correct strike types**: greater / greater_or_equal / less / between.

### Measured basis (7-day, corrected timing)
- **Pure venue basis** (BRTI 60s-avg vs Coinbase 60s-avg): ~+1.5 bps mean,
  ~2.5 bps std — tiny and stable.
- **Endpoint basis** (settle vs Coinbase instantaneous@close): ~8 bps std. The
  extra ~7.5 bps is within-minute endpoint-vs-average noise = real model
  uncertainty at τ→0.

(v2's apparent ~11 bps came mostly from a 1-minute forward-timing bug in the old
kline lookup; fixed in `build_corpus.py::price_end_at`.)

## Pipeline

1. `build_corpus.py` — pull settled markets (both families, multi-asset) with
   `result` + `expiration_value` (true settle) + Coinbase klines →
   `corpus.csv`; measure basis → `basis_report.json`.
2. `fit_v3.py` — time-split; **Stage A** recovers the physics per asset
   (`μ_b`, `σ_b`, `vol_mult`, `ν`) by regressing `(ln settle − ln spot_t)²` on
   `σ_t²·g(τ)` (slope→vol, intercept→σ_b²) + kurtosis→ν; **Stage B** fits a
   gentle residual Platt per (asset, horizon bucket) → `calibration_v3.json`.
3. `eval_v3.py` — honest OOS: v2 vs v3-raw vs v3-cal on the held-out split,
   Brier/log-loss by family / horizon / moneyness + reliability tables.

The live model is `../fair_price_model_v3.py` (dependency-free; hot-reloads
`calibration_v3.json`).

## Results (held-out OOS: train 7d, test latest ~2d, 258k samples)

Tradeable-band (0.05–0.95) Brier, **v2 → v3 (physics-only, no Platt)**:

| segment            | v2     | v3      | Δ      |
|--------------------|--------|---------|--------|
| BTC                | 0.1392 | 0.1184  | −15%   |
| ETH                | 0.1538 | 0.1466  | −4.7%  |
| SOL                | 0.1581 | 0.1499  | −5.2%  |
| XRP                | 0.1044 | 0.1019  | −2.4%  |
| 15m family         | 0.1691 | 0.1654  | −2.2%  |
| hourly family      | 0.2323 | 0.2241  | −3.5%  |
| every horizon bkt  | better at all of 0-12 / 12-25 / 25-50 / 50-100m |

v3 physics beats v2 on **every asset and every horizon bucket**, with no
overfitting. The biggest single fix is calibration: v2 mis-prices a large
near-ATM cluster by up to ~19 points (predicts 0.50, realizes 0.31); v3 cuts
reliability error to a few points.

What drove it (ablation): the **validation-gated `sd_mult ≈ 1.2`** (v2 was
~20% overconfident) + the **60s-average horizon shave** + **correct strike
types / index basis**. Per-asset fat-tail `ν` ranges from 4 (BTC-15m, very fat)
to ~Gaussian (BTC-hourly, SOL, XRP).

### Two honest caveats
1. **Stage-B Platt is OFF by default.** It improved the *aggregate* only by
   absorbing a DOGE downtrend that persisted train→test — fragile regime-fitting
   that *hurt* every well-behaved asset out of sample. `--with-platt` re-enables
   it; don't, unless you specifically want a (fragile) DOGE drift term. DOGE
   near-ATM is an honest coinflip in the physics model (0.25 Brier = irreducible).
2. **Sub-minute (τ < 1 min) — FIXED & validated on Binance 1s data.** The τ<1
   variance was originally `σ²·τ³/3` (future part of the averaging window only),
   which collapses to ~0 at τ→0 → wildly overconfident. The correct closed form
   adds the *elapsed-window* term:
   `Var = σ²·(τ³ + (1−τ)³)/3`  — at τ→0 this floors at `σ²/3` (≈0.577·σ), the
   endpoint-vs-trailing-average noise. Validated against 1s ground truth
   (`build_corpus_1s.py` → `validate_1s.py`, 14.9k fine-horizon BTC/ETH/SOL
   samples): the old law's z² hit **793** at 5s-to-close (predicted ≈0 variance);
   the new law sits near 1. Binary **log-loss at τ<1 improved −22.7%** (0.568 →
   0.439) across all three assets. The result is a correct **U-shaped** last-minute
   sd (high at τ→0 and τ→1, min at τ=0.5).

   **Partial-average enhancement (implemented, opt-in).** `fair_p(...,
   use_partial_history=True)` (or pass `partial_avg=`) feeds the running 60s-avg of
   the elapsed window into the τ<1 predictor: mean → `(1-τ)·ln(avg)+τ·ln(spot)`,
   variance → `σ²·τ³/3` (the elapsed term vanishes — it's now observed). Validated
   within-venue on 1s data: at 5s-to-close the settlement-prediction sd collapses
   from 3.55→0.12 bps. BUT its *binary* benefit needs a low-basis feed: the 1s
   validation uses Binance, whose basis to BRTI is −10±4.2 bps (USDT vs USD) and
   swamps the collapsed variance → partial looked worse on Binance. Coinbase↔BRTI
   is only ~2–3 bps, so live (Coinbase feed) should benefit. **Off by default**;
   enable only with the Coinbase feed + record_price() fed. `live` feed for the
   actual settlement index (BRTI) would make it strictly better still.

## Caveat (carried from the project memory)

A better-calibrated fair value does **not** imply tradeable edge on liquid Kalshi
crypto: strategies fire where the model most disagrees with the market, and on
efficient markets that disagreement is dominated by *model* error, not market
error. v3 is built and validated as an accuracy artifact. Whether any residual
edge survives must be tested separately against the market mid (not done here;
the user scoped this to accuracy).
