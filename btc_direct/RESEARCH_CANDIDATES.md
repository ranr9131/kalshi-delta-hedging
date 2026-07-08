# BTC direct-trading strategy candidates (deep research, 2026-07-04)

Source: 106-agent deep-research run — 24 sources fetched, 115 claims extracted,
25 top claims sent to 3-vote adversarial verification: **16 confirmed, 0 refuted,
9 unverified** (verifier agents hit API session limits — claims stand but lack
the adversarial pass; treat as one confidence tier lower). Synthesis step also
rate-limited; this ranking is Claude's own synthesis of the verified claims.

## Ranked for our stack (net edge × capacity at <$10k × implementation cost)

### 1. Turn-of-the-candle anomaly — TEST FIRST (free, uses data we have)
Verified 3-0: BTC returns of ~+0.58 bps/min concentrate in minutes 0/15/30/45
of each hour (the opening minute of 15-min candles); other minutes average
negative. t-stat > 9 on all 7 exchanges sampled in 2021; survives quantile
regression, TGARCH, and out-of-sample through Aug 2022.
[PMC10015199](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/)
- Unverified (rate-limited, not refuted): $5k capital → claimed 74% p.a. net
  vs 60% buy-and-hold, Sharpe ≈5.
- Fit: our 1-min corpus tests this in an afternoon. CAUTION: +0.58bp/min gross
  vs our 5bp taker cost means it only survives with maker fills or multi-minute
  holds — cost modeling is the whole game. Also: published 2023 = decay risk.

### 2. Funding-rate harvesting / cash-and-carry (delta-neutral income)
Unverified tier (verifiers rate-limited) but from BIS WP 1087 (Management
Science) and a 2025 peer-reviewed funding-arb study:
- BTC/ETH basis averaged ~7% p.a. 2019–2024, peaks >40%; funding-rate arb
  claimed +115.9%/6mo max-loss-1.92% in one study; uncorrelated with HODL.
- Structural caveats: Jan-2024 spot-ETF launch compressed carry by ~3pp
  (36% of the mean; ~97% on CME) — edge has structurally shrunk. At 10x
  leverage the short-futures leg would have been liquidated in >half the
  months 2018–2024 → run at 1–2x only.
- Fit: retail-accessible, no latency race, capacity fine at $10k. Blocker:
  needs a perp venue available to us (US constraint) — Coinbase/Kraken US
  perp access or CME micro futures. This is income (single-digit-to-low-teens
  % p.a. at safe leverage), not a get-rich signal.

### 3. Intraday time-series momentum (first half-hour → last half-hour)
Verified 3-0 (Financial Review 2022): first 30-min return predicts last
30-min return (t=4.38, OOS R²≈1.1–1.6%) — BUT the authors' own breakeven
costs are 3–10 bps/trade unlevered, i.e. marginal at our 5 bps, and the value
concentrates in down years (2014/15/18). Cheap to test alongside #1 since the
harness is identical. Low expectations.

### 4. Volatility risk premium (short BTC variance on Deribit) — LATER
Unverified tier: BTC VRP ≈ +14% annualized at 27-day horizon (~7x the S&P's),
OTM puts systematically rich. Genuine, widely-documented premium — but it's
short-tail-risk (one 2021-style crash erases months), needs an options venue
and margin, and $10k is thin for Deribit BTC options. Revisit if/when capital
and appetite for tail risk exist.

### 5. Daily-frequency simple technical rules (portfolio-of-rules)
Verified 3-0: Deprez & Frömmel 2024 (75,360 rules, snooping-corrected,
cost-inclusive) find simple-rule portfolios beat buy-and-hold OOS on
risk-adjusted basis. Counterweight (also verified 3-0): Hudson & Urquhart's
15k rules made NOTHING on BTC out-of-sample in 2018 — indicator edges on BTC
decay. Net read: daily-horizon rules are about drawdown control vs HODL, not
alpha. Low turnover → costs negligible. Worth a small harness eventually.

## Confirmed dead ends (don't revisit)
- **Cross-exchange arbitrage/lead-lag**: deviations documented through early
  2018 largely disappeared from Apr 2018 onward (J. Financial Markets study).
- **Triangular arb**: 4,879 opportunities in one week of 2024 Binance quotes →
  18 profitable after fees (~2% total). Dead at retail fees.
- **Sub-second order-flow imbalance**: real predictability (OFI R²≈3% at 1s,
  10–37% at 500ms across venues) but capturing it requires HFT-grade
  execution — the one confirmed edge we structurally cannot reach.

## Overfitting guardrails (both verified studies + Bailey/López de Prado)
- BTC-specific OOS decay is the norm: in-sample rule profits died OOS on BTC
  even when they persisted on alts.
- Published anomalies compress after publication (and after the 2024 ETF).
- Multiple-hypothesis corrections (Bonferroni/BH) + walk-forward with frozen
  params are mandatory before believing any candidate above.

Full claim list with quotes: session task output wzvy17d8x (scratchpad tasks dir).
