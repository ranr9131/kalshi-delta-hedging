# Kalshi Crypto Trading

Fair-value modeling and execution research for **Kalshi crypto prediction
markets** — primarily the 15-minute BTC/ETH/SOL/XRP "above-strike" binaries,
plus some daily/long-dated and sports experiments.

> **Status (2026-06-14):** research + **paper/shadow** stage. The primary
> strategies run live in paper mode against real quotes; **no meaningful real
> capital is deployed yet.** Backtest and shadow numbers below are encouraging
> but not yet validated on real fills — see *Execution Validation*.

---

## What are these markets?

Kalshi runs binary prediction markets on crypto every 15 minutes, 24/7. Each
market asks one yes/no question: **will the asset be above a fixed price at the
end of this 15-minute window?** That fixed price — the *floor strike* — is
Kalshi's oracle snapshot of the asset at the moment the window opens. A contract
pays $1 if you're right, $0 if wrong. You can buy Yes or No.

Kalshi prices contracts as probabilities: a Yes at $0.70 implies a 70% chance of
finishing above the floor. **Kalshi takes ~7% on gross winnings**, which — with
per-order rounding — is a major drag at small size.

The same machinery is pointed at four 15-min series (`KXBTC15M`, `KXETH15M`,
`KXSOL15M`, `KXXRP15M`) and, experimentally, daily markets and sports.

---

## Fair-value models (three generations)

Everything keys off one number: the **fair probability** the contract finishes
Yes, compared against Kalshi's quote to find edge. Three approaches have been
built, in increasing sophistication:

1. **Empirical 2D table** (original) — win rate bucketed by *(BTC move from
   floor, minute into window)* from ~6,370 historical markets. Simple, robust,
   powers the delta-hedge ("DH") trader. See *The Core Signal* below.
2. **`live/fair_price_model_v3.py`** — analytic digital-option pricing:
   log-normal/Student-t binary with **fat tails**, **basis drift**, martingale
   correction, correct strike-type handling, and a 60s settlement window.
   Recalibrated daily (`live/calibrate_daily.py`, 613k+ samples).
3. **Neural nets (`nn/`)** — transformer (`TSWinPredictor`) over the 15-minute
   feature sequence. **NN14 is the current best** by Brier and ROI. Variants:
   NN16 (NN14 + v3 features), and per-asset models (eth/sol/xrp).

---

## The Core Signal

Crypto moves within a 15-minute window have strong momentum. If the asset is up
0.3% from the floor four minutes in, it tends to still be above at close — and
**the larger the move, the more predictive it is**. Win rate depends on:

1. **How far** the asset has moved from the floor strike, and
2. **How many minutes** into the window we are.

A 0.02% move at minute 4 is barely better than a coin flip; a 0.4% move at
minute 10 wins ~95% of the time. The empirical 2D table captures this:

| Move from floor | min 4 | min 7 | min 10 | min 13 |
|---|---|---|---|---|
| < 0.05% | 57% | 61% | 63% | 69% |
| 0.05–0.10% | 66% | 73% | 80% | 88% |
| 0.10–0.20% | 75% | 83% | 91% | 94% |
| 0.20–0.50% | 86% | 92% | 96% | 94% |
| > 0.50% | 95% | 98% | 99% | — |

---

## Bet Sizing

Bets are sized by two independent signals multiplied together:

- **Magnitude multiplier** — smooth sigmoid over the size of the move (≈0× for
  tiny moves, up to 3× for large ones). No cliff edges.
- **Mispricing multiplier** — compares the model's fair price to Kalshi's quote.
  1× at zero edge, up to 2× when Kalshi underprices our side, <1× when it's
  against us.

> **final stake = base × magnitude × mispricing** (capped ~6×; usually 1–3×).

---

## Continuously Updating Position (Delta Hedging)

The DH trader re-evaluates every minute from minute 4 through 13. Each minute it
recomputes fair value and Kalshi's quote and decides whether to add.

- **Target mode** (default) maintains a desired exposure per side and bets only
  the gap — never piling on beyond what the signal justifies, shrinking if the
  asset drifts back toward the floor.
- **Additive mode** bets the full computed amount each minute (more volume,
  more variance).

The name is a loose options analogy — we continuously update a *directional*
bet, not hedge risk in the classical sense.

---

## Neural Models & Evaluation (`nn/`)

**NN14** — `TSWinPredictor` transformer: d_model 32, 4 heads, 2 layers, 14
features × 15 minutes. Trained with BCE/AdamW, early-decision augmentation
(random truncation), decision minutes 10–13. Checkpoint:
`nn/checkpoints/best_v2_small.pt` (stores weights + `feature_mean/std`).

**Walkforward** (`nn/walkforward_v2p.py`, 4 folds):

| Model | Brier@10 | ROI | Notes |
|---|---|---|---|
| **NN14** | **~0.136** | **+78–80%** | best balance |
| NN16 | ~0.131 | +76% | NN14 + v3 features; best Brier |
| V3 (analytic) | ~0.133 | +77% | highest raw P&L |
| Market mid | ~0.121 | — | baseline |

**No decay on fresh data:** retrained through 2026-06-14, the post-May-25
out-of-sample fold held at Brier 0.135 / +92% ROI (`nn/build_dataset_cacheonly.py`
→ `augment_dataset_v3.py` → `walkforward_v2p.py`).

**Sharpe** (`nn/sharpe_eval.py`, `nn/sharpe_live_dd.py`): backtest Sharpe is
~22 but capacity- and adverse-selection-blind; live *shadow* ~12–15; **honest
post-haircut estimate ≈ 4–6.** Max drawdown is tiny in-sample (sign that the sim
hasn't seen a real regime shock).

---

## Execution Validation (the honest part)

Backtests assume you get filled at a flat slippage. Reality (adverse selection,
queue, fill probability) is the gap between paper and money. Tools here measure
it **without real capital:**

- **`nn/tier23_taker.py`** — taker execution + adverse selection from the live
  shadow fill log. Finding: real fills landed **~5c worse** than the backtest's
  `mid+4c`; the model's Brier on the trades it *fires* is ~0.26 vs 0.136 overall
  (adverse selection is real).
- **`nn/slippage_sensitivity.py`** — re-runs NN14 across slippage levels. **The
  edge survives realistic cost:** even at the measured ~9c, NN14 still returns
  **+53% ROI**; break-even slippage is far higher.
- **`nn/replay_recordings.py`** — the decisive test: reconstructs the real order
  book + trade tape from `recorder.py` output and replays NN14 against it
  (taker walks actual depth; maker fills only when a trade prints through a
  resting quote, back-of-queue). **Run on the box where `recordings/` lives.**
  Closes the last gap: *fill probability*.

---

## Data & Recorder

- **`live/recorder.py`** (`kalshi-recorder.service`) — passive WS recorder
  dumping every `orderbook_delta` + `ticker` and Coinbase spot to
  `recordings/YYYY-MM-DD/{kalshi,crypto}.jsonl[.gz]`. `RETENTION_DAYS=5`.
- **`live/fv/corpus.csv`** — processed backtest corpus built from recordings
  (`build_corpus.py`).
- **`nn/data/dataset_v2p.npz`** — 16-feature training set (14 base + v3_fair +
  v3_minus_mid), Mar 23 → May 25 baseline.
- Candle/market caches under `data/cache/`.

---

## Strategy Portfolio (see `STRATEGY_LOG.md`)

| Strategy | State |
|---|---|
| DH trader (2D table + delta hedge) | **Primary**, paper-live |
| NN traders (BTC/ETH/SOL/XRP) | Shadow / paper (`kalshi-trader-nn*`) |
| Conversion maker / taker | Shadow (`kalshi-conv-*`) |
| Market maker (thin coin markets) | Built, not active |
| Touch-market copy-trade | Active research (best non-NN candidate) |
| LIP farmer | Live (small cap) |
| Momentum scalper | Exploration |
| Sports latency taker | Tested → killed (model error, not lag) |

`STRATEGY_LOG.md` is the source of truth for what's live, paper, or shelved.

---

## Repo Layout

```
live/                 live services, fair-value models, recorder, dashboards
  fair_price_model_v3.py   analytic fair value (production)
  recorder.py              WS book/tape recorder
  fv/                      processed corpus + builders
  *.service                systemd units (deployed on EC2)
nn/                   neural models, datasets, eval/validation
  model.py                 TSWinPredictor
  walkforward_v2p.py       fold backtest (NN14/NN16/V3)
  sharpe_eval.py / sharpe_live_dd.py   risk metrics
  tier23_taker.py / slippage_sensitivity.py / replay_recordings.py   validation
  checkpoints/             trained models
data/cache/          Kalshi market + candle caches
analyze*.py          historical analysis scripts
```

---

## Running It

```bash
cd live
pip install -r ../requirements.txt
python trader.py            # DH trader; set PAPER_MODE=true in live/.env
```

`PAPER_MODE=true` runs the full loop (timing, fair value, P&L) without placing
real orders. Deployed components run as the `kalshi-*.service` systemd units on
EC2.

**Validation quickstart:**
```bash
python nn/walkforward_v2p.py            # backtest NN14/NN16/V3
python nn/slippage_sensitivity.py       # does the edge survive real slippage?
python nn/replay_recordings.py --asset BTC --mode both   # run where recordings/ exists
```
