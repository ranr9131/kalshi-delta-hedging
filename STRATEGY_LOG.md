# Strategy Log

Running record of every strategy in this repo: what it does, current state, config that's actually in `.env`, recent code changes, and live performance. Update this whenever a strategy's state changes or you tweak its config.

---

## 1. DH trader (PRIMARY — currently paper-mode live)

**File:** [live/trader.py](live/trader.py) (1062 lines)
**Backtest engine:** [simulate_dh.py](simulate_dh.py)
**State:** **Paper-mode live** since 2026-05-18; restarted 2026-05-29 03:26 UTC with new two-sided config.

### What it does
Trades Kalshi KXBTC15M (every 15-min BTC binary). Each minute T+4..T+13, computes target exposure on YES or NO from a 2D empirical fair-price table (minute × |BTC move| bucket) and either bets the gap (target mode) or the full computed amount (additive). Bet size = base × sigmoid(magnitude) × sigmoid(mispricing) × time-decay multiplier.

### Current live config (`live/.env`)
| Setting | Value | Notes |
|---|---|---|
| `PAPER_MODE` | `true` | Logs orders, doesn't place |
| `MODE` | `dh-target` | Only bet gap to target |
| `BASE_STAKE` | `10.0` | Per-window base; scaled by sigmoids |
| `MIN_BET` | `1` | Skip bets smaller than $1 |
| `STARTING_BALANCE` | `117.8204` | Bootstrap balance |
| `SIDE_FILTER` | (empty) | Both sides allowed — flipped from `yes_only` 5/29 |
| `RH_MINUTE` | `10` | Reversal hedge at T+10 (matches backtest as of 5/29) |
| `RH_TRIGGER` | `1.0` | $1 wrong-side trigger; scaled from sim's $10 trigger on 10× larger stake |
| `TIME_DECAY` | `true` | × 0.4 / × 0.8 / × 1.2 by minute |
| `MIN_EDGE_CENTS` | `1.0` (default in code) | Skip bets with edge < 1¢ after buffer |
| `FILL_BUFFER_CENTS` | `3` (in `kalshi_trade.py`) | Above ask, absorbs ~300ms slip |
| `CAP_FRACTION_OF_BALANCE` | `0.33` | Per-window dollar cap = balance × 0.33 |
| `MAX_WINDOW_WAGERED` | `30` | Fallback cap |
| `MAX_FILL_PRICE` | `0.97` (code) | Skip entry bets with fill above this |
| `MAX_HEDGE_FILL_PRICE` | `0.80` (code) | Skip RH hedges above this |
| `ACTIVE_HOURS` | (empty) | All 24h |

### Backtest expectation (canonical config, matches live risk controls)
`python3 simulate_dh.py --minutes 4-13 --fair-price-2d --time-decay --reversal-hedge 10 --slippage-cents 4 --min-edge-cents 1 --early-skip 5 0.05 --max-hedge-fill 0.80 --max-legs 2`

- Target mode: **+22.86% ROI** on 7,023 windows, **~1.4 bets/window**
- Additive mode: +19.48% ROI
- Includes the two live-only caps (`--max-hedge-fill 0.80`, `--max-legs 2`) that
  used to be in live but missing from sim. Earlier "+24.79%" without those caps
  overstates expected live performance by ~1.9pp.

### Live performance to date
Paper-mode totals across all live runs:

| Period | Windows | Acted | Wagered | P&L | ROI |
|---|---|---|---|---|---|
| Pre-filter (5/18–5/21) | 174 | 173 | $3,475 | -$142 | -4.1% |
| Post-filter (5/22–5/28) | 81 | 29 | $240 | -$3 | -1.3% |
| Restart-fresh (5/29+) | — | — | — | — | — |

Pre-filter data is **not comparable** to current strategy — the MIN_EDGE_CENTS filter and side filter both landed after that period. Post-filter sample is too small to draw firm conclusions (29 acted windows, ~9pp stdev on total ROI).

### Recent code changes
| Commit | Date | Change |
|---|---|---|
| `ad28461` | 5/22 | MIN_EDGE_CENTS filter added, FILL_BUFFER 5¢ → 3¢ |
| `3fb4404` | 5/22 | Early-skip variant in shadow sim + simulate_dh |
| `af28913` | 5/19 | T+14 sample + hedge-specific fill cap |
| `4b780d0` | 5/18 | Dynamic window cap from balance + wider fill buffer |
| `a7916bc` | 5/18 | RH overlay env-controlled via RH_MINUTE/RH_TRIGGER |

### Session changes (this conversation, 5/29)
- Diagnosed that backtest's +39% ROI was at `slippage=0`; realistic +24% ROI at slippage=4¢
- Identified that 5/18-5/21 live losses were from pre-MIN_EDGE_CENTS code, not strategy failure
- Flipped `SIDE_FILTER=yes_only` → empty (was overfitting to BTC uptrend regime)
- Restarted trader to pick up new config
- Added `monitor_table_health.py` (weekly cron candidate) for table staleness checks
- Rebuilt 2D table; added "0.50%+" magnitude bucket for minutes 4-11 (now have enough samples)
- Turned RH on (RH_MINUTE=10, RH_TRIGGER=1.0) to match backtest; restarted trader
- Added `--max-hedge-fill` and `--max-legs` flags to simulate_dh.py to match
  live's MAX_HEDGE_FILL_PRICE=0.80 and MAX_LEGS_PER_WINDOW=2. New honest
  backtest baseline: +22.86% ROI (down from +24.79% under non-matching sim)

---

## 2. Market maker (BUILT, NOT ACTIVE)

**File:** [live/market_maker.py](live/market_maker.py) (458 lines)
**Commit:** `2176448` (5/22)
**State:** Built paper-mode-first, never deployed.

Posts two-sided quotes inside the book on KXBTC15M with inventory skew. Config in `live/.env` under `MM_*` keys.

User decision (5/29): "becoming a market maker is out of the books" — keeping the code for reference but not pursuing.

---

## 3. Momentum scalper (EXPLORATION, UNCOMMITTED)

**File:** [live/momentum_scalper.py](live/momentum_scalper.py) (346 lines, untracked)
**Log:** [live/momentum_scalper.log](live/momentum_scalper.log) (5/22 only)
**State:** Last run 5/22, not active, untracked in git.

Standalone scalper. Status unclear from logs alone. Tag as abandoned unless we revisit.

---

## 4. Untracked simulators (EXPLORATION)

All untracked in git as of 5/29:
- [simulate_momentum.py](simulate_momentum.py) (210 lines)
- [simulate_momentum_sweep.py](simulate_momentum_sweep.py) (352 lines)
- [simulate_strategy_zoo.py](simulate_strategy_zoo.py) (673 lines, multi-strategy comparison)
- [simulate_yesonly.py](simulate_yesonly.py) (261 lines)
- [live/scan_markets.py](live/scan_markets.py) (164 lines)

None of these have produced a strategy that beat DH. Keep until we decide to prune the repo.

---

## 5. Delayed-hedge "arb" (TESTED & REJECTED — 5/29)

Inspired by a Polymarket bot post. Buy directional at T+entry, wait for opposite side to drop below threshold, hedge to lock in spread. Tested in [simulate_delayed_hedge.py](simulate_delayed_hedge.py) (since deleted).

**Result:** -3.8% to -4.4% ROI at T+4 entry; barely positive (+2.2%) only at T+10 entry with threshold 0.20. Strictly dominated by DH at +24%. Confirms the strategy needs wide-spread illiquid books, which KXBTC15M doesn't have.

---

## Table staleness monitor

**File:** [monitor_table_health.py](monitor_table_health.py) (5/29)

Weekly job that rebuilds the 2D table, diffs vs prior, runs full backtest, and writes `data/logs/TABLE_STALE` if last-7d ROI < 10% over ≥ 50 windows. Trader can check for this marker on startup (not yet wired in).

Manual run: `python3 monitor_table_health.py`. To schedule weekly: see plist snippet in conversation.

---

## Open TODOs / known mismatches

- [ ] **Wire `TABLE_STALE` check into trader startup** — current monitor writes the file but trader doesn't read it.
- [ ] **Prune untracked exploration files** once we've decided what's keeper vs dead end.
- [ ] **Track actual paper P&L vs backtest expectation weekly** — extend monitor or add separate script.
- [ ] **Get enough live samples to validate** — at current stake size, need ~5 days of two-sided paper data before drawing conclusions.

---

## How to use this file

When you change anything material — config tweak, code change, strategy decision — add a one-line entry to the relevant strategy's "Recent changes" or to a new session log block. Keep performance numbers fresh by re-running the canonical backtest and updating the row when you change config.
