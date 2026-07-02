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

### Revisit (6/06): MM on thin coin markets — Step 1 scan done
The 5/29 rejection was KXBTC15M-specific (too tight). Reopened for low-volume
coins, where spreads are wide. New scanner:
[live/scan_mm_crypto.py](live/scan_mm_crypto.py) — discovers all crypto series
(22 found: KX{BTC,ETH,SOL,XRP,HYPE,BNB,DOGE} × {spot,D,15M,Y}), pulls real
quotes per-series (bulk list returns "0.0000" — quotes only populate under a
`series_ticker` filter), and ranks by a fee-aware, volume-weighted MM score.

**Key economic finding — maker fees are negligible vs spread.** Round-trip
maker fee is **0.17–0.87¢** (maker ≈ 25% of taker = `0.07·P·(1−P)`). A maker
quote keeps essentially the full spread. This is the enabler the all-taker
framing missed: a *passive* sniper (post inside, rest) converts the sniper's
taker fee + 3¢ fill-buffer into a maker credit. ⚠️ 25% figure is from secondary
sources (official PDF bot-walled) — verify against real fills before sizing.

**But "low-volume coins" is structurally capped — the wide spreads have no
flow.** Scan snapshot (6/06):
- BTCD (NOT low-vol): 40 candidates, 739k vol24, tight 2–7¢ spreads, ~$5.3k/day
  *optimistic ceiling*. The real pool, but you'd compete with pro MMs.
- XRPD: 8 candidates, ~1.8k vol24, 12.5¢ avg spread, **~$63/day ceiling**.
- DOGE: 21¢ spread, ~$52/day. SOLD: ~$50/day. All tiny.
- Widest-spread books are **one-sided** (XRP T1.0999 bid_sz=1/ask_sz=208; DOGE
  B0.082 bid_sz=254/ask_sz=2) — can't round-trip; you'd fill one leg and hold
  directional/pin risk. The 28¢ "spread" is not capturable.

$/day numbers are CEILINGS (30% flow capture, **zero adverse selection**).

**Step 2 (6/06): passive-fill shadow logger BUILT + collecting.**
[live/mm_shadow_logger.py](live/mm_shadow_logger.py) +
[live/kalshi-mm-shadow.service](live/kalshi-mm-shadow.service). Tracks XRPD/DOGE
near-the-money two-sided markets; simulates a resting maker quote 1¢ inside with
**sticky quotes + REQUOTE_SEC reaction latency** (so it does NOT cheat by
cancelling stale quotes — fills against a stale price are the adverse-selection
events). Detects fills off the public `trade` channel
(`taker_side`+`yes_price_dollars`), logs a continuous book+fair snapshot stream,
and `--report` computes markout / adverse selection at 15/60/300s horizons.
- **Verdict rule:** passive MM survives only if **markout stays positive**. If
  immediate edge is positive but mk60 ≈ 0 or negative → spread is an illusion,
  you're being picked off.
- **Caveat:** fair_price_model_v2 is calibrated for 15M, not multi-day. Far-dated
  XRPD ladders (mins_left ~8780 = 6 days) have unreliable *level* (immediate
  edge), but markout (Δfair) is a difference and stays usable for adverse-sel.
- Fills are RARE on thin books (0 trades in 22s even on top BTCD) but BURST near
  expiry. Needs to run hours→days. Check with `python3 mm_shadow_logger.py --report`.
- Bug fixes 6/06: (a) a resting quote now fills ONCE then is consumed until
  re-quote (was re-filling on every print in a taker sweep); (b) dedup fills by
  `trade_id` (sweeps print the same id twice). Both inflated fill counts.

**Daily calibration (6/06): [live/calibrate_daily.py] + calibration_daily.json.**
Refits Platt PER HORIZON BUCKET (0-12/12-25/25-50/50-100/100+min) from settled
KX{coin}D outcomes vs historical Coinbase klines. Groups strikes by close_time
(fetch klines once per ladder, ~20× fewer requests). Wired into the model as
`fair_price_model_v2.fair_p_yes_daily()`; shadow logger now uses it.
- **Finding:** over 24,005 settled XRPD markets / 216k samples (14d), the RAW
  model is already well-calibrated (±0.08 per bin). My earlier live "65¢ vs 82¢"
  gap was NOT a calibration offset — likely a live σ-estimation artifact.
- Must fit on the TRADEABLE BAND (0.03<raw<0.97); else 213k trivial extreme
  strikes drown the fit. Band-restricted Platt: small Brier gains every bucket,
  but the 0-12m bucket has b=1.52 → meaningfully sharpens near-expiry ITM
  (+9c at 8min on the example strike), exactly where fills occur.
- Refit fast from dumped samples: `--from-csv cal_daily_xrp_samples.csv`.
  Extend to other coins: `--series KXSOLD,KXDOGED,...`.
**EC2 deploy DONE (6/06):** running 24/7 as `kalshi-mm-shadow.service` (enabled)
on 35.153.141.9. Daily-cal logic refactored into STANDALONE
[live/daily_calibration.py] (imports fair_price_model_v2 read-only) so the
production model file was NOT modified — live sniper unaffected. Local collector
stopped (server is authoritative). Check results:
`ssh -i ~/Desktop/kalshi-key-v2.pem ec2-user@35.153.141.9` then
`cd ~/kalshi-delta-hedging/live && python3.11 mm_shadow_logger.py --report`.
Status: collecting 24/7 on server.

**RESULT (6/07, 158 fills / 135 settled, ~1.5d) — NEGATIVE, as built:**
- Model markout stays POSITIVE (imm +6.0¢ → mk300 +5.3¢, adverse sel +0.7¢) →
  NOT being adversely selected / out-sped.
- REAL settlement P&L: **−627¢ / 135 fills = −4.6¢/fill = −$62.70** @10 contracts.
- **Divergence is the finding:** model *thinks* it has +5-6¢ edge, but the
  positions LOSE at expiry. Markout can't catch this — it measures movement in
  the model's OWN units, so it's blind to model bias. Only settlement caught it.
- Diagnosis: **fair-value model is wrong in LIVE conditions** (looked calibrated
  historically, mis-fires live — likely live σ-estimation artifact). Bottleneck
  is MODEL ACCURACY, not speed/adverse-selection. ~1.8σ from zero (lean, not yet
  ironclad); logger still running to firm it up.
- settle_pnl.py on server computes real P&L from fills + Kalshi settlement results.

User decision (6/07): (1) investigate the live model gap (σ estimation); (2) AUDIT
THE LIVE SNIPER for the same historical-vs-live overconfidence.

### MODEL-GAP / SNIPER AUDIT (6/07) — root cause: SELECTION EFFECT, not bad model
(Both snipers are PAPER_MODE=true; multi-account live sniper NOT running → no real
money at risk. NN traders separate.)

Analysis files (local /tmp, reproducible): audit_sniper2.py joins snipes.csv ⋈
settlements.csv; uncond_cal.py (on server) computes unconditional 15M calibration
from window_log.csv (strike≈btc_t0 open price, fair_p@t5).

- **Unconditional 15M calibration is GOOD**: Brier 0.205 < 0.250 (coinflip), errors
  ±0.05. The model is genuinely predictive on ALL windows.
- **Snipe-conditional calibration is GARBAGE**: realized hugs ~50% across every
  fair_p bin (model 17%→real 53%; model 83%→real 48%; errors ±0.35). Predicted
  edge +10.9¢/contract → realized **+1.8¢ gross** (≈ breakeven-to-losing after
  ~1.0-1.75¢ fee). Win rate 49.9%. HYPE −3.8¢. Biggest predicted edges (15¢+)
  had the WORST realized (+0.4¢) — more confidence = more error.
- **Mechanism:** the sniper fires where model disagrees most with market; on
  liquid crypto that's where the MODEL is wrong, not the market. Trading the
  model's "edge" = trading the model's own error. Explains the MM −4.6¢/fill
  (fill selection / adverse selection) AND the sniper ~0 edge (signal selection)
  AND the backtest-vs-live gap: backtests evaluate the model UNCONDITIONALLY
  (calibrated); live P&L is conditional on ACTING, which selects for model error.

**Implication: the fair-value-vs-market edge is largely illusory on liquid Kalshi
crypto. +22% DH backtest overstates live edge — do NOT size real money off it.**
Caveats: ~5d data; paper fills may be optimistic (reality likely worse); specific
to liquid crypto 15M/daily.

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

## 6. Touch markets: copy-trade sharky6999 (ACTIVE RESEARCH — best current candidate)

**Files:** [live/touch_shadow_logger.py](live/touch_shadow_logger.py) (collector, running),
[live/touch_backtest.py](live/touch_backtest.py), [live/touch_copytrade_sim.py](live/touch_copytrade_sim.py),
[live/copy_monitor.py](live/copy_monitor.py) (**running**, paper).
**State (6/10):** model-based replication REJECTED (settlement backtest −35¢/trade, 36% win vs 72% breakeven —
correlated June 1-7 dip blowups). COPYTRADING his actual fills is +EV: +$15.4k/6.2d at 0 slip, +$7.9k at 1¢,
breakeven ~2¢; measured real copy-slippage = **+1.0¢ if ≤10 min late** (own-snapshot join), Poly maker fee = 0.
copy_monitor.py logs his fills, signal lag, and copyable depth at +1¢. Decision gate: 2-3 days of monitor data
→ paper copy → live at 10-25% of his size (~$5-10k). Full evidence chain in memory + Research Log 6/10.

---

## 7. LIP farmer (LIVE since 6/10 16:12 ET — $180 cap)

**Files:** [live/lip_farmer.py](live/lip_farmer.py) (daemon), [live/lip_scoring.py](live/lip_scoring.py) (exact replica of Kalshi's snapshot scoring + quote optimizer, 9 unit tests in [live/test_lip_scoring.py](live/test_lip_scoring.py)), [live/lip_allocator.py](live/lip_allocator.py), [live/lip_quoter.py](live/lip_quoter.py), [live/lip_api.py](live/lip_api.py), [live/lip_report.py](live/lip_report.py), [live/kalshi-lip.service](live/kalshi-lip.service)

### What it does
Farms Kalshi's Liquidity Incentive Program: rests two-sided depth in program markets (2,295 active, ~$36k/day total pool, sunset 2026-09-01). Revenue needs **zero fills** — per-second snapshots score resting bids by `0.5^(¢ behind best bid) × size` within the target-size qualifying window, normalized per side; payout = time-averaged share × period reward. Snapshot pays nobody unless BOTH sides hold ≥ target_size — so completing dead books 24/7 is the edge. Allocator ranks all programs by est $/day under caps; optimizer picks prices maximizing share subject to a hard per-market loss bound (default $30 → caps bids ~3¢ at size 1000) plus mid-anchored caps, join-only (never improves the touch), post-only, self-cross/99¢ guards. Fills → pull quotes, 30-min cooldown, hold position (cheap fills = bounded lottery tickets), 20-fills/day circuit breaker, `lip_kill` file kill switch.

### State
**Paper daemon running locally** since 6/10 (`lip_farmer.out`, `lip_report.py` for dashboard). First cycles: 8-9 markets, est ~$90-100/day on ~$500 capital, $240-270 bounded worst-case (mostly 1-day $80 KXNHLPRICE/KXTRUEV programs with near-empty books — a competitor 1¢-farms them but sits 11 ticks behind reference ≈ zero share; we bid 3¢ and take 5-10× their score). Estimates assume scanned books persist — paper accrual (`lip_accrual.csv`) over 2-3 days is the honest number. Config all env-overridable (`LIP_*`, see [live/lip_config.py](live/lip_config.py)); defaults: 40 mkts / $2k capital / $500 total worst-loss. Decision gate: paper accrual ≥ ~$30/day stable → EC2 deploy → `LIP_PAPER=0` at $500 cap → reconcile real payouts vs accrual.

---

## 7. Sports latency taker (TESTED 6/10 → KILLED 6/11 — model error, not market lag)

**VERDICT (settled, 41 paper takes, 1 day):** −2,270¢ total, 3/41 wins. The "edges" were OUR
Gaussian WP model mispricing variance, not stale quotes: MLB books repriced big events ≤10s
(our sampling floor); the NBA "gap" persisted an hour with $100-200k at the touch (= the market
pricing comeback variance correctly — Knicks then came back from 24 down and won, settling all
31 Spurs takes at −$21). Kalshi posted ZERO two-sided in-game quotes all day (incl. NBA Finals)
→ kills xmarket_arb v1's premise too. The RN1-tier version needs websocket books, 1s tick data
and a calibrated WP model just to MEASURE the window. Cost of test: $0. Files: latency_taker.py,
latency_takes.csv, latency_quotes.csv (5k+ snapshots), runs 1-2 logs.

**File:** [live/latency_taker.py](live/latency_taker.py) (**running**, PID in latency_taker.out).
RN1-pattern strategy: league CDN feeds (NBA liveData ~1-3s, MLB statsapi GUMBO) beat stream-watchers
(15-60s behind). On every score change it snapshots Kalshi + Polymarket books; if a touch deviates from
the prior-anchored WP model by >5¢+fee it logs a simulated take (with real touch size), tracks
time-to-reprice for 180s, and settles at the final score. Outputs latency_takes.csv / latency_quotes.csv;
`--report` summarizes. Quirks: cdn.nba.com needs browser UA+Referer (separate session — Kalshi 403s those
headers); gamma game-market outcomes are [away, home]; NBA pbp end = actionType game/subType end.
First session 6/10: 15 MLB games (from 1:10pm ET) + NBA Finals G4 SAS@NYK (8:30pm ET).
Decision gate: ≥0 takes with positive settled P&L and reprice_s ≫ our reaction time → consider live.

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
