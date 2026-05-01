# BTC-Kalshi Live Trader

Automated trading system for **KXBTC15M** binary prediction markets on **Kalshi**. Uses real-time BTC/USD price from Coinbase to infer whether the current 15-minute window will resolve Yes or No, then sizes and places orders using a combined sigmoid stake function.

---

## Overview

**KXBTC15M** markets settle every 15 minutes at UTC :00, :15, :30, and :45. Each market resolves **Yes** if BTC is above the floor strike at close, **No** if below. The floor strike is Kalshi's oracle reference price recorded at market open.

The trader runs a continuous loop: wait for the right entry point within each window, fetch the current BTC price and Kalshi quote, decide direction and stake size, place the bet, wait for settlement, compute P&L, and log results.

---

## Trading Modes

Set `MODE` in `.env` to switch strategy:

| Mode | Entry | Description |
|------|-------|-------------|
| `t+5` | T+5 | Single bet per window at minute 5. Waits for T+15 settlement. |
| `dh-target` | T+4 | Delta hedge loop T+4..T+13. Only bets the gap between current exposure and computed target. Capital-efficient — self-limits to avoid doubling down. **(Default, recommended)** |
| `dh-additive` | T+4 | Delta hedge loop T+4..T+13. Bets the full computed stake at every interval regardless of prior exposure. Higher volume, higher variance. |

**dh-target vs dh-additive**: Target mode averages ~3.3 bets/window; additive averages ~9.0. Target is more capital-efficient and achieves similar or better ROI with less wagered per window. Additive can place both Yes and No bets in the same window if BTC reverses direction mid-loop, which target mode dampens by only closing gaps.

---

## Files

```
live/
├── trader.py         Main orchestrator — entry point
├── strategy.py       Sigmoid stake sizing functions
├── kalshi_trade.py   Kalshi REST API: market data, order placement, settlement polling
├── kalshi_auth.py    RSA-SHA256-PSS authentication for Kalshi
├── btc_feed.py       Real-time BTC/USD via Coinbase Exchange WebSocket
├── .env              Runtime configuration (mode, keys, stake, flags)
├── requirements.txt  Python dependencies
├── trade_log.csv     Per-bet log (one row per individual order placed)
└── window_log.csv    Per-window summary (one row per 15-min window)
```

---

## Configuration (.env)

```ini
KALSHI_API_KEY_ID=<your-api-key-id>
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"

PAPER_MODE=true        # true = no real orders placed (safe for testing)
MODE=dh-target         # t+5 | dh-target | dh-additive
BASE_STAKE=100.0       # base dollar amount per bet (scaled by sigmoid multipliers)
MIN_BET=5.0            # skip bets below this dollar threshold (avoids fee drag)
ACTIVE_HOURS=          # comma-separated UTC hours to trade, e.g. 13,14,18,22 — empty = all 24h
```

**PAPER_MODE=true** runs the entire loop — timing, market polling, P&L computation, logging — but skips actual order submission. Use this to verify timing and logic before going live.

---

## Startup

```bash
cd live
pip install -r requirements.txt
python trader.py
```

The trader:
1. Reads `.env` for all config
2. Loads the 2D fair price table from `data/logs/minute_analysis_2d.csv`
3. Starts the Coinbase WebSocket feed (`btc_feed.start()`) and waits up to 30s for first price
4. Fetches Kalshi balance (live mode only)
5. Enters the main loop

To stop cleanly, press **Ctrl+C** — the shutdown flag causes the loop to exit after the current window finishes rather than mid-trade.

---

## Main Loop Logic

```
while not shutdown:
    elapsed = seconds since last UTC :00/:15/:30/:45 boundary
    if elapsed < DECISION_OFFSET_SECS:
        enter current window immediately (elapsed still before T+entry)
    else:
        wait for next boundary
    run_window()
```

`DECISION_OFFSET_SECS` = 240s for DH modes (T+4), 300s for `t+5` mode. After settlement, if `trader.py` is still within the T+4 window (e.g., only 24s past the boundary), it enters the *current* window rather than skipping ahead to the next one, preventing missed windows after slightly-delayed settlements.

---

## run_window() Flow

### Shared startup (all modes)
1. Compute `window_ts` = current UTC :00/:15/:30/:45 boundary
2. Sleep until `DECISION_OFFSET_SECS` from `window_ts` (T+4 or T+5)
3. Fetch open KXBTC15M market from Kalshi REST API
4. Read `floor_strike` from market → this is `btc_t0` (the oracle BTC reference)
5. Read current BTC price from Coinbase WebSocket → `btc_entry`

### t+5 mode
6. Determine side: Yes if `btc_entry > btc_t0`, No otherwise
7. Compute stake via `strategy.compute_stake()` (see Stake Sizing below)
8. Place one limit order at the ask (crosses spread for immediate fill)
9. Wait for `close_time`, poll settlement every 10s
10. Compute P&L, log one row to `trade_log.csv` and `window_log.csv`

### dh-target / dh-additive mode
6. Call `run_dh_loop(window_ts, btc_t0, ticker, close_time)` — returns all bets and exposures
7. After loop: fetch `btc_t10` for logging
8. Wait for `close_time`, poll settlement
9. Compute summed P&L across all yes/no bets
10. Log one row to `window_log.csv`; each individual bet was already logged to `trade_log.csv` inside `run_dh_loop()`

---

## Delta Hedge Loop (run_dh_loop)

Runs for each minute in `DH_MINUTES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` (T+4 through T+13).

For each minute:

1. **Sleep until `window_ts + timedelta(minutes=minute)`** — 1s chunks so Ctrl+C is responsive
2. Fetch fresh BTC price from Coinbase (`btc_now`)
3. Fetch fresh market quote from Kalshi (`get_open_market()`) — refreshed every minute for up-to-date bid/ask
4. Compute `f_btc = sigmoid_btc(abs_pct_move)` — magnitude multiplier
5. Look up `fair = get_fair_price_2d(minute, abs_pct_move)` — 2D empirical win rate for this (minute, magnitude) cell
6. Compute mispricing and `g_misprice = sigmoid_mispricing(mispricing)` — mispricing multiplier
7. Compute target Yes and No stakes: `BASE_STAKE * f_btc * g_misprice`
8. **dh-target**: bet only `max(0, target - current_exposure)` per side
9. **dh-additive**: bet full target each interval
10. Fill Yes at `yes_ask`; fill No at `1.0 - yes_bid` (crossing the respective ask)
11. Log each bet to `trade_log.csv` immediately

The dead zone is **implicit** in the 2D table: cells where BTC moved very little have empirically lower win rates (~56-63%), so `mispricing` naturally shrinks or goes negative, suppressing stakes without a hard cutoff. No explicit DEAD_ZONE_PCT check is needed.

---

## Stake Sizing

**Combined sigmoid formula**: `stake = BASE_STAKE × f(BTC_magnitude) × g(Kalshi_mispricing)`

### f(BTC_magnitude) — sigmoid_btc()
Scales stake up as BTC moves further from the floor strike:
- Center: 0.10% move (inflection point)
- Steepness k=20.0
- Range: 0 → 3.0× (max 3× base stake for very large moves)

Small moves near the floor strike produce near-zero multiplier, keeping stakes small when signal is weak.

### g(Kalshi_mispricing) — sigmoid_mispricing()
Scales stake based on how wrong Kalshi's price is relative to fair value:
- `mispricing = fair_price - kalshi_yes_price` (Yes side) or `kalshi_yes_price - (1 - fair_price)` (No side)
- Steepness k=8.0
- Range: 0 → 2.0×, equals 1.0 at zero mispricing
- Positive mispricing = Kalshi is underpriced for our direction = bet more

### get_fair_price_2d() — 2D empirical win rate table
In DH modes, fair price is looked up from a 2D table indexed by **(minute, BTC magnitude bucket)**, loaded from `data/logs/minute_analysis_2d.csv` (generated by `analyze_minutes_2d.py` on 90 days / 6,360 markets).

**Magnitude buckets**: `0.00-0.05%`, `0.05-0.10%`, `0.10-0.20%`, `0.20-0.50%`, `0.50%+`

Sample values (win rate per cell):

| Minute | <0.05% | 0.05-0.10% | 0.10-0.20% | 0.20-0.50% | 0.50%+ |
|--------|--------|-----------|-----------|-----------|--------|
| T+4  | 0.569 | 0.662 | 0.750 | 0.855 | 0.947 |
| T+7  | 0.607 | 0.727 | 0.832 | 0.919 | 0.977 |
| T+10 | 0.625 | 0.796 | 0.908 | 0.959 | 0.990 |
| T+13 | 0.692 | 0.883 | 0.935 | 0.944 | — |

For cells with fewer than 30 samples, falls back to a 1D per-minute table. This 2D approach replaces the former flat dead zone filter — tiny-move cells have lower fair prices, which naturally reduces mispricing and stake without a hard cutoff.

Backtest result (90d, dh-target, no explicit dead zone): **+39.06% ROI**, 3.3 bets/window.

In `t+5` mode, the fixed `FAIR_PRICE = 0.698` from `strategy.py` is used instead.

---

## P&L Formula

Kalshi charges a **7% fee on gross winnings** (FEE_RATE = 0.07).

```
count = stake / fill_price          # fractional contracts

Win:  pnl = count × (1 - fill_price) × 0.93
Loss: pnl = -(count × fill_price) = -stake
```

Applied independently to each Yes and No bet, summed for the window total.

---

## Order Placement

**Limit orders at the ask** (immediate fill semantics):
- Yes buy: `yes_price = yes_ask_dollars` (crossing the Yes ask)
- No buy: `yes_price = yes_bid_dollars`; effective No price = `1.0 - yes_bid` (crossing the No ask)

Orders use `fractional_trading_enabled=true` — count is rounded to 2 decimal places, minimum 0.01 contracts. Kalshi's API uses `yes_price` in cents for both Yes and No orders.

Order placement retries once on failure (2 attempts total, 2s between attempts). Paper mode bypasses placement entirely.

---

## Authentication

Kalshi's REST API uses **RSA-SHA256-PSS** signatures. Per-request headers:
- `KALSHI-ACCESS-KEY` — API key ID
- `KALSHI-ACCESS-TIMESTAMP` — Unix milliseconds
- `KALSHI-ACCESS-SIGNATURE` — base64 of RSA-PSS signature over `(timestamp_ms + METHOD + path)`

Salt length = SHA256 digest size (32 bytes). `kalshi_auth.py` loads the PEM key from `.env` at startup.

Market data endpoints (`/markets`, `/markets/{ticker}`) require **no authentication**. Only order placement and balance queries use signed headers.

---

## BTC Price Feed

Real-time **BTC-USD** from **Coinbase Exchange WebSocket** (`wss://ws-feed.exchange.coinbase.com`). Subscribes to the `ticker` channel for `BTC-USD`. Runs in a daemon thread; auto-reconnects every 5s on disconnect.

Staleness guard: `MAX_PRICE_AGE_SECS = 10`. If the cached price is older than 10 seconds, `get_btc_with_retry()` waits up to 3 attempts × 5s before giving up and skipping the interval. This prevents acting on stale data during WebSocket disruptions.

---

## Log Files

### trade_log.csv — one row per individual bet

| Field | Description |
|-------|-------------|
| `window_ts` | UTC ISO timestamp of window T+0 boundary |
| `mode` | Trading mode (t+5, dh-target, dh-additive) |
| `ticker` | Kalshi market ticker (e.g. KXBTC15M-26APR302130-30) |
| `close_time` | ISO timestamp when market closes |
| `dh_minute` | Which minute in window this bet was placed (4-13) |
| `btc_t0` | Floor strike (BTC oracle price at window open) |
| `btc_now` | Live BTC price at this minute |
| `btc_price_age_secs` | Age of BTC quote in seconds |
| `abs_pct_move` | Cumulative % BTC move from btc_t0 to btc_now |
| `yes_bid` / `yes_ask` | Kalshi order book (fraction 0-1) |
| `spread` | yes_ask - yes_bid |
| `kalshi_yes_mid` | (yes_bid + yes_ask) / 2 |
| `direction` | "yes" (BTC up) or "no" (BTC down) |
| `yes_target` / `no_target` | Computed target stakes |
| `yes_exposure_before` / `no_exposure_before` | Existing exposure before this bet |
| `bet_side` | Which side was actually bet |
| `mispricing` | fair - kalshi_mid (positive = good for us) |
| `f_btc` / `g_misprice` | Sigmoid multipliers |
| `stake` | Dollar amount of this bet |
| `fill_price` | Effective fill (ask price as fraction) |
| `count` | Contracts purchased |
| `order_id` | Kalshi order ID or "paper" |
| `order_result` | "ok", "paper", or error message |

### window_log.csv — one row per 15-minute window

| Field | Description |
|-------|-------------|
| `window_ts` | UTC ISO timestamp of window T+0 boundary |
| `mode` | Trading mode |
| `ticker` / `close_time` | Market identifiers |
| `btc_t0` | Floor strike |
| `btc_t5` | BTC price at entry (T+4 for DH, T+5 for t+5) |
| `btc_t10` | BTC price at T+10 (end of DH loop) |
| `n_yes_bets` / `n_no_bets` / `total_bets` | Bet counts |
| `total_yes_stake` / `total_no_stake` / `total_wagered` | Dollar totals |
| `settlement_ts` | When settlement was confirmed |
| `market_winner` | "yes" or "no" |
| `yes_pnl` / `no_pnl` / `total_pnl` | Net P&L breakdown |
| `outcome` | "net_win", "net_loss", "win", "loss", or "unknown" |
| `cumulative_pnl` | Running session total across all windows |

---

## Backtest Results (reference)

From `simulate_dh.py` on 30-day data (2,819 markets), `--minutes 4-13 --dynamic-fair-price --dead-zone 0.05`:

| Configuration | Mode | ROI | Avg bets/window |
|---|---|---|---|
| 1D dynamic + dead zone 0.05% (30d) | dh-target | +35.69% | 1.3 |
| **2D table, no dead zone (90d)** | **dh-target** | **+39.06%** | 3.3 |
| 2D table, no dead zone (90d) | dh-additive | +19.67% | 9.0 |

*Backtests use $100 base stake. Live trading uses the same stake and parameters. Past results do not guarantee future performance.*

---

## Console Output (example window)

```
2026-05-01T14:00:01Z [INFO] Window T+0: 14:00:00 UTC | T+4 in 239s
2026-05-01T14:04:01Z [INFO] cutoff=$94,210.00 | BTC T+4=$94,451.00 (age=0.3s) | Kalshi bid/ask=0.620/0.640 spread=0.020 | ticker=KXBTC15M-...
2026-05-01T14:04:01Z [INFO] DH T+4: YES | cutoff=$94,210 now=$94,451 (+0.2560%) | f=2.891 g=1.284 mis=+0.110 | target_yes=$371 target_no=$0 | gap_yes=$371 gap_no=$0
2026-05-01T14:05:01Z [INFO] DH T+5: YES | cutoff=$94,210 now=$94,389 (+0.1900%) | f=2.384 g=1.251 | target_yes=$298 gap=$0 → no bet (target met)
2026-05-01T14:14:01Z [INFO] Waiting 56s for market to close at 14:15:00 UTC...
2026-05-01T14:15:12Z [INFO] RESULT: NET_WIN | market=yes | yes_pnl=+$151.27 no_pnl=$0.00 | total=+$151.27 | session=+$151.27 | bets=1Y+0N wagered=$371.00
```

---

## Error Handling

- **BTC price unavailable**: retries 3× with 5s delay; skips interval on exhaustion
- **Kalshi market fetch fails**: logs warning, skips interval
- **Order placement fails**: retries once; logs error, continues loop (bet counted as skipped)
- **Settlement timeout**: gives up after 120s, logs "unknown" outcome
- **Unhandled exception in run_window()**: caught at main loop level, logged with traceback, trader continues to next window

---

## Dependencies

```
requests>=2.31.0         Kalshi REST API
websocket-client>=1.7.0  Coinbase BTC feed
cryptography>=42.0.0     RSA-PSS signing
python-dotenv>=1.0.0     .env loading
numpy>=1.24.0            (used by strategy if needed)
```

Install: `pip install -r requirements.txt`

---

## Notes

- **Always start with PAPER_MODE=true** and verify timing/logic before setting it to false
- Kalshi fractional trading is enabled — minimum contract count is 0.01
- The floor strike (`floor_strike`) is Kalshi's own oracle price, not the live BTC price at open. They are close but not identical.
- `window_log.csv` field `btc_t5` records the BTC price at the actual entry minute (T+4 for DH modes), not necessarily T+5
- `trade_log.csv` and `window_log.csv` are append-only — each session adds rows to the same files, accumulating history across restarts
- To reset session P&L tracking, restart `trader.py` (the in-memory `_cumulative_pnl` resets to 0; CSV files retain history)
