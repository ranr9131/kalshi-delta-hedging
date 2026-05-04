# BTC-Kalshi Trading Strategy

## What are these markets?

Kalshi runs binary prediction markets on BTC every 15 minutes, 24/7. Each market asks a single yes/no question: **will BTC be above a fixed price at the end of this 15-minute window?** That fixed price — called the floor strike — is Kalshi's oracle snapshot of BTC at the moment the window opens. The market pays $1 per contract if you're right, nothing if you're wrong. You can buy Yes contracts or No contracts depending on which way you think BTC will move.

Kalshi prices these contracts as probabilities. A Yes contract at $0.70 means the market thinks there's a 70% chance BTC will be above the floor at close. Kalshi takes a 7% fee on gross winnings.

---

## The Core Signal

BTC price movements within a 15-minute window have strong momentum. If BTC has moved up 0.3% from the floor strike four minutes into the window, it tends to still be above the floor when the window closes — not always, but far more often than chance. The key insight is that **the larger the move, the more predictive it is.**

We validated this empirically across 6,334 historical KXBTC15M markets (90 days of data). The win rate — how often betting in the direction of BTC's current move produces a winning outcome — depends heavily on two things:

1. **How far BTC has moved** from the floor strike
2. **How many minutes into the window we are**

A small move early in the window (say, 0.02% at minute 4) barely predicts the outcome — the win rate is barely above 50%. A large move late in the window (say, 0.4% at minute 10) is highly predictive — the win rate approaches 95%. This 2D relationship is captured in an empirical lookup table built from the historical data.

---

## How Bets Are Sized

We don't bet a flat amount. We size each bet using two independent signals multiplied together.

### Signal 1: How confident are we? (win-rate multiplier)

Rather than scaling directly on raw BTC magnitude, we use the 2D empirical win rate as the confidence signal — the same table entry that tells us the fair price also tells us how strong the bet should be.

- Win rate near 50% (tiny move, at the floor) → multiplier ~0× — nothing is bet.
- Win rate at 65% → multiplier ~1.5× (the natural break-even zone).
- Win rate at 80–95% → multiplier climbs toward 3×.

This is more accurate than scaling on magnitude alone because it accounts for both how far BTC has moved *and* how many minutes remain. A 0.15% move at minute 4 and the same move at minute 12 produce the same magnitude but very different win rates — the signal sizes them differently.

### Signal 2: How mispriced is Kalshi? (mispricing multiplier)

Once we know BTC's direction, we compare Kalshi's current price to the 2D fair price. For example, if BTC is up 0.2% at minute 7, historical data says Yes should be worth ~92%. If Kalshi is only pricing Yes at 0.75, we have roughly 17 cents of edge on each dollar bet.

We scale the bet size by how large this mispricing is — again using a smooth S-curve. At zero mispricing (Kalshi is correctly priced), we still bet, but at a neutral multiplier of 1×. The more Kalshi underprices our side, the more we bet — up to a 2× bonus. If Kalshi is actually overpriced against us, the multiplier drops below 1×, shrinking the bet.

### Signal 3: Velocity multiplier (optional)

When enabled via `VEL_SOFT_K`, bet size is also scaled by how aligned BTC's recent momentum is with the direction of the bet. A BTC price moving strongly in the direction of the bet amplifies it (up to 2×); a BTC price moving against it shrinks it (down toward 0×). The steepness of this S-curve is controlled by `VEL_SOFT_K` — higher values are more aggressive.

### Combined

The final bet size is:

> **base stake × win-rate multiplier × mispricing multiplier × velocity multiplier**

At maximum (high-confidence situation, heavily mispriced Kalshi, aligned velocity): 3× × 2× × 2× = **12× the base stake**. In practice most bets fall in the 1–3× range.

---

## The 2D Fair Price Table

The "fair price" — what a Yes contract should theoretically be worth — is not a fixed number. It depends on both how far BTC has moved and how far into the window we are.

We built a lookup table from 6,334 historical markets, bucketed into **6 magnitude ranges**. Unlike the 1-minute table, this one is keyed at **15-second resolution** — T+4:00, T+4:15, T+4:30, etc. — capturing the fact that win rates shift measurably even within a single minute. Sample values from the 0.10–0.20% bucket:

| Time offset | Win rate |
|---|---|
| T+5:00 | 78.2% |
| T+5:15 | 81.0% |
| T+5:30 | 81.6% |
| T+5:45 | 81.3% |
| T+10:00 | 91.4% |

Magnitude buckets at minute 7:

| BTC move from floor | Win rate | Notes |
|---|---|---|
| Under 0.01% | ~51% | At the floor — coin flip, near-zero bet |
| 0.01–0.05% | ~60% | Small but real signal |
| 0.05–0.10% | ~73% | |
| 0.10–0.20% | ~83% | |
| 0.20–0.50% | ~92% | |
| Over 0.50% | ~98% | |

The near-floor bucket (`0.00–0.01%`, moves under ~$8 on $78k BTC) is important: BTC hovering just above or below the floor has no real directional signal. By giving these observations their own bucket, the table correctly shows ~50% win rate there, which causes the win-rate multiplier to shrink the bet to nearly zero — no hard filter needed.

---

## Continuously Updating Position (Delta Hedging)

Rather than placing one bet at minute 5 and waiting, the system re-evaluates at configurable intervals from T+4:00 through T+14:30. At each interval it looks at the current BTC price, recomputes fair value, checks Kalshi's current quote, and decides whether to add to its position.

We use a strategy called delta hedging, borrowed from options trading, where you continuously adjust a position as new information arrives rather than committing to a fixed bet upfront.

**Interval frequency** is set by `BETS_PER_MIN` in `.env`:
- `1` — one check per minute (T+4, T+5, ... T+14), 11 intervals
- `2` — every 30 seconds (T+4:00, T+4:30, ...), 21 intervals
- `4` — every 15 seconds (T+4:00, T+4:15, ...), 41 intervals

**Target mode** (the default) maintains a separate desired total exposure for Yes and No, and only bets the gap. Yes exposure and No exposure are tracked independently — once you've committed $X to Yes, that cap is filled regardless of whether direction subsequently flips to No. This prevents runaway accumulation in oscillating windows where BTC bounces across the floor:

- BTC crosses floor up → first Yes bet placed, Yes cap filled
- BTC crosses floor down → first No bet placed, No cap filled
- Subsequent flips → both caps already at or above target, no further bets

In a window where BTC moves clearly in one direction, the target grows with each interval (later offset + larger magnitude = higher win rate = bigger target), so the loop continuously tops up the position as confidence increases.

**Additive mode** ignores prior exposure and bets the full computed amount at every interval, leading to much higher volume (and variance).

---

## Why Wins and Losses Look Asymmetric

Binary contracts have an inherent asymmetry. If you buy a Yes contract at $0.75 and win, you collect $0.25 profit per dollar wagered (×0.93 after fees). If you lose, you lose the full $0.75. A 75-cent contract must win roughly 77% of the time just to break even.

This means:
- **Winning windows produce modest profits** relative to the amount wagered
- **Losing windows produce larger losses** relative to the amount wagered
- The strategy is profitable because win rates — especially on larger moves at later minutes — consistently exceed the break-even threshold

---

## Hour-of-Day Patterns

Not all hours are equally active. Analysis of the historical data shows that every UTC hour is profitable (worst hours still produce +30% ROI), but some hours significantly outperform:

| UTC Hours | ET Equivalent | Why active |
|---|---|---|
| 13:00–14:00 | 9–10am ET | US market open — highest BTC volatility |
| 18:00 | 2pm ET | US afternoon session |
| 22:00–23:00 | 6–7pm ET | Asian session open |

We support an optional filter to only trade specific hours, configurable without code changes via `ACTIVE_HOURS` in `.env`.

---

## Backtest Results

Tested on 6,334 markets with 15-second Kalshi trade data ($10 base stake):

| Strategy | ROI | Avg bets/window |
|---|---|---|
| Single bet at minute 5 | ~3% | 1 |
| DH, 1-min intervals, magnitude-f | +38.5% | 3.3 |
| DH, 1-min intervals, win-rate-f | +44.2% | 3.6 |
| DH, 15-sec intervals, magnitude-f | +46.2% | 6.8 |
| DH, 15-sec intervals, win-rate-f | +51.5% | 6.7 |
| DH, 15-sec, win-rate-f, skip negative mispricing | +57.0% | 5.5 |
| **DH, 15-sec, win-rate-f, skip neg-mis, velocity filter** | **+60.5%** | **4.8** |

Key improvements and what they add:
- **Win-rate-f vs magnitude-f** (+7.7pp): sizing based on the full 2D win-rate signal avoids over-betting small uncertain moves and under-betting late high-confidence ones.
- **15-second intervals vs 1-minute** (+7.7pp): the 15s table captures within-minute variation in win rates, and more frequent checks mean less lag when BTC makes a decisive move.
- **Skip negative mispricing** (+5.5pp): when Kalshi's ask price exceeds our fair value, the edge is negative — skip rather than bet against ourselves.
- **Near-floor bucket** (+1.9pp): isolating the `0.00–0.01%` magnitude range into its own table cell correctly shows ~50% win rate there, preventing the model from treating a $0.30 floor crossing the same as a $30 move.

---

## Running It

```bash
cd live
pip install -r requirements.txt
python trader.py
```

Set `PAPER_MODE=true` in `live/.env` to run the full loop without placing real orders. All timing, fair price lookups, and P&L math runs exactly as in live mode — only the order submission is skipped.

### `.env` reference

| Variable | Default | Description |
|---|---|---|
| `PAPER_MODE` | `true` | Set `false` to place real orders |
| `BASE_STAKE` | `100.0` | Base dollar amount before multipliers |
| `MODE` | `dh-target` | `t+5` / `dh-target` / `dh-additive` |
| `MIN_BET` | `5.0` | Skip bets smaller than this dollar amount |
| `MAX_FILL` | `0.97` | Skip bets where fill price exceeds this (illiquid contracts near 0/1) |
| `BETS_PER_MIN` | `4` | Intervals per minute: `1`, `2`, or `4` |
| `ACTIVE_HOURS` | *(empty)* | Comma-separated UTC hours to trade, e.g. `13,14,22`. Empty = all 24h |
| `VEL_SOFT_K` | *(empty)* | Velocity multiplier steepness. Empty = disabled. Recommended: `10` |
