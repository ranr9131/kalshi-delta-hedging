# BTC-Kalshi Trading Strategy

## What are these markets?

Kalshi runs binary prediction markets on BTC every 15 minutes, 24/7. Each market asks a single yes/no question: **will BTC be above a fixed price at the end of this 15-minute window?** That fixed price — called the floor strike — is Kalshi's oracle snapshot of BTC at the moment the window opens. The market pays $1 per contract if you're right, nothing if you're wrong. You can buy Yes contracts or No contracts depending on which way you think BTC will move.

Kalshi prices these contracts as probabilities. A Yes contract at $0.70 means the market thinks there's a 70% chance BTC will be above the floor at close. Kalshi takes a 7% fee on gross winnings.

---

## The Core Signal

BTC price movements within a 15-minute window have strong momentum. If BTC has moved up 0.3% from the floor strike four minutes into the window, it tends to still be above the floor when the window closes — not always, but far more often than chance. The key insight is that **the larger the move, the more predictive it is.**

We validated this empirically across 6,370 historical KXBTC15M markets (90 days of data). The win rate — how often betting in the direction of BTC's current move produces a winning outcome — depends heavily on two things:

1. **How far BTC has moved** from the floor strike
2. **How many minutes into the window we are**

A small move early in the window (say, 0.02% at minute 4) barely predicts the outcome — the win rate is barely above 50%. A large move late in the window (say, 0.4% at minute 10) is highly predictive — the win rate approaches 95%. This 2D relationship is captured in an empirical lookup table built from the historical data.

---

## How Bets Are Sized

We don't bet a flat amount. We size each bet using two independent signals multiplied together.

### Signal 1: How confident are we? (win-rate multiplier)

Rather than scaling directly on raw BTC magnitude, we use the 2D empirical win rate as the confidence signal — the same table entry that tells us the fair price also tells us how strong the bet should be.

- Win rate near 57% (tiny move, early minute) → multiplier ~0.5× — almost nothing is bet.
- Win rate at 65% → multiplier ~1.5× (the natural break-even zone).
- Win rate at 80-95% → multiplier climbs toward 3×.

This is more accurate than scaling on magnitude alone because it accounts for both how far BTC has moved *and* how many minutes remain. A 0.15% move at minute 4 and the same move at minute 12 produce the same magnitude but very different win rates — the signal sizes them differently.

### Signal 2: How mispriced is Kalshi? (mispricing multiplier)

Once we know BTC's direction, we compare Kalshi's current price to the 2D fair price. For example, if BTC is up 0.2% at minute 7, historical data says Yes should be worth ~92%. If Kalshi is only pricing Yes at 0.75, we have roughly 17 cents of edge on each dollar bet.

We scale the bet size by how large this mispricing is — again using a smooth S-curve. At zero mispricing (Kalshi is correctly priced), we still bet, but at a neutral multiplier of 1×. The more Kalshi underprices our side, the more we bet — up to a 2× bonus. If Kalshi is actually overpriced against us, the multiplier drops below 1×, shrinking the bet.

### Combined

The final bet size is:

> **base stake × win-rate multiplier × mispricing multiplier**

At maximum (high-confidence situation, heavily mispriced Kalshi): 3× × 2× = **6× the base stake**. In practice most bets fall in the 1-3× range.

---

## The 2D Fair Price Table

The "fair price" — what a Yes contract should theoretically be worth — is not a fixed number. It depends on both how far BTC has moved and how far into the window we are.

We built a lookup table from 6,370 historical markets, bucketed into 5 magnitude ranges. Unlike the 1-minute table, this one is keyed at **15-second resolution** — T+4:00, T+4:15, T+4:30, etc. — capturing the fact that win rates shift measurably even within a single minute. Sample values from the 0.10–0.20% bucket:

| Time offset | Win rate |
|---|---|
| T+5:00 | 78.2% |
| T+5:15 | 81.0% |
| T+5:30 | 81.6% |
| T+5:45 | 81.3% |
| T+10:00 | 91.4% |

Magnitude buckets at minute 7:

| BTC move from floor | Win rate |
|---|---|
| Under 0.05% | ~61% |
| 0.05–0.10% | ~73% |
| 0.10–0.20% | ~83% |
| 0.20–0.50% | ~92% |
| Over 0.50% | ~98% |

The 2D table replaces an old "dead zone" hard filter that blocked all bets when BTC was too close to the floor. With the 2D table we don't need it — tiny-move cells naturally produce fair prices close to Kalshi's price, so the computed mispricing is small and the bet shrinks to nothing automatically.

---

## Continuously Updating Position (Delta Hedging)

Rather than placing one bet at minute 5 and waiting, the system re-evaluates at configurable intervals from T+4:00 through T+13:45. At each interval it looks at the current BTC price, recomputes fair value, checks Kalshi's current quote, and decides whether to add to its position.

We use a strategy called delta hedging, borrowed from options trading, where you continuously adjust a position as new information arrives rather than committing to a fixed bet upfront. The name is a loose analogy — we're not hedging in the traditional risk-reduction sense, we're continuously updating a directional bet as BTC's price evolves.

**Interval frequency** is set by `BETS_PER_MIN` in `.env`:
- `1` — one check per minute (T+4, T+5, ... T+13), 10 intervals
- `2` — every 30 seconds (T+4:00, T+4:30, ...), 20 intervals
- `4` — every 15 seconds (T+4:00, T+4:15, ...), 40 intervals (default)

**Target mode** (the default) maintains a desired total exposure for each side and only bets the gap. If at T+4:00 we compute a target of $200 on Yes and place $200, then at T+4:15 the target recomputes to $180 (BTC moved a little less), we place nothing — we're already above target. If at T+4:30 it rises to $250, we place the $50 gap. This means:

- We never pile on beyond what the signal justifies
- If BTC drifts back toward the floor, the target shrinks and we stop adding
- In a window where BTC moves clearly in one direction, we might place 1-3 bets total; in a choppy window we place more as the target oscillates

**Additive mode** ignores prior exposure and bets the full computed amount at every interval, leading to much higher volume (and variance).

---

## Why Wins and Losses Look Asymmetric

Binary contracts have an inherent asymmetry. If you buy a Yes contract at $0.75 and win, you collect $0.25 profit per dollar wagered (×0.93 after fees). If you lose, you lose the full $0.75. A 75-cent contract must win roughly 77% of the time just to break even.

This means:
- **Winning windows produce modest profits** relative to the amount wagered
- **Losing windows produce larger losses** relative to the amount wagered
- The strategy is profitable because win rates — especially on larger moves at later minutes — consistently exceed the break-even threshold

A window where BTC moved clearly but you had $300 in No exposure that lost looks like a big loss. But the same window with Yes exposure would have returned $75-100 in profit. The losses are structurally larger in dollar terms but happen less often. Over enough windows, the expectation is strongly positive — the backtest shows +39% ROI over 6,360 windows.

---

## Hour-of-Day Patterns

Not all hours are equally active. Analysis of the historical data shows that every UTC hour is profitable (worst hours still produce +30% ROI), but some hours significantly outperform:

| UTC Hours | ET Equivalent | Why active |
|---|---|---|
| 13:00–14:00 | 9–10am ET | US market open — highest BTC volatility |
| 18:00 | 2pm ET | US afternoon session |
| 22:00–23:00 | 6–7pm ET | Asian session open |

We support an optional filter to only trade specific hours, configurable without code changes. Currently set to trade all 24 hours.

---

## Backtest Results

Tested on 6,334 markets with 15-second Kalshi trade data ($10 base stake):

| Strategy | ROI | Avg bets per window |
|---|---|---|
| Single bet at minute 5 | ~3% | 1 |
| Delta hedge, 1-min intervals, magnitude-f | +49.3% | 3.3 |
| Delta hedge, 1-min intervals, win-rate-f | +58.0% | 3.7 |
| Delta hedge, 15-sec intervals, magnitude-f | +60.4% | 6.7 |
| **Delta hedge, 15-sec intervals, win-rate-f** | **+66.3%** | 6.7 |

The two improvements compound independently:
- **Win-rate-f vs magnitude-f** (+8.7pp): sizing based on the full 2D win-rate signal rather than raw magnitude avoids over-betting small uncertain moves and under-betting late high-confidence ones.
- **15-second intervals vs 1-minute** (+11.1pp): the 15s table captures within-minute variation in win rates, and more frequent checks mean less lag when BTC makes a decisive move near the close.

---

## Running It

```bash
cd live
pip install -r requirements.txt
python trader.py
```

Set `PAPER_MODE=true` in `live/.env` to run the full loop without placing real orders. All timing, fair price lookups, and P&L math runs exactly as in live mode — only the order submission is skipped.
