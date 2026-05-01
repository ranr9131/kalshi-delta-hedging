# BTC-Kalshi Trading Strategy

## What are these markets?

Kalshi runs binary prediction markets on BTC every 15 minutes, 24/7. Each market asks a single yes/no question: **will BTC be above a fixed price at the end of this 15-minute window?** That fixed price — called the floor strike — is Kalshi's oracle snapshot of BTC at the moment the window opens. The market pays $1 per contract if you're right, nothing if you're wrong. You can buy Yes contracts or No contracts depending on which way you think BTC will move.

Kalshi prices these contracts as probabilities. A Yes contract at $0.70 means the market thinks there's a 70% chance BTC will be above the floor at close. Kalshi takes a 7% fee on gross winnings.

---

## The Core Signal

BTC price movements within a 15-minute window have strong momentum. If BTC has moved up 0.3% from the floor strike four minutes into the window, it tends to still be above the floor when the window closes — not always, but far more often than chance. The key insight is that **the larger the move, the more predictive it is.**

This was validated empirically across 6,370 historical KXBTC15M markets (90 days of data). The win rate — how often betting in the direction of BTC's current move produces a winning outcome — depends heavily on two things:

1. **How far BTC has moved** from the floor strike
2. **How many minutes into the window we are**

A small move early in the window (say, 0.02% at minute 4) barely predicts the outcome — the win rate is barely above 50%. A large move late in the window (say, 0.4% at minute 10) is highly predictive — the win rate approaches 95%. This 2D relationship is captured in an empirical lookup table built from the historical data.

---

## How Bets Are Sized

We don't bet a flat amount. We size each bet using two independent signals multiplied together.

### Signal 1: How far has BTC moved?

The further BTC has moved from the floor strike, the more confident we are in the direction. We scale the bet size using a smooth S-curve (sigmoid function) over the magnitude of the BTC move.

- A tiny move (under ~0.05%) produces a near-zero multiplier — almost nothing is bet.
- Around 0.10% the multiplier starts climbing quickly.
- Large moves (0.3%+) push the multiplier to its maximum of 3×.

The smooth curve avoids cliff edges — there's no sudden jump from "no bet" to "full bet" at an arbitrary threshold.

### Signal 2: How mispriced is Kalshi?

Once we know BTC's direction, we compare Kalshi's current price to what the price *should* be given the observed move. The "fair price" comes from the 2D empirical table — for example, if BTC is up 0.2% at minute 7, historical data says the win rate is about 92%. If Kalshi is only pricing Yes at 0.75, we have roughly 17 cents of edge on each dollar bet.

We scale the bet size by how large this mispricing is — again using a smooth S-curve. At zero mispricing (Kalshi is correctly priced), we still bet, but at a neutral multiplier of 1×. The more Kalshi underprices our side, the more we bet — up to a 2× bonus. If Kalshi is actually overpriced against us, the multiplier drops below 1×, shrinking the bet.

### Combined

The final bet size is:

> **base stake × magnitude multiplier × mispricing multiplier**

At maximum (large BTC move, heavily mispriced Kalshi): 3× × 2× = **6× the base stake**. In practice most bets fall in the 1-3× range.

---

## The 2D Fair Price Table

The "fair price" — what a Yes contract should theoretically be worth — is not a fixed number. It depends on both how far BTC has moved and how many minutes are left in the window.

We built a lookup table from 6,370 historical markets, bucketed into 5 magnitude ranges and 14 minute intervals. Sample values:

| BTC move from floor | At minute 4 | At minute 7 | At minute 10 | At minute 13 |
|---|---|---|---|---|
| Under 0.05% | 57% | 61% | 63% | 69% |
| 0.05–0.10% | 66% | 73% | 80% | 88% |
| 0.10–0.20% | 75% | 83% | 91% | 94% |
| 0.20–0.50% | 86% | 92% | 96% | 94% |
| Over 0.50% | 95% | 98% | 99% | — |

This replaces a simpler 1D table (which only knew what minute it was, not how far BTC had moved). The 2D table correctly identifies that a 0.02% move at minute 10 is still only worth a 63% fair price, while the old table would have blindly used 80%. This more accurate fair price leads to more accurate mispricing detection — we only size up when Kalshi is genuinely wrong.

It also makes the old "dead zone" filter (which hard-blocked bets when BTC was too close to the floor) unnecessary. Tiny-move cells naturally have fair prices close to Kalshi's price, so mispricing is small, so the computed bet is tiny — below the minimum threshold and skipped automatically.

---

## Continuously Updating Position (Delta Hedging)

Rather than placing one bet at minute 5 and waiting, the system re-evaluates every minute from minute 4 through minute 13. Each minute it looks at the current BTC price, recomputes fair value, checks Kalshi's current quote, and decides whether to add to its position.

This is called delta hedging — borrowing the concept from options trading where you continuously adjust a position to track a changing signal. The name here is loose: we're not hedging in the traditional risk-reduction sense, we're continuously updating a directional bet as new information arrives.

**Target mode** (the default) maintains a desired total exposure for each side and only bets the gap. If at minute 4 we compute a target of $200 on Yes and place $200, then at minute 5 the target recomputes to $180 (BTC moved a little less), we place nothing — we're already above target. If at minute 6 it rises to $250, we place the $50 gap. This means:

- We never pile on beyond what the signal justifies
- If BTC drifts back toward the floor, the target shrinks and we stop adding
- In a window where BTC moves clearly in one direction all the way to close, we might place 1-2 bets; in a choppy window we might place more as the target oscillates

**Additive mode** ignores prior exposure and bets the full computed amount every minute, leading to much higher volume (and variance).

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

The system supports an optional filter to only trade specific hours, configurable without code changes. Currently set to trade all 24 hours.

---

## Backtest Results

Tested on 6,360 markets (90 days, Feb–May 2026, $100 base stake):

| Strategy | ROI | Avg bets per window |
|---|---|---|
| Single bet at minute 5 | ~3% | 1 |
| Delta hedge, 1D fair price, hard dead zone | +35.7% | 1.3 |
| **Delta hedge, 2D fair price, no dead zone** | **+39.1%** | 3.3 |

The improvement from 1D to 2D fair price comes from more accurate edge detection — the system no longer over-bets on weak signals (small moves) or under-bets on strong signals (large moves at the same minute).

---

## Running It

```bash
cd live
pip install -r requirements.txt
python trader.py
```

Set `PAPER_MODE=true` in `live/.env` to run the full loop without placing real orders. All timing, fair price lookups, and P&L math runs exactly as in live mode — only the order submission is skipped.
