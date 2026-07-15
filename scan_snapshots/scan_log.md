# Cross-venue inefficiency scan log

Loop: scan Polymarket + Kalshi for exploitable inefficiencies. Self-paced.

## Snapshot 2026-06-21 ~13:10 ET (scan #1)

Implied spot: BTC ~ $64.1k (Kalshi Jun21 2pm), ~$63.5-64k (PM Jun22). Consistent.

### Findings
- **No risk-free cross-venue arb above fees.** Venues tightly aligned.
- Fed Jul-2026: best lock = Kalshi YES hike25 @19 + PM NO @80 = 99c gross (~1c, < Kalshi fees). Efficient.
- BTC/ETH: implied levels agree across venues; apparent gaps are different expiries, not arb.
- Intra-platform: zero YES+NO<100 on either venue (~3700 Kalshi + ~200 PM).
- Real edge = spread capture on thin Kalshi books (market-making, not arb):
  - KXTAYLORSWIFT Blake Lively 24/36 (12c)
  - KXU3MAX-27-6 unemp>6% 2/11 (9c)
  - KXGREENLAND NOACQ 79/84 (5c)

### Key prices for diffing next round
- Kalshi BTC Jun21 2pm: >=64000=83, >=64100=57, >=64200=29
- Kalshi Fed Jul26: maintain=80, hike25=17/19, cut25=1/2
- PM Fed Jul26: nochange=78.5, hike25=20.1, cut25=1.1
- PM BTC Jun22: >62k=95.2, >66k=6.3

### TODO for next pass
- Match BTC/ETH by identical expiry timestamp for true cross-venue arb
- Pull order-book DEPTH on Fed legs to size the ~1c edge realistically
- Diff vs this snapshot to flag fast moves / new mispricings

## Snapshot 2026-06-21 ~13:30 ET (scan #2 — SPORTS + POLITICS)

### LARGE cross-venue gaps (thin PM books — VERIFY DEPTH, likely stale PM quotes)
- GOP Nominee 2028 Rubio: Kalshi 29/30 vs PM 22.1 -> PM YES@22.1 + Kalshi NO@71 = ~6.9c gross
- F1 2026 Hamilton: Kalshi 20/21 vs PM 15.3 -> PM YES@15.3 + Kalshi NO@80 = ~4.7c gross
- GOP Nominee 2028 Vance: Kalshi 41/42 vs PM 37.4 -> PM YES@37.4 + Kalshi NO@59 = ~3.6c gross
  PM 24h vol tiny (Rubio $6.8k, Hamilton $4.7k) => high stale risk. #1 verify task.

### SMALL gaps (deep books both sides, trustworthy ~1c, World Cup)
- Netherlands WC: Kalshi 7.0/7.1 vs PM 5.6 -> 1.4c (PM YES + Kalshi NO)
- USA WC: Kalshi 5.2/5.3 vs PM 4.0 -> 1.2c (PM YES + Kalshi NO)
- Argentina WC: Kalshi 10.7/10.8 vs PM 11.9 -> 1.1c (Kalshi YES + PM NO)
- 2028 Pres Vance: Kalshi 18/19 vs PM 20.5 -> 1.5c (Kalshi YES + PM NO)
- 2028 Pres Newsom: Kalshi 13/14 vs PM 15.4 -> 1.4c (Kalshi YES + PM NO)
- Pattern: PM prices USA & Netherlands cheaper, Argentina dearer, vs Kalshi across WC book.

### Fee reality
- Polymarket ~0 fee (gas). Kalshi fee ~ 0.07*P*(1-P)/contract: ~0.5c @7c leg, ~1.7c @59c NO leg.
- WC ~1c gross -> ~0.3-0.5c net (marginal, capital locked to July).
- Big politics/F1 gaps -> several c net IF PM quote live. Verify depth.

### Efficient (no leg): Newsom Dem-nom (23 vs 23.2), Antonelli F1 (60 vs 60.9), Wyndham Clark golf.

### Scope going forward: crypto + Fed + sports + politics every pass.

## Snapshot 2026-06-21 ~13:55 ET (scan #3 — DEPTH VERIFICATION + diff)

### Verdict on the 3 large gaps: ALL REAL & FILLABLE (not stale), but poor carry
Mechanics: buy PM YES + Kalshi NO (=100-YES bid); pays $1 complementary; net = 100 - PM_ask - Kalshi_NO_ask - ~2c fee.
- Rubio GOP-nom: ~16,069 contracts, 2.67c net VW -> $429 riskless, ~$15.1k capital, resolves ~2028 => ~1.1%/yr
- Vance GOP-nom: ~5,254 contracts, 1.52c net -> $80, ~$5.1k capital, ~2028 => ~0.8%/yr
- Hamilton F1: ~1,983 contracts (capped by PM ask depth 811 shares), 1.22c -> $24, resolves Dec 2026 => ~2.5%/yr
=> All below ~4-5% risk-free. As hold-to-expiry arbs NOT worth it. Only attractive as CONVERGENCE trades
   (unwind when PM thin book realigns to Kalshi) or with pre-positioned capital on both venues.
Earlier scan overstated edge: Kalshi NO ask moved 1c worse + 2c fee not in gross.

### Diff vs scan #1/#2: markets STABLE
- BTC ~flat ~$64,050 (Kalshi 2pm tails just decayed near expiry, not a real move).
- No WC/Pres gap newly widened past 1.5c. NL still ~1.55c (unchanged).

### Standing insight: deep books (WC, Fed, BTC) arbed to ~1c; gaps live in THIN PM books
(GOP-nom, F1) where PM quotes lag Kalshi. Hunt thin-PM-book / liquid-Kalshi pairs near settlement
(shorter capital lock = better annualized).

## Snapshot 2026-06-21 ~14:25 ET (scan #4 — NEAR-DATED HUNT + convergence)

### Near-dated hunt: NO executable short-lock arb found. Two structural reasons:
1. GEOPOLITICS targets RULED OUT permanently: PM Iran/Taiwan/Hormuz binaries have NO true Kalshi
   equivalent. Kalshi defines them differently (US-Iran nuclear deal, recognize Pahlavi, single-day
   Hormuz ship-count, US-recognize-Taiwan). Different resolution criteria => not arbitrable. Dead thesis.
2. Matching events lack executable Kalshi liquidity: WC winner + WC match books (1-day lock, ideal)
   reported EMPTY Kalshi orderbooks. No PM Sept-2026 FOMC. BTC strikes don't overlap (Kalshi 200/250k
   vs PM 90/110k) + different mechanics (index snapshot vs daily close).

### DATA DISCREPANCY to resolve next round:
scan #4 hunter saw Kalshi WC orderbooks EMPTY, but scan #2 showed WC tight bid/ask + 19M vol, and
scan #3 depth check found real Kalshi depth on Rubio (politics). Either Kalshi /orderbook endpoint
behaves differently than market-summary bid/ask (auth/endpoint quirk), or WC books genuinely thin.
=> The small ~1c WC gaps (scan #2) may NOT be executable at size. NEEDS dedicated WC depth probe.

### Convergence: 3 gaps HOLDING (Rubio/Vance/Hamilton moved <0.5c/leg). No unwind signal. BTC flat ~$64.0k.

### State of the hunt after 4 scans: landscape largely mapped. Real edges only in thin politics/F1
books with poor annualized carry. Near-dated liquid matching pairs don't currently exist. Diminishing
returns -> consider slower cadence or event-triggered monitoring vs continuous re-scan.

## Snapshot 2026-06-21 ~14:40 ET (scan #5 — WC DEPTH RESOLVED + wind-down)

### WC gaps RESOLVED: NOT executable. Fees kill them.
Walked real Kalshi+PM books for FR/ES/NL/US/AR. After Kalshi fee (1-2c/ct) every team nets <=0:
France -1.3c, Spain -0.9c, Argentina -0.1c, Netherlands +0.0c (5,763 ct = $0.06 dust), USA +0.05c (68 ct = $0.03).
=> The ~1-1.5c WC "gaps" from scan #2 were GROSS; they were never real after fees. Dead.

### Data reliability SETTLED: Kalshi books are real & deep (France NO 81.0c x 57,783 ct; USA NO 94.7c x 245,560).
The scan #4 "empty orderbook" was a TRANSIENT PARSE ERROR, not an empty book. Displayed bid/ask IS backed by size.
Note: Kalshi `liquidity_dollars` always "0.0000" for WC event -> ignore it; use orderbook/*_size_fp for depth.

### FINAL CONCLUSION (5 scans): the two venues are essentially efficient to retail AFTER FEES.
- WC gaps: dead (fees). Geopolitics: dead (different event definitions). Crypto/Fed: efficient (~1c gross, sub-fee).
- ONLY fillable edge = thin-PM-book politics/F1 (Rubio ~$429 / Vance ~$80 / Hamilton ~$24), poor annualized
  carry (locked 2026-2028), worth it only as convergence trade or with pre-positioned capital. All HOLDING.

### MODE CHANGE: switched to SLOW / event-triggered (hourly heartbeat, alert only on gap >3c net /
>10% annualized / BTC move >1%) per user choice. Continuous 25-min re-scan retired (diminishing returns).

## ALERT 2026-06-21 ~18:30 ET (slow-mode hourly check #1 — MATERIAL CHANGE)
Gaps WIDENED (PM YES dropped vs Kalshi -> cheaper lock, better entry):
- Rubio: net 2.67c -> 4.85c (+2.18, crossed 3c threshold). Best edge + only one w/ real size (~$15k).
- Hamilton: net 1.22c -> 2.75c (+1.53).
- Vance: holding 1.65c. BTC flat ~$63.6k (-0.6%).
Action: better CONVERGENCE entry; but verify PM ask still has resting depth at widened price before acting
(thin-book widening can be a stale print). Pending user decision on depth re-verify.
