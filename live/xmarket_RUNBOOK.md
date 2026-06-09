# Cross-venue sports arb system — runbook

Buy the under-priced favourite on **Kalshi** (its book lags live games), hedge
the other outcome on **Polymarket** (sharp, deep). Net of the two = locked
profit if both fill.

## Components
| file | role |
|---|---|
| `xmarket_logger.py` | 24/7 data logger (service `kalshi-xmarket`); also provides matching/quotes |
| `xmarket_ob.py` | fixed single-subscription Kalshi WS book |
| `xmarket_arb.py` | **the executor** — detect → leg Kalshi (IOC) → hedge Poly (FAK) |
| `xmarket_poly_exec.py` | Polymarket CLOB order client |
| `xmarket_filltest.py` | minimal Kalshi-only fill test |
| `xmarket_scan.py` / `analyze_xmarket.py` | offline opportunity analysis |

## Hard safety caps (in `xmarket_arb.py`)
- `MAX_CONTRACTS = 25` per trade, `MAX_OPEN_EXPOSURE_USD = 250`
- Kill switch: `touch ~/STOP_ARB` halts immediately
- Re-checks live Kalshi ask at execution; aborts if it moved
- Per-game cooldown 60s; risk gate requires depth on BOTH sides ≥ size

## Recommended sequence (cheap → real)

### 1. Paper — watch it work, risk nothing (no keys)
```bash
python3.11 xmarket_arb.py --paper --contracts 5      # runs during live games
```
Detects real opportunities and logs simulated fills to `xmarket_trades.csv`.

### 2. Kalshi-only fill test — answer "does the stale ask fill?"
```bash
python3.11 xmarket_arb.py --live --yes --kalshi-only --contracts 5 --game <GAME> --once
```
Real Kalshi IOC, no auto hedge (prints the manual Poly hedge). If `k_filled>0`,
the edge is real. ~$3 at risk.

### 3. Set up Polymarket execution (only after step 2 proves fills)
Add to `~/kalshi-delta-hedging/live/.env` **yourself** (key never leaves the box):
```
POLY_PRIVATE_KEY=0x...     # your Polygon wallet private key — CONTROLS FUNDS
POLY_FUNDER=0x...          # your Polymarket deposit/proxy address
POLY_SIG_TYPE=1            # 0=MetaMask EOA, 1=email/magic proxy, 2=Gnosis safe
```
Fund the account with USDC on Polygon, then once:
```bash
python3.11 -c "import xmarket_poly_exec as p; c=p.make(paper=False); c.set_allowances()"
python3.11 xmarket_arb.py --check       # should show POLY usdc balance
```

### 4. Live, tiny, then scale
```bash
python3.11 xmarket_arb.py --live --yes --contracts 2 --once     # 1 trade, both legs
# inspect xmarket_trades.csv; if clean, raise --contracts within caps
```

## Notes
- Soccer (3-way) is detect-only; executor trades 2-way games (MLB/NBA/NHL/WNBA).
- Polymarket execution code is written to the py-clob-client spec but is
  UNVALIDATED until you fund a wallet and test 1–2 shares.
- Legging risk window = between the Kalshi fill and the Poly hedge; Kalshi-first
  + IOC + immediate FAK hedge minimises it. Underhedged fills are logged.
- An agent will not run `--live`. You execute real orders.
