"""
Stake sizing via combined sigmoid: f(BTC magnitude) * g(Kalshi mispricing).
Direct port of parameters from analyze_scaled.py — do not change without re-running backtests.
"""

import math

# Empirical from 30-day backtest (2817 windows, 69.8% signal accuracy)
FAIR_PRICE       = 0.698   # expected Kalshi Yes price given correct signal

# f(BTC magnitude): sigmoid over abs % BTC move from T+0 to T+5
SIGMOID_CENTER   = 0.10    # 0.10% move = inflection point
SIGMOID_K        = 20.0    # steepness
SIGMOID_MAX_MULT = 3.0     # max multiplier (3x base stake)

# g(mispricing): sigmoid over (fair_price - actual_kalshi_price)
MISPRICING_K     = 8.0     # steepness: ±0.15 mispricing → ~1.6x or ~0.4x
MISPRICING_MAX   = 2.0     # range (0, 2.0); equals 1.0 at zero mispricing


def sigmoid_btc(abs_pct_move: float) -> float:
    """f signal: 0 → SIGMOID_MAX_MULT based on magnitude of BTC move."""
    x = SIGMOID_K * (abs_pct_move - SIGMOID_CENTER)
    return SIGMOID_MAX_MULT / (1.0 + math.exp(-x))


def sigmoid_mispricing(mispricing: float) -> float:
    """g signal: 0 → MISPRICING_MAX based on how mispriced Kalshi is. Equals 1.0 at zero."""
    x = MISPRICING_K * mispricing
    return MISPRICING_MAX / (1.0 + math.exp(-x))


def compute_stake(
    btc_t0: float,
    btc_t5: float,
    kalshi_yes_price: float,
    side: str,
    base_stake: float,
) -> tuple[float, float, float, float, float]:
    """
    Compute decision leg stake and intermediate signal values.

    btc_t0, btc_t5: BTC price in USD at T+0 and T+5
    kalshi_yes_price: current Kalshi Yes price as fraction (0.0-1.0)
    side: "yes" (BTC up, buying Yes) or "no" (BTC down, buying No)
    base_stake: base dollar amount (e.g. 100.0)

    Returns: (stake, abs_pct_move, mispricing, f_btc, g_misprice)
      stake:        dollars to wager on this bet
      abs_pct_move: absolute BTC % move (e.g. 0.15 = 0.15%)
      mispricing:   positive = Kalshi price is good for us
      f_btc:        BTC magnitude multiplier (0 to SIGMOID_MAX_MULT)
      g_misprice:   mispricing multiplier (0 to MISPRICING_MAX)
    """
    abs_pct_move = abs(btc_t5 - btc_t0) / btc_t0 * 100.0

    if side == "yes":
        mispricing = FAIR_PRICE - kalshi_yes_price        # positive = Yes is cheap
    else:
        mispricing = kalshi_yes_price - (1.0 - FAIR_PRICE)  # positive = No is cheap

    f_btc     = sigmoid_btc(abs_pct_move)
    g_misprice = sigmoid_mispricing(mispricing)
    stake      = base_stake * f_btc * g_misprice

    return stake, abs_pct_move, mispricing, f_btc, g_misprice


if __name__ == "__main__":
    # Verify against known backtest values
    stake, pct, mis, f, g = compute_stake(82000, 82164, 0.55, "yes", 100.0)
    print(f"BTC +0.20%, Yes at 0.55 (underpriced by 0.148):")
    print(f"  abs_pct_move = {pct:.4f}%")
    print(f"  mispricing   = {mis:.4f}")
    print(f"  f_btc        = {f:.4f}  (3.0 = max)")
    print(f"  g_misprice   = {g:.4f}  (1.0 = fair, 2.0 = max)")
    print(f"  stake        = ${stake:.2f}")
