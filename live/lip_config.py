"""
LIP farmer configuration. Every knob is env-overridable (see .env / systemd
Environment= lines). Defaults are deliberately conservative: PAPER mode on,
small caps, join-only quoting (never improve the touch).

Program mechanics this config is built around (CFTC filings Sept 2025 +
Feb 2026 amendment, effective 2026-02-28):
  - Per-second random snapshots; per side, levels qualify by walking down from
    the best bid accumulating size until cumulative >= target_size.
  - Score per level = discount_factor^(cents behind best bid) * size,
    normalized per side per snapshot. Period payout = your time-averaged
    share x period_reward (min $1.00 to be paid at all).
  - Snapshot excluded unless BOTH sides have >= target_size resting.
  - Program sunsets 2026-09-01 (Kalshi may end it earlier).
"""

import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------- mode
# Fail safe: live only on an explicit false value. A typo must stay paper.
PAPER = os.environ.get("LIP_PAPER", "1").strip().lower() not in \
    ("0", "false", "no", "off")
CANCEL_ON_EXIT    = _b("LIP_CANCEL_ON_EXIT", True)
KILL_FILE         = _s("LIP_KILL_FILE", os.path.join(os.path.dirname(__file__), "lip_kill"))

# ---------------------------------------------------------------- portfolio caps
MAX_MARKETS           = _i("LIP_MAX_MARKETS", 40)
MAX_TOTAL_CAPITAL     = _f("LIP_MAX_TOTAL_CAPITAL", 2000.0)   # $ collateral resting
MAX_TOTAL_WORST_LOSS  = _f("LIP_MAX_TOTAL_WORST_LOSS", 500.0) # $ sum of per-market worst cases
MAX_LOSS_PER_MARKET   = _f("LIP_MAX_LOSS_PER_MARKET", 30.0)   # $ bound on one-sided adverse fill
MIN_BALANCE           = _f("LIP_MIN_BALANCE", 100.0)          # stop placing below this

# ---------------------------------------------------------------- market filters
MIN_REWARD_PER_DAY    = _f("LIP_MIN_REWARD_PER_DAY", 5.0)     # $/day program rate floor
MIN_EXPECTED_PAYOUT   = _f("LIP_MIN_EXPECTED_PAYOUT", 1.50)   # $ expected for remainder of period
                                                              # (Kalshi pays nothing under $1.00)
MIN_HOURS_TO_CLOSE    = _f("LIP_MIN_HOURS_TO_CLOSE", 2.0)
SERIES_BLACKLIST      = [s.strip().upper() for s in
                         _s("LIP_SERIES_BLACKLIST", "").split(",") if s.strip()]

# ---------------------------------------------------------------- schedule-safe windows
# A Kalshi market can stay open after its answer is already observable.  The
# LIP farmer must leave before that information deadline, not merely before
# close_time.  Unknown series fail closed by default; extend the scheduled or
# weather pattern lists only after their exchange metadata has been verified.
SAFE_WINDOW_ENABLED       = _b("LIP_SAFE_WINDOW_ENABLED", True)
SAFE_REQUIRE_CLASSIFIED   = _b("LIP_SAFE_REQUIRE_CLASSIFIED", True)
SAFE_MIN_REMAINING_HOURS  = _f("LIP_SAFE_MIN_REMAINING_HOURS", 2.0)
SAFE_SCHEDULE_BUFFER_HOURS = _f("LIP_SAFE_SCHEDULE_BUFFER_HOURS", 3.0)
SAFE_WEATHER_BUFFER_HOURS = _f("LIP_SAFE_WEATHER_BUFFER_HOURS", 2.0)
SAFE_GENERIC_CLOSE_BUFFER_HOURS = _f("LIP_SAFE_GENERIC_CLOSE_BUFFER_HOURS", 24.0)
MAX_MARKETS_PER_EVENT     = _i("LIP_MAX_MARKETS_PER_EVENT", 1)

SAFE_SCHEDULE_PATTERNS = [s.strip().upper() for s in _s(
    "LIP_SAFE_SCHEDULE_PATTERNS", "KXWCSTART"
).split(",") if s.strip()]
SAFE_WEATHER_PATTERNS = [s.strip().upper() for s in _s(
    "LIP_SAFE_WEATHER_PATTERNS", "KXRAIN"
).split(",") if s.strip()]
SAFE_BLOCK_PATTERNS = [s.strip().upper() for s in _s(
    "LIP_SAFE_BLOCK_PATTERNS",
    "MENTION,ELIMINATION,AAAGAS,FIRSTSONG,FINALSONG,TWEET,"
    "TRUTHSOCIAL,WCPRICE,H100WS,USFLYCAN"
).split(",") if s.strip()]

# ---------------------------------------------------------------- quoting policy
SIZE_MULT             = _f("LIP_SIZE_MULT", 1.0)   # our size per side = mult x target_size
ALLOW_IMPROVE         = _b("LIP_ALLOW_IMPROVE", False)  # never penny the best bid by default
MAX_BEHIND_TICKS      = _i("LIP_MAX_BEHIND_TICKS", 5)   # deepest candidate behind the reference
EMPTY_BOOK_PRICE      = _i("LIP_EMPTY_BOOK_PRICE", 2)   # cents, our bid when a side is empty
MID_MARGIN_CENTS      = _i("LIP_MID_MARGIN_CENTS", 3)   # stay this far below credible mid
MAX_CREDIBLE_SPREAD   = _i("LIP_MAX_CREDIBLE_SPREAD", 15)  # mid trusted only if spread <= this
SELF_CROSS_GAP        = _i("LIP_SELF_CROSS_GAP", 2)     # our yes_bid + no_bid <= 100 - gap
REPRICE_TOLERANCE     = _i("LIP_REPRICE_TOLERANCE", 1)  # don't churn for <= this many cents

# ---------------------------------------------------------------- cadence (seconds)
PROGRAM_REFRESH_SEC   = _i("LIP_PROGRAM_REFRESH_SEC", 600)
ALLOC_INTERVAL_SEC    = _i("LIP_ALLOC_INTERVAL_SEC", 300)
QUOTE_REFRESH_SEC     = _i("LIP_QUOTE_REFRESH_SEC", 30)
FILL_POLL_SEC         = _i("LIP_FILL_POLL_SEC", 20)
STATE_FLUSH_SEC       = _i("LIP_STATE_FLUSH_SEC", 60)
SCAN_TOP_N            = _i("LIP_SCAN_TOP_N", 150)   # books scanned per alloc cycle (by reward)
EXPLORE_N             = _i("LIP_EXPLORE_N", 30)     # extra random books scanned per cycle
HYSTERESIS            = _f("LIP_HYSTERESIS", 1.20)  # challenger must beat incumbent by 20%

# ---------------------------------------------------------------- risk reactions
FILL_COOLDOWN_SEC     = _i("LIP_FILL_COOLDOWN_SEC", 1800)  # pause market after a fill
SERIES_COOLDOWN_SEC   = _i("LIP_SERIES_COOLDOWN_SEC", 6 * 3600)  # a fill quarantines the
    # ENTIRE series: informed flow is correlated across an event's sibling
    # markets (learned 2026-06-10: Love Island elimination sweep, 5 markets
    # picked off in 19 min as the allocator rotated within the series)
FLATTEN_ON_FILL       = _b("LIP_FLATTEN_ON_FILL", False)   # default: hold cheap fills to settle
MAX_FILLS_PER_DAY     = _i("LIP_MAX_FILLS_PER_DAY", 8)     # global circuit breaker

# ---------------------------------------------------------------- rate limits (req/s)
PUBLIC_RPS            = _f("LIP_PUBLIC_RPS", 6.0)
AUTH_RPS              = _f("LIP_AUTH_RPS", 4.0)
WRITE_RPS             = _f("LIP_WRITE_RPS", 1.0)   # order place/cancel; Kalshi 429s fast bursts

# ---------------------------------------------------------------- paths
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATE_FILE   = _s("LIP_STATE_FILE", os.path.join(_HERE, "lip_state.json"))
FILLS_CSV    = _s("LIP_FILLS_CSV", os.path.join(_HERE, "lip_fills.csv"))
ALLOC_CSV    = _s("LIP_ALLOC_CSV", os.path.join(_HERE, "lip_alloc.csv"))
ACCRUAL_CSV  = _s("LIP_ACCRUAL_CSV", os.path.join(_HERE, "lip_accrual.csv"))
LOG_FILE     = _s("LIP_LOG_FILE", os.path.join(_HERE, "lip_farmer.log"))

# LIP scoring constants (from the rule filings; per-program values come from
# the API, these are only fallbacks/sanity bounds)
DEFAULT_DISCOUNT_FACTOR = 0.50
PROGRAM_SUNSET          = "2026-09-01T00:00:00Z"
