"""Fail-closed information-deadline policy for the Kalshi LIP farmer.

``close_time`` is not a safety boundary.  Sports lineups, broadcasts,
weather observations, and published indices can become known while their
markets remain tradable.  This module converts verified schedule metadata
into an earlier server-side order expiration and refuses series whose safe
window cannot be classified.

The policy deliberately supports only two live-safe classes initially:

* scheduled lineup series whose ``occurrence_datetime`` has been verified;
* KXRAIN markets, which leave before the measured local calendar day starts.

Known post-observation traps are blocked explicitly.  Unknown series fail
closed unless ``LIP_SAFE_REQUIRE_CLASSIFIED=0`` is intentionally configured.
"""

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import lip_config as cfg


@dataclass(frozen=True)
class SafeWindowDecision:
    allowed: bool
    deadline_ts: Optional[float]
    policy: str
    reason: str


# Final ticker component -> the location whose calendar day is measured.
_RAIN_TIMEZONES = {
    "NYC": "America/New_York", "CHI": "America/Chicago",
    "AUS": "America/Chicago", "MIA": "America/New_York",
    "DEN": "America/Denver", "PHIL": "America/New_York",
    "LAX": "America/Los_Angeles", "LV": "America/Los_Angeles",
    "NOLA": "America/Chicago", "SFO": "America/Los_Angeles",
    "DC": "America/New_York", "SEA": "America/Los_Angeles",
    "BOS": "America/New_York", "PHX": "America/Phoenix",
    "ATL": "America/New_York", "MIN": "America/Chicago",
    "DAL": "America/Chicago", "SATX": "America/Chicago",
    "HOU": "America/Chicago", "OKC": "America/Chicago",
}


def _parse_ts(raw) -> Optional[float]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _series(program, market: dict) -> str:
    return str(
        getattr(program, "series_ticker", "")
        or market.get("series_ticker")
        or getattr(program, "ticker", "").split("-")[0]
    ).upper()


def _matches(series: str, patterns: list[str]) -> bool:
    return any(pattern in series for pattern in patterns)


def series_supported(program) -> bool:
    """Cheap pre-scan gate so toxic programs never consume book requests."""
    if not cfg.SAFE_WINDOW_ENABLED:
        return True
    series = str(
        getattr(program, "series_ticker", "")
        or getattr(program, "ticker", "").split("-")[0]
    ).upper()
    if _matches(series, cfg.SAFE_BLOCK_PATTERNS):
        return False
    if _matches(series, cfg.SAFE_WEATHER_PATTERNS):
        return True
    if _matches(series, cfg.SAFE_SCHEDULE_PATTERNS):
        return True
    return not cfg.SAFE_REQUIRE_CLASSIFIED


def risk_group(program, market: dict) -> str:
    """Underlying catalyst key used to prevent correlated sibling quoting.

    Kalshi can split one match into team-specific event_tickers.  The first
    two ticker components remain the common real-world event/date key for the
    recurring LIP series observed so far.
    """
    ticker = str(getattr(program, "ticker", "") or market.get("ticker") or "")
    parts = ticker.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return str(market.get("event_ticker") or ticker)


def _rain_deadline(ticker: str) -> Optional[float]:
    parts = ticker.upper().split("-")
    if len(parts) < 3:
        return None
    zone = _RAIN_TIMEZONES.get(parts[-1])
    if not zone:
        return None
    try:
        measured_day = datetime.strptime(parts[1], "%y%b%d").date()
    except ValueError:
        return None
    local_start = datetime.combine(
        measured_day, dt_time.min, tzinfo=ZoneInfo(zone)
    )
    return (local_start - timedelta(
        hours=cfg.SAFE_WEATHER_BUFFER_HOURS
    )).timestamp()


def assess(program, market: dict, now_ts: Optional[float] = None) -> SafeWindowDecision:
    """Return whether a market may be quoted now and its hard expiration."""
    now_ts = datetime.now().timestamp() if now_ts is None else float(now_ts)
    if not cfg.SAFE_WINDOW_ENABLED:
        return SafeWindowDecision(True, None, "disabled", "safe-window disabled")

    ticker = str(getattr(program, "ticker", "") or market.get("ticker") or "")
    series = _series(program, market)

    if _matches(series, cfg.SAFE_BLOCK_PATTERNS):
        return SafeWindowDecision(
            False, None, "blocked-series", f"{series} matches toxic-series blocklist"
        )

    deadline = None
    policy = ""
    if _matches(series, cfg.SAFE_WEATHER_PATTERNS):
        deadline = _rain_deadline(ticker)
        policy = "weather-local-day"
        if deadline is None:
            return SafeWindowDecision(
                False, None, policy, "weather location/date could not be classified"
            )
    elif _matches(series, cfg.SAFE_SCHEDULE_PATTERNS):
        occurrence = _parse_ts(market.get("occurrence_datetime"))
        if occurrence is None:
            return SafeWindowDecision(
                False, None, "scheduled-event", "missing occurrence_datetime"
            )
        deadline = occurrence - cfg.SAFE_SCHEDULE_BUFFER_HOURS * 3600.0
        policy = "scheduled-event"
    elif cfg.SAFE_REQUIRE_CLASSIFIED:
        return SafeWindowDecision(
            False, None, "unclassified", f"{series or ticker} has no verified safe-window rule"
        )
    else:
        close_ts = _parse_ts(market.get("close_time"))
        if close_ts is None:
            return SafeWindowDecision(
                False, None, "generic-close", "missing close_time"
            )
        deadline = close_ts - cfg.SAFE_GENERIC_CLOSE_BUFFER_HOURS * 3600.0
        policy = "generic-close"

    remaining = deadline - now_ts
    if remaining <= 0:
        return SafeWindowDecision(
            False, deadline, policy, "information deadline has passed"
        )
    minimum = cfg.SAFE_MIN_REMAINING_HOURS * 3600.0
    if remaining < minimum:
        return SafeWindowDecision(
            False, deadline, policy,
            f"only {remaining / 3600.0:.2f} safe hours remain"
        )
    return SafeWindowDecision(
        True, deadline, policy, f"{remaining / 3600.0:.2f} safe hours remain"
    )
