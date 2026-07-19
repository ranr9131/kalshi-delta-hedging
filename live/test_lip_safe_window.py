"""Regression tests for the schedule-safe LIP quoting layer."""

from datetime import datetime, timezone
from types import SimpleNamespace

import lip_api
import lip_safe_window as safe


def ts(raw):
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def program(ticker, series):
    return SimpleNamespace(ticker=ticker, series_ticker=series)


def test_houston_rain_leaves_before_measured_day():
    p = program("KXRAIN-26JUL15-HOU", "KXRAIN")
    m = {"ticker": p.ticker}
    early = safe.assess(p, m, now_ts=ts("2026-07-14T22:00:00Z"))
    assert early.allowed and early.policy == "weather-local-day"
    # Houston midnight is 05:00Z in July; the two-hour safety buffer makes
    # 03:00Z the hard deadline.
    assert early.deadline_ts == ts("2026-07-15T03:00:00Z")
    filled = safe.assess(p, m, now_ts=ts("2026-07-15T15:56:02Z"))
    assert not filled.allowed


def test_world_cup_lineup_leaves_three_hours_before_kickoff():
    p = program("KXWCSTART-26JUL15ENGARG-ENG-MGUEHI6", "KXWCSTART")
    m = {"ticker": p.ticker, "occurrence_datetime": "2026-07-15T20:00:00Z"}
    early = safe.assess(p, m, now_ts=ts("2026-07-15T12:00:00Z"))
    assert early.allowed
    assert early.deadline_ts == ts("2026-07-15T17:00:00Z")
    lineup_release = safe.assess(p, m, now_ts=ts("2026-07-15T17:18:31Z"))
    assert not lineup_release.allowed


def test_known_observable_series_are_blocked():
    cases = [
        ("KXWNBAMENTION-26JUL15GSIND-DOUB", "KXWNBAMENTION"),
        ("KXAAAGASW-26JUL20-3.900", "KXAAAGASW"),
        ("KXLIUKELIMINATION-26JUN10-LOL", "KXLIUKELIMINATION"),
    ]
    for ticker, series in cases:
        d = safe.assess(program(ticker, series), {"ticker": ticker}, now_ts=0)
        assert not d.allowed and d.policy == "blocked-series", (ticker, d)


def test_unknown_series_fails_closed():
    p = program("KXUNVERIFIED-26JUL20-X", "KXUNVERIFIED")
    d = safe.assess(p, {"ticker": p.ticker}, now_ts=0)
    assert not d.allowed and d.policy == "unclassified"


def test_sibling_markets_share_one_risk_group():
    france = program("KXWCSTART-26JUL18FRAENG-FRA-KMBAPP10", "KXWCSTART")
    england = program("KXWCSTART-26JUL18FRAENG-ENG-HKANE9", "KXWCSTART")
    assert safe.risk_group(france, {}) == "KXWCSTART-26JUL18FRAENG"
    assert safe.risk_group(france, {}) == safe.risk_group(england, {})


def test_series_prescan_is_fail_closed():
    assert safe.series_supported(program("KXRAIN-26JUL18-NYC", "KXRAIN"))
    assert safe.series_supported(program("KXWCSTART-X-Y", "KXWCSTART"))
    assert not safe.series_supported(program("KXAAAGASW-X-Y", "KXAAAGASW"))
    assert not safe.series_supported(program("KXUNKNOWN-X-Y", "KXUNKNOWN"))


def test_expiring_order_payload(monkeypatch=None):
    captured = {}

    class Response:
        status_code = 201
        ok = True
        text = ""

        @staticmethod
        def json():
            return {"order": {"order_id": "test-order"}}

    original = lip_api._write
    try:
        def fake_write(method, path, body, private_key, api_key_id, retries=3):
            captured.update(body)
            return Response()

        lip_api._write = fake_write
        out = lip_api.place_resting_bid(
            None, None, "TEST", "yes", 1, 1000,
            expiration_ts=1784144400,
        )
    finally:
        lip_api._write = original

    assert out["order_id"] == "test-order"
    assert captured["expiration_time"] == 1784144400
    assert captured["time_in_force"] == "good_till_canceled"
    assert captured["cancel_order_on_pause"] is True
    assert captured["post_only"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
