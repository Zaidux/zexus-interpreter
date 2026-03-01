"""Tests for stdlib DateTimeModule."""

from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from src.zexus.stdlib.datetime import DateTimeModule


@pytest.fixture
def dt():
    return DateTimeModule()


# ── now / utc_now / timestamp ────────────────────────────────────────────
class TestCurrentTime:
    def test_now_returns_datetime(self, dt):
        result = dt.now()
        assert isinstance(result, datetime)

    def test_utc_now_returns_datetime(self, dt):
        result = dt.utc_now()
        assert isinstance(result, datetime)

    def test_timestamp_is_number(self, dt):
        result = dt.timestamp()
        assert isinstance(result, (int, float))
        assert result > 0


# ── from_timestamp ───────────────────────────────────────────────────────
class TestFromTimestamp:
    def test_from_timestamp_utc(self, dt):
        result = dt.from_timestamp(1577836800, utc=True)
        assert isinstance(result, datetime)
        assert result.year == 2020


# ── parse / format / iso_format ──────────────────────────────────────────
class TestParseFormat:
    def test_parse_and_format(self, dt):
        parsed = dt.parse("2025-06-15 10:30:00", "%Y-%m-%d %H:%M:%S")
        assert isinstance(parsed, datetime)
        formatted = dt.format(parsed, "%Y-%m-%d")
        assert formatted == "2025-06-15"

    def test_iso_format(self, dt):
        parsed = dt.parse("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        iso = dt.iso_format(parsed)
        assert "2025-01-01" in iso


# ── to_dict ──────────────────────────────────────────────────────────────
class TestToDict:
    def test_to_dict(self, dt):
        parsed = dt.parse("2025-03-15 14:30:45", "%Y-%m-%d %H:%M:%S")
        d = dt.to_dict(parsed)
        assert isinstance(d, dict)
        assert d["year"] == 2025
        assert d["month"] == 3
        assert d["day"] == 15
        assert d["hour"] == 14
        assert d["minute"] == 30
        assert d["second"] == 45


# ── add_* ────────────────────────────────────────────────────────────────
class TestAddFunctions:
    def test_add_days(self, dt):
        base = dt.parse("2025-01-01 00:00:00")
        result = dt.add_days(base, 10)
        assert dt.format(result, "%Y-%m-%d") == "2025-01-11"

    def test_add_hours(self, dt):
        base = dt.parse("2025-01-01 00:00:00")
        result = dt.add_hours(base, 25)
        d = dt.to_dict(result)
        assert d["day"] == 2
        assert d["hour"] == 1

    def test_add_minutes(self, dt):
        base = dt.parse("2025-01-01 00:00:00")
        result = dt.add_minutes(base, 90)
        d = dt.to_dict(result)
        assert d["hour"] == 1
        assert d["minute"] == 30

    def test_add_seconds(self, dt):
        base = dt.parse("2025-01-01 00:00:00")
        result = dt.add_seconds(base, 3661)
        d = dt.to_dict(result)
        assert d["hour"] == 1
        assert d["minute"] == 1
        assert d["second"] == 1


# ── diff_* ───────────────────────────────────────────────────────────────
class TestDiffFunctions:
    def test_diff_days(self, dt):
        d1 = dt.parse("2025-01-01 00:00:00")
        d2 = dt.parse("2025-01-11 00:00:00")
        assert dt.diff_days(d1, d2) == pytest.approx(10.0, abs=0.01)

    def test_diff_seconds(self, dt):
        d1 = dt.parse("2025-01-01 00:00:00")
        d2 = dt.parse("2025-01-01 00:01:00")
        assert dt.diff_seconds(d1, d2) == pytest.approx(60.0, abs=0.01)


# ── comparison ───────────────────────────────────────────────────────────
class TestComparison:
    def test_is_before(self, dt):
        d1 = dt.parse("2025-01-01 00:00:00")
        d2 = dt.parse("2025-12-31 00:00:00")
        assert dt.is_before(d1, d2) is True
        assert dt.is_before(d2, d1) is False

    def test_is_after(self, dt):
        d1 = dt.parse("2025-12-31 00:00:00")
        d2 = dt.parse("2025-01-01 00:00:00")
        assert dt.is_after(d1, d2) is True

    def test_is_between(self, dt):
        start = dt.parse("2025-01-01 00:00:00")
        mid = dt.parse("2025-06-15 00:00:00")
        end = dt.parse("2025-12-31 00:00:00")
        assert dt.is_between(mid, start, end) is True
        assert dt.is_between(start, mid, end) is False


# ── start/end of day/month ──────────────────────────────────────────────
class TestTruncation:
    def test_start_of_day(self, dt):
        base = dt.parse("2025-03-15 14:30:45")
        result = dt.start_of_day(base)
        d = dt.to_dict(result)
        assert d["hour"] == 0 and d["minute"] == 0 and d["second"] == 0

    def test_end_of_day(self, dt):
        base = dt.parse("2025-03-15 14:30:45")
        result = dt.end_of_day(base)
        d = dt.to_dict(result)
        assert d["hour"] == 23 and d["minute"] == 59 and d["second"] == 59

    def test_start_of_month(self, dt):
        base = dt.parse("2025-03-15 14:30:45")
        result = dt.start_of_month(base)
        d = dt.to_dict(result)
        assert d["day"] == 1 and d["hour"] == 0


# ── weekday / month names ───────────────────────────────────────────────
class TestNames:
    def test_weekday_name(self, dt):
        d = dt.parse("2025-01-01 00:00:00")
        name = dt.weekday_name(d)
        assert isinstance(name, str) and len(name) > 0

    def test_month_name(self, dt):
        d = dt.parse("2025-03-15 00:00:00")
        name = dt.month_name(d)
        assert name == "March"


# ── sleep (mocked) ──────────────────────────────────────────────────────
class TestSleep:
    @patch("time.sleep")
    def test_sleep_calls_time_sleep(self, mock_sleep, dt):
        dt.sleep(0.5)
        mock_sleep.assert_called_once_with(0.5)
