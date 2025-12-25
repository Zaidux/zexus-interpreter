"""DateTime module for Zexus standard library."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import time


class DateTimeModule:
    """Provides date and time operations."""

    @staticmethod
    def now() -> datetime:
        """Get current datetime."""
        return datetime.now()

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)

    @staticmethod
    def timestamp() -> float:
        """Get current Unix timestamp."""
        return time.time()

    @staticmethod
    def from_timestamp(ts: float, utc: bool = False) -> datetime:
        """Create datetime from Unix timestamp."""
        if utc:
            return datetime.fromtimestamp(ts, timezone.utc)
        return datetime.fromtimestamp(ts)

    @staticmethod
    def parse(date_string: str, format: str = '%Y-%m-%d %H:%M:%S') -> datetime:
        """Parse date string to datetime."""
        return datetime.strptime(date_string, format)

    @staticmethod
    def format(dt: datetime, format: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format datetime to string."""
        return dt.strftime(format)

    @staticmethod
    def iso_format(dt: datetime) -> str:
        """Format datetime to ISO 8601 string."""
        return dt.isoformat()

    @staticmethod
    def to_dict(dt: datetime) -> Dict[str, Any]:
        """Convert datetime to dictionary."""
        return {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute,
            'second': dt.second,
            'microsecond': dt.microsecond,
            'weekday': dt.weekday(),
            'timestamp': dt.timestamp()
        }

    @staticmethod
    def add_days(dt: datetime, days: int) -> datetime:
        """Add days to datetime."""
        return dt + timedelta(days=days)

    @staticmethod
    def add_hours(dt: datetime, hours: int) -> datetime:
        """Add hours to datetime."""
        return dt + timedelta(hours=hours)

    @staticmethod
    def add_minutes(dt: datetime, minutes: int) -> datetime:
        """Add minutes to datetime."""
        return dt + timedelta(minutes=minutes)

    @staticmethod
    def add_seconds(dt: datetime, seconds: int) -> datetime:
        """Add seconds to datetime."""
        return dt + timedelta(seconds=seconds)

    @staticmethod
    def diff_days(dt1: datetime, dt2: datetime) -> int:
        """Get difference in days between two datetimes."""
        return (dt2 - dt1).days

    @staticmethod
    def diff_seconds(dt1: datetime, dt2: datetime) -> float:
        """Get difference in seconds between two datetimes."""
        return (dt2 - dt1).total_seconds()

    @staticmethod
    def is_before(dt1: datetime, dt2: datetime) -> bool:
        """Check if dt1 is before dt2."""
        return dt1 < dt2

    @staticmethod
    def is_after(dt1: datetime, dt2: datetime) -> bool:
        """Check if dt1 is after dt2."""
        return dt1 > dt2

    @staticmethod
    def is_between(dt: datetime, start: datetime, end: datetime) -> bool:
        """Check if datetime is between start and end."""
        return start <= dt <= end

    @staticmethod
    def start_of_day(dt: datetime) -> datetime:
        """Get start of day (00:00:00)."""
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def end_of_day(dt: datetime) -> datetime:
        """Get end of day (23:59:59)."""
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    @staticmethod
    def start_of_month(dt: datetime) -> datetime:
        """Get start of month."""
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def sleep(seconds: float) -> None:
        """Sleep for specified seconds."""
        time.sleep(seconds)

    @staticmethod
    def weekday_name(dt: datetime) -> str:
        """Get weekday name."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[dt.weekday()]

    @staticmethod
    def month_name(dt: datetime) -> str:
        """Get month name."""
        return dt.strftime('%B')


# Export functions for easy access
now = DateTimeModule.now
utc_now = DateTimeModule.utc_now
timestamp = DateTimeModule.timestamp
from_timestamp = DateTimeModule.from_timestamp
parse = DateTimeModule.parse
format_dt = DateTimeModule.format
iso_format = DateTimeModule.iso_format
to_dict = DateTimeModule.to_dict
add_days = DateTimeModule.add_days
add_hours = DateTimeModule.add_hours
add_minutes = DateTimeModule.add_minutes
add_seconds = DateTimeModule.add_seconds
diff_days = DateTimeModule.diff_days
diff_seconds = DateTimeModule.diff_seconds
is_before = DateTimeModule.is_before
is_after = DateTimeModule.is_after
is_between = DateTimeModule.is_between
start_of_day = DateTimeModule.start_of_day
end_of_day = DateTimeModule.end_of_day
start_of_month = DateTimeModule.start_of_month
sleep = DateTimeModule.sleep
weekday_name = DateTimeModule.weekday_name
month_name = DateTimeModule.month_name
