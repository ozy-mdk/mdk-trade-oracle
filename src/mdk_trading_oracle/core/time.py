"""Standardized Turkish Time (TRT / Europe/Istanbul) datetime utilities.

Turkey operates on permanent UTC+3 (no daylight saving adjustments since 2016).
All market trading timestamps, window definitions, model executions, and logging
must strictly adhere to Europe/Istanbul time.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

TURKEY_TZ = ZoneInfo("Europe/Istanbul")


def get_turkey_timezone() -> ZoneInfo:
    """Return Europe/Istanbul (TRT / UTC+3) ZoneInfo instance."""
    return TURKEY_TZ


def now_turkey() -> datetime:
    """Return the current datetime in Turkish Time (timezone-aware)."""
    return datetime.now(TURKEY_TZ)


def now_turkey_naive() -> datetime:
    """Return the current datetime in Turkish Time without tzinfo (for naive SQL TIMESTAMP columns)."""
    return datetime.now(TURKEY_TZ).replace(tzinfo=None)


def today_turkey() -> date:
    """Return current trading date in Turkish Time."""
    return datetime.now(TURKEY_TZ).date()


def to_turkey_time(dt: datetime) -> datetime:
    """Convert a datetime instance to Turkish Time (Europe/Istanbul)."""
    if dt.tzinfo is None:
        # Assume it is already Turkish time if naive
        return dt.replace(tzinfo=TURKEY_TZ)
    return dt.astimezone(TURKEY_TZ)


def format_turkey_timestamp(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime in Turkish Time string representation."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(TURKEY_TZ)
    return dt.strftime(fmt)
