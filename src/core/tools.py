"""Deterministic tools for requests that need live system information."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from src.config import CHAT_TIMEZONE, TIME_DISPLAY_FORMAT


@tool
def get_current_time() -> str:
    """Return the current date and time in the chatbot's configured timezone."""
    try:
        current_time = datetime.now(ZoneInfo(CHAT_TIMEZONE))
        timezone_label = CHAT_TIMEZONE
    except ZoneInfoNotFoundError:
        # Windows may not have the IANA database until the tzdata dependency is installed.
        current_time = datetime.now().astimezone()
        timezone_label = current_time.tzname() or "local timezone"
    return f"The current time is {current_time.strftime(TIME_DISPLAY_FORMAT)} ({timezone_label})."


def call_time_tool() -> str:
    """Invoke the current-time tool and return its user-facing result."""
    return str(get_current_time.invoke({}))
