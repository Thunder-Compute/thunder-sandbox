"""Bounded API polling helpers shared by the native resource handles.

These live apart from :mod:`.sandbox` so that any handle can poll the API the
same way, with the same backoff and outage grace, without importing it.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from datetime import datetime
from typing import Literal, overload
from urllib.parse import quote

from .._common.exceptions import InvalidRequestError

# The fractional seconds of an RFC 3339 timestamp, before its offset or end.
_FRACTION = re.compile(r"\.(\d+)(?=[+-]\d{2}:\d{2}$|$)")

OUTAGE_GRACE_SECONDS = 30.0
# The API refuses a readiness wait held open longer than this per request.
WAIT_WINDOW_MAX_SECONDS = 30.0
# How much longer than its window one wait request may take to answer before
# the client gives up on it, covering transit and a server that has stalled.
WAIT_REPLY_GRACE_SECONDS = 15.0


def _path_segment(value: str) -> str:
    if not value:
        raise InvalidRequestError("sandbox ID cannot be empty")
    return quote(value, safe="")


def _earliest_deadline(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


@overload
def _datetime(value: object, optional: Literal[False] = False) -> datetime: ...


@overload
def _datetime(value: object, optional: Literal[True]) -> datetime | None: ...


def _datetime(value: object, optional: bool = False) -> datetime | None:
    if value in (None, ""):
        return None if optional else datetime.fromtimestamp(0).astimezone()
    text = str(value).replace("Z", "+00:00")
    # The API trims trailing zeros from nanosecond timestamps, so any number
    # of fractional digits arrives; Python 3.10 accepts only 3 or 6. Use six,
    # truncating below the microsecond.
    text = _FRACTION.sub(lambda match: "." + match.group(1).ljust(6, "0")[:6], text)
    return datetime.fromisoformat(text)


def _wait_window(deadline: float | None) -> float:
    """Seconds to ask the API to hold one readiness wait open.

    The client deadline, not the server's maximum, is the binding bound once
    it is nearer. Rounded to the millisecond so the query string stays plain,
    and never below one: the API rejects a window of zero, and a deadline that
    has just passed is reported by the caller, not by a 400.
    """
    window = WAIT_WINDOW_MAX_SECONDS
    if deadline is not None:
        window = min(window, deadline - time.monotonic())
    window = max(0.001, round(window, 3))
    assert 0.0 < window <= WAIT_WINDOW_MAX_SECONDS, window
    return window


async def _sleep_until_next_poll(
    deadline: float | None, delay: float, retry_after: float | None = None
) -> None:
    delay = max(delay, retry_after or 0.0)
    delay += random.uniform(0.0, min(delay * 0.2, 1.0))
    if deadline is not None:
        delay = max(0.0, min(delay, deadline - time.monotonic()))
    await asyncio.sleep(delay)
