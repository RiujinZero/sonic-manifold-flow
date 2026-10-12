"""Public demo availability windows.

The public world demo is only meant to be reachable during a few short
windows a day, both to bound GPU usage on the host machine and to avoid
leaving an unattended research prototype answering questions around the
clock. The dev machine's own system clock is set to US Eastern, but the
deployed container's is not (Docker containers default to UTC
regardless of the Windows host's timezone) -- so this checks wall-clock
time in "America/New_York" explicitly rather than trusting whatever
local time the process happens to be running under. That is what "use
the windows system time, which is east coast" actually requires once
the schedule has to hold inside a container with its own clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Window:
    start: time
    end: time
    label: str


WINDOWS: tuple[Window, ...] = (
    Window(time(9, 0), time(10, 0), "9:00 AM – 10:00 AM"),
    Window(time(14, 0), time(16, 0), "2:00 PM – 4:00 PM"),
    Window(time(18, 0), time(20, 0), "6:00 PM – 8:00 PM"),
)


def _eastern_now() -> datetime:
    return datetime.now(_EASTERN)


def is_online(now: datetime | None = None) -> bool:
    current = (now or _eastern_now()).timetz().replace(tzinfo=None)
    return any(window.start <= current < window.end for window in WINDOWS)


def status(now: datetime | None = None) -> dict[str, object]:
    current = now or _eastern_now()
    return {
        "online": is_online(current),
        "windows": [window.label for window in WINDOWS],
        "timezone": "US Eastern (America/New_York)",
        "server_time": current.isoformat(timespec="seconds"),
    }
