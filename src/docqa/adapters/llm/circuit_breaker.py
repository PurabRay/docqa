"""CircuitBreaker: stop calling a provider that keeps failing.

closed -> (failures_to_open failures) -> open for open_seconds -> half_open (one trial
call) -> closed on success, open again on failure. A daily-quota 429 opens the breaker
until the next UTC midnight, since retrying before then is pointless.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from docqa.domain.errors import DocQAError, RateLimitedError

State = Literal["closed", "open", "half_open"]
Listener = Callable[[str, State, State], None]  # (provider, old state, new state)


def next_utc_midnight(now: float) -> float:
    """Epoch seconds of the next 00:00 UTC after ``now``."""
    day = datetime.fromtimestamp(now, UTC).date() + timedelta(days=1)
    return datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()


class CircuitBreaker:
    """One breaker per provider. ``clock`` returns epoch seconds (injected for tests)."""

    def __init__(
        self,
        name: str,
        failures_to_open: int,
        open_seconds: float,
        clock: Callable[[], float] = time.time,
        listeners: list[Listener] | None = None,
    ) -> None:
        self.name = name
        self._failures_to_open = failures_to_open
        self._open_seconds = open_seconds
        self._clock = clock
        self._listeners = listeners or []
        self._failures = 0
        self._open_until: float | None = None

    @property
    def state(self) -> State:
        """Current state, derived from the clock."""
        if self._open_until is None:
            return "closed"
        return "open" if self._clock() < self._open_until else "half_open"

    def is_open(self) -> bool:
        """True while calls should be skipped."""
        return self.state == "open"

    def record_success(self) -> None:
        """A call worked: close the breaker."""
        old = self.state
        self._failures, self._open_until = 0, None
        self._notify(old)

    def record_failure(self, err: DocQAError) -> None:
        """A call failed: maybe open the breaker."""
        old = self.state
        self._failures += 1
        if isinstance(err, RateLimitedError) and err.daily_quota:
            self._open_until = next_utc_midnight(self._clock())
        elif old == "half_open" or self._failures >= self._failures_to_open:
            self._open_until = self._clock() + self._open_seconds
        self._notify(old)

    def _notify(self, old: State) -> None:
        new = self.state
        if new != old:
            for listener in self._listeners:
                listener(self.name, old, new)
