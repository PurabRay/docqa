"""Per-session token bucket: limits.questions_per_minute, then HTTP 429."""

from __future__ import annotations

import time
from collections.abc import Callable

from docqa.domain.errors import RateLimitedError

SECONDS_PER_MINUTE = 60


class RateLimiter:
    """One bucket per key; each holds ``per_minute`` tokens and refills continuously."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(per_minute)
        self._refill_per_s = per_minute / SECONDS_PER_MINUTE
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last time)

    def allow(self, key: str) -> bool:
        """Take one token for ``key``; False if the bucket is empty."""
        now = self._clock()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill_per_s)
        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1, now)
        return True

    def check(self, key: str) -> None:
        """Raise RateLimitedError if ``key`` has no token left."""
        if not self.allow(key):
            raise RateLimitedError("Too many questions; please wait a moment.")
