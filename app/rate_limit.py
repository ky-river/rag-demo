from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: int = 0


class SlidingWindowRateLimiter:
    def __init__(self, clock: Callable[[], float] | None = None):
        self.clock = clock or time.monotonic
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._checks = 0

    def check(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitResult(allowed=True, remaining=0)

        now = self.clock()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= limit:
                retry_after = max(1, math.ceil(events[0] + window_seconds - now))
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            events.append(now)
            self._checks += 1
            if self._checks % 1000 == 0:
                self._prune_empty_keys(cutoff)
            return RateLimitResult(
                allowed=True,
                remaining=max(0, limit - len(events)),
            )

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._checks = 0

    def _prune_empty_keys(self, cutoff: float) -> None:
        empty_keys = [
            key
            for key, events in self._events.items()
            if not events or events[-1] <= cutoff
        ]
        for key in empty_keys:
            self._events.pop(key, None)
