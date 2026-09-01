"""
Lightweight in-process rate limiter for API protection.

Features:
- Sliding-window timestamp tracking per client identifier.
- Thread-safe operations via locking.
- Exposes retry-after duration for standard HTTP 429 response headers.
- Deterministic reset functionality for automated testing.
- Automatic eviction of expired timestamps to maintain bounded memory footprint.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List, Tuple


class InMemoryRateLimiter:
    """
    Sliding window rate limiter tracking request timestamps per client identifier.
    """

    def __init__(self, default_max_requests: int = 60, default_window_seconds: int = 60):
        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(
        self,
        client_id: str,
        max_requests: int | None = None,
        window_seconds: int | None = None,
        current_time: float | None = None,
    ) -> Tuple[bool, int]:
        """
        Evaluates whether a request from client_id is allowed under the current rate limit.

        Args:
            client_id: Unique client key (e.g. IP address or client hash).
            max_requests: Maximum allowed requests within window (defaults to config).
            window_seconds: Duration of the sliding window in seconds (defaults to config).
            current_time: Optional explicit timestamp for deterministic testing.

        Returns:
            Tuple of (allowed: bool, retry_after: int)
            If allowed is True, retry_after is 0.
            If allowed is False, retry_after is the number of seconds until the oldest request expires.
        """
        limit = max_requests if max_requests is not None else self.default_max_requests
        window = window_seconds if window_seconds is not None else self.default_window_seconds
        now = current_time if current_time is not None else time.time()
        window_start = now - window

        with self._lock:
            # 1. Clean timestamps older than current window
            timestamps = self._records[client_id]
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) < limit:
                # Request allowed
                valid_timestamps.append(now)
                self._records[client_id] = valid_timestamps
                return True, 0

            # Request rejected - calculate retry-after based on oldest timestamp in window
            oldest_timestamp = valid_timestamps[0]
            retry_after = max(1, int(oldest_timestamp + window - now + 0.999))
            self._records[client_id] = valid_timestamps
            return False, retry_after

    def reset(self) -> None:
        """Clears all stored rate limiter records (useful for deterministic tests)."""
        with self._lock:
            self._records.clear()

    def get_client_count(self, client_id: str, window_seconds: int | None = None) -> int:
        """Returns the number of active requests in the current window for a client."""
        window = window_seconds if window_seconds is not None else self.default_window_seconds
        now = time.time()
        window_start = now - window
        with self._lock:
            return len([t for t in self._records.get(client_id, []) if t > window_start])


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()
