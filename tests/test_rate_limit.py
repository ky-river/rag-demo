import unittest

from app.rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SlidingWindowRateLimiterTest(unittest.TestCase):
    def test_blocks_after_limit_and_reports_retry_after(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(clock=clock)

        first = limiter.check("user", limit=2, window_seconds=10)
        clock.now = 2
        second = limiter.check("user", limit=2, window_seconds=10)
        clock.now = 3
        blocked = limiter.check("user", limit=2, window_seconds=10)

        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 1)
        self.assertTrue(second.allowed)
        self.assertEqual(second.remaining, 0)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after_seconds, 7)

    def test_expired_events_release_capacity(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(clock=clock)
        limiter.check("user", limit=1, window_seconds=10)

        clock.now = 10.1
        allowed = limiter.check("user", limit=1, window_seconds=10)

        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.remaining, 0)

    def test_different_subjects_have_independent_limits(self):
        limiter = SlidingWindowRateLimiter()

        limiter.check("first", limit=1, window_seconds=60)
        second = limiter.check("second", limit=1, window_seconds=60)

        self.assertTrue(second.allowed)


if __name__ == "__main__":
    unittest.main()
