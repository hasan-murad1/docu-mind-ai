"""Tests for VisitorLimiter: per-minute and per-day quotas, refunds and memory cap."""

from datetime import date

import pytest

from usage_limiter import (
    LIMIT_DAY,
    LIMIT_MINUTE,
    MAX_QUESTIONS_PER_DAY,
    MAX_QUESTIONS_PER_MINUTE,
    MAX_UPLOADS_PER_DAY,
    MAX_UPLOADS_PER_MINUTE,
    VisitorLimiter,
    question_limiter,
    upload_limiter,
)


class FakeTime:
    """Controllable clock and calendar day."""

    def __init__(self):
        self.now = 1000.0
        self.day = date(2026, 11, 17)

    def clock(self):
        return self.now

    def today(self):
        return self.day


def make(per_minute=3, per_day=5, max_visitors=100):
    fake = FakeTime()
    limiter = VisitorLimiter(per_minute, per_day, max_visitors=max_visitors,
                             clock=fake.clock, today=fake.today)
    return limiter, fake


def test_allows_up_to_per_minute_then_blocks():
    limiter, _ = make(per_minute=3)
    assert [limiter.try_acquire("v") for _ in range(3)] == [None, None, None]
    assert limiter.try_acquire("v") == LIMIT_MINUTE


def test_minute_window_slides():
    limiter, fake = make(per_minute=3)
    for _ in range(3):
        limiter.try_acquire("v")
    fake.now += 61
    assert limiter.try_acquire("v") is None


def test_daily_limit_blocks_even_after_a_minute():
    limiter, fake = make(per_minute=3, per_day=5)
    for _ in range(5):
        assert limiter.try_acquire("v") is None
        fake.now += 61
    assert limiter.try_acquire("v") == LIMIT_DAY


def test_new_day_resets_the_count():
    limiter, fake = make(per_minute=10, per_day=2)
    limiter.try_acquire("v")
    limiter.try_acquire("v")
    assert limiter.try_acquire("v") == LIMIT_DAY
    fake.day = date(2026, 11, 18)
    assert limiter.try_acquire("v") is None


def test_visitors_are_independent():
    limiter, _ = make(per_minute=1)
    assert limiter.try_acquire("a") is None
    assert limiter.try_acquire("a") == LIMIT_MINUTE
    assert limiter.try_acquire("b") is None


def test_blocked_attempt_is_not_recorded():
    limiter, _ = make(per_minute=1, per_day=5)
    limiter.try_acquire("v")
    assert limiter.try_acquire("v") == LIMIT_MINUTE
    assert limiter.remaining_today("v") == 4


def test_refund_gives_back_one_action():
    limiter, _ = make(per_minute=1, per_day=5)
    assert limiter.try_acquire("v") is None
    limiter.refund("v")
    assert limiter.try_acquire("v") is None  # the minute slot came back too
    assert limiter.remaining_today("v") == 4


def test_refund_is_safe_for_unknown_visitors_and_never_goes_negative():
    limiter, _ = make(per_day=5)
    limiter.refund("nobody")
    limiter.try_acquire("v")
    limiter.refund("v")
    limiter.refund("v")
    assert limiter.remaining_today("v") == 5


def test_remaining_today_for_a_new_visitor():
    limiter, _ = make(per_day=5)
    assert limiter.remaining_today("new") == 5


@pytest.mark.parametrize("bad_id", ["", "   ", None, 5])
def test_rejects_bad_visitor_ids(bad_id):
    limiter, _ = make()
    with pytest.raises(ValueError):
        limiter.try_acquire(bad_id)


def test_long_visitor_ids_are_truncated():
    limiter, _ = make(per_minute=1)
    assert limiter.try_acquire("a" * 200 + "x") is None
    assert limiter.try_acquire("a" * 200 + "y") == LIMIT_MINUTE  # same first 200 chars


@pytest.mark.parametrize("per_minute, per_day", [(0, 5), (3, 0), (True, 5), ("3", 5), (3.0, 5)])
def test_rejects_bad_settings(per_minute, per_day):
    with pytest.raises(ValueError):
        VisitorLimiter(per_minute, per_day)


def test_memory_is_capped():
    limiter, fake = make(max_visitors=3)
    for i in range(5):
        limiter.try_acquire(f"v{i}")
        fake.now += 1
    assert len(limiter._visitors) == 3
    assert "v0" not in limiter._visitors  # least recent visitor was dropped


def test_visitors_from_old_days_are_evicted_first():
    limiter, fake = make(max_visitors=2)
    limiter.try_acquire("old")
    fake.day = date(2026, 11, 18)
    limiter.try_acquire("a")
    limiter.try_acquire("b")
    assert set(limiter._visitors) == {"a", "b"}


def test_demo_limits():
    assert (question_limiter.per_day, upload_limiter.per_day) == (5, 3)
    assert (MAX_QUESTIONS_PER_DAY, MAX_UPLOADS_PER_DAY) == (5, 3)
    # No separate per-minute limit for the demo: it equals the daily total
    assert MAX_QUESTIONS_PER_MINUTE == MAX_QUESTIONS_PER_DAY
    assert MAX_UPLOADS_PER_MINUTE == MAX_UPLOADS_PER_DAY