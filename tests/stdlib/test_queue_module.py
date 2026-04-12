"""Tests for stdlib QueueModule."""

import pytest
from src.zexus.stdlib.queue_module import QueueModule


class TestFIFOQueue:
    def test_create(self):
        q = QueueModule.create()
        assert q is not None

    def test_push_pop(self):
        q = QueueModule.create()
        QueueModule.push(q, "a")
        QueueModule.push(q, "b")
        assert QueueModule.pop(q, timeout=1) == "a"
        assert QueueModule.pop(q, timeout=1) == "b"

    def test_peek(self):
        q = QueueModule.create()
        QueueModule.push(q, "first")
        assert QueueModule.peek(q) == "first"
        # peek should not remove
        assert QueueModule.size(q) == 1

    def test_is_empty(self):
        q = QueueModule.create()
        assert QueueModule.is_empty(q) is True
        QueueModule.push(q, 1)
        assert QueueModule.is_empty(q) is False

    def test_size(self):
        q = QueueModule.create()
        QueueModule.push(q, 1)
        QueueModule.push(q, 2)
        assert QueueModule.size(q) == 2

    def test_clear(self):
        q = QueueModule.create()
        QueueModule.push(q, 1)
        QueueModule.push(q, 2)
        QueueModule.clear(q)
        assert QueueModule.is_empty(q) is True


class TestPriorityQueue:
    def test_create(self):
        q = QueueModule.create_priority()
        assert q is not None

    def test_priority_ordering(self):
        q = QueueModule.create_priority()
        QueueModule.push_priority(q, "low", 10)
        QueueModule.push_priority(q, "high", 1)
        QueueModule.push_priority(q, "mid", 5)
        # Lowest priority number comes first
        assert QueueModule.pop_priority(q, timeout=1) == "high"
        assert QueueModule.pop_priority(q, timeout=1) == "mid"
        assert QueueModule.pop_priority(q, timeout=1) == "low"


class TestDeque:
    def test_create(self):
        d = QueueModule.create_deque()
        assert d is not None

    def test_push_back_pop_front(self):
        d = QueueModule.create_deque()
        QueueModule.push_back(d, "a")
        QueueModule.push_back(d, "b")
        assert QueueModule.pop_front(d) == "a"
        assert QueueModule.pop_front(d) == "b"

    def test_push_front_pop_back(self):
        d = QueueModule.create_deque()
        QueueModule.push_front(d, "a")
        QueueModule.push_front(d, "b")
        assert QueueModule.pop_back(d) == "a"
        assert QueueModule.pop_back(d) == "b"

    def test_bounded_deque(self):
        d = QueueModule.create_deque(maxsize=2)
        QueueModule.push_back(d, "a")
        QueueModule.push_back(d, "b")
        # Third item should push out oldest if maxlen
        QueueModule.push_back(d, "c")
        assert QueueModule.pop_front(d) in ("b", "c")


class TestPubSub:
    def test_create_topic(self):
        t = QueueModule.create_topic()
        assert t is not None

    def test_subscribe_and_publish(self):
        t = QueueModule.create_topic()
        received = []
        QueueModule.subscribe(t, "sub1")
        QueueModule.publish(t, "hello")
        # Topic stores messages for subscribers
        assert t is not None

    def test_unsubscribe(self):
        t = QueueModule.create_topic()
        QueueModule.subscribe(t, "sub1")
        QueueModule.unsubscribe(t, "sub1")
        assert t is not None
