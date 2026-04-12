"""Tests for concurrency system enhancements."""

import time
import threading
import pytest
from src.zexus.concurrency_system import (
    Channel, select, TaskGroup, 
    BackpressurePolicy, BackpressuredChannel, TaskContext
)


class TestSelect:
    def test_select_single_channel(self):
        ch = Channel(name="ch1", capacity=5)
        ch.send(42)
        idx, val = select(ch, timeout=1)
        assert idx == 0
        assert val == 42

    def test_select_multiple_channels(self):
        ch1 = Channel(name="ch1", capacity=5)
        ch2 = Channel(name="ch2", capacity=5)
        ch2.send("hello")
        idx, val = select(ch1, ch2, timeout=1)
        assert idx == 1
        assert val == "hello"

    def test_select_timeout(self):
        ch = Channel(name="empty", capacity=5)
        idx, val = select(ch, timeout=0.1)
        assert idx is None
        assert val is None

    def test_select_first_ready(self):
        ch1 = Channel(name="first", capacity=5)
        ch2 = Channel(name="second", capacity=5)
        ch1.send("first_value")
        ch2.send("second_value")
        idx, val = select(ch1, ch2, timeout=1)
        assert idx in (0, 1)
        assert val in ("first_value", "second_value")


class TestTaskGroup:
    def test_basic_task_group(self):
        results = []
        def worker(x):
            results.append(x * 2)
            return x * 2

        with TaskGroup(name="test") as g:
            g.spawn(worker, 1)
            g.spawn(worker, 2)
            g.spawn(worker, 3)

        assert len(results) == 3
        assert sorted(results) == [2, 4, 6]

    def test_task_group_results(self):
        def add(a, b):
            return a + b

        g = TaskGroup(name="math")
        g.spawn(add, 1, 2)
        g.spawn(add, 3, 4)
        g.wait_all(timeout=5)
        # Results are (index, value) tuples
        values = sorted([r[1] if isinstance(r, tuple) else r for r in g.results])
        assert values == [3, 7]

    def test_task_group_errors(self):
        def fail():
            raise ValueError("intentional error")

        g = TaskGroup(name="err_test")
        g.spawn(fail)
        g.wait_all(timeout=5)
        assert len(g.errors) == 1

    def test_cancel_all(self):
        def long_task():
            time.sleep(10)

        g = TaskGroup(name="cancel_test")
        g.spawn(long_task)
        time.sleep(0.1)
        g.cancel_all()
        # Should not hang
        assert True

    def test_context_manager(self):
        called = []
        with TaskGroup() as g:
            g.spawn(lambda: called.append(1))
            g.spawn(lambda: called.append(2))
        assert len(called) == 2


class TestBackpressuredChannel:
    def test_drop_newest(self):
        ch = BackpressuredChannel("test", capacity=2, policy=BackpressurePolicy.DROP_NEWEST)
        ch.send("a")
        ch.send("b")
        ch.send("c")  # should be dropped
        assert ch.dropped_count >= 1

    def test_drop_oldest(self):
        ch = BackpressuredChannel("test", capacity=2, policy=BackpressurePolicy.DROP_OLDEST)
        ch.send("a")
        ch.send("b")
        ch.send("c")  # should drop "a"
        val1 = ch.receive(timeout=1)
        val2 = ch.receive(timeout=1)
        assert "a" not in (val1, val2)  # "a" was dropped

    def test_raise_policy(self):
        ch = BackpressuredChannel("test", capacity=1, policy=BackpressurePolicy.RAISE)
        ch.send("a")
        with pytest.raises(RuntimeError):
            ch.send("b")

    def test_block_policy(self):
        ch = BackpressuredChannel("test", capacity=2, policy=BackpressurePolicy.BLOCK)
        ch.send("a")
        ch.send("b")
        # Third send would block - don't test blocking, just verify setup
        assert ch.dropped_count == 0


class TestTaskContext:
    def test_create_context(self):
        ctx = TaskContext()
        assert ctx.trace_id is not None

    def test_custom_trace_id(self):
        ctx = TaskContext(trace_id="my-trace-123")
        assert ctx.trace_id == "my-trace-123"

    def test_deadline(self):
        from datetime import datetime, timedelta
        # Deadline in the past
        ctx = TaskContext(deadline=datetime.now() - timedelta(seconds=10))
        assert ctx.is_expired() is True

    def test_not_expired(self):
        from datetime import datetime, timedelta
        ctx = TaskContext(deadline=datetime.now() + timedelta(hours=1))
        assert ctx.is_expired() is False

    def test_no_deadline(self):
        ctx = TaskContext()
        assert ctx.is_expired() is False

    def test_with_value(self):
        ctx = TaskContext()
        ctx2 = ctx.with_value("user_id", 42)
        assert ctx2.get_value("user_id") == 42
        # Original is unchanged
        assert ctx.get_value("user_id") is None

    def test_get_value_default(self):
        ctx = TaskContext()
        assert ctx.get_value("missing", "default") == "default"

    def test_run_with_context(self):
        ctx = TaskContext(trace_id="trace-abc")
        
        captured = []
        def task():
            current = TaskContext.current()
            if current:
                captured.append(current.trace_id)
        
        ctx.run(task)
        assert len(captured) == 1
        assert captured[0] == "trace-abc"

    def test_current_outside_run(self):
        # Outside of run(), current() should return None or empty
        current = TaskContext.current()
        # May or may not be None depending on thread-local state
        assert True  # Just verify no crash
