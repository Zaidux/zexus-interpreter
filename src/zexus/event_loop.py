"""
Zexus Shared Event Loop

Provides a single, persistent asyncio event loop running on a dedicated
background thread.  Every async operation in the evaluator and VM submits
work to this loop instead of creating throw-away loops with asyncio.run().

Benefits:
  - Tasks can communicate via shared state / asyncio primitives
  - asyncio.gather, asyncio.wait, Semaphore, etc. all work correctly
  - No overhead of creating + tearing down a loop per await
  - True cooperative concurrency between Zexus async actions

Usage:
    from zexus.event_loop import get_event_loop, submit, spawn, shutdown

    # Submit a coroutine and block for the result
    result = submit(some_coro())

    # Fire-and-forget
    spawn(background_work())

    # Graceful shutdown (called once at interpreter exit)
    shutdown()
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()
_started = threading.Event()


def _loop_thread_main() -> None:
    """Entry-point for the background thread that owns the event loop."""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _started.set()
    _loop.run_forever()
    # After run_forever returns (shutdown was called) — clean up pending tasks
    pending = asyncio.all_tasks(_loop)
    if pending:
        _loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    _loop.run_until_complete(_loop.shutdown_asyncgens())
    _loop.close()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start the background loop thread if it hasn't been started yet."""
    global _thread
    if _started.is_set() and _loop is not None and _loop.is_running():
        return _loop
    with _lock:
        if _started.is_set() and _loop is not None and _loop.is_running():
            return _loop
        _thread = threading.Thread(
            target=_loop_thread_main,
            name="zexus-event-loop",
            daemon=True,
        )
        _thread.start()
        _started.wait()  # block until the loop is actually running
    assert _loop is not None
    return _loop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the shared Zexus event loop (starting it lazily if needed)."""
    return _ensure_loop()


def submit(coro: Coroutine[Any, Any, T], *, timeout: Optional[float] = None) -> T:
    """Submit *coro* to the shared loop and **block** until the result is ready.

    This is safe to call from any non-event-loop thread (including the main
    interpreter thread).  If you call it from inside the event loop thread
    you'll deadlock — use ``await`` directly in that case.

    Parameters
    ----------
    coro : coroutine
        The coroutine to run.
    timeout : float | None
        Maximum seconds to wait.  ``None`` = wait forever.

    Returns
    -------
    The coroutine's return value.

    Raises
    ------
    Any exception the coroutine raises.
    TimeoutError if *timeout* expires.
    """
    loop = _ensure_loop()
    future: ConcurrentFuture[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def submit_async(coro: Coroutine[Any, Any, T]) -> ConcurrentFuture[T]:
    """Submit *coro* and return a :class:`concurrent.futures.Future`.

    The caller can poll / wait on the future at their leisure.
    """
    loop = _ensure_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Fire-and-forget: schedule *coro* as a task on the shared loop.

    Returns the :class:`asyncio.Task` handle (which lives on the shared
    loop's thread — do **not** await it from the main thread; use
    :func:`submit` instead if you need the result).
    """
    loop = _ensure_loop()

    # asyncio.run_coroutine_threadsafe returns a Future, but we want the Task
    # object so callers can cancel / inspect it.  We wrap via call_soon_threadsafe.
    task_holder: list[asyncio.Task] = []
    ready = threading.Event()

    def _create_task() -> None:
        task = loop.create_task(coro)
        task_holder.append(task)
        ready.set()

    loop.call_soon_threadsafe(_create_task)
    ready.wait()
    return task_holder[0]


def call_soon(callback, *args) -> None:
    """Schedule a plain callback on the shared loop (thread-safe)."""
    loop = _ensure_loop()
    loop.call_soon_threadsafe(callback, *args)


def is_loop_thread() -> bool:
    """Return True if the caller is on the event-loop thread."""
    return _thread is not None and threading.current_thread() is _thread


def task_count() -> int:
    """Return the number of currently pending tasks on the shared loop."""
    if _loop is None or not _loop.is_running():
        return 0
    return len(asyncio.all_tasks(_loop))


def shutdown(timeout: float = 5.0) -> None:
    """Stop the shared event loop and join the background thread.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _loop, _thread
    with _lock:
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
        if _thread is not None and _thread.is_alive():
            _thread.join(timeout=timeout)
        _loop = None
        _thread = None
        _started.clear()


# Register shutdown at interpreter exit so the background thread doesn't
# prevent a clean exit.
atexit.register(shutdown)
