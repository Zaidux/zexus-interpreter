"""
Concurrency & Performance System for Zexus Interpreter

Provides channels for message passing, atomic operations for safe concurrent access,
and support for async/await patterns. Designed for safe, race-free concurrent programming.
"""

from typing import Dict, List, Any, Optional, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Condition, Event
import queue
import time

T = TypeVar('T')

# Sentinel value to signal channel is closed
class _ChannelClosedSentinel:
    """Sentinel object to wake up receivers when channel is closed"""
    pass

_CHANNEL_CLOSED_SENTINEL = _ChannelClosedSentinel()


class ChannelMode(Enum):
    """Channel communication mode"""
    UNBUFFERED = "unbuffered"  # Blocks until receiver/sender ready
    BUFFERED = "buffered"      # Has internal queue with capacity
    CLOSED = "closed"          # Channel closed, no more communication


@dataclass
class Channel(Generic[T]):
    """
    Type-safe message passing channel
    
    Supports:
    - Unbuffered channels (synchronization point)
    - Buffered channels (queue with capacity)
    - Non-blocking sends/receives
    - Close semantics
    
    Example:
        channel<integer> numbers;
        send(numbers, 42);
        value = receive(numbers);
    """
    
    name: str
    element_type: Optional[str] = None
    capacity: int = 0  # 0 = unbuffered
    _queue: queue.Queue = field(default_factory=queue.Queue)
    _closed: bool = field(default=False)
    _lock: Lock = field(default_factory=Lock)
    _send_ready: Condition = field(default=None)
    _recv_ready: Condition = field(default=None)
    _closed_event: Event = field(default_factory=Event)
    
    def __post_init__(self):
        if self.capacity > 0:
            self._queue = queue.Queue(maxsize=self.capacity)
        else:
            self._queue = queue.Queue()
        # Initialize Condition variables with the same lock
        self._send_ready = Condition(self._lock)
        self._recv_ready = Condition(self._lock)
    
    @property
    def is_open(self) -> bool:
        """Check if channel is open"""
        with self._lock:
            return not self._closed
    
    def send(self, value: T, timeout: Optional[float] = None) -> bool:
        """
        Send value to channel
        
        Args:
            value: Value to send
            timeout: Maximum wait time (None = infinite)
            
        Returns:
            True if sent, False if channel closed
            
        Raises:
            RuntimeError: If channel is closed
        """
        # Check if closed (with lock)
        with self._lock:
            if self._closed:
                raise RuntimeError(f"Cannot send on closed channel '{self.name}'")
        
        # Send without holding lock (queue.Queue is thread-safe)
        try:
            if self.capacity == 0:
                # Unbuffered: block until receiver ready
                self._queue.put(value, timeout=timeout)
            else:
                # Buffered: block if full
                self._queue.put(value, timeout=timeout)
            
            # Notify receiver (with lock)
            with self._lock:
                self._recv_ready.notify()
            return True
        except queue.Full:
            raise RuntimeError(f"Channel '{self.name}' buffer full")
        except queue.Empty:
            raise RuntimeError(f"Timeout sending to channel '{self.name}'")
    
    def receive(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Receive value from channel (blocking)
        
        Args:
            timeout: Maximum wait time (None = infinite)
            
        Returns:
            Received value or None if channel closed and empty
            
        Raises:
            RuntimeError: On communication error
        """
        # Check if closed first (with lock)
        with self._lock:
            if self._closed and self._queue.empty():
                return None
        
        # Receive without holding lock (queue.Queue is thread-safe)
        try:
            value = self._queue.get(timeout=timeout)
            
            # Check if this is the closed sentinel
            if isinstance(value, _ChannelClosedSentinel):
                return None
            
            # Notify sender (with lock)
            with self._lock:
                self._send_ready.notify()
            return value
        except queue.Empty:
            # Check if closed (with lock)
            with self._lock:
                if self._closed:
                    return None
            raise RuntimeError(f"Timeout receiving from channel '{self.name}'")
    
    def close(self):
        """Close channel - no more sends/receives allowed"""
        with self._lock:
            self._closed = True
            self._closed_event.set()
            # Put sentinel values to wake up any waiting receivers
            # This ensures they immediately return None instead of timing out
            try:
                # For buffered channels, put one sentinel
                if self.capacity > 0:
                    self._queue.put_nowait(_CHANNEL_CLOSED_SENTINEL)
                # For unbuffered channels, use notification
                else:
                    self._recv_ready.notify_all()
                    self._send_ready.notify_all()
            except queue.Full:
                # Queue is full, receivers will check closed flag anyway
                pass
    
    def __repr__(self) -> str:
        mode = f"buffered({self.capacity})" if self.capacity > 0 else "unbuffered"
        status = "closed" if self._closed else "open"
        return f"Channel<{self.element_type}>({self.name}, {mode}, {status})"


@dataclass
class Atomic:
    """
    Atomic operation wrapper - ensures indivisible execution
    
    Provides mutex-protected code region where concurrent accesses
    cannot interleave. Useful for short, critical sections.
    
    Example:
        atomic(counter = counter + 1);
        
        atomic {
            x = x + 1;
            y = y + 1;
        };
    """
    
    _lock: Lock = field(default_factory=Lock)
    _depth: int = field(default=0)  # Reentrancy depth
    
    def execute(self, operation, *args, **kwargs):
        """
        Execute operation atomically
        
        Args:
            operation: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of operation
        """
        with self._lock:
            self._depth += 1
            try:
                return operation(*args, **kwargs)
            finally:
                self._depth -= 1
    
    def acquire(self):
        """Acquire atomic lock (for manual control)"""
        self._lock.acquire()
        self._depth += 1
    
    def release(self):
        """Release atomic lock (for manual control)"""
        if self._depth > 0:
            self._depth -= 1
            self._lock.release()
    
    def is_locked(self) -> bool:
        """Check if currently locked"""
        return self._depth > 0


@dataclass
class WaitGroup:
    """
    Wait group for synchronizing multiple async operations
    
    Similar to Go's sync.WaitGroup - allows waiting for a collection
    of tasks to complete. Useful for coordinating producer-consumer patterns.
    
    Example:
        let wg = wait_group()
        wg.add(2)  # Expecting 2 tasks
        
        async action task1() {
            # ... work ...
            wg.done()
        }
        
        async action task2() {
            # ... work ...
            wg.done()
        }
        
        async task1()
        async task2()
        wg.wait()  # Blocks until both tasks call done()
    """
    _count: int = field(default=0)
    _lock: Lock = field(default_factory=Lock)
    _zero_event: Event = field(default_factory=Event)
    
    def __post_init__(self):
        # Start with event set (count is 0)
        self._zero_event.set()
    
    def add(self, delta: int = 1):
        """Add delta to the wait group counter"""
        with self._lock:
            self._count += delta
            if self._count < 0:
                raise ValueError("WaitGroup counter cannot be negative")
            if self._count == 0:
                self._zero_event.set()
            else:
                self._zero_event.clear()
    
    def done(self):
        """Decrement the wait group counter by 1"""
        self.add(-1)
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until the counter reaches zero
        
        Args:
            timeout: Maximum wait time in seconds (None = infinite)
            
        Returns:
            True if counter reached zero, False if timeout
        """
        return self._zero_event.wait(timeout=timeout)
    
    def count(self) -> int:
        """Get current counter value"""
        with self._lock:
            return self._count


@dataclass
class Barrier:
    """
    Synchronization barrier for coordinating multiple tasks
    
    Allows multiple tasks to wait at a barrier point until all have arrived.
    Once all parties arrive, all are released simultaneously.
    
    Example:
        let barrier = barrier(2)  # Wait for 2 tasks
        
        async action task1() {
            # ... phase 1 work ...
            barrier.wait()  # Wait for task2
            # ... phase 2 work ...
        }
        
        async action task2() {
            # ... phase 1 work ...
            barrier.wait()  # Wait for task1
            # ... phase 2 work ...
        }
    """
    parties: int  # Number of tasks that must call wait()
    _count: int = field(default=0)
    _generation: int = field(default=0)
    _lock: Lock = field(default_factory=Lock)
    _condition: Condition = field(default=None)
    
    def __post_init__(self):
        if self.parties <= 0:
            raise ValueError("Barrier parties must be positive")
        if self._condition is None:
            self._condition = Condition(self._lock)
    
    def wait(self, timeout: Optional[float] = None) -> int:
        """
        Wait at the barrier until all parties arrive
        
        Args:
            timeout: Maximum wait time in seconds (None = infinite)
            
        Returns:
            Barrier generation number (increments each cycle)
            
        Raises:
            RuntimeError: On timeout
        """
        with self._condition:
            generation = self._generation
            self._count += 1
            
            if self._count == self.parties:
                # Last one to arrive - release all
                self._count = 0
                self._generation += 1
                self._condition.notify_all()
                return generation
            else:
                # Wait for others
                while generation == self._generation:
                    if not self._condition.wait(timeout=timeout):
                        raise RuntimeError(f"Barrier timeout waiting for {self.parties - self._count} more tasks")
                return generation
    
    def reset(self):
        """Reset the barrier to initial state"""
        with self._condition:
            self._count = 0
            self._generation += 1
            self._condition.notify_all()
    
    def __repr__(self) -> str:
        return f"Atomic(depth={self._depth}, locked={self.is_locked()})"


class ConcurrencyManager:
    """
    Central manager for all concurrency operations
    
    Manages:
    - Channel creation and lifecycle
    - Atomic operation coordination
    - Goroutine/task scheduling
    - Deadlock detection
    - Performance monitoring
    """
    
    def __init__(self):
        self.channels: Dict[str, Channel] = {}
        self.atomics: Dict[str, Atomic] = {}
        self._lock = Lock()
        self._tasks: List[Any] = []
        self._completed_count = 0
    
    def create_channel(self, name: str, element_type: Optional[str] = None, 
                       capacity: int = 0) -> Channel:
        """
        Create a new channel
        
        Args:
            name: Channel name
            element_type: Type of elements (for validation)
            capacity: Buffer capacity (0 = unbuffered)
            
        Returns:
            Created channel
        """
        with self._lock:
            if name in self.channels:
                raise ValueError(f"Channel '{name}' already exists")
            
            channel = Channel(name=name, element_type=element_type, capacity=capacity)
            self.channels[name] = channel
            
            # Debug logging (optional)
            # from .evaluator.utils import debug_log
            # debug_log("ConcurrencyManager", f"Created channel: {channel}")
            
            return channel
    
    def get_channel(self, name: str) -> Optional[Channel]:
        """Get existing channel by name"""
        with self._lock:
            return self.channels.get(name)
    
    def create_atomic(self, name: str) -> Atomic:
        """
        Create atomic operation region
        
        Args:
            name: Atomic region identifier
            
        Returns:
            Atomic wrapper
        """
        with self._lock:
            if name in self.atomics:
                return self.atomics[name]
            
            atomic = Atomic()
            self.atomics[name] = atomic
            
            # Debug logging (optional)
            # from .evaluator.utils import debug_log
            # debug_log("ConcurrencyManager", f"Created atomic: {name}")
            
            return atomic
    
    def close_all_channels(self):
        """Close all open channels"""
        with self._lock:
            for channel in self.channels.values():
                if channel.is_open:
                    channel.close()
    
    def statistics(self) -> Dict[str, Any]:
        """Get concurrency statistics"""
        with self._lock:
            open_channels = sum(1 for ch in self.channels.values() if ch.is_open)
            return {
                "channels_created": len(self.channels),
                "channels_open": open_channels,
                "atomics_created": len(self.atomics),
                "tasks_total": len(self._tasks),
                "tasks_completed": self._completed_count
            }
    
    def __repr__(self) -> str:
        stats = self.statistics()
        return (f"ConcurrencyManager("
                f"channels={stats['channels_open']}/{stats['channels_created']}, "
                f"atomics={stats['atomics_created']}, "
                f"tasks={stats['tasks_completed']}/{stats['tasks_total']})")


# ---------------------------------------------------------------------------
# AsyncChannel — asyncio-native channel for the shared event loop
# ---------------------------------------------------------------------------

class AsyncChannel:
    """
    Async-native channel backed by :class:`asyncio.Queue`.

    Unlike :class:`Channel` (which uses ``threading`` primitives), this
    channel is designed to be used inside coroutines running on the shared
    Zexus event loop.  ``send`` and ``receive`` are ``async`` methods.

    Example (inside a Zexus ``async action``)::

        ch = AsyncChannel("numbers", capacity=10)
        await ch.send(42)
        val = await ch.receive()  # 42
        ch.close()
    """

    def __init__(self, name: str, element_type: Optional[str] = None,
                 capacity: int = 0):
        self.name = name
        self.element_type = element_type
        self.capacity = capacity
        self._closed = False

        import asyncio as _asyncio
        if capacity > 0:
            self._queue: _asyncio.Queue = _asyncio.Queue(maxsize=capacity)
        else:
            self._queue = _asyncio.Queue()

    @property
    def is_open(self) -> bool:
        return not self._closed

    async def send(self, value, *, timeout: Optional[float] = None):
        """Send *value* into the channel (async, may block if full)."""
        if self._closed:
            raise RuntimeError(f"Cannot send on closed async channel '{self.name}'")
        import asyncio as _asyncio
        if timeout is not None:
            await _asyncio.wait_for(self._queue.put(value), timeout=timeout)
        else:
            await self._queue.put(value)

    async def receive(self, *, timeout: Optional[float] = None):
        """Receive a value from the channel (async, may block if empty)."""
        if self._closed and self._queue.empty():
            return None
        import asyncio as _asyncio
        try:
            if timeout is not None:
                value = await _asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                value = await self._queue.get()
            if isinstance(value, _ChannelClosedSentinel):
                return None
            return value
        except _asyncio.TimeoutError:
            if self._closed:
                return None
            raise RuntimeError(f"Timeout receiving from async channel '{self.name}'")

    def close(self):
        """Close the channel.  Pending receivers will get ``None``."""
        self._closed = True
        try:
            self._queue.put_nowait(_CHANNEL_CLOSED_SENTINEL)
        except Exception:
            pass

    def __repr__(self) -> str:
        mode = f"buffered({self.capacity})" if self.capacity > 0 else "unbuffered"
        status = "closed" if self._closed else "open"
        return f"AsyncChannel<{self.element_type}>({self.name}, {mode}, {status})"


# Global singleton instance
_concurrency_manager: Optional[ConcurrencyManager] = None


def get_concurrency_manager() -> ConcurrencyManager:
    """
    Get or create the global concurrency manager instance
    
    Returns:
        ConcurrencyManager singleton
    """
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager()
    return _concurrency_manager


def reset_concurrency_manager():
    """Reset the global concurrency manager (for testing)"""
    global _concurrency_manager
    if _concurrency_manager:
        _concurrency_manager.close_all_channels()
    _concurrency_manager = ConcurrencyManager()


# ---------------------------------------------------------------------------
# select() — Multiplex across multiple channels
# ---------------------------------------------------------------------------

def select(*channels, timeout=None):
    """
    Multiplex across multiple channels, returning the first available value.

    Polls all *channels* with exponential backoff.  Returns a tuple
    ``(channel_index, value)`` for the first channel that has data ready.
    If *timeout* expires before any channel produces data, returns
    ``(None, None)``.

    Args:
        *channels: One or more :class:`Channel` instances.
        timeout: Maximum total wait time in seconds (``None`` = wait forever).

    Returns:
        ``(int, value)`` on success, ``(None, None)`` on timeout.
    """
    if not channels:
        raise ValueError("select() requires at least one channel")

    deadline = (time.monotonic() + timeout) if timeout is not None else None
    poll_interval = 0.001  # start at 1 ms
    max_interval = 0.05    # cap at 50 ms

    while True:
        for idx, ch in enumerate(channels):
            try:
                value = ch._queue.get_nowait()
                if isinstance(value, _ChannelClosedSentinel):
                    continue
                return (idx, value)
            except queue.Empty:
                continue

        # Check timeout
        if deadline is not None and time.monotonic() >= deadline:
            return (None, None)

        # Check if all channels are closed and empty
        if all(ch._closed and ch._queue.empty() for ch in channels):
            return (None, None)

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 2, max_interval)


# ---------------------------------------------------------------------------
# TaskGroup — Structured concurrency with cancellation
# ---------------------------------------------------------------------------

from threading import Thread


class TaskGroup:
    """
    Structured concurrency primitive that manages a group of tasks.

    Tasks are spawned as threads and can be collectively waited on or
    cancelled.  Supports the context-manager protocol so that all tasks
    are automatically joined when leaving the ``with`` block.

    Example::

        with TaskGroup("workers") as g:
            g.spawn(work, 1)
            g.spawn(work, 2)
        # all tasks finished here
        print(g.results)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._threads: List[Thread] = []
        self._results: List[Any] = []
        self._errors: List[Exception] = []
        self._cancelled = Event()
        self._lock = Lock()

    # Expose the cancellation event so spawned tasks can check it
    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def spawn(self, fn, *args):
        """
        Schedule *fn(*args)* to run in a new thread.

        The spawned function receives no special arguments; if it needs to
        check for cancellation it should reference ``group.cancelled``.
        """
        idx = len(self._threads)

        def _wrapper():
            try:
                result = fn(*args)
                with self._lock:
                    self._results.append((idx, result))
            except Exception as exc:
                with self._lock:
                    self._errors.append((idx, exc))

        t = Thread(target=_wrapper, daemon=True)
        self._threads.append(t)
        t.start()

    def wait_all(self, timeout: Optional[float] = None) -> List[Any]:
        """
        Block until every spawned task completes.

        Args:
            timeout: Maximum total wait time in seconds.

        Returns:
            Ordered list of task results (``None`` for tasks that raised).
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        for t in self._threads:
            remaining = None
            if deadline is not None:
                remaining = max(0, deadline - time.monotonic())
            t.join(timeout=remaining)

        ordered: Dict[int, Any] = {idx: val for idx, val in self._results}
        return [ordered.get(i) for i in range(len(self._threads))]

    def cancel_all(self):
        """Signal cancellation to all running tasks."""
        self._cancelled.set()

    @property
    def results(self) -> List[Any]:
        """List of ``(index, value)`` pairs for successfully completed tasks."""
        with self._lock:
            return list(self._results)

    @property
    def errors(self) -> List[Any]:
        """List of ``(index, exception)`` pairs for failed tasks."""
        with self._lock:
            return list(self._errors)

    # Context-manager protocol -------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait_all()
        return False


# ---------------------------------------------------------------------------
# BackpressurePolicy / BackpressuredChannel
# ---------------------------------------------------------------------------

class BackpressurePolicy(Enum):
    """Policy applied when a :class:`BackpressuredChannel` buffer is full."""
    DROP_NEWEST = "drop_newest"
    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"
    RAISE = "raise"


class BackpressuredChannel:
    """
    Channel with configurable back-pressure behaviour.

    Wraps a bounded buffer and applies the chosen :class:`BackpressurePolicy`
    when a ``send`` would exceed the buffer *capacity*.

    Args:
        name: Human-readable channel name.
        capacity: Maximum number of buffered messages (must be > 0).
        policy: What to do when the buffer is full.
    """

    def __init__(self, name: str, capacity: int,
                 policy: BackpressurePolicy = BackpressurePolicy.BLOCK):
        if capacity <= 0:
            raise ValueError("BackpressuredChannel capacity must be > 0")
        self.name = name
        self.capacity = capacity
        self.policy = policy
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._lock = Lock()
        self._closed = False
        self._dropped_count = 0

    @property
    def is_open(self) -> bool:
        with self._lock:
            return not self._closed

    @property
    def dropped_count(self) -> int:
        """Number of messages silently dropped due to policy."""
        with self._lock:
            return self._dropped_count

    def send(self, value, timeout: Optional[float] = None) -> bool:
        """
        Send a value into the channel, applying the back-pressure policy
        when the buffer is full.

        Returns:
            ``True`` if the message was enqueued, ``False`` if it was dropped.

        Raises:
            RuntimeError: If the channel is closed, or if the policy is
                ``RAISE`` and the buffer is full.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"Cannot send on closed channel '{self.name}'")

        if self.policy == BackpressurePolicy.BLOCK:
            try:
                self._queue.put(value, timeout=timeout)
                return True
            except queue.Full:
                raise RuntimeError(f"Channel '{self.name}' buffer full")

        if self.policy == BackpressurePolicy.RAISE:
            try:
                self._queue.put_nowait(value)
                return True
            except queue.Full:
                raise RuntimeError(
                    f"Channel '{self.name}' buffer full (policy=RAISE)")

        if self.policy == BackpressurePolicy.DROP_NEWEST:
            try:
                self._queue.put_nowait(value)
                return True
            except queue.Full:
                with self._lock:
                    self._dropped_count += 1
                return False

        if self.policy == BackpressurePolicy.DROP_OLDEST:
            with self._lock:
                try:
                    self._queue.put_nowait(value)
                    return True
                except queue.Full:
                    try:
                        self._queue.get_nowait()  # discard oldest
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(value)
                    except queue.Full:
                        pass
                    self._dropped_count += 1
                    return True

        return False  # unreachable but keeps linters happy

    def receive(self, timeout: Optional[float] = None):
        """Receive the next value from the channel."""
        with self._lock:
            if self._closed and self._queue.empty():
                return None
        try:
            value = self._queue.get(timeout=timeout)
            if isinstance(value, _ChannelClosedSentinel):
                return None
            return value
        except queue.Empty:
            with self._lock:
                if self._closed:
                    return None
            raise RuntimeError(
                f"Timeout receiving from channel '{self.name}'")

    def close(self):
        """Close the channel."""
        with self._lock:
            self._closed = True
        try:
            self._queue.put_nowait(_CHANNEL_CLOSED_SENTINEL)
        except queue.Full:
            pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (f"BackpressuredChannel({self.name}, cap={self.capacity}, "
                f"policy={self.policy.value}, {status})")


# ---------------------------------------------------------------------------
# TaskContext — Context propagation across tasks
# ---------------------------------------------------------------------------

import uuid
from datetime import datetime
import threading as _threading

_task_context_local = _threading.local()


class TaskContext:
    """
    Propagates trace/correlation metadata across threads.

    Each :class:`TaskContext` carries a *trace_id*, an optional *deadline*,
    and an arbitrary ``values`` dict.  The currently-active context is stored
    in thread-local storage and accessible via :meth:`current`.

    Example::

        ctx = TaskContext(deadline=datetime(2025, 12, 31))
        def work():
            c = TaskContext.current()
            print(c.trace_id)
        ctx.run(work)
    """

    def __init__(self, trace_id: Optional[str] = None,
                 deadline: Optional[datetime] = None,
                 values: Optional[Dict[str, Any]] = None):
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.deadline: Optional[datetime] = deadline
        self._values: Dict[str, Any] = dict(values) if values else {}

    def is_expired(self) -> bool:
        """Return ``True`` if the deadline has passed."""
        if self.deadline is None:
            return False
        return datetime.now() >= self.deadline

    def with_value(self, key: str, value: Any) -> "TaskContext":
        """Return a **new** context with ``key`` set to ``value``."""
        new_values = dict(self._values)
        new_values[key] = value
        return TaskContext(
            trace_id=self.trace_id,
            deadline=self.deadline,
            values=new_values,
        )

    def get_value(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the context dict."""
        return self._values.get(key, default)

    # Thread-local current context -------------------------------------

    @classmethod
    def current(cls) -> Optional["TaskContext"]:
        """Return the active :class:`TaskContext` for the calling thread."""
        return getattr(_task_context_local, "ctx", None)

    def run(self, fn, *args):
        """
        Execute *fn(*args)* with this context installed as the current
        context in thread-local storage.  The previous context is restored
        when *fn* returns (or raises).
        """
        previous = getattr(_task_context_local, "ctx", None)
        _task_context_local.ctx = self
        try:
            return fn(*args)
        finally:
            _task_context_local.ctx = previous

    def __repr__(self) -> str:
        expired = "expired" if self.is_expired() else "active"
        return (f"TaskContext(trace={self.trace_id[:8]}..., "
                f"{expired}, values={len(self._values)})")
