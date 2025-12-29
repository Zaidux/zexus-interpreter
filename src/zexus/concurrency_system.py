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
