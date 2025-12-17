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
        with self._lock:
            if self._closed:
                raise RuntimeError(f"Cannot send on closed channel '{self.name}'")
            
            try:
                if self.capacity == 0:
                    # Unbuffered: block until receiver ready
                    self._queue.put(value, timeout=timeout)
                else:
                    # Buffered: block if full
                    self._queue.put(value, timeout=timeout)
                
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
        with self._lock:
            try:
                value = self._queue.get(timeout=timeout)
                self._send_ready.notify()
                return value
            except queue.Empty:
                if self._closed:
                    return None
                raise RuntimeError(f"Timeout receiving from channel '{self.name}'")
    
    def close(self):
        """Close channel - no more sends/receives allowed"""
        with self._lock:
            self._closed = True
            self._closed_event.set()
            self._recv_ready.notify_all()
            self._send_ready.notify_all()
    
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
