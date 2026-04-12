"""Message queue abstraction module for Zexus standard library."""

import queue
import collections
import itertools
from typing import Any, Dict, List, Optional, Tuple


class QueueModule:
    """Provides message queue, priority queue, deque, and pub/sub operations."""

    # ── FIFO Queue ──────────────────────────────────────────────────────

    @staticmethod
    def create(maxsize: int = 0) -> queue.Queue:
        """Create a new FIFO queue.

        Args:
            maxsize: Maximum number of items (0 for unlimited).

        Returns:
            A new FIFO queue instance.
        """
        return queue.Queue(maxsize=maxsize)

    @staticmethod
    def push(q: queue.Queue, item: Any) -> None:
        """Push an item onto the FIFO queue.

        Args:
            q: The queue to push to.
            item: The item to add.
        """
        q.put(item)

    @staticmethod
    def pop(q: queue.Queue, timeout: Optional[float] = None) -> Any:
        """Pop an item from the FIFO queue.

        Args:
            q: The queue to pop from.
            timeout: Seconds to wait before raising queue.Empty (None for blocking).

        Returns:
            The next item from the queue.

        Raises:
            queue.Empty: If the queue is empty and timeout expires.
        """
        return q.get(timeout=timeout)

    @staticmethod
    def peek(q: queue.Queue) -> Any:
        """Peek at the next item without removing it.

        Args:
            q: The queue to peek into.

        Returns:
            The next item in the queue.

        Raises:
            queue.Empty: If the queue is empty.
        """
        with q.mutex:
            if q.queue:
                return q.queue[0]
            raise queue.Empty("Queue is empty")

    @staticmethod
    def size(q: queue.Queue) -> int:
        """Return the number of items in the queue.

        Args:
            q: The queue to check.

        Returns:
            The current queue size.
        """
        return q.qsize()

    @staticmethod
    def is_empty(q: queue.Queue) -> bool:
        """Check whether the queue is empty.

        Args:
            q: The queue to check.

        Returns:
            True if the queue has no items.
        """
        return q.empty()

    @staticmethod
    def clear(q: queue.Queue) -> None:
        """Remove all items from the queue.

        Args:
            q: The queue to clear.
        """
        with q.mutex:
            q.queue.clear()
            q.unfinished_tasks = 0
            q.all_tasks_done.notify_all()
            q.not_full.notify_all()

    # ── Priority Queue ──────────────────────────────────────────────────

    _priority_counter = itertools.count()

    @staticmethod
    def create_priority(maxsize: int = 0) -> queue.PriorityQueue:
        """Create a new priority queue (lowest priority number comes first).

        Args:
            maxsize: Maximum number of items (0 for unlimited).

        Returns:
            A new priority queue instance.
        """
        return queue.PriorityQueue(maxsize=maxsize)

    @staticmethod
    def push_priority(q: queue.PriorityQueue, item: Any, priority: int) -> None:
        """Push an item onto the priority queue.

        Args:
            q: The priority queue.
            item: The item to add.
            priority: Numeric priority (lower values are dequeued first).
        """
        q.put((priority, next(QueueModule._priority_counter), item))

    @staticmethod
    def pop_priority(q: queue.PriorityQueue, timeout: Optional[float] = None) -> Any:
        """Pop the highest-priority (lowest number) item.

        Args:
            q: The priority queue.
            timeout: Seconds to wait before raising queue.Empty (None for blocking).

        Returns:
            The item with the lowest priority number.

        Raises:
            queue.Empty: If the queue is empty and timeout expires.
        """
        _priority, _seq, item = q.get(timeout=timeout)
        return item

    # ── Deque (double-ended queue) ──────────────────────────────────────

    @staticmethod
    def create_deque(maxsize: int = 0) -> collections.deque:
        """Create a new double-ended queue.

        Args:
            maxsize: Maximum number of items (0 for unlimited).

        Returns:
            A new deque instance.
        """
        return collections.deque(maxlen=maxsize if maxsize > 0 else None)

    @staticmethod
    def push_front(q: collections.deque, item: Any) -> None:
        """Push an item to the front of the deque.

        Args:
            q: The deque.
            item: The item to add.
        """
        q.appendleft(item)

    @staticmethod
    def push_back(q: collections.deque, item: Any) -> None:
        """Push an item to the back of the deque.

        Args:
            q: The deque.
            item: The item to add.
        """
        q.append(item)

    @staticmethod
    def pop_front(q: collections.deque) -> Any:
        """Pop an item from the front of the deque.

        Args:
            q: The deque.

        Returns:
            The front item.

        Raises:
            IndexError: If the deque is empty.
        """
        return q.popleft()

    @staticmethod
    def pop_back(q: collections.deque) -> Any:
        """Pop an item from the back of the deque.

        Args:
            q: The deque.

        Returns:
            The back item.

        Raises:
            IndexError: If the deque is empty.
        """
        return q.pop()

    # ── Topic-based Pub/Sub ─────────────────────────────────────────────

    @staticmethod
    def create_topic() -> Dict[str, List[str]]:
        """Create a new topic for pub/sub messaging.

        Returns:
            A topic dict with a subscribers list, used by subscribe/publish/unsubscribe.
        """
        return {"subscribers": []}

    @staticmethod
    def subscribe(topic: Dict[str, List[str]], handler_id: str) -> None:
        """Subscribe a handler to a topic.

        Args:
            topic: The topic (from create_topic).
            handler_id: Unique identifier for the subscriber.
        """
        if handler_id not in topic["subscribers"]:
            topic["subscribers"].append(handler_id)

    @staticmethod
    def publish(topic: Dict[str, List[str]], message: Any) -> List[Tuple[str, Any]]:
        """Publish a message to all subscribers of a topic.

        Args:
            topic: The topic (from create_topic).
            message: The message payload to deliver.

        Returns:
            A list of (handler_id, message) tuples representing deliveries.
        """
        return [(hid, message) for hid in topic["subscribers"]]

    @staticmethod
    def unsubscribe(topic: Dict[str, List[str]], handler_id: str) -> bool:
        """Remove a subscriber from a topic.

        Args:
            topic: The topic (from create_topic).
            handler_id: The subscriber identifier to remove.

        Returns:
            True if the handler was found and removed, False otherwise.
        """
        if handler_id in topic["subscribers"]:
            topic["subscribers"].remove(handler_id)
            return True
        return False


# Export functions for easy access
create = QueueModule.create
push = QueueModule.push
pop = QueueModule.pop
peek = QueueModule.peek
size = QueueModule.size
is_empty = QueueModule.is_empty
clear = QueueModule.clear
create_priority = QueueModule.create_priority
push_priority = QueueModule.push_priority
pop_priority = QueueModule.pop_priority
create_deque = QueueModule.create_deque
push_front = QueueModule.push_front
push_back = QueueModule.push_back
pop_front = QueueModule.pop_front
pop_back = QueueModule.pop_back
create_topic = QueueModule.create_topic
subscribe = QueueModule.subscribe
publish = QueueModule.publish
unsubscribe = QueueModule.unsubscribe
