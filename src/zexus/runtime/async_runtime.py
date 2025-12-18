"""
Async Runtime System for Zexus
Provides EventLoop, Task management, and async execution coordination
"""

import threading
import queue
import time
from collections import deque
from typing import Any, Callable, Optional, List, Dict


class Task:
    """
    Represents an async task in the event loop
    Wraps a coroutine with metadata and state management
    """
    
    # Task states
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    
    _task_id_counter = 0
    _lock = threading.Lock()
    
    def __init__(self, coroutine, name=None, priority=0):
        """
        Create a new task
        
        Args:
            coroutine: The coroutine/generator to execute
            name: Optional task name for debugging
            priority: Task priority (higher = more priority)
        """
        with Task._lock:
            Task._task_id_counter += 1
            self.id = Task._task_id_counter
        
        self.coroutine = coroutine
        self.name = name or f"Task-{self.id}"
        self.priority = priority
        self.state = Task.PENDING
        self.result = None
        self.error = None
        self.cancelled = False
        self.dependencies = []  # Other tasks this task depends on
        self.dependents = []    # Tasks that depend on this task
        self.awaiting_promise = None  # Promise this task is currently waiting for
    
    def cancel(self):
        """Cancel the task"""
        if self.state in (Task.PENDING, Task.RUNNING):
            self.state = Task.CANCELLED
            self.cancelled = True
            return True
        return False
    
    def is_complete(self):
        """Check if task is complete (success, failure, or cancelled)"""
        return self.state in (Task.COMPLETED, Task.FAILED, Task.CANCELLED)
    
    def is_ready(self):
        """Check if task is ready to run (all dependencies complete)"""
        if self.cancelled:
            return False
        return all(dep.is_complete() for dep in self.dependencies)
    
    def add_dependency(self, task):
        """Add a task dependency"""
        if task not in self.dependencies:
            self.dependencies.append(task)
            task.dependents.append(self)
    
    def __lt__(self, other):
        """Compare tasks by priority (for priority queue)"""
        return self.priority > other.priority
    
    def __repr__(self):
        return f"Task({self.name}, state={self.state}, priority={self.priority})"


class EventLoop:
    """
    Event loop for managing async task execution
    Provides task scheduling, execution, and coordination
    """
    
    def __init__(self):
        """Initialize the event loop"""
        self.task_queue = deque()  # Tasks ready to run
        self.waiting_tasks = {}    # Tasks waiting for promises: {promise_id: [tasks]}
        self.all_tasks = {}        # All tasks by ID
        self.running = False
        self.current_task = None
        self.lock = threading.Lock()
    
    def create_task(self, coroutine, name=None, priority=0):
        """
        Create and schedule a new task
        
        Args:
            coroutine: The coroutine to execute
            name: Optional task name
            priority: Task priority
            
        Returns:
            Task: The created task
        """
        task = Task(coroutine, name, priority)
        with self.lock:
            self.all_tasks[task.id] = task
            self._schedule_task(task)
        return task
    
    def _schedule_task(self, task):
        """Add a task to the ready queue if it's ready to run"""
        if task.is_ready() and not task.is_complete():
            # Insert task in priority order
            inserted = False
            for i, queued_task in enumerate(self.task_queue):
                if task.priority > queued_task.priority:
                    self.task_queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self.task_queue.append(task)
    
    def _run_task_step(self, task):
        """
        Run one step of a task's coroutine
        
        Returns:
            (is_done, value): Whether task is complete and the yielded/returned value
        """
        try:
            task.state = Task.RUNNING
            self.current_task = task
            
            # Check if task has a coroutine (might be a Coroutine object)
            if hasattr(task.coroutine, 'resume'):
                # It's a Coroutine object
                is_done, value = task.coroutine.resume()
                
                if is_done:
                    task.state = Task.COMPLETED
                    task.result = value
                    return (True, value)
                else:
                    # Yielded a value (might be a Promise)
                    return (False, value)
            else:
                # It's a generator
                value = next(task.coroutine)
                return (False, value)
                
        except StopIteration as e:
            # Coroutine completed
            task.state = Task.COMPLETED
            task.result = e.value if hasattr(e, 'value') else None
            return (True, task.result)
            
        except Exception as e:
            # Coroutine error
            task.state = Task.FAILED
            task.error = e
            return (True, None)
        
        finally:
            self.current_task = None
    
    def _handle_task_yield(self, task, yielded_value):
        """
        Handle a value yielded by a task
        
        Args:
            task: The task that yielded
            yielded_value: The value that was yielded
        """
        # Check if it's a Promise
        if hasattr(yielded_value, 'type') and yielded_value.type() == "PROMISE":
            promise = yielded_value
            
            # If promise is already resolved, reschedule task immediately
            if promise.is_resolved():
                with self.lock:
                    self._schedule_task(task)
            else:
                # Add task to waiting list for this promise
                promise_id = id(promise)
                
                with self.lock:
                    if promise_id not in self.waiting_tasks:
                        self.waiting_tasks[promise_id] = []
                    
                    self.waiting_tasks[promise_id].append(task)
                    task.awaiting_promise = promise
                
                # Set up promise callback to reschedule task when resolved
                def on_promise_resolved(value):
                    with self.lock:
                        if promise_id in self.waiting_tasks:
                            waiting = self.waiting_tasks.pop(promise_id)
                            for waiting_task in waiting:
                                waiting_task.awaiting_promise = None
                                self._schedule_task(waiting_task)
                
                promise.then(on_promise_resolved)
                promise.catch(on_promise_resolved)
        else:
            # Unknown yield value, reschedule task
            with self.lock:
                self._schedule_task(task)
    
    def _complete_task(self, task):
        """Handle task completion and notify dependents"""
        with self.lock:
            # Schedule dependent tasks that are now ready
            for dependent in task.dependents:
                if dependent.is_ready():
                    self._schedule_task(dependent)
    
    def run_until_complete(self, coroutine):
        """
        Run the event loop until the given coroutine completes
        
        Args:
            coroutine: The main coroutine to execute
            
        Returns:
            The result of the coroutine
        """
        # Create main task
        main_task = self.create_task(coroutine, name="main", priority=100)
        
        self.running = True
        
        try:
            while self.running:
                # Check if main task is complete
                if main_task.is_complete():
                    if main_task.state == Task.FAILED:
                        raise main_task.error
                    return main_task.result
                
                # Get next task from queue
                with self.lock:
                    if not self.task_queue:
                        # No tasks ready - check if any tasks are waiting
                        if not self.waiting_tasks:
                            # No waiting tasks either - we're done
                            break
                        
                        # Wait a bit for promises to resolve
                        time.sleep(0.001)
                        continue
                    
                    task = self.task_queue.popleft()
                
                # Run one step of the task
                is_done, value = self._run_task_step(task)
                
                if is_done:
                    # Task completed
                    self._complete_task(task)
                else:
                    # Task yielded a value
                    self._handle_task_yield(task, value)
            
            # If we exit the loop without completing main task, check its state
            if main_task.state == Task.FAILED:
                raise main_task.error
            elif main_task.state == Task.CANCELLED:
                raise Exception("Main task was cancelled")
            
            return main_task.result
            
        finally:
            self.running = False
    
    def stop(self):
        """Stop the event loop"""
        self.running = False
    
    def get_task_count(self):
        """Get count of tasks in various states"""
        with self.lock:
            return {
                'total': len(self.all_tasks),
                'ready': len(self.task_queue),
                'waiting': sum(len(tasks) for tasks in self.waiting_tasks.values())
            }


# Global event loop instance
_global_event_loop = None
_global_event_loop_lock = threading.Lock()


def get_event_loop():
    """Get the global event loop instance"""
    global _global_event_loop
    
    with _global_event_loop_lock:
        if _global_event_loop is None:
            _global_event_loop = EventLoop()
        
        return _global_event_loop


def set_event_loop(loop):
    """Set the global event loop instance"""
    global _global_event_loop
    
    with _global_event_loop_lock:
        _global_event_loop = loop


def new_event_loop():
    """Create a new event loop"""
    return EventLoop()
