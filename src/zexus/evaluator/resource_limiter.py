# src/zexus/evaluator/resource_limiter.py
"""
Resource Limiter for Zexus Interpreter

Prevents resource exhaustion attacks by enforcing limits on:
- Loop iterations (prevents infinite loops)
- Execution time (prevents DoS via slow operations)
- Call stack depth (prevents stack overflow)
- Memory usage (prevents memory exhaustion)

Security Fix #7: Resource Limits
"""

import time
import signal
import sys


class ResourceError(Exception):
    """Raised when a resource limit is exceeded"""
    pass


class TimeoutError(ResourceError):
    """Raised when execution timeout is exceeded"""
    pass


class ResourceLimiter:
    """
    Enforces resource limits during program execution.
    
    Limits:
    - max_iterations: Maximum loop iterations across all loops (default: 1,000,000)
    - timeout_seconds: Maximum execution time (default: 30 seconds)
    - max_call_depth: Maximum call stack depth (default: 1000)
    - max_memory_mb: Maximum memory usage (default: 500 MB, not enforced by default)
    
    Usage:
        limiter = ResourceLimiter(max_iterations=100000, timeout_seconds=10)
        limiter.start()  # Start timeout timer
        
        # In loops:
        limiter.check_iterations()
        
        # On function calls:
        limiter.enter_call()
        # ... function body ...
        limiter.exit_call()
        
        limiter.stop()  # Stop timeout timer
    """
    
    # Default limits
    DEFAULT_MAX_ITERATIONS = 1_000_000  # 1 million iterations
    DEFAULT_TIMEOUT_SECONDS = 30  # 30 seconds
    DEFAULT_MAX_CALL_DEPTH = 100  # 100 nested calls (Python interpreter uses many stack frames per Zexus call)
    DEFAULT_MAX_MEMORY_MB = 500  # 500 MB (not enforced by default)
    
    def __init__(self, 
                 max_iterations=None,
                 timeout_seconds=None,
                 max_call_depth=None,
                 max_memory_mb=None,
                 enable_timeout=False,
                 enable_memory_check=False):
        """
        Initialize resource limiter.
        
        Args:
            max_iterations: Maximum total loop iterations (default: 1,000,000)
            timeout_seconds: Maximum execution time (default: 30)
            max_call_depth: Maximum call stack depth (default: 1000)
            max_memory_mb: Maximum memory usage in MB (default: 500)
            enable_timeout: Enable timeout enforcement (default: False, Linux only)
            enable_memory_check: Enable memory checking (default: False, requires psutil)
        """
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self.max_call_depth = max_call_depth or self.DEFAULT_MAX_CALL_DEPTH
        self.max_memory_mb = max_memory_mb or self.DEFAULT_MAX_MEMORY_MB
        
        # Feature flags
        self.enable_timeout = enable_timeout
        self.enable_memory_check = enable_memory_check
        
        # Runtime counters
        self.iteration_count = 0
        self.call_depth = 0
        self.start_time = None
        self.timeout_handler = None
        
        # Memory checking (optional, requires psutil)
        self.psutil_available = False
        if enable_memory_check:
            try:
                import psutil
                self.psutil = psutil
                self.psutil_available = True
                self.process = psutil.Process()
            except ImportError:
                print("⚠️  Warning: psutil not available, memory checking disabled")
                self.enable_memory_check = False
    
    def start(self):
        """
        Start resource monitoring (timeout timer, etc.)
        Should be called at the beginning of script execution.
        """
        self.start_time = time.time()
        self.iteration_count = 0
        self.call_depth = 0
        
        # Set timeout handler (Linux/Unix only)
        if self.enable_timeout and hasattr(signal, 'SIGALRM'):
            self._set_timeout_alarm()
    
    def stop(self):
        """
        Stop resource monitoring and cleanup.
        Should be called at the end of script execution.
        """
        # Cancel timeout alarm
        if self.enable_timeout and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
    
    def reset(self):
        """Reset iteration counter (useful for multiple script executions)"""
        self.iteration_count = 0
        self.call_depth = 0
        self.start_time = None
    
    def check_iterations(self):
        """
        Check if iteration limit has been exceeded.
        Should be called at the beginning of each loop iteration.
        
        Raises:
            ResourceError: If iteration limit exceeded
        """
        self.iteration_count += 1
        
        if self.iteration_count > self.max_iterations:
            raise ResourceError(
                f"Iteration limit exceeded: {self.max_iterations:,} iterations\n"
                f"This prevents infinite loops and resource exhaustion.\n\n"
                f"Suggestion: Review your loop conditions or increase the limit with:\n"
                f"  zx-run --max-iterations 10000000 script.zx"
            )
    
    def check_timeout(self):
        """
        Check if execution timeout has been exceeded.
        Should be called periodically during long operations.
        
        Raises:
            TimeoutError: If timeout exceeded
        """
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"Execution timeout exceeded: {elapsed:.2f}s > {self.timeout_seconds}s\n"
                f"This prevents denial-of-service via slow operations.\n\n"
                f"Suggestion: Optimize your code or increase timeout with:\n"
                f"  zx-run --timeout 60 script.zx"
            )
    
    def check_memory(self):
        """
        Check if memory limit has been exceeded.
        Should be called periodically (e.g., every 1000 iterations).
        
        Raises:
            ResourceError: If memory limit exceeded
        """
        if not self.enable_memory_check or not self.psutil_available:
            return
        
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                raise ResourceError(
                    f"Memory limit exceeded: {memory_mb:.2f}MB > {self.max_memory_mb}MB\n"
                    f"This prevents memory exhaustion attacks.\n\n"
                    f"Suggestion: Reduce memory usage or increase limit with:\n"
                    f"  zx-run --max-memory 1000 script.zx"
                )
        except Exception as e:
            # Don't crash on memory check failure
            print(f"⚠️  Warning: Memory check failed: {e}")
    
    def enter_call(self, function_name=None):
        """
        Called when entering a function/action call.
        Tracks call depth to prevent stack overflow.
        
        Args:
            function_name: Optional name of function being called
        
        Raises:
            ResourceError: If call depth limit exceeded
        """
        self.call_depth += 1
        
        if self.call_depth > self.max_call_depth:
            func_info = f" ({function_name})" if function_name else ""
            raise ResourceError(
                f"Call depth limit exceeded: {self.max_call_depth} nested calls{func_info}\n"
                f"This prevents stack overflow from excessive recursion.\n\n"
                f"Suggestion: Review your recursion or increase limit with:\n"
                f"  zx-run --max-call-depth 5000 script.zx"
            )
    
    def exit_call(self):
        """
        Called when exiting a function/action call.
        Decrements call depth counter.
        """
        if self.call_depth > 0:
            self.call_depth -= 1
    
    def get_stats(self):
        """
        Get current resource usage statistics.
        
        Returns:
            dict: Resource usage stats
        """
        stats = {
            'iterations': self.iteration_count,
            'max_iterations': self.max_iterations,
            'iteration_percent': (self.iteration_count / self.max_iterations) * 100,
            'call_depth': self.call_depth,
            'max_call_depth': self.max_call_depth,
        }
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            stats['elapsed_seconds'] = elapsed
            stats['timeout_seconds'] = self.timeout_seconds
            stats['timeout_percent'] = (elapsed / self.timeout_seconds) * 100
        
        if self.enable_memory_check and self.psutil_available:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                stats['memory_mb'] = memory_mb
                stats['max_memory_mb'] = self.max_memory_mb
                stats['memory_percent'] = (memory_mb / self.max_memory_mb) * 100
            except:
                pass
        
        return stats
    
    def _set_timeout_alarm(self):
        """
        Set SIGALRM timeout handler (Linux/Unix only).
        This is automatically called by start() if enable_timeout is True.
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Execution timeout: {self.timeout_seconds}s exceeded\n"
                f"This prevents denial-of-service via slow operations.\n\n"
                f"Suggestion: Optimize your code or increase timeout with:\n"
                f"  zx-run --timeout 60 script.zx"
            )
        
        self.timeout_handler = timeout_handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)


# Default global limiter (can be overridden)
_default_limiter = None


def get_default_limiter():
    """Get the default global resource limiter"""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = ResourceLimiter()
    return _default_limiter


def set_default_limiter(limiter):
    """Set the default global resource limiter"""
    global _default_limiter
    _default_limiter = limiter
