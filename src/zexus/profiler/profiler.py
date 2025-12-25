"""Performance profiler for Zexus code."""

import time
import tracemalloc
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FunctionProfile:
    """Profile data for a function."""
    name: str
    calls: int = 0
    total_time: float = 0.0
    self_time: float = 0.0
    memory_allocated: int = 0
    memory_peak: int = 0


@dataclass
class ProfileReport:
    """Complete profiling report."""
    total_time: float
    total_calls: int
    memory_peak: int
    functions: Dict[str, FunctionProfile] = field(default_factory=dict)
    hotspots: List[FunctionProfile] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'total_time': self.total_time,
            'total_calls': self.total_calls,
            'memory_peak': self.memory_peak,
            'functions': {
                name: {
                    'calls': prof.calls,
                    'total_time': prof.total_time,
                    'self_time': prof.self_time,
                    'memory_allocated': prof.memory_allocated,
                    'memory_peak': prof.memory_peak
                }
                for name, prof in self.functions.items()
            },
            'hotspots': [
                {
                    'name': prof.name,
                    'calls': prof.calls,
                    'total_time': prof.total_time,
                    'self_time': prof.self_time
                }
                for prof in self.hotspots
            ]
        }


class Profiler:
    """Performance profiler for Zexus code execution."""

    def __init__(self):
        self.enabled = False
        self.start_time = 0.0
        self.functions = {}
        self.call_stack = []
        self.memory_enabled = False
        
    def start(self, enable_memory: bool = True):
        """Start profiling."""
        self.enabled = True
        self.memory_enabled = enable_memory
        self.start_time = time.time()
        self.functions = {}
        self.call_stack = []
        
        if enable_memory:
            tracemalloc.start()
    
    def stop(self) -> ProfileReport:
        """Stop profiling and generate report."""
        self.enabled = False
        total_time = time.time() - self.start_time
        
        memory_peak = 0
        if self.memory_enabled:
            _, peak = tracemalloc.get_traced_memory()
            memory_peak = peak
            tracemalloc.stop()
        
        # Calculate total calls
        total_calls = sum(prof.calls for prof in self.functions.values())
        
        # Identify hotspots (functions taking most time)
        hotspots = sorted(
            self.functions.values(),
            key=lambda p: p.total_time,
            reverse=True
        )[:10]  # Top 10 hotspots
        
        return ProfileReport(
            total_time=total_time,
            total_calls=total_calls,
            memory_peak=memory_peak,
            functions=self.functions,
            hotspots=hotspots
        )
    
    def enter_function(self, name: str):
        """Record entering a function."""
        if not self.enabled:
            return
        
        if name not in self.functions:
            self.functions[name] = FunctionProfile(name=name)
        
        profile = self.functions[name]
        profile.calls += 1
        
        # Record memory if enabled
        if self.memory_enabled:
            current, _ = tracemalloc.get_traced_memory()
            profile.memory_allocated = current
        
        # Push onto call stack
        self.call_stack.append({
            'name': name,
            'start_time': time.time(),
            'start_memory': profile.memory_allocated if self.memory_enabled else 0
        })
    
    def exit_function(self, name: str):
        """Record exiting a function."""
        if not self.enabled or not self.call_stack:
            return
        
        # Pop from call stack
        call_info = self.call_stack.pop()
        
        if call_info['name'] != name:
            # Stack mismatch - log warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Profile stack mismatch. Expected {call_info['name']}, got {name}")
            return
        
        # Calculate elapsed time
        elapsed = time.time() - call_info['start_time']
        
        profile = self.functions[name]
        profile.total_time += elapsed
        profile.self_time += elapsed
        
        # Update memory if enabled
        if self.memory_enabled:
            current, peak = tracemalloc.get_traced_memory()
            profile.memory_peak = max(profile.memory_peak, peak)
        
        # Subtract time from parent if exists
        if self.call_stack:
            parent_name = self.call_stack[-1]['name']
            if parent_name in self.functions:
                self.functions[parent_name].self_time -= elapsed
    
    def print_report(self, report: ProfileReport, top_n: int = 20):
        """Print profiling report."""
        print("\n" + "="*80)
        print("ZEXUS PERFORMANCE PROFILE REPORT")
        print("="*80)
        print(f"\nTotal Time: {report.total_time:.4f} seconds")
        print(f"Total Calls: {report.total_calls}")
        print(f"Peak Memory: {report.memory_peak / 1024 / 1024:.2f} MB")
        
        print(f"\n{'Function':<40} {'Calls':>10} {'Total Time':>15} {'Self Time':>15} {'Avg Time':>15}")
        print("-"*100)
        
        # Sort by total time
        sorted_functions = sorted(
            report.functions.values(),
            key=lambda p: p.total_time,
            reverse=True
        )
        
        for prof in sorted_functions[:top_n]:
            avg_time = prof.total_time / prof.calls if prof.calls > 0 else 0
            print(f"{prof.name:<40} {prof.calls:>10} {prof.total_time:>15.4f}s {prof.self_time:>15.4f}s {avg_time:>15.6f}s")
        
        print("\n" + "="*80)
        print("TOP HOTSPOTS (by total time)")
        print("="*80)
        
        for i, prof in enumerate(report.hotspots[:10], 1):
            pct = (prof.total_time / report.total_time * 100) if report.total_time > 0 else 0
            print(f"{i}. {prof.name}: {prof.total_time:.4f}s ({pct:.1f}%)")
        
        print("="*80 + "\n")


# Global profiler instance
_profiler = Profiler()


def start_profiling(enable_memory: bool = True):
    """Start global profiler."""
    _profiler.start(enable_memory)


def stop_profiling() -> ProfileReport:
    """Stop global profiler and get report."""
    return _profiler.stop()


def print_profile():
    """Print current profile."""
    report = stop_profiling()
    _profiler.print_report(report)
    return report


def profile_function(func):
    """Decorator to profile a function."""
    def wrapper(*args, **kwargs):
        _profiler.enter_function(func.__name__)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            _profiler.exit_function(func.__name__)
    return wrapper


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    return _profiler
