"""
Zexus Safety Module - Safer than Rust

This module provides comprehensive safety features:
- Memory safety (bounds checking, use-after-free prevention)
- Type safety (runtime type validation)
- Concurrency safety (deadlock detection, race condition prevention)
- Resource safety (automatic cleanup, leak detection)
"""

from .memory_safety import (
    # Core safety classes
    SafeArray,
    ReferenceCounted,
    MemoryGuard,
    SafePointer,
    
    # Exceptions
    MemorySafetyError,
    BoundsViolation,
    UseAfterFree,
    StackOverflow,
    HeapCorruption,
    
    # Globals
    get_memory_guard,
    reset_memory_guard
)

__all__ = [
    'SafeArray',
    'ReferenceCounted',
    'MemoryGuard',
    'SafePointer',
    'MemorySafetyError',
    'BoundsViolation',
    'UseAfterFree',
    'StackOverflow',
    'HeapCorruption',
    'get_memory_guard',
    'reset_memory_guard'
]
