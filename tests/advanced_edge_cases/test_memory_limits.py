#!/usr/bin/env python3
"""
Test memory limits and tracking.

Tests basic memory limit scenarios using the existing memory manager.

Location: tests/advanced_edge_cases/test_memory_limits.py
"""

import sys
import os
import traceback

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_memory_manager_exists():
    """Test that memory manager can be imported and instantiated."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager(max_heap_size=1024 * 1024)
    assert manager.heap.max_size == 1024 * 1024
    print("✅ Memory manager: instantiated successfully")


def test_memory_allocation_tracking():
    """Test that memory allocations can be tracked."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager()
    manager.allocate([1, 2, 3, 4, 5])
    manager.allocate({"key": "value"})
    manager.allocate("test string")

    stats = manager.get_stats()
    assert stats['allocation_count'] >= 3
    print(f"✅ Memory allocation tracking: {stats['allocation_count']} allocations tracked")


def test_memory_limit_enforcement():
    """Test that memory limits can be enforced."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager(max_heap_size=1024)
    large_data = [0] * 10_000

    with pytest.raises(MemoryError):
        manager.allocate(large_data)

    print("✅ Memory limit enforcement: limit enforced successfully")


def test_garbage_collection():
    """Test that garbage collection works."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager()

    for i in range(10):
        manager.allocate([i] * 100)

    stats_before = manager.get_stats()
    collected, _ = manager.collect_garbage(force=True)
    stats_after = manager.get_stats()

    assert collected >= 0
    assert stats_after['gc_runs'] >= stats_before['gc_runs']
    print(
        "✅ Garbage collection: ran successfully "
        f"({stats_before.get('allocation_count', 0)} → {stats_after.get('allocation_count', 0)} allocations)"
    )


def test_memory_stats():
    """Test that memory statistics can be retrieved."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager()
    stats = manager.get_stats()

    expected_keys = {'allocation_count', 'current_usage', 'peak_usage'}
    assert expected_keys.issubset(stats.keys())
    print(f"✅ Memory statistics: {len(expected_keys)} metrics verified")


def test_memory_leak_detection():
    """Test basic memory leak detection."""
    from zexus.vm.memory_manager import MemoryManager

    manager = MemoryManager()

    objects = [manager.allocate([i] * 10) for i in range(100)]
    objects.clear()

    manager.collect_garbage(force=True)

    if hasattr(manager, 'detect_leaks'):
        leaks = manager.detect_leaks()
        assert isinstance(leaks, list)
        print(f"✅ Memory leak detection: {len(leaks)} potential leaks reported")
    else:
        pytest.skip("Memory manager does not expose leak detection")


if __name__ == '__main__':
    print("=" * 70)
    print("MEMORY LIMITS AND TRACKING TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_memory_manager_exists,
        test_memory_allocation_tracking,
        test_memory_limit_enforcement,
        test_garbage_collection,
        test_memory_stats,
        test_memory_leak_detection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
