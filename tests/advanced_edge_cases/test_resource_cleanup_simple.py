#!/usr/bin/env python3
"""
Test resource cleanup (simplified).

Basic resource cleanup tests without long-running operations.

Location: tests/advanced_edge_cases/test_resource_cleanup_simple.py
"""

import sys
import os
import gc
import traceback
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.environment import Environment


def test_environment_cleanup():
    """Test that environments are cleaned up properly."""
    # Create an environment and get a weak reference
    env = Environment()
    env.set("test", "value")
    weak_ref = weakref.ref(env)
    
    # Delete the environment
    del env
    gc.collect()
    
    # Check if it was collected
    assert weak_ref() is None, "Environment should be garbage collected"
    print("✅ Environment cleanup: environments garbage collected properly")


def test_object_reference_cleanup():
    """Test that object references don't cause leaks."""
    initial = len(gc.get_objects())
    
    # Create and destroy environments
    for _ in range(10):
        env = Environment()
        env.set("data", [1, 2, 3, 4, 5])
        del env
    
    gc.collect()
    final = len(gc.get_objects())
    growth = final - initial
    
    assert growth < 50, f"Object references retained unexpectedly ({growth})"
    print(f"✅ Object reference cleanup: minimal growth ({growth} objects)")


def test_nested_scope_cleanup():
    """Test that nested scopes are cleaned up."""
    outer = Environment()
    outer.set("x", 10)
    
    inner = Environment(outer=outer)
    inner.set("y", 20)
    
    _ = weakref.ref(outer)  # Keep reference for consistency
    weak_inner = weakref.ref(inner)
    
    del inner
    gc.collect()
    
    assert weak_inner() is None, "Inner scope should be collected"
    print("✅ Nested scope cleanup: inner scope collected")

    del outer
    gc.collect()


def test_circular_reference():
    """Test handling of potential circular references."""
    class Container:
        def __init__(self):
            self.ref = None
    
    obj1 = Container()
    obj2 = Container()
    obj1.ref = obj2
    obj2.ref = obj1
    
    weak1 = weakref.ref(obj1)
    weak2 = weakref.ref(obj2)
    
    del obj1, obj2
    gc.collect()
    
    assert weak1() is None and weak2() is None, "Circular reference objects still alive"
    print("✅ Circular reference: Python GC handles circular refs")


def test_exception_cleanup():
    """Test cleanup when exceptions occur."""
    initial = len(gc.get_objects())
    
    for _ in range(10):
        try:
            env = Environment()
            env.set("data", [1, 2, 3])
            raise ValueError("Test error")
        except ValueError:
            pass
    
    gc.collect()
    final = len(gc.get_objects())
    growth = final - initial
    
    assert growth < 30, f"Objects retained after exceptions: {growth}"
    print(f"✅ Exception cleanup: minimal growth on errors ({growth} objects)")


if __name__ == '__main__':
    print("=" * 70)
    print("RESOURCE CLEANUP TESTS (SIMPLIFIED)")
    print("=" * 70)
    print()
    
    tests = [
        test_environment_cleanup,
        test_object_reference_cleanup,
        test_nested_scope_cleanup,
        test_circular_reference,
        test_exception_cleanup,
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
