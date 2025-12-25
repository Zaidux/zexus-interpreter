#!/usr/bin/env python3
"""
Test null and falsy value reassignment in Environment.assign().

Tests that the assign() method correctly handles reassignment of variables
with null and other falsy values.

Location: tests/edge_cases/test_null_reassignment.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.environment import Environment


def test_null_reassignment():
    """Test that variables with null values can be reassigned."""
    env = Environment()
    
    # Set a variable to null
    env.set("x", None)
    assert env.get("x") is None
    
    # Reassign it to a non-null value
    env.assign("x", 42)
    result = env.get("x")
    assert result is not None
    assert result == 42
    
    print("✅ Null reassignment: variables with null can be reassigned")


def test_zero_reassignment():
    """Test that variables with value 0 can be reassigned."""
    env = Environment()
    
    # Set a variable to 0
    env.set("count", 0)
    assert env.get("count") == 0
    
    # Reassign it
    env.assign("count", 10)
    result = env.get("count")
    assert result == 10
    
    print("✅ Zero reassignment: variables with 0 can be reassigned")


def test_empty_string_reassignment():
    """Test that variables with empty strings can be reassigned."""
    env = Environment()
    
    # Set a variable to empty string
    env.set("text", "")
    assert env.get("text") == ""
    
    # Reassign it
    env.assign("text", "hello")
    result = env.get("text")
    assert result == "hello"
    
    print("✅ Empty string reassignment: variables with '' can be reassigned")


def test_false_reassignment():
    """Test that variables with False can be reassigned."""
    env = Environment()
    
    # Set a variable to False
    env.set("flag", False)
    assert env.get("flag") is False
    
    # Reassign it
    env.assign("flag", True)
    result = env.get("flag")
    assert result is True
    
    print("✅ False reassignment: variables with False can be reassigned")


def test_null_reassignment_in_outer_scope():
    """Test that null variables in outer scopes can be reassigned."""
    outer = Environment()
    outer.set("value", None)
    
    inner = Environment(outer=outer)
    
    # Reassign the null value from inner scope
    inner.assign("value", 100)
    
    # Check it was updated in outer scope
    result = outer.get("value")
    assert result == 100
    
    print("✅ Null outer scope reassignment: null values in outer scopes work")


def test_zero_reassignment_in_outer_scope():
    """Test that zero variables in outer scopes can be reassigned."""
    outer = Environment()
    outer.set("counter", 0)
    
    inner = Environment(outer=outer)
    
    # Reassign from inner scope
    inner.assign("counter", 5)
    
    # Check it was updated in outer scope
    result = outer.get("counter")
    assert result == 5
    
    print("✅ Zero outer scope reassignment: zero values in outer scopes work")


if __name__ == '__main__':
    print("=" * 70)
    print("NULL AND FALSY VALUE REASSIGNMENT TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_null_reassignment,
        test_zero_reassignment,
        test_empty_string_reassignment,
        test_false_reassignment,
        test_null_reassignment_in_outer_scope,
        test_zero_reassignment_in_outer_scope,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
