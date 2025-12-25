#!/usr/bin/env python3
"""
Test input validation module.

Tests comprehensive input validation for all public APIs.

Location: tests/advanced_edge_cases/test_input_validation.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.input_validation import (
    validate_string_input,
    validate_integer_input,
    validate_number_input,
    validate_collection_input,
    validate_index,
    validate_file_path,
    validate_enum_input,
    validate_not_none,
    validate_positive_integer,
    validate_non_negative_integer,
    validate_non_empty_string,
    validate_percentage,
)


def test_string_validation():
    """Test string input validation."""
    # Valid strings
    assert validate_string_input("hello") == "hello"
    assert validate_string_input("") == ""
    
    # Invalid type
    try:
        validate_string_input(123)
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    # Empty not allowed
    try:
        validate_string_input("", allow_empty=False)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Max length
    try:
        validate_string_input("toolong", max_length=5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ String validation: works correctly")


def test_integer_validation():
    """Test integer input validation."""
    # Valid integers
    assert validate_integer_input(42) == 42
    assert validate_integer_input(-10) == -10
    
    # Invalid type
    try:
        validate_integer_input("42")
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    # Min value
    try:
        validate_integer_input(5, min_value=10)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Max value
    try:
        validate_integer_input(15, max_value=10)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ Integer validation: works correctly")


def test_number_validation():
    """Test numeric input validation."""
    # Valid numbers
    assert validate_number_input(42) == 42
    assert validate_number_input(3.14) == 3.14
    
    # Invalid type
    try:
        validate_number_input("42")
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    # Range checks
    assert validate_number_input(50, min_value=0, max_value=100) == 50
    
    print("✅ Number validation: works correctly")


def test_collection_validation():
    """Test collection input validation."""
    # Valid collections
    assert validate_collection_input([1, 2, 3]) == [1, 2, 3]
    assert validate_collection_input((1, 2)) == (1, 2)
    
    # Invalid type
    try:
        validate_collection_input(42)
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    # Length checks
    try:
        validate_collection_input([1, 2], min_length=5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ Collection validation: works correctly")


def test_index_validation():
    """Test index validation."""
    collection = [1, 2, 3, 4, 5]
    
    # Valid indices
    assert validate_index(0, collection) == 0
    assert validate_index(4, collection) == 4
    assert validate_index(-1, collection) == 4  # Negative index
    
    # Out of bounds
    try:
        validate_index(10, collection)
        assert False, "Should raise IndexError"
    except IndexError:
        pass
    
    # Invalid type
    try:
        validate_index("0", collection)
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    print("✅ Index validation: works correctly")


def test_file_path_validation():
    """Test file path validation."""
    # Valid path
    assert validate_file_path("/tmp/test.txt") == "/tmp/test.txt"
    
    # Empty path
    try:
        validate_file_path("")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Invalid type
    try:
        validate_file_path(123)
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    print("✅ File path validation: works correctly")


def test_enum_validation():
    """Test enum validation."""
    # Valid value
    assert validate_enum_input("GET", ["GET", "POST", "PUT"]) == "GET"
    
    # Invalid value
    try:
        validate_enum_input("DELETE", ["GET", "POST", "PUT"])
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ Enum validation: works correctly")


def test_not_none_validation():
    """Test not-none validation."""
    # Valid value
    assert validate_not_none(42) == 42
    assert validate_not_none("") == ""
    
    # None
    try:
        validate_not_none(None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ Not-none validation: works correctly")


def test_convenience_validators():
    """Test convenience validator functions."""
    # Positive integer
    assert validate_positive_integer(5) == 5
    try:
        validate_positive_integer(0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Non-negative integer
    assert validate_non_negative_integer(0) == 0
    assert validate_non_negative_integer(5) == 5
    
    # Non-empty string
    assert validate_non_empty_string("hello") == "hello"
    try:
        validate_non_empty_string("")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Percentage
    assert validate_percentage(50) == 50
    try:
        validate_percentage(150)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✅ Convenience validators: work correctly")


if __name__ == '__main__':
    print("=" * 70)
    print("INPUT VALIDATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_string_validation,
        test_integer_validation,
        test_number_validation,
        test_collection_validation,
        test_index_validation,
        test_file_path_validation,
        test_enum_validation,
        test_not_none_validation,
        test_convenience_validators,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
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
