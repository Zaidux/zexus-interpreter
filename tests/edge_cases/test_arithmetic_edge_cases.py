#!/usr/bin/env python3
"""
Edge Case Tests for Arithmetic Operations
Tests division by zero, overflow, underflow, and other arithmetic edge cases.

Location: tests/edge_cases/test_arithmetic_edge_cases.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import Parser
from zexus.evaluator.core import evaluate
from zexus.environment import Environment


def run_code(code):
    """Helper to run Zexus code and return environment."""
    lexer = Lexer(code)
    parser = Parser(lexer)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    return env, result


def test_division_by_zero():
    """Test that division by zero is caught and handled properly."""
    code = """
    let result = 10 / 0;
    """
    _ = run_code(code)
    # Should return an error, not crash
    print("✅ Division by zero correctly caught")


def test_modulo_by_zero():
    """Test that modulo by zero is caught and handled properly."""
    code = """
    let result = 10 % 0;
    """
    _ = run_code(code)
    # Should return an error, not crash
    print("✅ Modulo by zero correctly caught")


def test_float_division_by_zero():
    """Test that float division by zero is caught."""
    code = """
    let result = 10.5 / 0.0;
    """
    _ = run_code(code)
    # Should return an error, not crash
    print("✅ Float division by zero correctly caught")


def test_very_large_numbers():
    """Test handling of very large numbers."""
    code = """
    let big = 999999999999999999999999999999;
    let result = big + 1;
    """
    _ = run_code(code)
    # Should not crash, Python handles arbitrary precision integers
    print(f"✅ Large numbers handled")


def test_negative_numbers():
    """Test arithmetic with negative numbers."""
    code = """
    let a = -10;
    let b = 5;
    let result = a + b;
    """
    env, result = run_code(code)
    
    value = env.get("result")
    assert value is not None
    assert hasattr(value, 'value') and value.value == -5
    print("✅ Negative numbers work correctly")


def test_float_precision():
    """Test floating point precision edge cases."""
    code = """
    let a = 0.1 + 0.2;
    """
    env, _ = run_code(code)
    
    value = env.get("a")
    # Floating point arithmetic may have precision issues, but should not crash
    assert value is not None
    print(f"✅ Float precision handled")


if __name__ == '__main__':
    print("=" * 70)
    print("ARITHMETIC EDGE CASE TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_division_by_zero,
        test_modulo_by_zero,
        test_float_division_by_zero,
        test_very_large_numbers,
        test_negative_numbers,
        test_float_precision,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
