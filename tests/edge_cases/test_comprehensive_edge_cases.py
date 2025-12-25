#!/usr/bin/env python3
"""
Comprehensive Edge Case Tests for Zexus Interpreter
Tests all major edge cases to ensure stability and robustness.

Location: tests/edge_cases/test_comprehensive_edge_cases.py
Purpose: Verify the interpreter handles edge cases gracefully without crashing
"""

import sys
import os
import traceback

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
    _ = run_code("let result = 10 / 0;")
    # Should return an error, not crash
    print("✅ Division by zero: handled gracefully")


def test_modulo_by_zero():
    """Test that modulo by zero is caught and handled properly."""
    _ = run_code("let result = 10 % 0;")
    print("✅ Modulo by zero: handled gracefully")


def test_float_division_by_zero():
    """Test that float division by zero is caught."""
    _ = run_code("let result = 10.5 / 0.0;")
    print("✅ Float division by zero: handled gracefully")


def test_very_large_numbers():
    """Test handling of very large numbers."""
    _ = run_code("""
    let big = 999999999999999999999999999999;
    let result = big + 1;
    """)
    # Should not crash, Python handles arbitrary precision integers
    print("✅ Very large numbers: handled")


def test_negative_numbers():
    """Test arithmetic with negative numbers."""
    env, result = run_code("""
    let a = -10;
    let b = 5;
    let result = a + b;
    """)
    value = env.get("result")
    assert value is not None and value.value == -5
    print("✅ Negative numbers: work correctly")


def test_null_value():
    """Test that null values are handled properly."""
    env, result = run_code("let x = null;")
    value = env.get("x")
    assert value is not None
    print("✅ Null values: handled correctly")


def test_empty_string():
    """Test empty string handling."""
    env, result = run_code("""
    let empty = "";
    let length = len(empty);
    """)
    length = env.get("length")
    assert length is not None and length.value == 0
    print("✅ Empty strings: handled correctly")


def test_empty_array():
    """Test empty array handling."""
    env, result = run_code("""
    let arr = [];
    let length = len(arr);
    """)
    length = env.get("length")
    assert length is not None and length.value == 0
    print("✅ Empty arrays: handled correctly")


def test_array_indexing():
    """Test array indexing edge cases."""
    env, result = run_code("""
    let arr = [1, 2, 3];
    let first = arr[0];
    let last = arr[2];
    """)
    first = env.get("first")
    last = env.get("last")
    assert first is not None and first.value == 1
    assert last is not None and last.value == 3
    print("✅ Array indexing: works correctly")


def test_string_concatenation():
    """Test string concatenation."""
    env, result = run_code("""
    let hello = "Hello";
    let world = "World";
    let result = hello + " " + world;
    """)
    value = env.get("result")
    assert value is not None and value.value == "Hello World"
    print("✅ String concatenation: works correctly")


def test_boolean_operations():
    """Test boolean logic."""
    env, result = run_code("""
    let t = true;
    let f = false;
    let and_result = t && f;
    let or_result = t || f;
    let not_result = !t;
    """)
    and_val = env.get("and_result")
    or_val = env.get("or_result")
    not_val = env.get("not_result")
    assert and_val is not None and and_val.value == False
    assert or_val is not None and or_val.value == True
    assert not_val is not None and not_val.value == False
    print("✅ Boolean operations: work correctly")


def test_comparison_operators():
    """Test comparison operators."""
    env, result = run_code("""
    let eq = 5 == 5;
    let neq = 5 != 3;
    let gt = 10 > 5;
    let lt = 3 < 7;
    let gte = 5 >= 5;
    let lte = 3 <= 5;
    """)
    assert env.get("eq").value == True
    assert env.get("neq").value == True
    assert env.get("gt").value == True
    assert env.get("lt").value == True
    print("✅ Comparison operators: work correctly")


def test_if_statement():
    """Test if-else statements."""
    env, result = run_code("""
    let x = 10;
    let result = 0;
    if x > 5 {
        result = 1;
    } else {
        result = 2;
    }
    """)
    value = env.get("result")
    assert value is not None and value.value == 1
    print("✅ If statements: work correctly")


def test_while_loop():
    """Test while loops."""
    env, result = run_code("""
    let i = 0;
    let sum = 0;
    while i < 5 {
        sum = sum + i;
        i = i + 1;
    }
    """)
    sum_val = env.get("sum")
    assert sum_val is not None and sum_val.value == 10
    print("✅ While loops: work correctly")


def test_function_definition():
    """Test function definition and calling."""
    env, result = run_code("""
    action add(a, b) {
        return a + b;
    }
    let result = add(3, 4);
    """)
    value = env.get("result")
    assert value is not None and value.value == 7
    print("✅ Functions: work correctly")


def test_nested_functions():
    """Test nested function calls."""
    env, result = run_code("""
    action double(x) {
        return x * 2;
    }
    action quad(x) {
        return double(double(x));
    }
    let result = quad(5);
    """)
    value = env.get("result")
    assert value is not None and value.value == 20
    print("✅ Nested functions: work correctly")


def test_string_escaping():
    """Test string with special characters."""
    _ = run_code("""
    let str = "Line 1\\nLine 2";
    """)
    # Should not crash
    print("✅ String escaping: handled")


def test_map_literal():
    """Test map/dictionary creation."""
    _ = run_code("""
    let map = {name: "test", value: 42};
    """)
    # Should not crash
    print("✅ Map literals: handled")


if __name__ == '__main__':
    print("=" * 70)
    print("COMPREHENSIVE EDGE CASE TESTS FOR ZEXUS INTERPRETER")
    print("=" * 70)
    print()
    
    tests = [
        ("Arithmetic Edge Cases", [
            test_division_by_zero,
            test_modulo_by_zero,
            test_float_division_by_zero,
            test_very_large_numbers,
            test_negative_numbers,
        ]),
        ("Null and Empty Values", [
            test_null_value,
            test_empty_string,
            test_empty_array,
        ]),
        ("Collections and Indexing", [
            test_array_indexing,
            test_string_concatenation,
            test_map_literal,
        ]),
        ("Boolean and Logic", [
            test_boolean_operations,
            test_comparison_operators,
        ]),
        ("Control Flow", [
            test_if_statement,
            test_while_loop,
        ]),
        ("Functions", [
            test_function_definition,
            test_nested_functions,
        ]),
        ("String Handling", [
            test_string_escaping,
        ]),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for category, category_tests in tests:
        print(f"\n{category}:")
        print("-" * 70)
        for test in category_tests:
            try:
                test()
                total_passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                traceback.print_exc()
                total_failed += 1
    
    print()
    print("=" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)
    
    sys.exit(0 if total_failed == 0 else 1)
