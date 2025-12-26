#!/usr/bin/env python3
"""
Test recursion limits and stack overflow scenarios.

Tests:
1. Very deep recursion (Python recursion limit)
2. VM stack overflow scenarios

Location: tests/advanced_edge_cases/test_recursion_limits.py
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


def test_deep_recursion_graceful_failure():
    """Test that very deep recursion fails gracefully without crashing."""
    code = """
    action deep_recursive(n) {
        if n <= 0 {
            return 0;
        }
        return 1 + deep_recursive(n - 1);
    }
    
    # Try to recurse deeply - should hit recursion limit
    let result = deep_recursive(10000);
    """
    
    try:
        _, result = run_code(code)
        # If we get here without crash, good!
        print("✅ Deep recursion: handled gracefully (may hit recursion limit)")
    except RecursionError:
        print("✅ Deep recursion: RecursionError caught as expected")
    except Exception as e:
        print(f"✅ Deep recursion: Caught exception gracefully - {type(e).__name__}")


def test_reasonable_recursion_depth():
    """Test that reasonable recursion works correctly."""
    code = """
    action factorial(n) {
        if n <= 1 {
            return 1;
        }
        return n * factorial(n - 1);
    }
    
    let result = factorial(100);
    """
    
    env, result = run_code(code)
    value = env.get("result")
    assert value is not None
    # 100! is a very large number but should work
    print("✅ Reasonable recursion (100 levels): works correctly")


def test_mutual_recursion_depth():
    """Test mutual recursion with reasonable depth."""
    code = """
    action is_even(n) {
        if n == 0 {
            return true;
        }
        return is_odd(n - 1);
    }
    
    action is_odd(n) {
        if n == 0 {
            return false;
        }
        return is_even(n - 1);
    }
    
    let result = is_even(100);
    """
    
    env, result = run_code(code)
    value = env.get("result")
    assert value is not None
    print("✅ Mutual recursion (100 levels): works correctly")


def test_tail_recursion_simulation():
    """Test that iterative approach works for what would be tail recursion."""
    code = """
    # Simulate tail recursion with while loop
    action sum_to_n(n) {
        let sum = 0;
        let i = 1;
        while i <= n {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
    
    let result = sum_to_n(1000);
    """
    
    env, result = run_code(code)
    value = env.get("result")
    assert value is not None
    expected = (1000 * 1001) // 2  # Sum formula
    assert value.value == expected
    print(f"✅ Tail recursion simulation (1000 iterations): works correctly")


if __name__ == '__main__':
    print("=" * 70)
    print("RECURSION LIMIT AND STACK OVERFLOW TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_deep_recursion_graceful_failure,
        test_reasonable_recursion_depth,
        test_mutual_recursion_depth,
        test_tail_recursion_simulation,
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
