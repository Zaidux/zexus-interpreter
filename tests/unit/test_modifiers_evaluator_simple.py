#!/usr/bin/env python3
"""Test that evaluator honors modifiers on action statements (pytest-free)."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.environment import Environment


def test_inline_modifier():
    """Test that 'inline' modifier sets is_inlined flag on action."""
    code = "inline action foo() { ret 42; }"
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    ast = parser.parse()
    
    env = Environment()
    evaluator = Evaluator()
    evaluator.eval(ast, env)
    
    action = env.get('foo')
    assert action is not None, "Action 'foo' not defined"
    assert hasattr(action, 'is_inlined'), "Action should have is_inlined attribute"
    assert action.is_inlined is True, "is_inlined should be True for inline modifier"
    print("✓ inline modifier test passed")


def test_async_modifier():
    """Test that 'async' modifier sets is_async flag on action."""
    code = "async action bar() { ret 1; }"
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    ast = parser.parse()
    
    env = Environment()
    evaluator = Evaluator()
    evaluator.eval(ast, env)
    
    action = env.get('bar')
    assert action is not None, "Action 'bar' not defined"
    assert hasattr(action, 'is_async'), "Action should have is_async attribute"
    assert action.is_async is True, "is_async should be True for async modifier"
    print("✓ async modifier test passed")


def test_secure_modifier():
    """Test that 'secure' modifier sets is_secure flag on action."""
    code = "secure action baz() { ret 2; }"
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    ast = parser.parse()
    
    env = Environment()
    evaluator = Evaluator()
    evaluator.eval(ast, env)
    
    action = env.get('baz')
    assert action is not None, "Action 'baz' not defined"
    assert hasattr(action, 'is_secure'), "Action should have is_secure attribute"
    assert action.is_secure is True, "is_secure should be True for secure modifier"
    print("✓ secure modifier test passed")


def test_multiple_modifiers():
    """Test that multiple modifiers all set their flags."""
    code = "secure inline async action multi() { ret 5; }"
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    ast = parser.parse()
    
    env = Environment()
    evaluator = Evaluator()
    evaluator.eval(ast, env)
    
    action = env.get('multi')
    assert action is not None, "Action 'multi' not defined"
    assert action.is_secure is True, "is_secure should be True"
    assert action.is_inlined is True, "is_inlined should be True"
    assert action.is_async is True, "is_async should be True"
    print("✓ multiple modifiers test passed")


def test_no_modifiers():
    """Test that actions without modifiers work normally."""
    code = "action nomod() { ret 7; }"
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    ast = parser.parse()
    
    env = Environment()
    evaluator = Evaluator()
    evaluator.eval(ast, env)
    
    action = env.get('nomod')
    assert action is not None, "Action 'nomod' not defined"
    # Should not have flags set or flags should be None/False
    assert not getattr(action, 'is_inlined', False), "Unmodified action should not have is_inlined"
    assert not getattr(action, 'is_async', False), "Unmodified action should not have is_async"
    assert not getattr(action, 'is_secure', False), "Unmodified action should not have is_secure"
    print("✓ no modifiers test passed")


if __name__ == '__main__':
    try:
        test_inline_modifier()
        test_async_modifier()
        test_secure_modifier()
        test_multiple_modifiers()
        test_no_modifiers()
        print("\n✅ All evaluator modifier tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
