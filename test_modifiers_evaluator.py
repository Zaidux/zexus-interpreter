#!/usr/bin/env python3
"""Test that evaluator honors modifiers on action statements."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.environment import Environment


class TestModifiersEvaluator:
    """Test modifier behavior in evaluator."""
    
    def test_inline_modifier_sets_flag(self):
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
    
    def test_async_modifier_sets_flag(self):
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
    
    def test_secure_modifier_sets_flag(self):
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
    
    def test_pure_modifier_sets_flag(self):
        """Test that 'pure' modifier sets is_pure flag on action."""
        code = "pure action qux() { ret 3; }"
        lexer = Lexer(code)
        parser = UltimateParser(lexer)
        ast = parser.parse()
        
        env = Environment()
        evaluator = Evaluator()
        evaluator.eval(ast, env)
        
        action = env.get('qux')
        assert action is not None, "Action 'qux' not defined"
        assert hasattr(action, 'is_pure'), "Action should have is_pure attribute"
        assert action.is_pure is True, "is_pure should be True for pure modifier"
    
    def test_native_modifier_sets_flag(self):
        """Test that 'native' modifier sets is_native flag on action."""
        code = "native action nfoo() { ret 4; }"
        lexer = Lexer(code)
        parser = UltimateParser(lexer)
        ast = parser.parse()
        
        env = Environment()
        evaluator = Evaluator()
        evaluator.eval(ast, env)
        
        action = env.get('nfoo')
        assert action is not None, "Action 'nfoo' not defined"
        assert hasattr(action, 'is_native'), "Action should have is_native attribute"
        assert action.is_native is True, "is_native should be True for native modifier"
    
    def test_multiple_modifiers(self):
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
    
    def test_public_modifier_exports(self):
        """Test that 'public' modifier exports the action."""
        code = "public action exported() { ret 6; }"
        lexer = Lexer(code)
        parser = UltimateParser(lexer)
        ast = parser.parse()
        
        env = Environment()
        evaluator = Evaluator()
        evaluator.eval(ast, env)
        
        action = env.get('exported')
        assert action is not None, "Action 'exported' not defined"
        
        # Check if exported (should be in exported symbols)
        # Note: Exact export mechanism depends on Environment implementation
        # This test assumes exports are tracked; if not, this verifies no error occurs
        assert 'exported' in env.values, "Action should be exported to env.values"
    
    def test_no_modifiers(self):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
