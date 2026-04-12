"""Tests for LSP and DAP enhancements."""

import pytest


class TestLSPCompletionUserSymbols:
    def test_import_completion_provider(self):
        from src.zexus.lsp.completion_provider import get_user_defined_symbols
        assert callable(get_user_defined_symbols)

    def test_detects_let_declarations(self):
        from src.zexus.lsp.completion_provider import get_user_defined_symbols
        source = '''
let counter = 0
let name = "hello"
'''
        symbols = get_user_defined_symbols(source)
        labels = [s["label"] for s in symbols]
        assert "counter" in labels
        assert "name" in labels

    def test_detects_action_declarations(self):
        from src.zexus.lsp.completion_provider import get_user_defined_symbols
        source = '''
action greet(name) {
    print("Hello " ++ name)
}

action calculate(a, b) {
    ret a + b
}
'''
        symbols = get_user_defined_symbols(source)
        labels = [s["label"] for s in symbols]
        assert "greet" in labels
        assert "calculate" in labels

    def test_detects_entity_declarations(self):
        from src.zexus.lsp.completion_provider import get_user_defined_symbols
        source = '''
entity User {
    let name = ""
}
'''
        symbols = get_user_defined_symbols(source)
        labels = [s["label"] for s in symbols]
        assert "User" in labels

    def test_detects_contract_declarations(self):
        from src.zexus.lsp.completion_provider import get_user_defined_symbols
        source = '''
contract Token {
    state balance = 0
}
'''
        symbols = get_user_defined_symbols(source)
        labels = [s["label"] for s in symbols]
        assert "Token" in labels


class TestLSPFormatter:
    def test_import_formatter(self):
        from src.zexus.lsp.formatter import ZexusFormatter
        assert ZexusFormatter is not None

    def test_basic_formatting(self):
        from src.zexus.lsp.formatter import ZexusFormatter
        source = "let   x  =  1"
        formatted = ZexusFormatter.format_document(source)
        assert isinstance(formatted, str)

    def test_indentation(self):
        from src.zexus.lsp.formatter import ZexusFormatter
        source = """action greet() {
print("hello")
}"""
        formatted = ZexusFormatter.format_document(source)
        # Should have some indentation inside the block
        lines = formatted.strip().split('\n')
        assert len(lines) >= 2

    def test_trailing_whitespace_removed(self):
        from src.zexus.lsp.formatter import ZexusFormatter
        source = "let x = 1   \nlet y = 2   \n"
        formatted = ZexusFormatter.format_document(source)
        for line in formatted.split('\n'):
            assert line == line.rstrip()


class TestDAPConditionalBreakpoint:
    def test_import_conditional_breakpoint(self):
        from src.zexus.dap.debug_engine import ConditionalBreakpoint
        assert ConditionalBreakpoint is not None

    def test_create_conditional_breakpoint(self):
        from src.zexus.dap.debug_engine import ConditionalBreakpoint
        bp = ConditionalBreakpoint("test.zx", 10, condition="x > 5")
        assert bp.file == "test.zx"
        assert bp.line == 10
        assert bp.condition == "x > 5"

    def test_should_break_no_condition(self):
        from src.zexus.dap.debug_engine import ConditionalBreakpoint
        bp = ConditionalBreakpoint("test.zx", 10)
        assert bp.should_break({}) is True

    def test_should_break_with_condition(self):
        from src.zexus.dap.debug_engine import ConditionalBreakpoint
        bp = ConditionalBreakpoint("test.zx", 10, condition="x > 5")
        assert bp.should_break({"x": 10}) is True
        assert bp.should_break({"x": 3}) is False

    def test_hit_count(self):
        from src.zexus.dap.debug_engine import ConditionalBreakpoint
        bp = ConditionalBreakpoint("test.zx", 10, hit_count=3)
        assert bp.should_break({}) is False  # hit 1
        assert bp.should_break({}) is False  # hit 2
        assert bp.should_break({}) is True   # hit 3


class TestDAPExpressionEval:
    def test_evaluate_expression_import(self):
        from src.zexus.dap.debug_engine import DebugEngine
        engine = DebugEngine.__new__(DebugEngine)
        # Just verify the method exists
        assert hasattr(engine, 'evaluate_expression')
