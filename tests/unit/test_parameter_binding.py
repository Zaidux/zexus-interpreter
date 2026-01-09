"""Pytest coverage for function parameter binding semantics."""

import os
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment


def _evaluate(code_str: str):
    """Parse and execute snippet, raising on failure."""
    lexer = Lexer(code_str)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    program = parser.parse_program()
    evaluator = Evaluator()
    env = Environment()
    return evaluator.eval_node(program, env)


_BASIC_CASES = [
    (
        "single parameter",
        """function greet(name) {
    print("Hello, " + name);
}
greet("Alice");""",
    ),
    (
        "multiple parameters",
        """function add(a, b) {
    print(a + b);
    return a + b;
}
result = add(5, 3);""",
    ),
    (
        "parameters in concatenation",
        """function registerPackage(name, version) {
    print("Registering " + name + "@" + version);
    return {"name": name, "version": version};
}
pkg = registerPackage("pkg", "1.0");""",
    ),
    (
        "nested function calls",
        """function double(x) { return x * 2; }
function quad(x) { return double(double(x)); }
result = quad(5);
print(result);""",
    ),
    (
        "parameter shadowing",
        """let x = "outer";
function test(x) {
    print("inner: " + x);
}
test("inner");
print("outer: " + x);""",
    ),
]


def _phase10_case():
    phase10_path = _SRC_PATH / "tests" / "test_phase10_ecosystem.zx"
    if not phase10_path.exists():
        return pytest.param(
            "phase 10 ecosystem (missing)",
            "",
            marks=pytest.mark.skip(reason="test_phase10_ecosystem.zx not available"),
        )
    return (
        "phase 10 ecosystem",
        phase10_path.read_text(encoding="utf-8"),
    )


_TEST_CASES = _BASIC_CASES + [_phase10_case()]


def _case_id(param):
    if hasattr(param, "values"):
        return param.values[0]
    return param[0]


@pytest.mark.parametrize("description, code_snippet", _TEST_CASES, ids=_case_id)
def test_parameter_binding(description, code_snippet):
    """All snippets should execute without raising."""
    _evaluate(code_snippet)
