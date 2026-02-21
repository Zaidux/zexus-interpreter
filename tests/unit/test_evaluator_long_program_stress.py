"""Evaluator/VM stress regression for long programs.

Goal: ensure the full pipeline remains stable on ~10k statement files.
This catches O(n^2) behavior and stack/IO-related misbehavior regressions.

We intentionally keep the program simple (one variable) to avoid huge memory
usage from thousands of distinct bindings.
"""

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import evaluate
from zexus.environment import Environment


def _make_increment_program(line_count: int = 10_000) -> str:
    assert line_count >= 2
    lines = ["let x = 0;"]
    lines.extend(["x = x + 1;"] * (line_count - 1))
    return "\n".join(lines) + "\n"


def test_long_program_10k_statements_eval_with_and_without_vm():
    code = _make_increment_program(10_000)

    # Parse with default settings (advanced parsing may auto-disable for large files).
    program = UltimateParser(Lexer(code)).parse(raise_on_error=True)

    # Evaluate with VM enabled.
    env_vm = Environment()
    evaluate(program, env_vm, use_vm=True)
    assert env_vm.get("x").value == 9_999

    # Evaluate without VM.
    env_no_vm = Environment()
    evaluate(program, env_no_vm, use_vm=False)
    assert env_no_vm.get("x").value == 9_999
