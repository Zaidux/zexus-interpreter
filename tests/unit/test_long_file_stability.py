"""Regression tests for long-file stability.

The lexer previously used recursion to skip comments/newlines, which could
hit Python's recursion limit (~1000) on long runs of blank/comment-only lines.

These tests ensure both the interpreter lexer and the compiler lexer handle
10k+ line sources without stack growth, and that parsing/execution still works.
"""

import sys

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import evaluate
from zexus.environment import Environment

from zexus.compiler.lexer import Lexer as CompilerLexer
from zexus.zexus_token import EOF


def _lex_all(lexer, *, hard_cap: int = 200_000):
    """Consume tokens until EOF, guarding against infinite loops."""
    steps = 0
    while True:
        tok = lexer.next_token()
        steps += 1
        if tok.type == EOF:
            return steps
        if steps > hard_cap:
            raise AssertionError("Lexer did not reach EOF (possible infinite loop)")


def test_long_comment_file_lex_parse_eval_no_recursion():
    # Force a low recursion limit to catch accidental recursion.
    prev_limit = sys.getrecursionlimit()
    try:
        # 1000 is the common default; keep it low for the test.
        sys.setrecursionlimit(min(prev_limit, 1000))

        code = ("# comment\n" * 10_000) + "let x = 1;\n"

        # Interpreter lexer should not recurse when skipping comments.
        _lex_all(Lexer(code))

        # Parser + evaluator should still work (with and without VM).
        program = UltimateParser(Lexer(code), enable_advanced_strategies=False).parse(raise_on_error=True)

        env_vm = Environment()
        evaluate(program, env_vm, use_vm=True)
        assert env_vm.get("x").value == 1

        env_no_vm = Environment()
        evaluate(program, env_no_vm, use_vm=False)
        assert env_no_vm.get("x").value == 1

    finally:
        sys.setrecursionlimit(prev_limit)


def test_compiler_lexer_long_comment_file_no_recursion():
    prev_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(min(prev_limit, 1000))

        code = ("# comment\n" * 10_000) + "let x = 1;\n"
        _lex_all(CompilerLexer(code))

    finally:
        sys.setrecursionlimit(prev_limit)
