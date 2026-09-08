"""Differential harness: the same .zx program must produce identical
results on the tree-walk evaluator AND the VM (ROADMAP Phase F rule #2).

This is the structural fix for engine drift: any construct whose VM and
interpreter behavior diverges fails here with both outputs shown, so a
fix must restore equality (not just "work on one engine").
"""
from __future__ import annotations

import io
import sys
import pathlib
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from zexus.lexer import Lexer                      # noqa: E402
from zexus.parser.parser import UltimateParser     # noqa: E402


def _parse(code: str):
    parser = UltimateParser(Lexer(code, filename="<diff>"), enable_advanced_strategies=False)
    program = parser.parse_program()
    assert not parser.errors, f"parse errors: {parser.errors[:3]}"
    return program


def _capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def run_treewalk(code: str) -> str:
    from zexus.evaluator.core import evaluate
    from zexus.environment import Environment

    return _capture(lambda: evaluate(_parse(code), Environment(), use_vm=False))


def run_vm(code: str) -> str:
    from zexus.vm.compiler import BytecodeCompiler
    from zexus.vm.vm import VM

    def _exec():
        vm = VM()
        vm.execute(BytecodeCompiler().compile(_parse(code)))
    return _capture(_exec)


def assert_equal(code: str) -> None:
    """Run on both engines; assert identical stdout."""
    tw = run_treewalk(code)
    vm = run_vm(code)
    assert tw == vm, (
        f"ENGINES DIVERGE:\n--- tree-walk ---\n{tw!r}\n--- VM ---\n{vm!r}"
    )


# ── The corpus: canonical grammar constructs, one program each ────────
# Each entry documents the construct and (historically) which issue
# covered it. New constructs get new entries — the corpus only grows.

CORPUS: dict[str, str] = {
    "literals": 'print(1)\nprint(1.5)\nprint("s")\nprint(true)\nprint(null)\nprint(0xFF)',
    "arithmetic": "print(1 + 2 * 3)\nprint(10 / 4)\nprint(7 % 3)\nprint(2.5 * 2)",
    "string_methods": (
        'print("hello".len())\nprint("hello".upper())\n'
        'print("a-b-c".split("-").len())\nprint("hi".to_hex())'
    ),
    "bytes": (
        'print(b"\\x01\\x02".len())\nprint(b"ab" + b"c")\n'
        'print(b"\\x41".to_string())\nprint(bytes_from_hex("00ff").to_hex())'
    ),
    "control_flow": (
        "let n = 0\n"
        "while n < 3 {\n  print(n)\n  n = n + 1\n}\n"
        "if n == 3 { print(\"done\") }"
    ),
    "for_each_list": (
        "let total = 0\n"
        "for each x in [1, 2, 3, 4] {\n  total = total + x\n}\n"
        "print(total)"
    ),
    "functions": (
        "function double(x) { return x * 2 }\n"
        "fn triple(x) { return x * 3 }\n"
        "print(double(21))\n"
        "print(triple(7))\n"
        "print(double(double(5)))"
    ),
    "maps_lists": (
        'let m = {"a": 1}\n'
        'm["b"] = 2\n'
        'print(m["a"] + m["b"])\n'
        'let l = [1, 2]\n'
        'l.push(3)\n'
        'print(l.len())'
    ),
    "not_keyword": (
        "let f = true\nprint(not f)\nprint(not false)\nprint(not null)"
    ),
    "range_loop": (
        "let total = 0\nfor i in 0..4 {\n  total = total + i\n}\nprint(total)"
    ),
    "range_expression": (
        "let r = 1..4\nprint(r.len())\nprint(r)"
    ),
    "contracts": (
        "contract Counter {\n"
        "    state { count: 0 }\n"
        "    action increment() { this.count = this.count + 1 }\n"
        "    action get() { return this.count }\n"
        "}\n"
        "let c = Counter()\n"
        "c.increment()\n"
        "c.increment()\n"
        "print(c.get())"
    ),
    "crypto_module": (
        'use "crypto"\n'
        'print(hash_sha256("abc"))'
    ),
    "json_module": (
        'use "json"\n'
        'let s = stringify({"ok": true, "n": 2})\n'
        'print(s)'
    ),
}


@pytest.mark.parametrize("name", sorted(CORPUS), ids=lambda n: n)
def test_engines_agree(name):
    assert_equal(CORPUS[name])


# ── Targeted regression pairs for the open V/R-series issues ──────────
# These are expected to FAIL until the corresponding fix lands; each is
# xfail with the issue ID so the suite stays green while making the
# remaining work visible.


def test_error_model_division_by_zero():
    """PHASE G (unified): an uncaught runtime error halts the program at
    the failing statement on BOTH engines — prior output kept, no raise
    past the executor. try/catch handles errors in-program."""
    code = "function boom() { return 1 / 0 }\nlet r = boom()\nprint(typeof(r))"
    assert_equal(code)


def test_error_model_halts_after_output():
    """Output BEFORE the failure is preserved on both engines."""
    code = "print(\"before\")\nlet x = 1 / 0\nprint(\"after\")"
    assert_equal(code)


@pytest.mark.xfail(reason="V-004/R-029: anonymous action closures do not parse as expressions", strict=False)
def test_anonymous_closure():
    code = "let f = action(a, b) { return a + b }\nprint(f(3, 4))"
    assert_equal(code)
