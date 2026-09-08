#!/usr/bin/env python3
"""Phase G hybrid-execution benchmark: tree-walk vs VM per construct.

Produces the tiering decision table (ROADMAP Phase G: "every tier
decision is a benchmark table in the repo"). Measures canonical
constructs on both engines with output-equality verification first.
"""
import sys
import time
import io
import pathlib
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.vm.compiler import BytecodeCompiler
from zexus.vm.vm import VM
from zexus.evaluator.core import evaluate
from zexus.environment import Environment

BENCHMARKS = {
    "fib(18) recursion": """
        fn fib(n) { if n < 2 { return n } return fib(n-1) + fib(n-2) }
        let r = fib(18)
        print(r)
    """,
    "loop 50k arithmetic": """
        let total = 0
        let i = 0
        while i < 50000 { total = total + i i = i + 1 }
        print(total)
    """,
    "loop 50k (tiered: 120+)": """
        let total = 0
        let i = 0
        while i < 50000 { total = total + i * 2 i = i + 1 }
        print(total)
    """,
    "string ops 5k": """
        let out = ""
        let i = 0
        while i < 5000 {
            out = "x".upper() + out.slice(0, 10)
            i = i + 1
        }
        print(out.len())
    """,
    "list push 10k": """
        let l = []
        let i = 0
        while i < 10000 { l.push(i) i = i + 1 }
        print(l.len())
    """,
    "map writes 10k": """
        let m = {}
        let i = 0
        while i < 10000 { m[i] = i i = i + 1 }
        print(m[9999])
    """,
    "contract actions 1k": """
        contract Counter {
            state { count: 0 }
            action increment() { this.count = this.count + 1 }
            action get() { return this.count }
        }
        let c = Counter()
        let i = 0
        while i < 1000 { c.increment() i = i + 1 }
        print(c.get())
    """,
}


def parse(code):
    p = UltimateParser(Lexer(code, filename="<bench>"), enable_advanced_strategies=False)
    program = p.parse_program()
    assert not p.errors, p.errors[:2]
    return program


def run_treewalk(code):
    buf = io.StringIO()
    with redirect_stdout(buf):
        evaluate(parse(code), Environment(), use_vm=False)
    return buf.getvalue().strip()


def run_vm(code):
    buf = io.StringIO()
    with redirect_stdout(buf):
        vm = VM()
        vm.execute(BytecodeCompiler().compile(parse(code)))
    return buf.getvalue().strip()


def timeit(fn, code, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(code)
        best = min(best, time.perf_counter() - t0)
    return best


print("=" * 68)
print("  PHASE G HYBRID-EXECUTION BENCHMARK (zexus 2.0)")
print("  tree-walk interpreter vs bytecode VM — best of 3")
print("=" * 68)
print(f"  {'construct':24} {'tree-walk':>10} {'VM':>10} {'winner':>8} {'parity':>7}")
print("-" * 68)

results = {}
for name, code in BENCHMARKS.items():
    tw_out = run_treewalk(code)
    vm_out = run_vm(code)
    parity = tw_out == vm_out
    tw_t = timeit(run_treewalk, code)
    vm_t = timeit(run_vm, code)
    winner = "VM" if vm_t < tw_t else "tree"
    speedup = tw_t / vm_t if vm_t > 0 else float("inf")
    results[name] = (tw_t, vm_t, parity, speedup)
    print(f"  {name:24} {tw_t*1000:8.1f}ms {vm_t*1000:8.1f}ms {winner:>8} "
          f"{'✓' if parity else '✗':>7} ({speedup:.2f}x)")

print("=" * 68)
vm_wins = [r for r in results.values() if r[1] < r[0]]
print(f"  VM wins: {len(vm_wins)}/{len(results)} constructs")
print("  Tiering implication: hot loops ≥120 iterations auto-promote to")
print("  the VM (unified_execution thresholds); the table above validates")
print("  which constructs actually benefit.")
if any(not r[2] for r in results.values()):
    print("  ⚠ PARITY FAILURE — investigate before shipping")
    sys.exit(1)
