"""
TEST 03: eval() Injection in JIT Compiler (jit.py)
===================================================

VULNERABILITY: The JIT optimizer uses eval() on string values that start with
a digit character. The check `a_val[0].isdigit()` is insufficient — a string
like "1+__import__('os').system('id')" starts with '1' (a digit) but contains
arbitrary Python code.

LOCATION: src/zexus/vm/jit.py, lines 512-513, 947

SEVERITY: CRITICAL - Code Execution

ATTACK VECTOR: If a Zexus program produces bytecode constants whose string
representation starts with a digit, the JIT constant-folding path will eval()
the full string.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_eval_injection_in_jit():
    """Demonstrate that eval() on digit-prefixed strings is dangerous."""
    results = []

    # Simulate the vulnerable pattern from jit.py lines 512-513:
    #   a_const = eval(a_val) if isinstance(a_val, str) and a_val[0].isdigit() else a_val

    # --- Payload 1: Simple arithmetic that sneaks through ---
    a_val = "1+1"
    passes_check = isinstance(a_val, str) and a_val[0].isdigit()
    try:
        result = eval(a_val) if passes_check else a_val  # noqa: S307
        results.append({
            "payload": a_val,
            "type": "arithmetic_eval",
            "exploited": result == 2,  # Should be string "1+1", not int 2
            "detail": f"eval({a_val!r}) = {result!r} (expected string, got computed value)",
        })
    except Exception as e:
        results.append({
            "payload": a_val, "type": "arithmetic_eval",
            "exploited": False, "detail": f"Exception: {e}",
        })

    # --- Payload 2: Code execution disguised as a number ---
    # This string starts with '9' (isdigit() → True) but is actually malicious code
    # We use a safe read-only probe: __import__('os').getpid()
    a_val_malicious = "9 if not __import__('os').getpid() else 42"
    passes_check2 = isinstance(a_val_malicious, str) and a_val_malicious[0].isdigit()
    try:
        # This would run in the JIT constant folding path
        result2 = eval(a_val_malicious) if passes_check2 else a_val_malicious  # noqa: S307
        exploited2 = isinstance(result2, int) and result2 == 42
        results.append({
            "payload": a_val_malicious,
            "type": "code_exec_via_eval",
            "exploited": exploited2,
            "detail": f"eval() returned {result2!r} — arbitrary Python code executed",
        })
    except Exception as e:
        results.append({
            "payload": a_val_malicious, "type": "code_exec_via_eval",
            "exploited": False, "detail": f"Exception: {e}",
        })

    # --- Payload 3: eval(f"{{a_val}} {{operator}} {{b_val}}") at line 947 ---
    # If a_val = "1", operator = "+", b_val = "__import__('os').getpid()"
    a_val3 = "1"
    operator3 = "+"
    b_val3 = "__import__('os').getpid()"
    expr3 = f"{a_val3} {operator3} {b_val3}"
    try:
        result3 = eval(expr3)  # noqa: S307
        results.append({
            "payload": expr3,
            "type": "format_string_eval",
            "exploited": isinstance(result3, int) and result3 > 1,
            "detail": f"eval({expr3!r}) = {result3!r} — computed 1 + getpid()",
        })
    except Exception as e:
        results.append({
            "payload": expr3, "type": "format_string_eval",
            "exploited": False, "detail": f"Exception: {e}",
        })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 03: eval() Injection in JIT Compiler")
    print("=" * 70)
    for r in test_eval_injection_in_jit():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
