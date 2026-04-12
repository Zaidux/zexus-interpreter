"""
TEST 03: eval() Injection in JIT Compiler (jit.py)
===================================================

VULNERABILITY: The JIT optimizer previously used eval() on string values
that start with a digit character. The check `a_val[0].isdigit()` was
insufficient — a string like "1+__import__('os').system('id')" starts
with '1' (a digit) but contains arbitrary Python code.

LOCATION: src/zexus/vm/jit.py, lines 512-513, 947

SEVERITY: CRITICAL - Code Execution

FIX: Replaced eval() with _safe_parse_number() which uses a strict regex
to accept only plain numeric literals, and _safe_binop() for compile-time
constant folding.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_eval_injection_in_jit():
    """Verify that the JIT now uses safe numeric parsing instead of eval()."""
    results = []

    # Import the actual fixed function
    from zexus.vm.jit import _safe_parse_number, _safe_binop

    # --- Payload 1: Simple arithmetic that should NOT be evaluated ---
    a_val = "1+1"
    parsed = _safe_parse_number(a_val)
    results.append({
        "payload": a_val,
        "type": "arithmetic_string_rejected",
        "exploited": parsed is not None and parsed == 2,
        "detail": f"_safe_parse_number({a_val!r}) = {parsed!r} — {'REJECTED (safe)' if parsed is None else 'EVALUATED (vulnerable)'}",
    })

    # --- Payload 2: Code execution disguised as a number ---
    a_val_malicious = "9 if not __import__('os').getpid() else 42"
    parsed2 = _safe_parse_number(a_val_malicious)
    results.append({
        "payload": a_val_malicious,
        "type": "code_exec_via_parse",
        "exploited": parsed2 is not None,
        "detail": f"_safe_parse_number() = {parsed2!r} — {'REJECTED (safe)' if parsed2 is None else 'PARSED (vulnerable)'}",
    })

    # --- Payload 3: Verify safe binop works for legitimate values ---
    result3 = _safe_binop(10, "+", 20)
    results.append({
        "payload": "_safe_binop(10, '+', 20)",
        "type": "safe_binop_works",
        "exploited": result3 != 30,  # If it DOESN'T work, that's a regression
        "detail": f"_safe_binop(10, '+', 20) = {result3!r} — {'CORRECT' if result3 == 30 else 'WRONG'}",
    })

    # --- Payload 4: Verify legitimate numbers are still parsed ---
    for num_str, expected in [("42", 42), ("3.14", 3.14), ("-7", -7), ("1e10", 1e10)]:
        parsed = _safe_parse_number(num_str)
        results.append({
            "payload": f"_safe_parse_number({num_str!r})",
            "type": "legitimate_number_parsed",
            "exploited": parsed != expected,
            "detail": f"Got {parsed!r}, expected {expected!r}",
        })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 03: eval() Injection in JIT Compiler (AFTER FIX)")
    print("=" * 70)
    for r in test_eval_injection_in_jit():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
