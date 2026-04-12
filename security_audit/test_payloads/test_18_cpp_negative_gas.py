"""
Test 18: C/C++ Integration — Negative Gas Charge Bypass
========================================================
Vulnerability: gas_charge() in native_runtime.cpp and cabi.c didn't validate
that the amount was non-negative. Passing a negative value would INCREASE
remaining gas via:
    new_gas = PyNumber_Subtract(cur, subtrahend)
A negative subtrahend means addition: gas actually goes UP.

Fix: Added validation that rejects negative amounts with an
"InvalidGasAmount" error dict.
"""
import os
import sys

SAFE = True

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "zexus", "vm")
for fname in ("native_runtime.cpp", "cabi.c"):
    fpath = os.path.join(SRC_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            source = f.read()
        if "InvalidGasAmount" not in source:
            print(f"VULNERABLE: {fname} not rejecting negative gas charges")
            SAFE = False
        if "Gas charge amount must be non-negative" not in source:
            print(f"VULNERABLE: {fname} missing negative gas validation")
            SAFE = False

if SAFE:
    print("SAFE: Negative gas charge bypass blocked in both files")
else:
    print("FAIL: Negative gas charge vulnerability persists")
    sys.exit(1)
