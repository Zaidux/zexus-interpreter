"""
Test 17: C/C++ Integration — Use-After-Free in atomic_add
==========================================================
Vulnerability: In both native_runtime.cpp and cabi.c, the atomic_add function
had:
    PyObject *delta_val = delta ? delta : PyLong_FromLong(0);
    if (!delta) Py_DECREF(delta_val);  // frees before use!
    PyObject *new_val = PyNumber_Add(current, delta_val);  // use-after-free

This decrefs delta_val BEFORE using it in PyNumber_Add, causing a
use-after-free when delta is NULL. Python's small integer caching masks
this for value 0, but it's still a refcount bug.

Fix: Changed to use a separate `delta_owned` pointer that is DECREF'd only
AFTER PyNumber_Add completes.
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
        # Verify the old buggy pattern is gone
        if "if (!delta) Py_DECREF(delta_val)" in source:
            print(f"VULNERABLE: {fname} has use-after-free in atomic_add")
            SAFE = False
        # Verify the fix uses delta_owned pattern
        if "delta_owned" not in source:
            print(f"VULNERABLE: {fname} missing delta_owned fix pattern")
            SAFE = False

if SAFE:
    print("SAFE: Use-after-free in atomic_add fixed in both files")
else:
    print("FAIL: Use-after-free vulnerability persists")
    sys.exit(1)
