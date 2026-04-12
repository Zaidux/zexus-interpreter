"""
Test 19: C/C++ Integration — Memory Leaks
==========================================
Vulnerability: Multiple memory leaks in native_runtime.cpp and cabi.c:
1. zexus_rt_read()/zexus_cabi_read(): PyUnicode_FromString("r") never decref'd
2. zexus_rt_write()/zexus_cabi_write(): PyUnicode_FromString("w") never decref'd
3. zexus_rt_define_entity()/zexus_cabi_define_entity():
   PyUnicode_FromString("entity") never decref'd

Fix: Changed to store temporary strings in local vars and Py_DECREF after use.
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
        # Check the old leaky patterns are gone
        # Old: PyTuple_Pack(2, path, PyUnicode_FromString("r"))
        if 'PyTuple_Pack(2, path, PyUnicode_FromString("r"))' in source:
            print(f"LEAK: {fname} still has inline PyUnicode_FromString in read()")
            SAFE = False
        if 'PyTuple_Pack(2, path, PyUnicode_FromString("w"))' in source:
            print(f"LEAK: {fname} still has inline PyUnicode_FromString in write()")
            SAFE = False
        # Check define_entity uses type_str pattern
        if 'PyDict_SetItemString(members, "_type", PyUnicode_FromString("entity"))' in source:
            print(f"LEAK: {fname} still has inline PyUnicode_FromString in define_entity()")
            SAFE = False
        # Verify the fixes use mode/type_str variables
        if "Py_DECREF(mode)" not in source:
            print(f"LEAK: {fname} missing Py_DECREF(mode) for file mode string")
            SAFE = False
        if "Py_DECREF(type_str)" not in source:
            print(f"LEAK: {fname} missing Py_DECREF(type_str) for entity type")
            SAFE = False

if SAFE:
    print("SAFE: Memory leaks fixed in both native_runtime.cpp and cabi.c")
else:
    print("FAIL: Memory leaks persist")
    sys.exit(1)
