"""
Test 16: C/C++ Integration — Unrestricted Module Import
=======================================================
Vulnerability: zexus_rt_import()/zexus_cabi_import() called PyImport_Import()
with no restrictions, allowing JIT-compiled code to import dangerous modules
(os, subprocess, ctypes, etc.) and bypass Python-level sandboxing.

Fix: Added ZX_IMPORT_BLOCKLIST and zx_is_import_blocked() that rejects
dangerous module names and their submodules.
"""
import os
import sys

SAFE = True
BLOCKED_MODULES = [
    "os", "subprocess", "shutil", "ctypes", "importlib",
    "sys", "signal", "socket", "http", "urllib",
    "pathlib", "tempfile", "glob", "fnmatch",
    "code", "codeop", "compile", "compileall",
    "multiprocessing", "threading", "concurrent",
]

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "zexus", "vm")
for fname in ("native_runtime.cpp", "cabi.c"):
    fpath = os.path.join(SRC_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            source = f.read()
        # Verify blocklist exists
        if "ZX_IMPORT_BLOCKLIST" not in source:
            print(f"VULNERABLE: {fname} missing import blocklist")
            SAFE = False
        if "zx_is_import_blocked" not in source:
            print(f"VULNERABLE: {fname} missing import block check")
            SAFE = False
        # Verify all critical modules are in the blocklist
        for mod in BLOCKED_MODULES:
            if f'"{mod}"' not in source:
                print(f"VULNERABLE: {fname} not blocking '{mod}'")
                SAFE = False

if SAFE:
    print("SAFE: Dangerous module imports blocked in both C/C++ files")
else:
    print("FAIL: Import restriction vulnerability persists")
    sys.exit(1)
