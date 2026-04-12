"""
TEST 13: Path Traversal via test_name in ZPICS snapshot files
==============================================================

VULNERABILITY: save_snapshot() and load_snapshot() used test_name
directly in file paths without sanitization. A test_name like
"../../etc/cron.d/evil" could write snapshot files outside the
snapshot directory.

FIX: Added _sanitize_test_name() that strips path separators and
".." sequences from test names.

LOCATION: src/zexus/testing/zpics.py, zpics_runtime.py

SEVERITY: MEDIUM - Path Traversal via test name
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_zpics_path_traversal():
    """Verify that test_name is sanitized in snapshot operations."""
    results = []

    from zexus.testing.zpics import _sanitize_test_name

    # --- Payload 1: ".." traversal in test name ---
    name1 = "../../etc/cron.d/evil"
    sanitized1 = _sanitize_test_name(name1)
    has_traversal = ".." in sanitized1 or "/" in sanitized1

    results.append({
        "payload": name1,
        "type": "dotdot_traversal",
        "exploited": has_traversal,
        "detail": f"Sanitized to '{sanitized1}' — {'STILL HAS TRAVERSAL (vulnerable)' if has_traversal else 'SAFE'}",
    })

    # --- Payload 2: backslash path separator ---
    name2 = "..\\..\\windows\\system32\\evil"
    sanitized2 = _sanitize_test_name(name2)
    has_traversal2 = "\\" in sanitized2 or ".." in sanitized2

    results.append({
        "payload": name2,
        "type": "backslash_traversal",
        "exploited": has_traversal2,
        "detail": f"Sanitized to '{sanitized2}' — {'STILL HAS TRAVERSAL (vulnerable)' if has_traversal2 else 'SAFE'}",
    })

    # --- Payload 3: Normal test names still work ---
    name3 = "my_test_function"
    sanitized3 = _sanitize_test_name(name3)
    works = sanitized3 == name3

    results.append({
        "payload": name3,
        "type": "normal_name_preserved",
        "exploited": not works,
        "detail": f"Sanitized to '{sanitized3}' — {'UNCHANGED (correct)' if works else 'INCORRECTLY MODIFIED'}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 13: ZPICS Path Traversal via test_name (AFTER FIX)")
    print("=" * 70)
    for r in test_zpics_path_traversal():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
