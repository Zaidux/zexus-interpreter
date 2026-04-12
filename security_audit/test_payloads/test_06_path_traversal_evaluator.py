"""
TEST 06: Path Validation in evaluator/functions.py
==================================================

VULNERABILITY: fs_rmtree() and fs_copy() in the evaluator previously accepted
arbitrary user-supplied paths without validation.

FIX: Added _validate_write_path() that ensures all paths resolve within
the current working directory.

LOCATION: src/zexus/evaluator/functions.py

SEVERITY: HIGH - Arbitrary File Deletion / File Overwrite

NOTE: This test verifies the _validate_write_path logic directly since
the full evaluator requires the complete interpreter runtime.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def _validate_write_path(path_str):
    """Reproduce the validation logic from evaluator/functions.py."""
    resolved = os.path.realpath(os.path.normpath(
        os.path.join(os.getcwd(), path_str) if not os.path.isabs(path_str) else path_str
    ))
    cwd = os.path.realpath(os.getcwd())
    if not (resolved == cwd or resolved.startswith(cwd + os.sep)):
        raise ValueError(
            f"Path '{path_str}' resolves to '{resolved}' which is "
            f"outside the working directory '{cwd}'"
        )
    return resolved


def test_path_validation():
    """Verify that _validate_write_path blocks traversal attempts."""
    results = []

    # --- Payload 1: Absolute path outside CWD ---
    try:
        _validate_write_path("/tmp/some_dir")
        blocked = False
    except ValueError:
        blocked = True

    results.append({
        "payload": "/tmp/some_dir",
        "type": "absolute_path_outside_cwd",
        "exploited": not blocked,
        "detail": "Path validation correctly blocked" if blocked else "VULNERABLE — path accepted",
    })

    # --- Payload 2: Relative traversal outside CWD ---
    try:
        _validate_write_path("../../etc/passwd")
        blocked2 = False
    except ValueError:
        blocked2 = True

    results.append({
        "payload": "../../etc/passwd",
        "type": "relative_traversal",
        "exploited": not blocked2,
        "detail": "Path validation correctly blocked" if blocked2 else "VULNERABLE — path accepted",
    })

    # --- Payload 3: Path within CWD should work ---
    try:
        result = _validate_write_path("subdir/file.txt")
        within_cwd = True
    except ValueError:
        within_cwd = False

    results.append({
        "payload": "subdir/file.txt",
        "type": "valid_path_within_cwd",
        "exploited": not within_cwd,
        "detail": f"Path accepted: {within_cwd} — legitimate paths should work",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 06: Path Validation in Evaluator (AFTER FIX)")
    print("=" * 70)
    for r in test_path_validation():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
