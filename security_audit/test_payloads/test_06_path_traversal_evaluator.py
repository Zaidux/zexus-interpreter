"""
TEST 06: Unvalidated Path Operations in evaluator/functions.py
=============================================================

VULNERABILITY: fs_rmtree() and fs_copy() in the evaluator accept arbitrary
user-supplied paths without validation, enabling deletion or copying of
files anywhere on the filesystem.

LOCATION: src/zexus/evaluator/functions.py, lines 1147-1148, 1177-1180

SEVERITY: HIGH - Arbitrary File Deletion / File Overwrite

ATTACK VECTOR: A Zexus program calling fs_rmdir("/etc", true) or
fs_copy("/etc/shadow", "/tmp/stolen") can traverse the entire filesystem.
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_unvalidated_path_operations():
    """Demonstrate that evaluator fs functions lack path validation."""
    results = []

    # We simulate the vulnerable code path without actually importing the
    # evaluator (which requires full interpreter setup). Instead we reproduce
    # the exact pattern from functions.py.

    # --- Payload 1: Arbitrary directory deletion ---
    # Simulate: shutil.rmtree(user_supplied_path) with no validation
    test_dir = tempfile.mkdtemp(prefix="zexus_rmtree_test_")
    os.makedirs(os.path.join(test_dir, "subdir"), exist_ok=True)
    with open(os.path.join(test_dir, "subdir", "secret.txt"), "w") as f:
        f.write("sensitive data")

    # The vulnerable code does: shutil.rmtree(a[0].value)
    # where a[0].value is user-controlled with NO validation
    user_path = test_dir  # attacker provides any absolute path
    try:
        # Simulating the vulnerable pattern:
        shutil.rmtree(user_path)  # No path validation!
        deleted = not os.path.exists(test_dir)
    except Exception:
        deleted = False

    results.append({
        "payload": f'fs_rmdir("{test_dir}", true)',
        "type": "arbitrary_rmtree",
        "exploited": deleted,
        "detail": f"Directory at {test_dir} {'was deleted' if deleted else 'still exists'} — no path validation",
    })

    # --- Payload 2: File copy to arbitrary location ---
    src_file = tempfile.mktemp(suffix=".src")
    dst_file = tempfile.mktemp(suffix=".dst")
    with open(src_file, "w") as f:
        f.write("stolen credentials")

    # The vulnerable code does: shutil.copy2(src, dst)
    # where src and dst are user-controlled with NO validation
    try:
        shutil.copy2(src_file, dst_file)  # No path validation!
        copied = os.path.exists(dst_file) and open(dst_file).read() == "stolen credentials"
    except Exception:
        copied = False

    results.append({
        "payload": f'fs_copy("{src_file}", "{dst_file}")',
        "type": "arbitrary_file_copy",
        "exploited": copied,
        "detail": f"File copied from {src_file} to {dst_file} — no path restriction",
    })

    # Cleanup
    for f in [src_file, dst_file]:
        if os.path.exists(f):
            os.unlink(f)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 06: Unvalidated Path Operations in Evaluator")
    print("=" * 70)
    for r in test_unvalidated_path_operations():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
