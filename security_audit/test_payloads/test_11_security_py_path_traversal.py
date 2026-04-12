"""
TEST 11: Path Traversal in security.py export_to_file() and trail sinks
========================================================================

VULNERABILITY: export_to_file() and trail file sinks accepted arbitrary
paths without validation, enabling writes outside the working directory.

FIX: Added realpath() boundary checks — export_to_file() enforces CWD
containment, trail sinks enforce AUDIT_DIR/STORAGE_DIR containment.

LOCATION: src/zexus/security.py lines 180-188, 447-453

SEVERITY: HIGH - Arbitrary File Write
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_security_path_traversal():
    """Verify that export_to_file() blocks path traversal."""
    results = []

    from zexus.security import AuditLog

    log = AuditLog()
    log.log("test_data", "read", "string", additional_context={"source": "security_test"})

    # --- Payload 1: Export to absolute path outside CWD ---
    result = log.export_to_file("/tmp/pwned_audit_log.json")
    wrote = os.path.exists("/tmp/pwned_audit_log.json")
    # Cleanup just in case
    if wrote:
        os.unlink("/tmp/pwned_audit_log.json")

    results.append({
        "payload": "/tmp/pwned_audit_log.json",
        "type": "absolute_path_export",
        "exploited": wrote,
        "detail": f"export_to_file('/tmp/...') {'WROTE FILE (vulnerable)' if wrote else 'BLOCKED (safe)'}",
    })

    # --- Payload 2: Relative traversal ---
    result2 = log.export_to_file("../../../tmp/escape.json")
    wrote2 = os.path.exists("/tmp/escape.json")
    if wrote2:
        os.unlink("/tmp/escape.json")

    results.append({
        "payload": "../../../tmp/escape.json",
        "type": "relative_traversal_export",
        "exploited": wrote2,
        "detail": f"export_to_file('../../tmp/...') {'WROTE FILE (vulnerable)' if wrote2 else 'BLOCKED (safe)'}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 11: Path Traversal in security.py (AFTER FIX)")
    print("=" * 70)
    for r in test_security_path_traversal():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
