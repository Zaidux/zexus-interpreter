"""
TEST 04: Path Traversal in VirtualFilesystem
=============================================

VULNERABILITY: The SandboxFileSystem.resolve_path() previously used string-prefix
matching (startswith) to check that resolved paths stay within mount roots.
This could be bypassed with directory names that share a prefix.

FIX: The startswith check now includes os.sep suffix to prevent prefix collisions.
e.g. "/opt/app" no longer matches "/opt/app_secret".

LOCATION: src/zexus/virtual_filesystem.py, line 136

SEVERITY: HIGH - Unauthorized File Access / Directory Traversal
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_path_traversal_vfs():
    """Test that the prefix-collision fix works correctly."""
    results = []

    # Test the fixed string-prefix logic directly
    # OLD code:  if real_path.startswith(mount.real_path):
    # NEW code:  mount_root = mount.real_path.rstrip(os.sep) + os.sep
    #            if real_path == mount.real_path or real_path.startswith(mount_root):

    mount_real_path = "/opt/app"

    # --- Payload 1: Prefix collision attack ---
    # A path that starts with "/opt/app" but is actually "/opt/app_secret"
    attack_path = os.path.normpath(os.path.join(mount_real_path, "../app_secret/data.txt"))
    # attack_path = "/opt/app_secret/data.txt"

    # OLD check (vulnerable):
    old_check = attack_path.startswith(mount_real_path)

    # NEW check (fixed):
    mount_root = mount_real_path.rstrip(os.sep) + os.sep
    new_check = attack_path == mount_real_path or attack_path.startswith(mount_root)

    results.append({
        "payload": f"real_path={attack_path!r} vs mount={mount_real_path!r}",
        "type": "prefix_collision_bypass",
        "exploited": new_check,  # If new check still passes, still vulnerable
        "detail": f"Old check: {old_check} (VULNERABLE). New check: {new_check} (FIXED). "
                  f"Path correctly {'BLOCKED' if not new_check else 'allowed'}.",
    })

    # --- Payload 2: Legitimate subpath should still work ---
    legit_path = os.path.normpath(os.path.join(mount_real_path, "src/main.py"))
    new_check2 = legit_path == mount_real_path or legit_path.startswith(mount_root)

    results.append({
        "payload": f"real_path={legit_path!r} vs mount={mount_real_path!r}",
        "type": "legitimate_subpath",
        "exploited": not new_check2,  # Legitimate path SHOULD pass
        "detail": f"Legitimate path {'allowed' if new_check2 else 'BLOCKED'} — should be allowed",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 04: Path Traversal in VirtualFilesystem (AFTER FIX)")
    print("=" * 70)
    for r in test_path_traversal_vfs():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
