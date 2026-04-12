"""
TEST 12: Pickle Serialization Removed from binary_bytecode.py
==============================================================

VULNERABILITY: _serialize_constant() used pickle.dumps() as a fallback
for non-JSON-serializable objects. If a malicious object with a custom
__reduce__ was serialized and later deserialized elsewhere, it could
achieve arbitrary code execution.

FIX: Replaced pickle.dumps() fallback with repr() — all non-JSON
objects are now stored as string representations.

LOCATION: src/zexus/vm/binary_bytecode.py lines 236-250

SEVERITY: MEDIUM - Removed attack surface for pickle deserialization
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_pickle_serialization_removed():
    """Verify that pickle is no longer used for serialization."""
    results = []

    # Check that the source code no longer imports pickle for serialization
    bytecode_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'zexus', 'vm', 'binary_bytecode.py'
    )
    with open(bytecode_path, "r") as f:
        source = f.read()

    # Count occurrences of pickle usage in serialization context
    pickle_dumps_count = source.count("pickle.dumps")

    results.append({
        "payload": "Check for pickle.dumps in binary_bytecode.py",
        "type": "pickle_serialization_removed",
        "exploited": pickle_dumps_count > 0,
        "detail": f"Found {pickle_dumps_count} occurrences of pickle.dumps — {'STILL PRESENT (vulnerable)' if pickle_dumps_count > 0 else 'REMOVED (safe)'}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 12: Pickle Serialization in binary_bytecode.py (AFTER FIX)")
    print("=" * 70)
    for r in test_pickle_serialization_removed():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
