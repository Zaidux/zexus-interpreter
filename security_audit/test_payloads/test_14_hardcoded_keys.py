"""
TEST 14: Hardcoded Placeholder Crypto Keys in external_bridge.py
================================================================

VULNERABILITY: generate_sphincs_keypair() returned static hardcoded
strings ("sphincs_pub_placeholder", "sphincs_priv_placeholder") as
crypto keys. Anyone who read the source code would know every key.

FIX: Replaced with secrets.token_hex() to generate unique random
keys at each invocation.

LOCATION: src/zexus/external_bridge.py lines 6-9

SEVERITY: MEDIUM - Predictable Cryptographic Material
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_hardcoded_keys():
    """Verify that crypto keys are no longer hardcoded."""
    results = []

    from zexus.external_bridge import external_functions

    keypair1 = external_functions["generate_sphincs_keypair"]()
    keypair2 = external_functions["generate_sphincs_keypair"]()

    # Check they're not the old placeholder values
    pub_is_placeholder = keypair1["public_key"] == "sphincs_pub_placeholder"
    priv_is_placeholder = keypair1["private_key"] == "sphincs_priv_placeholder"

    results.append({
        "payload": "generate_sphincs_keypair()",
        "type": "static_placeholder_check",
        "exploited": pub_is_placeholder or priv_is_placeholder,
        "detail": f"public_key={'PLACEHOLDER' if pub_is_placeholder else 'RANDOM'}, "
                  f"private_key={'PLACEHOLDER' if priv_is_placeholder else 'RANDOM'}",
    })

    # Check that two calls produce different keys (randomness)
    same_keys = (keypair1["public_key"] == keypair2["public_key"] and
                 keypair1["private_key"] == keypair2["private_key"])

    results.append({
        "payload": "Two sequential calls to generate_sphincs_keypair()",
        "type": "randomness_check",
        "exploited": same_keys,
        "detail": f"Keys are {'IDENTICAL (not random)' if same_keys else 'DIFFERENT (properly random)'}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 14: Hardcoded Crypto Keys in external_bridge.py (AFTER FIX)")
    print("=" * 70)
    for r in test_hardcoded_keys():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
