"""
TEST 05: Environment Variable Leakage via os_module.py
======================================================

VULNERABILITY: OSModule.listenv() returns ALL environment variables
including sensitive ones (API keys, tokens, credentials, secrets).
Additionally, setenv()/unsetenv() allow unrestricted modification.

LOCATION: src/zexus/stdlib/os_module.py, lines 75-77

SEVERITY: MEDIUM - Information Disclosure / Credential Theft

ATTACK VECTOR: A Zexus program can call os.listenv() to dump all
environment variables, which may include cloud credentials, API tokens,
database passwords, etc.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.stdlib.os_module import OSModule


SENSITIVE_PATTERNS = [
    "SECRET", "TOKEN", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL",
    "AUTH", "PRIVATE", "API_KEY", "AWS_", "AZURE_", "GCP_",
    "DATABASE_URL", "DB_PASS", "GITHUB_TOKEN", "NPM_TOKEN",
    "DOCKER_", "SSH_", "GPG_",
]


def test_env_var_leakage():
    """Demonstrate environment variable exposure."""
    results = []

    # --- Setup: Plant a fake secret to detect ---
    os.environ["FAKE_API_SECRET_KEY"] = "sk-test-supersecret-12345"
    os.environ["FAKE_DB_PASSWORD"] = "p@ssw0rd!unsafe"

    # --- Payload 1: listenv() exposes everything ---
    env_vars = OSModule.listenv()
    leaked_secrets = {
        k: v for k, v in env_vars.items()
        if any(pat in k.upper() for pat in SENSITIVE_PATTERNS)
    }

    results.append({
        "payload": "OSModule.listenv()",
        "type": "env_dump_secrets",
        "exploited": len(leaked_secrets) > 0,
        "detail": f"Found {len(leaked_secrets)} sensitive env vars: {list(leaked_secrets.keys())}",
    })

    # --- Payload 2: setenv() can modify critical variables ---
    original_path = os.environ.get("PATH", "")
    OSModule.setenv("PATH", "/tmp/evil:$PATH")
    modified = os.environ.get("PATH", "").startswith("/tmp/evil")
    os.environ["PATH"] = original_path  # Restore immediately

    results.append({
        "payload": 'OSModule.setenv("PATH", "/tmp/evil:$PATH")',
        "type": "env_modification",
        "exploited": modified,
        "detail": "PATH was modified — could redirect system commands to attacker-controlled binaries",
    })

    # --- Payload 3: unsetenv() can remove security-critical vars ---
    os.environ["FAKE_SECURITY_FLAG"] = "enabled"
    OSModule.unsetenv("FAKE_SECURITY_FLAG")
    removed = "FAKE_SECURITY_FLAG" not in os.environ

    results.append({
        "payload": 'OSModule.unsetenv("FAKE_SECURITY_FLAG")',
        "type": "env_deletion",
        "exploited": removed,
        "detail": "Security-critical env var was deleted without restriction",
    })

    # Cleanup
    os.environ.pop("FAKE_API_SECRET_KEY", None)
    os.environ.pop("FAKE_DB_PASSWORD", None)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 05: Environment Variable Leakage")
    print("=" * 70)
    for r in test_env_var_leakage():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
