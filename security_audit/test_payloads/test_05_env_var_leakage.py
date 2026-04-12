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
    try:
        OSModule.setenv("PATH", "/tmp/evil:$PATH")
        modified = True
    except PermissionError:
        modified = False

    results.append({
        "payload": 'OSModule.setenv("PATH", "/tmp/evil:$PATH")',
        "type": "env_modification",
        "exploited": modified,
        "detail": "PATH modification blocked by PermissionError" if not modified else "PATH was modified — VULNERABLE",
    })

    # --- Payload 3: unsetenv() on non-sensitive variable still works ---
    # Non-sensitive env vars should be modifiable (this is expected behavior)
    os.environ["ZEXUS_USER_SETTING"] = "enabled"
    try:
        OSModule.unsetenv("ZEXUS_USER_SETTING")
        removed = "ZEXUS_USER_SETTING" not in os.environ
    except PermissionError:
        removed = False

    results.append({
        "payload": 'OSModule.unsetenv("ZEXUS_USER_SETTING")',
        "type": "env_deletion_non_sensitive",
        "exploited": False,  # Non-sensitive vars SHOULD be deletable
        "detail": f"Non-sensitive env var {'removed' if removed else 'kept'} — this is expected behavior",
    })

    # --- Payload 4: unsetenv() blocked for sensitive variable ---
    os.environ["MY_SECRET_KEY"] = "s3cr3t"
    try:
        OSModule.unsetenv("MY_SECRET_KEY")
        secret_removed = "MY_SECRET_KEY" not in os.environ
    except PermissionError:
        secret_removed = False

    results.append({
        "payload": 'OSModule.unsetenv("MY_SECRET_KEY")',
        "type": "env_deletion_sensitive_blocked",
        "exploited": secret_removed,
        "detail": "Deletion of sensitive env var blocked" if not secret_removed else "VULNERABLE — sensitive var was deleted",
    })

    # Cleanup
    os.environ.pop("FAKE_API_SECRET_KEY", None)
    os.environ.pop("FAKE_DB_PASSWORD", None)
    os.environ.pop("MY_SECRET_KEY", None)

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
