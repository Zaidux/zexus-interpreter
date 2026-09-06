"""Phase C regression tests: capability store unification + Bytes type.

Capability unification: grants issued by `grant` previously landed in the
global capability manager while checks read an integration-private manager
(a *third* mismatch: entities didn't map and plugin names never expanded),
so every gated builtin (http_get, file_read_text, ...) was permanently
denied with no escape hatch. Grants now land where checks read them.

Bytes (GRAMMAR.md section 4): b"..." literals with raw \\xNN escapes, a
Bytes object with methods shared by both engines, concat/equality infix,
and bytes_from_hex.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import pytest


def _parse(code: str):
    from zexus.lexer import Lexer
    from zexus.parser.parser import UltimateParser

    parser = UltimateParser(Lexer(code, filename="<test>"), enable_advanced_strategies=False)
    program = parser.parse_program()
    return program


def _eval(code: str, env=None):
    from zexus.evaluator.core import evaluate
    from zexus.environment import Environment

    # NOTE: an empty Environment is falsy (__len__) — `env or Environment()`
    # would silently replace a passed-in env. Use an explicit None check.
    env = env if env is not None else Environment()
    return evaluate(_parse(code), env, debug_mode=False, use_vm=False)


@pytest.fixture(autouse=True)
def _clean_capability_state():
    """Isolate the global capability manager between tests."""
    from zexus.capability_system import get_capability_manager

    manager = get_capability_manager()
    saved = dict(manager.granted_capabilities)
    manager.granted_capabilities.clear()
    yield
    manager.granted_capabilities.clear()
    manager.granted_capabilities.update(saved)


# ── Capability store unification ───────────────────────────────────────


def test_integration_shares_global_capability_manager():
    from zexus.capability_system import get_capability_manager
    from zexus.evaluator.integration import get_integration

    assert get_integration().capability_manager is get_capability_manager()


def test_grant_lands_where_checks_read():
    from zexus.capability_system import get_capability_manager

    _eval("grant self network")
    manager = get_capability_manager()
    allowed, _ = manager.check_capability("default", "network.tcp")
    assert allowed, "grant self network did not grant network.tcp to the runtime context"
    allowed_http, _ = manager.check_capability("default", "network.http")
    assert allowed_http, "plugin set expansion missing network.http"


def test_revoke_removes_granted_capabilities():
    from zexus.capability_system import get_capability_manager

    _eval("grant self io_full")
    manager = get_capability_manager()
    allowed, _ = manager.check_capability("default", "io.read")
    assert allowed

    _eval("revoke self io_full")
    denied, _ = manager.check_capability("default", "io.read")
    assert not denied, "revoke did not undo the matching grant"


def test_gated_builtin_denied_then_granted_then_revoked(tmp_path):
    """End-to-end: file_read_text is denied by default, usable after a
    grant, denied again after revoke (the capability loop)."""
    from zexus.environment import Environment

    env = Environment()
    scratch = "._phase_c_io_test.txt"

    # Default: denied (safe by default)
    r = _eval(f'file_write_text("{scratch}", "data")', env)
    assert hasattr(r, "message") and "io.write" in r.message

    # Granted: works
    _eval("grant self io_full", env)
    r = _eval(f'file_write_text("{scratch}", "data")', env)
    assert not hasattr(r, "message") or "denied" not in str(getattr(r, "message", ""))
    r = _eval(f'file_read_text("{scratch}")', env)
    assert getattr(r, "value", r) == "data"

    # Revoked: denied again
    _eval("revoke self io_full", env)
    r = _eval(f'file_read_text("{scratch}")', env)
    assert hasattr(r, "message") and "io.read" in r.message

    pathlib.Path(scratch).unlink(missing_ok=True)


def test_grant_specific_capability_name():
    from zexus.capability_system import get_capability_manager

    _eval("grant self network.tcp")
    allowed, _ = get_capability_manager().check_capability("default", "network.tcp")
    assert allowed


# ── Bytes type ─────────────────────────────────────────────────────────


def test_bytes_literal_lexes():
    from zexus.lexer import Lexer

    lx = Lexer('b"\\x41"', filename="<t>")
    tok = lx.next_token()
    assert str(tok.type) == "BYTES"
    assert tok.literal == b"A"


def test_bytes_xnn_is_raw_byte_not_codepoint():
    r = _eval('b"\\xff".at(0)')
    assert r.value == 255


def test_bytes_len_concat_and_equality():
    assert _eval('b"\\x00\\x01\\xff".len()').value == 3
    joined = _eval('b"hi" + b"!"')
    assert isinstance(joined.value, bytes) and joined.value == b"hi!"
    assert _eval('b"ab" == b"ab"').value is True
    assert _eval('b"ab" == b"cd"').value is False
    assert _eval('b"ab" != b"cd"').value is True


def test_bytes_methods():
    assert _eval('b"hi".to_hex()').value == "6869"
    assert _eval('b"\\x41".to_string()').value == "A"
    assert _eval('b"abc".at(1)').value == 98
    assert _eval('b"abcdef".slice(1, 3).to_hex()').value == "6263"
    assert _eval('b"abcdef".contains(b"cd")').value is True
    assert _eval('b"abcdef".contains(b"zz")').value is False


def test_bytes_from_hex_builtin():
    r = _eval('bytes_from_hex("deadbeef")')
    assert isinstance(r.value, bytes) and r.value == b"\xde\xad\xbe\xef"
    assert _eval('bytes_from_hex("deadbeef").to_hex()').value == "deadbeef"


def test_bytes_u_escape_rejected():
    from zexus.lexer import Lexer

    with pytest.raises(Exception):
        lx = Lexer('b"\\u0041"', filename="<t>")
        lx.next_token()


def test_bytes_interpolation_rejected():
    from zexus.lexer import Lexer

    with pytest.raises(Exception):
        lx = Lexer('b"pre${x}"', filename="<t>")
        lx.next_token()


# ── VM parity: same results on both engines ────────────────────────────


def _vm_env(code: str):
    from zexus.vm.compiler import BytecodeCompiler
    from zexus.vm.vm import VM

    vm = VM()
    vm.execute(BytecodeCompiler().compile(_parse(code)))
    return vm.env


def test_vm_bytes_roundtrip():
    env = _vm_env('let h = b"hi".to_hex()\nlet p = b"\\x00\\x01"')
    assert env["h"] == "6869"
    assert bytes(env["p"].value) == b"\x00\x01"


def test_vm_builtin_crypto_module_reachable():
    """use "crypto" on the VM exposes the merged registry (sha256 was null
    before the lazy-init fix — builtin registry required an evaluator)."""
    env = _vm_env('use "crypto"\nlet h = sha256("abc")')
    assert env["h"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_vm_and_treewalk_agree_on_bytes_program():
    from zexus.evaluator.core import evaluate
    from zexus.environment import Environment

    program = 'let a = b"\\x01\\x02" + b"\\x03"\nlet h = a.to_hex()'
    env_vm = _vm_env(program)

    env_tw = Environment()
    evaluate(_parse(program), env_tw, use_vm=False)

    assert env_vm["h"] == "010203"
    assert env_tw.get("h").value == "010203"
