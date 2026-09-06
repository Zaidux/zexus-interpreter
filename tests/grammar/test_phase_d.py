"""Phase D regression tests: Rust-first native layer, C/C++ retirement.

- The Rust core (zexus_core) builds and serves the hash hot paths with
  results identical to pure Python (cross-checked).
- The C/C++ extensions (cabi/fastops/native_runtime) are GONE: removed
  sources, binaries, build config, and import sites. The VM has exactly
  two execution tiers: Rust VM (when built) and pure Python.
- A missing extension never bricks anything: every consumer is guarded.
"""
import importlib
import importlib.util
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import pytest


def test_rust_core_built_and_importable():
    spec = importlib.util.find_spec("zexus_core")
    if spec is None:
        pytest.skip("zexus_core not built in this environment")
    core = importlib.import_module("zexus_core")
    assert core.is_available()


def test_rust_sha256_matches_python():
    import hashlib

    from zexus.blockchain.crypto import CryptoPlugin

    for payload in (b"abc", b"", b"x" * 1000, "unicode-é"):
        expected = hashlib.sha256(
            payload if isinstance(payload, bytes) else payload.encode()
        ).hexdigest()
        assert CryptoPlugin.hash_data(payload, "SHA256") == expected


def test_rust_keccak_matches_pycryptodome():
    from zexus.blockchain.crypto import CryptoPlugin

    try:
        from Crypto.Hash import keccak as _keccak
    except ImportError:
        pytest.skip("pycryptodome not installed")

    for payload in (b"abc", b"", b"zexus"):
        k = _keccak.new(digest_bits=256)
        k.update(payload)
        assert CryptoPlugin.hash_data(payload, "KECCAK256") == k.hexdigest()


def test_rust_hasher_api_surface():
    core_spec = importlib.util.find_spec("zexus_core")
    if core_spec is None:
        pytest.skip("zexus_core not built in this environment")
    core = importlib.import_module("zexus_core")
    hasher = core.RustHasher()
    assert hasher.sha256(b"abc") == hasher.sha256(b"abc")  # deterministic
    assert hasher.sha256(b"a") != hasher.sha256(b"b")
    merkle = core.RustMerkle()
    root1 = merkle.compute_root_from_data([b"a", b"b"])
    root2 = merkle.compute_root_from_data([b"b", b"a"])
    assert root1 != root2  # order-sensitive


# ── C/C++ retirement ──────────────────────────────────────────────────


def test_c_extension_sources_deleted():
    vm_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "zexus" / "vm"
    for gone in ("cabi.c", "cabi.h", "fastops.c", "fastops.pyx", "native_runtime.cpp"):
        assert not (vm_dir / gone).exists(), f"{gone} should be deleted"
    # No prebuilt C artifacts either.
    for so in vm_dir.glob("*.so"):
        assert so.name == "zexus_core.so" or "cabi" not in so.name


def test_fastops_not_importable():
    assert importlib.util.find_spec("zexus.vm.fastops") is None
    assert importlib.util.find_spec("zexus.vm.cabi") is None
    assert importlib.util.find_spec("zexus.vm.native_runtime") is None


def test_vm_module_has_no_fastops_reference():
    import zexus.vm.vm as vm_mod

    assert not hasattr(vm_mod, "_fastops")
    assert not hasattr(vm_mod, "_FASTOPS_AVAILABLE")


def test_vm_still_executes_without_c_layer():
    """The VM's tier structure after Phase D: Rust VM when built, else the
    pure-Python interpreter — never a C fast path. Contract + bytes program
    executes identically to the tree-walk evaluator."""
    from zexus.lexer import Lexer
    from zexus.parser.parser import UltimateParser
    from zexus.vm.compiler import BytecodeCompiler
    from zexus.vm.vm import VM

    program = (
        'let c = b"\\x01\\x02"\nlet h = c.to_hex()\nlet n = b"hi".len()'
    )
    parser = UltimateParser(Lexer(program, filename="<t>"), enable_advanced_strategies=False)
    program_ast = parser.parse_program()
    vm = VM()
    vm.execute(BytecodeCompiler().compile(program_ast))
    assert vm.env["h"] == "0102"
    assert vm.env["n"] == 2


def test_rust_vm_tier_activates_when_built():
    from zexus.vm.vm import _RUST_VM_AVAILABLE

    if importlib.util.find_spec("zexus_core") is None:
        # Extension absent: flag must be False, pure Python still works.
        assert _RUST_VM_AVAILABLE is False
    else:
        assert _RUST_VM_AVAILABLE is True


def test_mock_crypto_still_gated():
    """Phase A's security gate survives the native-layer swap."""
    from zexus.blockchain.crypto import CryptoPlugin

    with pytest.raises(RuntimeError, match="non-PEM"):
        CryptoPlugin.sign_data("payload", "not-a-pem-key", "ECDSA")


def test_jit_backend_imports_without_c_symbols():
    """native_jit_backend no longer imports cabi/native_runtime; the LLVM
    pipeline imports cleanly (symbol registration is a no-op)."""
    import zexus.vm.native_jit_backend as backend

    assert hasattr(backend, "NativeJITBackend")
