"""Phase 1 — Binary Bytecode Format tests.

Tests round-trip correctness for Python serialize/deserialize,
Rust deserialization, .zxc file I/O, checksum verification, and
size comparison with Python tuple representation.
"""

import os
import struct
import time
import pytest

from zexus.vm.bytecode import Bytecode, Opcode, BytecodeBuilder
from zexus.vm.binary_bytecode import (
    serialize, deserialize, save_zxc, load_zxc,
    compare_sizes, ZXC_MAGIC, ZXC_VERSION,
    ConstTag, OperandType,
    serialize_multi, deserialize_multi,
    save_zxcm, load_zxcm,
    zxc_path_for, is_zxc_fresh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_bytecode():
    """let x = 10 + 20; return x;"""
    b = BytecodeBuilder()
    b.emit_constant(10)
    b.emit_constant(20)
    b.emit("ADD")
    b.emit_store("x")
    b.emit_load("x")
    b.emit("RETURN")
    return b.build()


def _make_loop_bytecode():
    """while i < n: total += i; i += 1; return total;"""
    bc = Bytecode()
    # Constants: 0=0, 1="total", 2="i", 3="n", 4=1
    bc.constants = [0, "total", "i", "n", 1]
    bc.instructions = [
        (Opcode.LOAD_CONST, 0),   # 0: push 0
        (Opcode.STORE_NAME, 1),   # 1: total = 0
        (Opcode.LOAD_CONST, 0),   # 2: push 0
        (Opcode.STORE_NAME, 2),   # 3: i = 0
        # loop start (ip=4)
        (Opcode.LOAD_NAME, 2),    # 4: push i
        (Opcode.LOAD_NAME, 3),    # 5: push n
        (Opcode.LT, None),        # 6: i < n
        (Opcode.JUMP_IF_FALSE, 14),  # 7: exit loop
        (Opcode.LOAD_NAME, 1),    # 8: push total
        (Opcode.LOAD_NAME, 2),    # 9: push i
        (Opcode.ADD, None),       # 10: total + i
        (Opcode.STORE_NAME, 1),   # 11: total = ...
        (Opcode.LOAD_NAME, 2),    # 12: push i
        (Opcode.LOAD_CONST, 4),   # 13: push 1
        (Opcode.ADD, None),       # 14 (actual 14): i + 1
        (Opcode.STORE_NAME, 2),   # 15: i = ...
        (Opcode.JUMP, 4),         # 16: back to loop
        # exit (ip=17 — relabeled, but actual jump target is fine)
        (Opcode.LOAD_NAME, 1),    # 17: push total
        (Opcode.RETURN, None),    # 18: return
    ]
    return bc


def _make_blockchain_bytecode():
    """Bytecode with blockchain-specific opcodes."""
    bc = Bytecode()
    bc.constants = ["balance", 100, "Transfer", "from", "to", 50]
    bc.instructions = [
        (Opcode.STATE_READ, 0),       # read "balance"
        (Opcode.LOAD_CONST, 1),       # push 100
        (Opcode.GTE, None),           # balance >= 100
        (Opcode.REQUIRE, 0),          # require(condition)
        (Opcode.TX_BEGIN, None),      # begin transaction
        (Opcode.LOAD_CONST, 5),       # push 50
        (Opcode.STATE_WRITE, 0),      # write to "balance"
        (Opcode.EMIT_EVENT, (2,)),    # emit "Transfer"
        (Opcode.GAS_CHARGE, 1000),    # charge gas
        (Opcode.TX_COMMIT, None),     # commit
        (Opcode.RETURN, None),
    ]
    return bc


def _make_rich_constants_bytecode():
    """Bytecode with various constant types."""
    bc = Bytecode()
    bc.constants = [
        None,               # null
        True,               # bool
        False,              # bool
        42,                 # int32
        2**40,              # int64 (large)
        -999,               # negative int32
        3.14159,            # float64
        "hello world",      # string
        "",                 # empty string
        "emoji: 🚀",       # unicode
        [1, 2, 3],          # list
        {"key": "value"},   # dict → func_desc
    ]
    bc.instructions = [
        (Opcode.LOAD_CONST, i) for i in range(len(bc.constants))
    ]
    bc.instructions.append((Opcode.RETURN, None))
    return bc


def _make_register_bytecode():
    """Bytecode with register operations (tuple operands)."""
    bc = Bytecode()
    bc.constants = [10, 20]
    bc.instructions = [
        (Opcode.LOAD_REG, (0, 0)),      # r0 = const[0] (pair)
        (Opcode.LOAD_REG, (1, 1)),      # r1 = const[1]
        (Opcode.ADD_REG, (2, 0, 1)),    # r2 = r0 + r1 (triple)
        (Opcode.PUSH_REG, (2,)),        # push r2 to stack (pair with 1 elem)
        (Opcode.RETURN, None),
    ]
    return bc


# ---------------------------------------------------------------------------
# Python Round-Trip Tests
# ---------------------------------------------------------------------------

class TestPythonRoundTrip:
    """Verify serialize → deserialize produces identical Bytecode."""

    def test_simple_bytecode(self):
        original = _make_simple_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        assert len(restored.instructions) == len(original.instructions)
        assert len(restored.constants) == len(original.constants)

    def test_loop_bytecode(self):
        original = _make_loop_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        assert len(restored.instructions) == len(original.instructions)
        # Check constants match
        for i, (orig, rest) in enumerate(zip(original.constants, restored.constants)):
            assert orig == rest, f"Constant {i}: {orig!r} != {rest!r}"

    def test_blockchain_opcodes(self):
        original = _make_blockchain_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        assert len(restored.instructions) == len(original.instructions)

    def test_rich_constants(self):
        original = _make_rich_constants_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        for i, (orig, rest) in enumerate(zip(original.constants, restored.constants)):
            if orig is None:
                assert rest is None, f"Constant {i}: expected None, got {rest!r}"
            elif isinstance(orig, float):
                assert abs(orig - rest) < 1e-10, f"Constant {i}: {orig} != {rest}"
            elif isinstance(orig, dict):
                assert isinstance(rest, dict), f"Constant {i}: expected dict, got {type(rest)}"
            else:
                assert orig == rest, f"Constant {i}: {orig!r} != {rest!r}"

    def test_register_operands(self):
        original = _make_register_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        assert len(restored.instructions) == len(original.instructions)

    def test_empty_bytecode(self):
        bc = Bytecode()
        data = serialize(bc)
        restored = deserialize(data)
        assert len(restored.instructions) == 0
        assert len(restored.constants) == 0

    def test_opcode_values_preserved(self):
        """Verify opcode enum values survive serialization."""
        original = _make_loop_bytecode()
        data = serialize(original)
        restored = deserialize(data)
        for i, (orig, rest) in enumerate(zip(original.instructions, restored.instructions)):
            orig_op = orig[0].value if hasattr(orig[0], 'value') else orig[0]
            rest_op = rest[0].value if hasattr(rest[0], 'value') else rest[0]
            assert orig_op == rest_op, f"Instr {i}: opcode {orig_op} != {rest_op}"


# ---------------------------------------------------------------------------
# Checksum Tests
# ---------------------------------------------------------------------------

class TestChecksum:

    def test_checksum_verification_passes(self):
        bc = _make_simple_bytecode()
        data = serialize(bc, include_checksum=True)
        # Should not raise
        deserialize(data, verify_checksum=True)

    def test_corrupted_data_detected(self):
        bc = _make_simple_bytecode()
        data = bytearray(serialize(bc, include_checksum=True))
        # Corrupt a byte in the middle
        mid = len(data) // 2
        data[mid] ^= 0xFF
        with pytest.raises(ValueError, match="Checksum mismatch"):
            deserialize(bytes(data), verify_checksum=True)

    def test_no_checksum_mode(self):
        bc = _make_simple_bytecode()
        data = serialize(bc, include_checksum=False)
        # Should work without checksum verification
        restored = deserialize(data, verify_checksum=False)
        assert len(restored.instructions) == len(bc.instructions)


# ---------------------------------------------------------------------------
# File I/O Tests
# ---------------------------------------------------------------------------

class TestFileIO:

    def test_save_and_load(self, tmp_path):
        bc = _make_loop_bytecode()
        path = str(tmp_path / "test.zxc")
        save_zxc(path, bc)
        assert os.path.exists(path)
        restored = load_zxc(path)
        assert len(restored.instructions) == len(bc.instructions)
        assert len(restored.constants) == len(bc.constants)

    def test_file_starts_with_magic(self, tmp_path):
        bc = _make_simple_bytecode()
        path = str(tmp_path / "magic.zxc")
        save_zxc(path, bc)
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == ZXC_MAGIC

    def test_version_in_header(self, tmp_path):
        bc = _make_simple_bytecode()
        path = str(tmp_path / "version.zxc")
        save_zxc(path, bc)
        with open(path, "rb") as f:
            f.read(4)  # skip magic
            version = struct.unpack("<H", f.read(2))[0]
        assert version == ZXC_VERSION


# ---------------------------------------------------------------------------
# Rust Deserializer Tests
# ---------------------------------------------------------------------------

class TestRustDeserializer:

    @pytest.fixture
    def reader(self):
        try:
            import zexus_core
            return zexus_core.RustBytecodeReader()
        except ImportError:
            pytest.skip("Rust core not available")

    def test_validate(self, reader):
        bc = _make_simple_bytecode()
        data = serialize(bc)
        assert reader.validate(data) is True

    def test_validate_bad_data(self, reader):
        assert reader.validate(b"not a zxc file") is False

    def test_header_info(self, reader):
        bc = _make_loop_bytecode()
        data = serialize(bc)
        info = reader.header_info(data)
        assert info["magic_ok"] is True
        assert info["version"] == ZXC_VERSION
        assert info["n_constants"] == len(bc.constants)
        assert info["n_instructions"] == len(bc.instructions)

    def test_deserialize_simple(self, reader):
        bc = _make_simple_bytecode()
        data = serialize(bc)
        result = reader.deserialize(data)
        assert result["n_constants"] == len(bc.constants)
        assert result["n_instructions"] == len(bc.instructions)
        assert len(result["constants"]) == len(bc.constants)
        assert len(result["instructions"]) == len(bc.instructions)

    def test_deserialize_rich_constants(self, reader):
        bc = _make_rich_constants_bytecode()
        data = serialize(bc)
        result = reader.deserialize(data)
        consts = result["constants"]
        assert consts[0] is None          # null
        assert consts[1] is True          # bool
        assert consts[2] is False         # bool
        assert consts[3] == 42            # int32
        assert consts[4] == 2**40         # int64
        assert consts[5] == -999          # negative
        assert abs(consts[6] - 3.14159) < 1e-10  # float
        assert consts[7] == "hello world" # string
        assert consts[8] == ""            # empty string
        assert consts[9] == "emoji: 🚀"  # unicode

    def test_deserialize_blockchain(self, reader):
        bc = _make_blockchain_bytecode()
        data = serialize(bc)
        result = reader.deserialize(data)
        # Check blockchain opcode values
        opcodes = [i["opcode"] for i in result["instructions"]]
        assert int(Opcode.STATE_READ) in opcodes
        assert int(Opcode.TX_BEGIN) in opcodes
        assert int(Opcode.GAS_CHARGE) in opcodes
        assert int(Opcode.TX_COMMIT) in opcodes

    def test_rust_checksum_verification(self, reader):
        bc = _make_simple_bytecode()
        data = bytearray(serialize(bc, include_checksum=True))
        data[len(data) // 2] ^= 0xFF
        assert reader.validate(bytes(data)) is False

    def test_python_rust_consistency(self, reader):
        """Verify Rust and Python deserialize to the same values."""
        bc = _make_loop_bytecode()
        data = serialize(bc)

        py_result = deserialize(data)
        rs_result = reader.deserialize(data)

        assert len(py_result.constants) == len(rs_result["constants"])
        assert len(py_result.instructions) == len(rs_result["instructions"])

        # Constants match
        for i, (py_c, rs_c) in enumerate(zip(py_result.constants, rs_result["constants"])):
            if isinstance(py_c, float):
                assert abs(py_c - rs_c) < 1e-10
            else:
                assert py_c == rs_c, f"Constant {i}: Python={py_c!r} Rust={rs_c!r}"


# ---------------------------------------------------------------------------
# Compiler Integration Tests 
# ---------------------------------------------------------------------------

class TestCompilerIntegration:
    """Test serializing bytecode produced by the real compiler."""

    def test_compiled_code_round_trips(self):
        """Parse real .zx code, compile to bytecode, serialize/deserialize."""
        from zexus.lexer import Lexer
        from zexus.parser import UltimateParser
        from zexus.vm.compiler import BytecodeCompiler

        source = """
            let x = 10;
            let y = 20;
            let z = x + y * 2;
            if z > 30 {
                return z;
            } else {
                return 0;
            }
        """
        lexer = Lexer(source)
        parser = UltimateParser(lexer, enable_advanced_strategies=False)
        program = parser.parse_program()
        assert parser.errors == [], f"Parse errors: {parser.errors}"

        compiler = BytecodeCompiler()
        bytecode = compiler.compile(program)
        assert bytecode is not None
        assert len(bytecode.instructions) > 0

        # Serialize and deserialize
        data = serialize(bytecode)
        restored = deserialize(data)
        assert len(restored.instructions) == len(bytecode.instructions)
        assert len(restored.constants) == len(bytecode.constants)

    def test_loop_compiled_round_trips(self):
        from zexus.lexer import Lexer
        from zexus.parser import UltimateParser
        from zexus.vm.compiler import BytecodeCompiler

        source = """
            let total = 0;
            let i = 0;
            while i < 100 {
                total = total + i;
                i = i + 1;
            }
            return total;
        """
        lexer = Lexer(source)
        parser = UltimateParser(lexer, enable_advanced_strategies=False)
        program = parser.parse_program()
        assert parser.errors == []

        compiler = BytecodeCompiler()
        bytecode = compiler.compile(program)
        data = serialize(bytecode)
        restored = deserialize(data)

        assert len(restored.instructions) == len(bytecode.instructions)

    def test_function_def_round_trips(self):
        from zexus.lexer import Lexer
        from zexus.parser import UltimateParser
        from zexus.vm.compiler import BytecodeCompiler

        source = """
            action add(a, b) {
                return a + b;
            }
            let result = add(10, 20);
        """
        lexer = Lexer(source)
        parser = UltimateParser(lexer, enable_advanced_strategies=False)
        program = parser.parse_program()
        assert parser.errors == []

        compiler = BytecodeCompiler()
        bytecode = compiler.compile(program)
        data = serialize(bytecode)
        restored = deserialize(data)
        assert len(restored.instructions) == len(bytecode.instructions)


# ---------------------------------------------------------------------------
# Size Comparison / Benchmark
# ---------------------------------------------------------------------------

class TestSizeComparison:

    def test_binary_is_compact(self):
        bc = _make_loop_bytecode()
        info = compare_sizes(bc)
        # Binary should be significantly smaller
        assert info["ratio"] < 1.0, f"Binary not smaller: ratio={info['ratio']:.2f}"
        assert info["binary_bytes"] > 0
        assert info["n_instructions"] == len(bc.instructions)

    def test_large_bytecode_size(self):
        """Generate a large bytecode and check compression ratio."""
        bc = Bytecode()
        bc.constants = [f"var_{i}" for i in range(100)]
        bc.instructions = []
        for i in range(1000):
            bc.instructions.append((Opcode.LOAD_CONST, i % 100))
            bc.instructions.append((Opcode.STORE_NAME, i % 100))
        info = compare_sizes(bc)
        print(f"\nLarge bytecode size: binary={info['binary_bytes']}B, "
              f"python={info['python_bytes']}B, ratio={info['ratio']:.2f}")
        assert info["binary_bytes"] < info["python_bytes"]


class TestSerializationPerformance:
    """Benchmark serialization/deserialization speed."""

    def test_serialization_speed(self):
        bc = Bytecode()
        bc.constants = list(range(200)) + [f"name_{i}" for i in range(200)]
        bc.instructions = [
            (Opcode.LOAD_CONST, i % 400) for i in range(5000)
        ] + [
            (Opcode.ADD, None) for _ in range(2000)
        ] + [(Opcode.RETURN, None)]

        # Serialize
        t0 = time.perf_counter()
        for _ in range(100):
            data = serialize(bc)
        ser_time = (time.perf_counter() - t0) / 100

        # Deserialize
        t0 = time.perf_counter()
        for _ in range(100):
            deserialize(data)
        deser_time = (time.perf_counter() - t0) / 100

        print(f"\n--- Serialization Benchmark (7001 instrs, 400 consts) ---")
        print(f"Binary size: {len(data)} bytes")
        print(f"Serialize:   {ser_time*1000:.2f}ms")
        print(f"Deserialize: {deser_time*1000:.2f}ms")
        print(f"---")

        # Should be fast
        assert ser_time < 0.1, f"Serialization too slow: {ser_time*1000:.1f}ms"
        assert deser_time < 0.1, f"Deserialization too slow: {deser_time*1000:.1f}ms"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_max_int32(self):
        bc = Bytecode(constants=[2**31 - 1, -(2**31)], instructions=[
            (Opcode.LOAD_CONST, 0),
            (Opcode.LOAD_CONST, 1),
        ])
        data = serialize(bc)
        restored = deserialize(data)
        assert restored.constants[0] == 2**31 - 1
        assert restored.constants[1] == -(2**31)

    def test_large_int64(self):
        bc = Bytecode(constants=[2**40, -(2**50)], instructions=[
            (Opcode.LOAD_CONST, 0),
        ])
        data = serialize(bc)
        restored = deserialize(data)
        assert restored.constants[0] == 2**40
        assert restored.constants[1] == -(2**50)

    def test_unicode_strings(self):
        bc = Bytecode(constants=["Hello 世界", "🚀💻", "Ñoño"], instructions=[])
        data = serialize(bc)
        restored = deserialize(data)
        for i, (orig, rest) in enumerate(zip(bc.constants, restored.constants)):
            assert orig == rest, f"Unicode constant {i}: {orig!r} != {rest!r}"

    def test_nested_list_constant(self):
        bc = Bytecode(constants=[[1, [2, 3], "nested"]], instructions=[])
        data = serialize(bc)
        restored = deserialize(data)
        assert restored.constants[0] == [1, [2, 3], "nested"]

    def test_invalid_magic(self):
        data = b"BAAD" + b"\x00" * 20
        with pytest.raises(ValueError, match="Invalid magic"):
            deserialize(data, verify_checksum=False)

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            deserialize(b"ZXC", verify_checksum=False)


# ---------------------------------------------------------------------------
# Multi-Bytecode Container Tests
# ---------------------------------------------------------------------------

class TestMultiContainer:

    def test_round_trip(self):
        bcs = [_make_simple_bytecode(), _make_loop_bytecode(), _make_rich_constants_bytecode()]
        data = serialize_multi(bcs)
        restored = deserialize_multi(data)
        assert len(restored) == 3
        for orig, rest in zip(bcs, restored):
            assert len(orig.instructions) == len(rest.instructions)
            assert len(orig.constants) == len(rest.constants)

    def test_empty_list(self):
        data = serialize_multi([])
        restored = deserialize_multi(data)
        assert restored == []

    def test_single_entry(self):
        bcs = [_make_simple_bytecode()]
        data = serialize_multi(bcs)
        restored = deserialize_multi(data)
        assert len(restored) == 1

    def test_file_io(self, tmp_path):
        bcs = [_make_simple_bytecode(), _make_loop_bytecode()]
        path = str(tmp_path / "multi.zxc")
        save_zxcm(path, bcs)
        restored = load_zxcm(path)
        assert len(restored) == 2

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid multi-bytecode"):
            deserialize_multi(b"BAADBAAD")


# ---------------------------------------------------------------------------
# Co-located .zxc Helper Tests
# ---------------------------------------------------------------------------

class TestColocated:

    def test_zxc_path_for(self):
        assert zxc_path_for("foo.zx") == "foo.zxc"
        assert zxc_path_for("/path/to/module.zx") == "/path/to/module.zxc"
        assert zxc_path_for("bar.zexus") == "bar.zxc"

    def test_is_zxc_fresh_missing(self, tmp_path):
        src = str(tmp_path / "test.zx")
        with open(src, "w") as f:
            f.write("let x = 1;")
        assert is_zxc_fresh(src) is False

    def test_is_zxc_fresh_exists(self, tmp_path):
        import time as _time
        src = str(tmp_path / "test.zx")
        with open(src, "w") as f:
            f.write("let x = 1;")
        _time.sleep(0.05)
        zxc = zxc_path_for(src)
        bc = _make_simple_bytecode()
        save_zxc(zxc, bc)
        assert is_zxc_fresh(src) is True


# ---------------------------------------------------------------------------
# Cache Integration Tests
# ---------------------------------------------------------------------------

class TestCacheIntegration:

    def test_disk_cache_uses_zxc(self, tmp_path):
        from zexus.vm.cache import BytecodeCache
        cache = BytecodeCache(
            max_size=10, persistent=True,
            cache_dir=str(tmp_path), debug=False,
        )
        from zexus import zexus_ast
        node = zexus_ast.IntegerLiteral(99)
        bc = _make_simple_bytecode()
        cache.put(node, bc)

        # Verify .zxc file was created (not .cache)
        from pathlib import Path
        zxc_files = list(Path(tmp_path).glob("*.zxc"))
        cache_files = list(Path(tmp_path).glob("*.cache"))
        assert len(zxc_files) == 1, f"Expected 1 .zxc file, got {len(zxc_files)}"
        assert len(cache_files) == 0, f"Expected 0 .cache files, got {len(cache_files)}"

    def test_disk_cache_round_trip(self, tmp_path):
        from zexus.vm.cache import BytecodeCache
        from zexus import zexus_ast

        cache1 = BytecodeCache(
            max_size=10, persistent=True,
            cache_dir=str(tmp_path), debug=False,
        )
        node = zexus_ast.IntegerLiteral(77)
        bc = _make_simple_bytecode()
        cache1.put(node, bc)

        # New cache instance should load from disk
        cache2 = BytecodeCache(
            max_size=10, persistent=True,
            cache_dir=str(tmp_path), debug=False,
        )
        restored = cache2.get(node)
        assert restored is not None
        assert len(restored.instructions) == len(bc.instructions)
