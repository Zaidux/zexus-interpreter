"""
Zexus VM — Binary Bytecode Format (Phase 1)

Compact binary representation of Zexus bytecode that can be deserialized
by both Python and Rust without GIL overhead.

File Format (.zxc)
==================

Header (16 bytes):
    magic:      4 bytes   b"ZXC\\x00"
    version:    2 bytes   uint16 LE (currently 1)
    flags:      2 bytes   uint16 LE (reserved)
    n_consts:   4 bytes   uint32 LE (number of constants)
    n_instrs:   4 bytes   uint32 LE (number of instructions)

Constants Section:
    For each constant:
        tag:    1 byte    (ConstTag enum)
        data:   variable  (depends on tag)

    ConstTag values:
        0x00  NULL       — no data
        0x01  BOOL       — 1 byte (0=false, 1=true)
        0x02  INT32      — 4 bytes int32 LE
        0x03  INT64      — 8 bytes int64 LE
        0x04  FLOAT64    — 8 bytes float64 LE (IEEE 754)
        0x05  STRING     — 4 bytes uint32 LE (length) + UTF-8 bytes
        0x06  FUNC_DESC  — 4 bytes uint32 LE (length) + UTF-8 bytes (JSON-encoded)
        0x07  LIST       — 4 bytes uint32 LE (count) + count × constant entries (recursive)
        0x08  MAP        — 4 bytes uint32 LE (count) + count × (key_const, val_const) entries
        0xFF  OPAQUE     — 4 bytes uint32 LE (length) + raw bytes (not Rust-readable)

Instructions Section:
    For each instruction (variable-width):
        opcode:   2 bytes   uint16 LE (Opcode IntEnum value)
        op_type:  1 byte    (OperandType enum)
        operand:  variable  (depends on op_type)

    OperandType values:
        0x00  NONE       — no operand data
        0x01  U32        — 4 bytes uint32 LE (constant index, jump target, etc.)
        0x02  I64        — 8 bytes int64 LE (immediate integer)
        0x03  PAIR_U32   — 8 bytes (2 × uint32 LE) — e.g. (name_idx, arg_count)
        0x04  TRIPLE_U32 — 12 bytes (3 × uint32 LE) — e.g. register ops (r1, r2, r3)

Footer (optional, for integrity):
    checksum:   4 bytes   CRC32 of everything before the checksum

Usage
-----
::

    from zexus.vm.binary_bytecode import serialize, deserialize

    # Convert Bytecode → bytes
    data = serialize(bytecode_obj)

    # Convert bytes → Bytecode
    bytecode_obj = deserialize(data)

    # File I/O
    save_zxc("contract.zxc", bytecode_obj)
    bytecode_obj = load_zxc("contract.zxc")
"""

from __future__ import annotations

import struct
import json
import zlib
from enum import IntEnum
from typing import Any, List, Tuple, Optional, Dict

from .bytecode import Bytecode, Opcode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZXC_MAGIC = b"ZXC\x00"
ZXC_VERSION = 1
ZXC_HEADER_SIZE = 16  # magic(4) + version(2) + flags(2) + n_consts(4) + n_instrs(4)


class ConstTag(IntEnum):
    """Type tags for the constant pool."""
    NULL = 0x00
    BOOL = 0x01
    INT32 = 0x02
    INT64 = 0x03
    FLOAT64 = 0x04
    STRING = 0x05
    FUNC_DESC = 0x06
    LIST = 0x07
    MAP = 0x08
    OPAQUE = 0xFF


class OperandType(IntEnum):
    """Operand encoding types for instructions."""
    NONE = 0x00
    U32 = 0x01
    I64 = 0x02
    PAIR_U32 = 0x03
    TRIPLE_U32 = 0x04


# ---------------------------------------------------------------------------
# Serialization (Python Bytecode → Binary)
# ---------------------------------------------------------------------------

class _Writer:
    """Buffered binary writer."""
    __slots__ = ("_parts",)

    def __init__(self):
        self._parts: List[bytes] = []

    def write(self, data: bytes):
        self._parts.append(data)

    def u8(self, val: int):
        self._parts.append(struct.pack("<B", val & 0xFF))

    def u16(self, val: int):
        self._parts.append(struct.pack("<H", val & 0xFFFF))

    def u32(self, val: int):
        self._parts.append(struct.pack("<I", val & 0xFFFFFFFF))

    def i32(self, val: int):
        self._parts.append(struct.pack("<i", val))

    def i64(self, val: int):
        self._parts.append(struct.pack("<q", val))

    def f64(self, val: float):
        self._parts.append(struct.pack("<d", val))

    def string(self, s: str):
        encoded = s.encode("utf-8")
        self.u32(len(encoded))
        self._parts.append(encoded)

    def raw_bytes(self, data: bytes):
        self.u32(len(data))
        self._parts.append(data)

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


class _Reader:
    """Binary reader with position tracking."""
    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def read(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise ValueError(f"Unexpected end of data at offset {self._pos}, need {n} bytes")
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def u8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def string(self) -> str:
        length = self.u32()
        return self.read(length).decode("utf-8")

    def raw_bytes(self) -> bytes:
        length = self.u32()
        return self.read(length)


def _serialize_constant(w: _Writer, value: Any):
    """Serialize a single constant to the writer."""
    if value is None:
        w.u8(ConstTag.NULL)
    elif isinstance(value, bool):
        # Must check bool before int (bool is a subclass of int)
        w.u8(ConstTag.BOOL)
        w.u8(1 if value else 0)
    elif isinstance(value, int):
        if -(2**31) <= value < 2**31:
            w.u8(ConstTag.INT32)
            w.i32(value)
        else:
            w.u8(ConstTag.INT64)
            w.i64(value)
    elif isinstance(value, float):
        w.u8(ConstTag.FLOAT64)
        w.f64(value)
    elif isinstance(value, str):
        w.u8(ConstTag.STRING)
        w.string(value)
    elif isinstance(value, (list, tuple)):
        # Check if it's a simple list of serializable values
        w.u8(ConstTag.LIST)
        w.u32(len(value))
        for item in value:
            _serialize_constant(w, item)
    elif isinstance(value, dict):
        # Serialize as JSON string for portability
        try:
            json_str = json.dumps(value, default=str)
            w.u8(ConstTag.FUNC_DESC)
            w.string(json_str)
        except (TypeError, ValueError):
            # Fall back to opaque
            import pickle
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            w.u8(ConstTag.OPAQUE)
            w.raw_bytes(data)
    else:
        # Callable, AST nodes, or other Python objects → opaque
        import pickle
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            w.u8(ConstTag.OPAQUE)
            w.raw_bytes(data)
        except (pickle.PicklingError, TypeError, AttributeError):
            # Truly unserializable → store repr as string
            w.u8(ConstTag.STRING)
            w.string(repr(value))


def _serialize_instruction(w: _Writer, opcode_val: int, operand: Any):
    """Serialize a single instruction to the writer."""
    w.u16(opcode_val)

    if operand is None:
        w.u8(OperandType.NONE)
    elif isinstance(operand, int):
        if operand >= 0 and operand < 2**32:
            w.u8(OperandType.U32)
            w.u32(operand)
        else:
            w.u8(OperandType.I64)
            w.i64(operand)
    elif isinstance(operand, tuple):
        if len(operand) == 2:
            w.u8(OperandType.PAIR_U32)
            w.u32(int(operand[0]) & 0xFFFFFFFF)
            w.u32(int(operand[1]) & 0xFFFFFFFF)
        elif len(operand) == 3:
            w.u8(OperandType.TRIPLE_U32)
            w.u32(int(operand[0]) & 0xFFFFFFFF)
            w.u32(int(operand[1]) & 0xFFFFFFFF)
            w.u32(int(operand[2]) & 0xFFFFFFFF)
        else:
            # Fallback — encode as I64 of first element
            w.u8(OperandType.I64)
            w.i64(int(operand[0]) if operand else 0)
    elif isinstance(operand, str):
        # String operands (label names, etc.) → treat as U32 after storing in constants
        # This shouldn't normally happen after compilation, but handle gracefully
        w.u8(OperandType.I64)
        w.i64(hash(operand) & 0x7FFFFFFFFFFFFFFF)
    else:
        # Numeric-like
        w.u8(OperandType.U32)
        w.u32(int(operand) & 0xFFFFFFFF)


def _resolve_opcode(op: Any) -> int:
    """Resolve an opcode to its integer value."""
    if isinstance(op, int):
        return op
    if isinstance(op, Opcode):
        return int(op)
    if isinstance(op, str):
        try:
            return int(Opcode[op])
        except KeyError:
            # Unknown opcode → NOP
            return int(Opcode.NOP)
    return int(Opcode.NOP)


def serialize(bytecode: Bytecode, *, include_checksum: bool = True) -> bytes:
    """Serialize a Bytecode object to compact binary format.

    Parameters
    ----------
    bytecode : Bytecode
        The bytecode to serialize.
    include_checksum : bool
        If True, append a CRC32 checksum.

    Returns
    -------
    bytes
        The binary-encoded bytecode.
    """
    w = _Writer()

    consts = bytecode.constants or []
    instrs = bytecode.instructions or []

    # Header
    w.write(ZXC_MAGIC)
    w.u16(ZXC_VERSION)
    w.u16(0)  # flags (reserved)
    w.u32(len(consts))
    w.u32(len(instrs))

    # Constants
    for c in consts:
        _serialize_constant(w, c)

    # Instructions
    for instr in instrs:
        if instr is None:
            continue
        if isinstance(instr, tuple) and len(instr) >= 2:
            op = instr[0]
            operand = instr[1]
        else:
            op = instr
            operand = None
        opcode_val = _resolve_opcode(op)
        _serialize_instruction(w, opcode_val, operand)

    body = w.getvalue()

    if include_checksum:
        crc = zlib.crc32(body) & 0xFFFFFFFF
        body += struct.pack("<I", crc)

    return body


# ---------------------------------------------------------------------------
# Deserialization (Binary → Python Bytecode)
# ---------------------------------------------------------------------------

def _deserialize_constant(r: _Reader) -> Any:
    """Deserialize a single constant from the reader."""
    tag = r.u8()

    if tag == ConstTag.NULL:
        return None
    elif tag == ConstTag.BOOL:
        return bool(r.u8())
    elif tag == ConstTag.INT32:
        return r.i32()
    elif tag == ConstTag.INT64:
        return r.i64()
    elif tag == ConstTag.FLOAT64:
        return r.f64()
    elif tag == ConstTag.STRING:
        return r.string()
    elif tag == ConstTag.FUNC_DESC:
        json_str = r.string()
        return json.loads(json_str)
    elif tag == ConstTag.LIST:
        count = r.u32()
        return [_deserialize_constant(r) for _ in range(count)]
    elif tag == ConstTag.MAP:
        count = r.u32()
        result = {}
        for _ in range(count):
            key = _deserialize_constant(r)
            val = _deserialize_constant(r)
            result[key] = val
        return result
    elif tag == ConstTag.OPAQUE:
        import pickle
        data = r.raw_bytes()
        return pickle.loads(data)
    else:
        raise ValueError(f"Unknown constant tag: 0x{tag:02x}")


def _deserialize_instruction(r: _Reader) -> Tuple[Any, Any]:
    """Deserialize a single instruction from the reader."""
    opcode_val = r.u16()
    op_type = r.u8()

    # Resolve opcode to Opcode enum
    try:
        opcode = Opcode(opcode_val)
    except ValueError:
        opcode = opcode_val  # Unknown opcode — keep as int

    if op_type == OperandType.NONE:
        return (opcode, None)
    elif op_type == OperandType.U32:
        return (opcode, r.u32())
    elif op_type == OperandType.I64:
        return (opcode, r.i64())
    elif op_type == OperandType.PAIR_U32:
        a = r.u32()
        b = r.u32()
        return (opcode, (a, b))
    elif op_type == OperandType.TRIPLE_U32:
        a = r.u32()
        b = r.u32()
        c = r.u32()
        return (opcode, (a, b, c))
    else:
        raise ValueError(f"Unknown operand type: 0x{op_type:02x}")


def deserialize(data: bytes, *, verify_checksum: bool = True) -> Bytecode:
    """Deserialize binary data into a Bytecode object.

    Parameters
    ----------
    data : bytes
        The binary-encoded bytecode.
    verify_checksum : bool
        If True, verify the trailing CRC32 checksum.

    Returns
    -------
    Bytecode
        The deserialized Bytecode object.

    Raises
    ------
    ValueError
        If the data is malformed or the checksum doesn't match.
    """
    if len(data) < ZXC_HEADER_SIZE:
        raise ValueError(f"Data too short for header: {len(data)} bytes")

    # Verify checksum first (last 4 bytes)
    if verify_checksum and len(data) > ZXC_HEADER_SIZE + 4:
        body = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"Checksum mismatch: stored=0x{stored_crc:08x}, "
                f"computed=0x{computed_crc:08x}"
            )
        data = body  # Strip checksum for reading

    r = _Reader(data)

    # Header
    magic = r.read(4)
    if magic != ZXC_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {ZXC_MAGIC!r}")

    version = r.u16()
    if version > ZXC_VERSION:
        raise ValueError(f"Unsupported version: {version} (max supported: {ZXC_VERSION})")

    _flags = r.u16()  # reserved
    n_consts = r.u32()
    n_instrs = r.u32()

    # Constants
    constants = [_deserialize_constant(r) for _ in range(n_consts)]

    # Instructions
    instructions = [_deserialize_instruction(r) for _ in range(n_instrs)]

    # Build Bytecode object
    bc = Bytecode(instructions=instructions, constants=constants)
    bc.metadata["source_file"] = None
    bc.metadata["version"] = f"binary-{version}"
    bc.metadata["created_by"] = "binary_deserializer"

    return bc


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_zxc(path: str, bytecode: Bytecode, *, include_checksum: bool = True):
    """Save a Bytecode object to a .zxc file.

    Parameters
    ----------
    path : str
        Output file path (typically ending in `.zxc`).
    bytecode : Bytecode
        The bytecode to save.
    include_checksum : bool
        If True, append CRC32 checksum.
    """
    data = serialize(bytecode, include_checksum=include_checksum)
    with open(path, "wb") as f:
        f.write(data)


def load_zxc(path: str, *, verify_checksum: bool = True) -> Bytecode:
    """Load a Bytecode object from a .zxc file.

    Parameters
    ----------
    path : str
        Input file path.
    verify_checksum : bool
        If True, verify CRC32 checksum.

    Returns
    -------
    Bytecode
        The deserialized Bytecode object.
    """
    with open(path, "rb") as f:
        data = f.read()
    return deserialize(data, verify_checksum=verify_checksum)


# ---------------------------------------------------------------------------
# Utility — size comparison
# ---------------------------------------------------------------------------

def compare_sizes(bytecode: Bytecode) -> Dict[str, Any]:
    """Compare binary size vs Python tuple representation size.

    Returns a dict with size information for analysis.
    """
    import sys

    binary = serialize(bytecode)
    binary_size = len(binary)

    # Estimate Python tuple size
    py_size = sys.getsizeof(bytecode.instructions)
    for instr in bytecode.instructions:
        py_size += sys.getsizeof(instr)
        if isinstance(instr, tuple):
            for item in instr:
                py_size += sys.getsizeof(item)
    py_size += sys.getsizeof(bytecode.constants)
    for c in bytecode.constants:
        py_size += sys.getsizeof(c)

    return {
        "binary_bytes": binary_size,
        "python_bytes": py_size,
        "ratio": binary_size / py_size if py_size > 0 else 0.0,
        "n_instructions": len(bytecode.instructions),
        "n_constants": len(bytecode.constants),
    }


# ---------------------------------------------------------------------------
# Multi-Bytecode Container (for file-level caching)
# ---------------------------------------------------------------------------
#
# Format: ZXCM magic (4B) + count (4B) + [length(4B) + zxc_data]*count
#

ZXCM_MAGIC = b"ZXCM"


def serialize_multi(bytecodes: List[Bytecode], *, include_checksum: bool = True) -> bytes:
    """Serialize a list of Bytecode objects into a single binary blob.

    Used for file-level caching where multiple Bytecodes represent
    individual statements from a single source file.
    """
    w = _Writer()
    w.write(ZXCM_MAGIC)
    w.u32(len(bytecodes))
    for bc in bytecodes:
        chunk = serialize(bc, include_checksum=include_checksum)
        w.u32(len(chunk))
        w.write(chunk)
    return w.getvalue()


def deserialize_multi(data: bytes, *, verify_checksum: bool = True) -> List[Bytecode]:
    """Deserialize a multi-bytecode container back to a list of Bytecodes."""
    if len(data) < 8:
        raise ValueError("Multi-bytecode data too short")
    if data[:4] != ZXCM_MAGIC:
        raise ValueError(f"Invalid multi-bytecode magic: {data[:4]!r}")
    count = struct.unpack_from("<I", data, 4)[0]
    offset = 8
    result: List[Bytecode] = []
    for _ in range(count):
        if offset + 4 > len(data):
            raise ValueError("Truncated multi-bytecode container")
        chunk_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if offset + chunk_len > len(data):
            raise ValueError("Truncated multi-bytecode chunk")
        chunk = data[offset:offset + chunk_len]
        result.append(deserialize(chunk, verify_checksum=verify_checksum))
        offset += chunk_len
    return result


def save_zxcm(path: str, bytecodes: List[Bytecode]):
    """Save multiple Bytecodes to a .zxc multi-container file."""
    data = serialize_multi(bytecodes)
    with open(path, "wb") as f:
        f.write(data)


def load_zxcm(path: str) -> List[Bytecode]:
    """Load multiple Bytecodes from a .zxc multi-container file."""
    with open(path, "rb") as f:
        data = f.read()
    return deserialize_multi(data)


# ---------------------------------------------------------------------------
# Co-located .zxc helper
# ---------------------------------------------------------------------------

import os


def zxc_path_for(source_path: str) -> str:
    """Return the .zxc path co-located with a .zx source file.

    ``foo.zx`` → ``foo.zxc``
    ``bar.zexus`` → ``bar.zxc``
    Other extensions just append ``.zxc``.
    """
    root, ext = os.path.splitext(source_path)
    return root + ".zxc"


def is_zxc_fresh(source_path: str) -> bool:
    """Return True if a co-located .zxc exists and is newer than the source."""
    zxc = zxc_path_for(source_path)
    try:
        if not os.path.exists(zxc):
            return False
        return os.path.getmtime(zxc) >= os.path.getmtime(source_path)
    except OSError:
        return False
