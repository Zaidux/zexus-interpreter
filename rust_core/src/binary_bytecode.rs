// ─────────────────────────────────────────────────────────────────────
// Zexus Blockchain — Binary Bytecode Format (Rust Deserializer)
// ─────────────────────────────────────────────────────────────────────
//
// Deserializes the .zxc binary bytecode format produced by Python's
// `binary_bytecode.serialize()`.  The format is intentionally simple
// (little-endian, no alignment padding) so both Python and Rust can
// read it without external dependencies.
//
// This module is GIL-free — it operates on raw `&[u8]` data that was
// already copied out of Python.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::ToPyObject;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Format constants ──────────────────────────────────────────────────

const ZXC_MAGIC: &[u8; 4] = b"ZXC\x00";
const ZXC_HEADER_SIZE: usize = 16;

// ── Constant tags ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ConstTag {
    Null = 0x00,
    Bool = 0x01,
    Int32 = 0x02,
    Int64 = 0x03,
    Float64 = 0x04,
    String = 0x05,
    FuncDesc = 0x06,
    List = 0x07,
    Map = 0x08,
    Opaque = 0xFF,
}

impl ConstTag {
    fn from_u8(val: u8) -> Option<Self> {
        match val {
            0x00 => Some(Self::Null),
            0x01 => Some(Self::Bool),
            0x02 => Some(Self::Int32),
            0x03 => Some(Self::Int64),
            0x04 => Some(Self::Float64),
            0x05 => Some(Self::String),
            0x06 => Some(Self::FuncDesc),
            0x07 => Some(Self::List),
            0x08 => Some(Self::Map),
            0xFF => Some(Self::Opaque),
            _ => None,
        }
    }
}

// ── Operand types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum OperandType {
    None = 0x00,
    U32 = 0x01,
    I64 = 0x02,
    PairU32 = 0x03,
    TripleU32 = 0x04,
}

impl OperandType {
    fn from_u8(val: u8) -> Option<Self> {
        match val {
            0x00 => Some(Self::None),
            0x01 => Some(Self::U32),
            0x02 => Some(Self::I64),
            0x03 => Some(Self::PairU32),
            0x04 => Some(Self::TripleU32),
            _ => None,
        }
    }
}

// ── Value types for the constant pool ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZxcValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    FuncDesc(String),       // JSON-encoded function descriptor
    List(Vec<ZxcValue>),
    Map(Vec<(ZxcValue, ZxcValue)>),
    Opaque(Vec<u8>),        // Python-pickled data (not interpreted in Rust)
}

impl fmt::Display for ZxcValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZxcValue::Null => write!(f, "null"),
            ZxcValue::Bool(b) => write!(f, "{}", b),
            ZxcValue::Int(i) => write!(f, "{}", i),
            ZxcValue::Float(v) => write!(f, "{}", v),
            ZxcValue::String(s) => write!(f, "\"{}\"", s),
            ZxcValue::FuncDesc(s) => write!(f, "func({})", s),
            ZxcValue::List(items) => write!(f, "[{} items]", items.len()),
            ZxcValue::Map(pairs) => write!(f, "{{{} pairs}}", pairs.len()),
            ZxcValue::Opaque(data) => write!(f, "<opaque {} bytes>", data.len()),
        }
    }
}

// ── Instruction operand ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operand {
    None,
    U32(u32),
    I64(i64),
    Pair(u32, u32),
    Triple(u32, u32, u32),
}

// ── Decoded instruction ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: u16,
    pub operand: Operand,
}

// ── Decoded bytecode module ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZxcModule {
    pub version: u16,
    pub flags: u16,
    pub constants: Vec<ZxcValue>,
    pub instructions: Vec<Instruction>,
}

impl ZxcModule {
    pub fn n_constants(&self) -> usize {
        self.constants.len()
    }

    pub fn n_instructions(&self) -> usize {
        self.instructions.len()
    }
}

// ── Binary reader ─────────────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(format!(
                "Unexpected end of data at offset {}, need {} bytes",
                self.pos, n
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.read(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let bytes = self.read(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.read(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, String> {
        let bytes = self.read(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i64(&mut self) -> Result<i64, String> {
        let bytes = self.read(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn f64(&mut self) -> Result<f64, String> {
        let bytes = self.read(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        let bytes = self.read(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| format!("Invalid UTF-8 string at offset {}: {}", self.pos - len, e))
    }

    fn raw_bytes(&mut self) -> Result<Vec<u8>, String> {
        let len = self.u32()? as usize;
        Ok(self.read(len)?.to_vec())
    }
}

// ── Deserialization functions ─────────────────────────────────────────

fn read_constant(r: &mut Reader) -> Result<ZxcValue, String> {
    let tag_byte = r.u8()?;
    let tag = ConstTag::from_u8(tag_byte)
        .ok_or_else(|| format!("Unknown constant tag: 0x{:02x}", tag_byte))?;

    match tag {
        ConstTag::Null => Ok(ZxcValue::Null),
        ConstTag::Bool => Ok(ZxcValue::Bool(r.u8()? != 0)),
        ConstTag::Int32 => Ok(ZxcValue::Int(r.i32()? as i64)),
        ConstTag::Int64 => Ok(ZxcValue::Int(r.i64()?)),
        ConstTag::Float64 => Ok(ZxcValue::Float(r.f64()?)),
        ConstTag::String => Ok(ZxcValue::String(r.string()?)),
        ConstTag::FuncDesc => Ok(ZxcValue::FuncDesc(r.string()?)),
        ConstTag::List => {
            let count = r.u32()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_constant(r)?);
            }
            Ok(ZxcValue::List(items))
        }
        ConstTag::Map => {
            let count = r.u32()? as usize;
            let mut pairs = Vec::with_capacity(count);
            for _ in 0..count {
                let key = read_constant(r)?;
                let val = read_constant(r)?;
                pairs.push((key, val));
            }
            Ok(ZxcValue::Map(pairs))
        }
        ConstTag::Opaque => Ok(ZxcValue::Opaque(r.raw_bytes()?)),
    }
}

fn read_instruction(r: &mut Reader) -> Result<Instruction, String> {
    let opcode = r.u16()?;
    let op_type_byte = r.u8()?;
    let op_type = OperandType::from_u8(op_type_byte)
        .ok_or_else(|| format!("Unknown operand type: 0x{:02x}", op_type_byte))?;

    let operand = match op_type {
        OperandType::None => Operand::None,
        OperandType::U32 => Operand::U32(r.u32()?),
        OperandType::I64 => Operand::I64(r.i64()?),
        OperandType::PairU32 => {
            let a = r.u32()?;
            let b = r.u32()?;
            Operand::Pair(a, b)
        }
        OperandType::TripleU32 => {
            let a = r.u32()?;
            let b = r.u32()?;
            let c = r.u32()?;
            Operand::Triple(a, b, c)
        }
    };

    Ok(Instruction { opcode, operand })
}

/// Deserialize a .zxc binary buffer into a ZxcModule.
///
/// This function is fully GIL-free — it operates on raw bytes.
pub fn deserialize_zxc(data: &[u8], verify_checksum: bool) -> Result<ZxcModule, String> {
    if data.len() < ZXC_HEADER_SIZE {
        return Err(format!(
            "Data too short for header: {} bytes (need {})",
            data.len(),
            ZXC_HEADER_SIZE
        ));
    }

    // Verify CRC32 checksum (last 4 bytes)
    let body = if verify_checksum && data.len() > ZXC_HEADER_SIZE + 4 {
        let body = &data[..data.len() - 4];
        let stored_bytes = &data[data.len() - 4..];
        let stored_crc = u32::from_le_bytes([
            stored_bytes[0],
            stored_bytes[1],
            stored_bytes[2],
            stored_bytes[3],
        ]);
        let computed_crc = crc32(body);
        if stored_crc != computed_crc {
            return Err(format!(
                "Checksum mismatch: stored=0x{:08x}, computed=0x{:08x}",
                stored_crc, computed_crc
            ));
        }
        body
    } else {
        data
    };

    let mut r = Reader::new(body);

    // Header
    let magic = r.read(4)?;
    if magic != ZXC_MAGIC {
        return Err(format!("Invalid magic: {:?}", magic));
    }

    let version = r.u16()?;
    if version > 1 {
        return Err(format!("Unsupported version: {}", version));
    }

    let flags = r.u16()?;
    let n_consts = r.u32()? as usize;
    let n_instrs = r.u32()? as usize;

    // Constants
    let mut constants = Vec::with_capacity(n_consts);
    for _ in 0..n_consts {
        constants.push(read_constant(&mut r)?);
    }

    // Instructions
    let mut instructions = Vec::with_capacity(n_instrs);
    for _ in 0..n_instrs {
        instructions.push(read_instruction(&mut r)?);
    }

    Ok(ZxcModule {
        version,
        flags,
        constants,
        instructions,
    })
}

// ── CRC32 (IEEE 802.3 / zlib-compatible) ──────────────────────────────

/// Compute CRC32 matching Python's `zlib.crc32()`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

// ── PyO3 wrapper ──────────────────────────────────────────────────────

/// Python-visible bytecode deserializer.
#[pyclass(name = "RustBytecodeReader")]
pub struct RustBytecodeReader;

#[pymethods]
impl RustBytecodeReader {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Deserialize a .zxc binary buffer and return a dict with the module contents.
    ///
    /// Returns a dict with keys: "version", "flags", "n_constants", "n_instructions",
    /// "constants" (list of dicts), "instructions" (list of dicts).
    #[pyo3(signature = (data, verify_checksum = true))]
    fn deserialize<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyBytes>,
        verify_checksum: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bytes = data.as_bytes();

        // Deserialize outside the GIL for CPU-heavy work
        let module = py.allow_threads(|| deserialize_zxc(bytes, verify_checksum))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        // Convert to Python dict
        let result = PyDict::new_bound(py);
        result.set_item("version", module.version)?;
        result.set_item("flags", module.flags)?;
        result.set_item("n_constants", module.n_constants())?;
        result.set_item("n_instructions", module.n_instructions())?;

        // Constants
        let py_consts = PyList::empty_bound(py);
        for c in &module.constants {
            py_consts.append(zxc_value_to_py(py, c)?)?;
        }
        result.set_item("constants", py_consts)?;

        // Instructions
        let py_instrs = PyList::empty_bound(py);
        for instr in &module.instructions {
            let d = PyDict::new_bound(py);
            d.set_item("opcode", instr.opcode)?;
            match &instr.operand {
                Operand::None => d.set_item("operand", py.None())?,
                Operand::U32(v) => d.set_item("operand", *v)?,
                Operand::I64(v) => d.set_item("operand", *v)?,
                Operand::Pair(a, b) => d.set_item("operand", (*a, *b))?,
                Operand::Triple(a, b, c) => d.set_item("operand", (*a, *b, *c))?,
            }
            py_instrs.append(d)?;
        }
        result.set_item("instructions", py_instrs)?;

        Ok(result)
    }

    /// Quick validation — check if binary data is a valid .zxc file.
    #[pyo3(signature = (data, verify_checksum = true))]
    fn validate(&self, py: Python<'_>, data: &Bound<'_, PyBytes>, verify_checksum: bool) -> PyResult<bool> {
        let bytes = data.as_bytes();
        let result = py.allow_threads(|| deserialize_zxc(bytes, verify_checksum));
        Ok(result.is_ok())
    }

    /// Get basic info from .zxc header without full deserialization.
    fn header_info<'py>(&self, py: Python<'py>, data: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyDict>> {
        let bytes = data.as_bytes();
        if bytes.len() < ZXC_HEADER_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err("Data too short"));
        }

        let magic = &bytes[0..4];
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let n_consts = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let n_instrs = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        let result = PyDict::new_bound(py);
        result.set_item("magic_ok", magic == ZXC_MAGIC)?;
        result.set_item("version", version)?;
        result.set_item("flags", flags)?;
        result.set_item("n_constants", n_consts)?;
        result.set_item("n_instructions", n_instrs)?;
        result.set_item("total_bytes", bytes.len())?;
        Ok(result)
    }
}

/// Convert a ZxcValue to a Python object.
fn zxc_value_to_py(py: Python<'_>, val: &ZxcValue) -> PyResult<PyObject> {
    match val {
        ZxcValue::Null => Ok(py.None()),
        ZxcValue::Bool(b) => Ok(b.to_object(py)),
        ZxcValue::Int(i) => Ok(i.to_object(py)),
        ZxcValue::Float(f) => Ok(f.to_object(py)),
        ZxcValue::String(s) => Ok(s.to_object(py)),
        ZxcValue::FuncDesc(s) => {
            // Parse JSON back to Python dict
            let json_val: serde_json::Value = serde_json::from_str(s)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
            json_to_py(py, &json_val)
        }
        ZxcValue::List(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(zxc_value_to_py(py, item)?)?;
            }
            Ok(list.to_object(py))
        }
        ZxcValue::Map(pairs) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in pairs {
                dict.set_item(zxc_value_to_py(py, k)?, zxc_value_to_py(py, v)?)?;
            }
            Ok(dict.to_object(py))
        }
        ZxcValue::Opaque(data) => {
            // Return raw bytes — Python can unpickle if needed
            Ok(PyBytes::new_bound(py, data).to_object(py))
        }
    }
}

/// Convert serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.to_object(py))
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.to_object(py))
        }
    }
}
