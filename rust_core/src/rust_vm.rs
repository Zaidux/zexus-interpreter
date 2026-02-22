// ─────────────────────────────────────────────────────────────────────
// Zexus Blockchain — Rust Bytecode Interpreter (Phase 2)
// ─────────────────────────────────────────────────────────────────────
//
// A complete stack-machine bytecode interpreter written in Rust that
// executes .zxc binary bytecode **without GIL overhead**.
//
// Architecture:
//
//   1.  Deserialize .zxc bytes → ZxcModule  (Phase 1, binary_bytecode.rs)
//   2.  Convert ZxcModule into VM-internal representation (VmProgram)
//   3.  Execute VmProgram on the RustVM stack machine
//   4.  Return result to Python via PyO3
//
// Value system:
//   The VM operates on `ZxValue` — a Rust enum covering all Zexus
//   runtime types including Null, Bool, Int, Float, String, List, Map.
//   Python objects that cannot be represented natively (e.g. callables,
//   AST nodes) are stored as `ZxValue::PyObject(PyObject)` and require
//   GIL acquisition only when accessed.
//
// Gas metering:
//   Each opcode has an associated gas cost (matching the Python VM).
//   The gas budget is checked before every instruction.  When gas is
//   exhausted an `OutOfGasError` is raised.
//
// Blockchain state:
//   STATE_READ / STATE_WRITE operate on a `HashMap<String, ZxValue>`
//   that is passed in from Python and returned after execution.
//   TX_BEGIN / TX_COMMIT / TX_REVERT provide snapshot-based
//   transactions.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList};
use pyo3::ToPyObject;
use std::collections::HashMap;
use std::fmt;

use crate::binary_bytecode::{self, Operand, ZxcModule, ZxcValue};

// ── Opcode constants (mirrors Python Opcode IntEnum) ─────────────────

#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Op {
    // Stack
    LOAD_CONST = 1,
    LOAD_NAME = 2,
    STORE_NAME = 3,
    STORE_FUNC = 4,
    POP = 5,
    DUP = 6,

    // Arithmetic
    ADD = 10,
    SUB = 11,
    MUL = 12,
    DIV = 13,
    MOD = 14,
    POW = 15,
    NEG = 16,

    // Comparison
    EQ = 20,
    NEQ = 21,
    LT = 22,
    GT = 23,
    LTE = 24,
    GTE = 25,

    // Logical
    AND = 30,
    OR = 31,
    NOT = 32,

    // Control flow
    JUMP = 40,
    JUMP_IF_FALSE = 41,
    JUMP_IF_TRUE = 42,
    RETURN = 43,

    // Calls
    CALL_NAME = 50,
    CALL_FUNC_CONST = 51,
    CALL_TOP = 52,
    CALL_BUILTIN = 53,
    CALL_METHOD = 54,

    // Collections
    BUILD_LIST = 60,
    BUILD_MAP = 61,
    BUILD_SET = 62,
    INDEX = 63,
    SLICE = 64,
    GET_ATTR = 65,

    // Async
    SPAWN = 70,
    AWAIT = 71,
    SPAWN_CALL = 72,

    // Events
    REGISTER_EVENT = 80,
    EMIT_EVENT = 81,

    // Modules
    IMPORT = 90,
    EXPORT = 91,

    // Blockchain
    HASH_BLOCK = 110,
    VERIFY_SIGNATURE = 111,
    MERKLE_ROOT = 112,
    STATE_READ = 113,
    STATE_WRITE = 114,
    TX_BEGIN = 115,
    TX_COMMIT = 116,
    TX_REVERT = 117,
    GAS_CHARGE = 118,
    LEDGER_APPEND = 119,

    // Security & Contracts
    REQUIRE = 130,
    DEFINE_CONTRACT = 131,
    DEFINE_ENTITY = 132,
    AUDIT_LOG = 136,
    RESTRICT_ACCESS = 137,

    // Exception handling
    SETUP_TRY = 140,
    POP_TRY = 141,
    THROW = 142,

    // Iteration
    FOR_ITER = 150,

    // System
    ENABLE_ERROR_MODE = 160,

    // I/O
    PRINT = 250,

    // Parallel (no-op markers)
    PARALLEL_START = 300,
    PARALLEL_END = 301,

    // NOP
    NOP = 255,

    // Unknown — anything not recognised
    UNKNOWN = 0xFFFF,
}

impl Op {
    fn from_u16(v: u16) -> Self {
        match v {
            1 => Op::LOAD_CONST,
            2 => Op::LOAD_NAME,
            3 => Op::STORE_NAME,
            4 => Op::STORE_FUNC,
            5 => Op::POP,
            6 => Op::DUP,
            10 => Op::ADD,
            11 => Op::SUB,
            12 => Op::MUL,
            13 => Op::DIV,
            14 => Op::MOD,
            15 => Op::POW,
            16 => Op::NEG,
            20 => Op::EQ,
            21 => Op::NEQ,
            22 => Op::LT,
            23 => Op::GT,
            24 => Op::LTE,
            25 => Op::GTE,
            30 => Op::AND,
            31 => Op::OR,
            32 => Op::NOT,
            40 => Op::JUMP,
            41 => Op::JUMP_IF_FALSE,
            42 => Op::JUMP_IF_TRUE,
            43 => Op::RETURN,
            50 => Op::CALL_NAME,
            51 => Op::CALL_FUNC_CONST,
            52 => Op::CALL_TOP,
            53 => Op::CALL_BUILTIN,
            54 => Op::CALL_METHOD,
            60 => Op::BUILD_LIST,
            61 => Op::BUILD_MAP,
            62 => Op::BUILD_SET,
            63 => Op::INDEX,
            64 => Op::SLICE,
            65 => Op::GET_ATTR,
            70 => Op::SPAWN,
            71 => Op::AWAIT,
            72 => Op::SPAWN_CALL,
            80 => Op::REGISTER_EVENT,
            81 => Op::EMIT_EVENT,
            90 => Op::IMPORT,
            91 => Op::EXPORT,
            110 => Op::HASH_BLOCK,
            111 => Op::VERIFY_SIGNATURE,
            112 => Op::MERKLE_ROOT,
            113 => Op::STATE_READ,
            114 => Op::STATE_WRITE,
            115 => Op::TX_BEGIN,
            116 => Op::TX_COMMIT,
            117 => Op::TX_REVERT,
            118 => Op::GAS_CHARGE,
            119 => Op::LEDGER_APPEND,
            130 => Op::REQUIRE,
            131 => Op::DEFINE_CONTRACT,
            132 => Op::DEFINE_ENTITY,
            136 => Op::AUDIT_LOG,
            137 => Op::RESTRICT_ACCESS,
            140 => Op::SETUP_TRY,
            141 => Op::POP_TRY,
            142 => Op::THROW,
            150 => Op::FOR_ITER,
            160 => Op::ENABLE_ERROR_MODE,
            250 => Op::PRINT,
            255 => Op::NOP,
            300 => Op::PARALLEL_START,
            301 => Op::PARALLEL_END,
            _ => Op::UNKNOWN,
        }
    }

    /// Gas cost for this opcode (matches Python GasCost IntEnum).
    fn gas_cost(self) -> u64 {
        match self {
            Op::NOP => 0,
            Op::LOAD_CONST => 1,
            Op::LOAD_NAME => 2,
            Op::STORE_NAME | Op::STORE_FUNC => 3,
            Op::POP | Op::DUP => 1,
            Op::ADD | Op::SUB => 3,
            Op::MUL => 5,
            Op::DIV | Op::MOD => 10,
            Op::POW => 20,
            Op::NEG => 2,
            Op::EQ | Op::NEQ | Op::LT | Op::GT | Op::LTE | Op::GTE => 2,
            Op::NOT | Op::AND | Op::OR => 2,
            Op::JUMP => 2,
            Op::JUMP_IF_FALSE | Op::JUMP_IF_TRUE => 3,
            Op::RETURN => 2,
            Op::BUILD_LIST | Op::BUILD_MAP | Op::BUILD_SET => 5,
            Op::INDEX | Op::GET_ATTR => 3,
            Op::SLICE => 5,
            Op::CALL_NAME | Op::CALL_TOP | Op::CALL_METHOD | Op::CALL_FUNC_CONST => 10,
            Op::CALL_BUILTIN => 8,
            Op::SPAWN | Op::SPAWN_CALL => 15,
            Op::AWAIT => 10,
            Op::HASH_BLOCK => 50,
            Op::VERIFY_SIGNATURE => 100,
            Op::MERKLE_ROOT => 30,
            Op::STATE_READ => 20,
            Op::STATE_WRITE => 50,
            Op::TX_BEGIN => 20,
            Op::TX_COMMIT => 30,
            Op::TX_REVERT => 20,
            Op::GAS_CHARGE => 2,
            Op::LEDGER_APPEND => 40,
            Op::REQUIRE => 5,
            Op::AUDIT_LOG => 15,
            Op::RESTRICT_ACCESS => 5,
            Op::DEFINE_CONTRACT | Op::DEFINE_ENTITY => 50,
            Op::EMIT_EVENT | Op::REGISTER_EVENT => 10,
            Op::SETUP_TRY => 3,
            Op::POP_TRY => 2,
            Op::THROW => 5,
            Op::ENABLE_ERROR_MODE => 2,
            Op::PRINT => 10,
            Op::IMPORT | Op::EXPORT => 10,
            Op::PARALLEL_START => 15,
            Op::PARALLEL_END => 10,
            Op::FOR_ITER => 3,
            Op::UNKNOWN => 5,
        }
    }
}

// ── VM Value type ─────────────────────────────────────────────────────

/// Runtime value for the Rust VM stack machine.
/// Covers all Zexus primitive types plus Python interop.
#[derive(Debug)]
pub enum ZxValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<ZxValue>),
    Map(Vec<(String, ZxValue)>),
    /// Python object that cannot be represented natively.
    /// Requires GIL for access — used for callables, AST nodes, etc.
    PyObj(PyObject),
}

impl Clone for ZxValue {
    fn clone(&self) -> Self {
        match self {
            ZxValue::Null => ZxValue::Null,
            ZxValue::Bool(b) => ZxValue::Bool(*b),
            ZxValue::Int(i) => ZxValue::Int(*i),
            ZxValue::Float(f) => ZxValue::Float(*f),
            ZxValue::Str(s) => ZxValue::Str(s.clone()),
            ZxValue::List(v) => ZxValue::List(v.clone()),
            ZxValue::Map(v) => ZxValue::Map(v.clone()),
            ZxValue::PyObj(obj) => {
                Python::with_gil(|py| ZxValue::PyObj(obj.clone_ref(py)))
            }
        }
    }
}

impl ZxValue {
    /// Truthiness (matches Python semantics).
    pub fn is_truthy(&self) -> bool {
        match self {
            ZxValue::Null => false,
            ZxValue::Bool(b) => *b,
            ZxValue::Int(i) => *i != 0,
            ZxValue::Float(f) => *f != 0.0,
            ZxValue::Str(s) => !s.is_empty(),
            ZxValue::List(v) => !v.is_empty(),
            ZxValue::Map(v) => !v.is_empty(),
            ZxValue::PyObj(_) => true,
        }
    }

    /// Convert to i64, defaulting to 0 for non-numeric types.
    pub fn as_int(&self) -> i64 {
        match self {
            ZxValue::Int(i) => *i,
            ZxValue::Float(f) => *f as i64,
            ZxValue::Bool(b) => if *b { 1 } else { 0 },
            _ => 0,
        }
    }

    /// Convert to f64, defaulting to 0.0 for non-numeric types.
    pub fn as_float(&self) -> f64 {
        match self {
            ZxValue::Float(f) => *f,
            ZxValue::Int(i) => *i as f64,
            ZxValue::Bool(b) => if *b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }

    /// String representation for PRINT and error messages.
    pub fn display_str(&self) -> String {
        match self {
            ZxValue::Null => "null".to_string(),
            ZxValue::Bool(b) => b.to_string(),
            ZxValue::Int(i) => i.to_string(),
            ZxValue::Float(f) => format!("{}", f),
            ZxValue::Str(s) => s.clone(),
            ZxValue::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.display_str()).collect();
                format!("[{}]", inner.join(", "))
            }
            ZxValue::Map(pairs) => {
                let inner: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.display_str()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
            ZxValue::PyObj(_) => "<python object>".to_string(),
        }
    }

    /// Check if this is a numeric type (Int or Float).
    fn is_numeric(&self) -> bool {
        matches!(self, ZxValue::Int(_) | ZxValue::Float(_))
    }

    /// Check if either operand is float (for promotion).
    fn needs_float_promotion(a: &ZxValue, b: &ZxValue) -> bool {
        matches!(a, ZxValue::Float(_)) || matches!(b, ZxValue::Float(_))
    }
}

impl PartialEq for ZxValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ZxValue::Null, ZxValue::Null) => true,
            (ZxValue::Bool(a), ZxValue::Bool(b)) => a == b,
            (ZxValue::Int(a), ZxValue::Int(b)) => a == b,
            (ZxValue::Float(a), ZxValue::Float(b)) => a == b,
            (ZxValue::Int(a), ZxValue::Float(b)) => (*a as f64) == *b,
            (ZxValue::Float(a), ZxValue::Int(b)) => *a == (*b as f64),
            (ZxValue::Str(a), ZxValue::Str(b)) => a == b,
            (ZxValue::Bool(a), ZxValue::Int(b)) => (if *a { 1i64 } else { 0 }) == *b,
            (ZxValue::Int(a), ZxValue::Bool(b)) => *a == (if *b { 1i64 } else { 0 }),
            _ => false,
        }
    }
}

impl PartialOrd for ZxValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ZxValue::Int(a), ZxValue::Int(b)) => a.partial_cmp(b),
            (ZxValue::Float(a), ZxValue::Float(b)) => a.partial_cmp(b),
            (ZxValue::Int(a), ZxValue::Float(b)) => (*a as f64).partial_cmp(b),
            (ZxValue::Float(a), ZxValue::Int(b)) => a.partial_cmp(&(*b as f64)),
            (ZxValue::Str(a), ZxValue::Str(b)) => a.partial_cmp(b),
            (ZxValue::Bool(a), ZxValue::Bool(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

// ── ZxcValue → ZxValue conversion ────────────────────────────────────

fn zxc_to_zx(v: &ZxcValue) -> ZxValue {
    match v {
        ZxcValue::Null => ZxValue::Null,
        ZxcValue::Bool(b) => ZxValue::Bool(*b),
        ZxcValue::Int(i) => ZxValue::Int(*i),
        ZxcValue::Float(f) => ZxValue::Float(*f),
        ZxcValue::String(s) => ZxValue::Str(s.clone()),
        ZxcValue::FuncDesc(s) => ZxValue::Str(s.clone()), // treated as string for now
        ZxcValue::List(items) => ZxValue::List(items.iter().map(zxc_to_zx).collect()),
        ZxcValue::Map(pairs) => {
            let converted: Vec<(String, ZxValue)> = pairs
                .iter()
                .map(|(k, v)| {
                    let key = match k {
                        ZxcValue::String(s) => s.clone(),
                        other => format!("{}", other),
                    };
                    (key, zxc_to_zx(v))
                })
                .collect();
            ZxValue::Map(converted)
        }
        ZxcValue::Opaque(_) => ZxValue::Null, // Opaque data not usable in Rust
    }
}

// ── ZxValue → PyObject conversion ────────────────────────────────────

fn zx_to_py(py: Python<'_>, val: &ZxValue) -> PyObject {
    match val {
        ZxValue::Null => py.None(),
        ZxValue::Bool(b) => b.to_object(py),
        ZxValue::Int(i) => i.to_object(py),
        ZxValue::Float(f) => f.to_object(py),
        ZxValue::Str(s) => s.to_object(py),
        ZxValue::List(items) => {
            let py_list = PyList::empty_bound(py);
            for item in items {
                let _ = py_list.append(zx_to_py(py, item));
            }
            py_list.to_object(py)
        }
        ZxValue::Map(pairs) => {
            let py_dict = PyDict::new_bound(py);
            for (k, v) in pairs {
                let _ = py_dict.set_item(k, zx_to_py(py, v));
            }
            py_dict.to_object(py)
        }
        ZxValue::PyObj(obj) => obj.clone_ref(py),
    }
}

/// Convert a PyObject to ZxValue.
fn py_to_zx(py: Python<'_>, obj: &PyObject) -> ZxValue {
    let bound = obj.bind(py);

    if bound.is_none() {
        return ZxValue::Null;
    }
    if let Ok(b) = bound.downcast::<PyBool>() {
        return ZxValue::Bool(b.is_true());
    }
    if let Ok(i) = bound.extract::<i64>() {
        return ZxValue::Int(i);
    }
    if let Ok(f) = bound.extract::<f64>() {
        return ZxValue::Float(f);
    }
    if let Ok(s) = bound.extract::<String>() {
        return ZxValue::Str(s);
    }
    if let Ok(list) = bound.downcast::<PyList>() {
        let items: Vec<ZxValue> = list
            .iter()
            .map(|item| py_to_zx(py, &item.to_object(py)))
            .collect();
        return ZxValue::List(items);
    }
    if let Ok(dict) = bound.downcast::<PyDict>() {
        let pairs: Vec<(String, ZxValue)> = dict
            .iter()
            .map(|(k, v)| {
                let key = k.extract::<String>().unwrap_or_else(|_| format!("{}", k));
                let val = py_to_zx(py, &v.to_object(py));
                (key, val)
            })
            .collect();
        return ZxValue::Map(pairs);
    }

    // Fallback: keep as PyObject
    ZxValue::PyObj(obj.clone_ref(py))
}

// ── VM Errors ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VmError {
    OutOfGas { used: u64, limit: u64, opcode: String },
    RequireFailed(String),
    RuntimeError(String),
    UnsupportedOpcode(u16),
    /// Opcode requires Python fallback (e.g. CALL_NAME, CALL_METHOD).
    NeedsPythonFallback,
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::OutOfGas { used, limit, opcode } => {
                write!(f, "Out of gas: used={}, limit={}, opcode={}", used, limit, opcode)
            }
            VmError::RequireFailed(msg) => write!(f, "Requirement failed: {}", msg),
            VmError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            VmError::UnsupportedOpcode(op) => write!(f, "Unsupported opcode: {}", op),
            VmError::NeedsPythonFallback => write!(f, "Opcode requires Python fallback"),
        }
    }
}

// ── Transaction snapshot ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TxSnapshot {
    state: HashMap<String, ZxValue>,
    pending: HashMap<String, ZxValue>,
}

// ── The Rust VM ───────────────────────────────────────────────────────

/// Stack-based bytecode interpreter for Zexus.
pub struct RustVM {
    // Program
    constants: Vec<ZxValue>,
    instructions: Vec<(Op, Operand)>,

    // Execution state
    stack: Vec<ZxValue>,
    ip: usize,

    // Environment
    env: HashMap<String, ZxValue>,
    blockchain_state: HashMap<String, ZxValue>,

    // Gas metering
    gas_limit: u64,
    gas_used: u64,
    gas_enabled: bool,

    // Transaction state
    in_transaction: bool,
    tx_snapshot: Option<HashMap<String, ZxValue>>,
    tx_pending: HashMap<String, ZxValue>,
    tx_stack: Vec<TxSnapshot>,

    // Exception handling
    try_stack: Vec<usize>,

    // Ledger
    ledger: Vec<ZxValue>,

    // Audit log
    audit_log: Vec<(String, ZxValue)>,

    // Output capture (for PRINT)
    output: Vec<String>,

    // Stats
    instructions_executed: u64,
}

impl RustVM {
    /// Create a new VM from a deserialized ZxcModule.
    pub fn from_module(module: &ZxcModule) -> Self {
        let constants: Vec<ZxValue> = module.constants.iter().map(zxc_to_zx).collect();
        let instructions: Vec<(Op, Operand)> = module
            .instructions
            .iter()
            .map(|i| (Op::from_u16(i.opcode), i.operand.clone()))
            .collect();

        RustVM {
            constants,
            instructions,
            stack: Vec::with_capacity(256),
            ip: 0,
            env: HashMap::new(),
            blockchain_state: HashMap::new(),
            gas_limit: 100_000_000,
            gas_used: 0,
            gas_enabled: false,
            in_transaction: false,
            tx_snapshot: None,
            tx_pending: HashMap::new(),
            tx_stack: Vec::new(),
            try_stack: Vec::new(),
            ledger: Vec::new(),
            audit_log: Vec::new(),
            output: Vec::new(),
            instructions_executed: 0,
        }
    }

    /// Set gas limit and enable gas metering.
    pub fn set_gas_limit(&mut self, limit: u64) {
        self.gas_limit = limit;
        self.gas_enabled = true;
    }

    /// Set initial environment variables.
    pub fn set_env(&mut self, env: HashMap<String, ZxValue>) {
        self.env = env;
    }

    /// Set initial blockchain state.
    pub fn set_blockchain_state(&mut self, state: HashMap<String, ZxValue>) {
        self.blockchain_state = state;
    }

    // ── Helpers ───────────────────────────────────────────────────────

    #[inline(always)]
    fn pop(&mut self) -> ZxValue {
        self.stack.pop().unwrap_or(ZxValue::Null)
    }

    #[inline(always)]
    fn push(&mut self, v: ZxValue) {
        self.stack.push(v);
    }

    #[inline(always)]
    fn peek(&self) -> &ZxValue {
        self.stack.last().unwrap_or(&ZxValue::Null)
    }

    #[inline(always)]
    fn const_val(&self, idx: u32) -> ZxValue {
        self.constants
            .get(idx as usize)
            .cloned()
            .unwrap_or(ZxValue::Null)
    }

    fn const_str(&self, idx: u32) -> String {
        match self.constants.get(idx as usize) {
            Some(ZxValue::Str(s)) => s.clone(),
            Some(other) => other.display_str(),
            None => String::new(),
        }
    }

    #[inline(always)]
    fn consume_gas(&mut self, op: Op) -> Result<(), VmError> {
        if !self.gas_enabled {
            return Ok(());
        }
        let cost = op.gas_cost();
        self.gas_used += cost;
        if self.gas_used > self.gas_limit {
            return Err(VmError::OutOfGas {
                used: self.gas_used,
                limit: self.gas_limit,
                opcode: format!("{:?}", op),
            });
        }
        Ok(())
    }

    fn revert_on_failure(&mut self) {
        if self.in_transaction {
            if let Some(snapshot) = self.tx_snapshot.take() {
                self.blockchain_state = snapshot;
            }
            self.in_transaction = false;
            self.tx_pending.clear();
        }
    }

    // ── Arithmetic helpers (with type promotion) ─────────────────────

    fn arith_add(a: ZxValue, b: ZxValue) -> ZxValue {
        match (&a, &b) {
            (ZxValue::Int(x), ZxValue::Int(y)) => ZxValue::Int(x.wrapping_add(*y)),
            (ZxValue::Float(x), ZxValue::Float(y)) => ZxValue::Float(x + y),
            (ZxValue::Int(x), ZxValue::Float(y)) => ZxValue::Float(*x as f64 + y),
            (ZxValue::Float(x), ZxValue::Int(y)) => ZxValue::Float(x + *y as f64),
            (ZxValue::Str(x), ZxValue::Str(y)) => {
                let mut s = x.clone();
                s.push_str(y);
                ZxValue::Str(s)
            }
            (ZxValue::Str(x), _) => {
                let mut s = x.clone();
                s.push_str(&b.display_str());
                ZxValue::Str(s)
            }
            (_, ZxValue::Str(y)) => {
                let mut s = a.display_str();
                s.push_str(y);
                ZxValue::Str(s)
            }
            (ZxValue::List(x), ZxValue::List(y)) => {
                let mut result = x.clone();
                result.extend(y.iter().cloned());
                ZxValue::List(result)
            }
            _ => ZxValue::Int(a.as_int().wrapping_add(b.as_int())),
        }
    }

    fn arith_sub(a: ZxValue, b: ZxValue) -> ZxValue {
        if ZxValue::needs_float_promotion(&a, &b) {
            ZxValue::Float(a.as_float() - b.as_float())
        } else {
            ZxValue::Int(a.as_int().wrapping_sub(b.as_int()))
        }
    }

    fn arith_mul(a: ZxValue, b: ZxValue) -> ZxValue {
        match (&a, &b) {
            (ZxValue::Str(s), ZxValue::Int(n)) | (ZxValue::Int(n), ZxValue::Str(s)) => {
                ZxValue::Str(s.repeat((*n).max(0) as usize))
            }
            _ if ZxValue::needs_float_promotion(&a, &b) => {
                ZxValue::Float(a.as_float() * b.as_float())
            }
            _ => ZxValue::Int(a.as_int().wrapping_mul(b.as_int())),
        }
    }

    fn arith_div(a: ZxValue, b: ZxValue) -> ZxValue {
        let bv = b.as_float();
        if bv == 0.0 {
            return ZxValue::Int(0);
        }
        if ZxValue::needs_float_promotion(&a, &b) {
            ZxValue::Float(a.as_float() / bv)
        } else {
            let ai = a.as_int();
            let bi = b.as_int();
            if bi == 0 {
                ZxValue::Int(0)
            } else {
                ZxValue::Int(ai / bi)
            }
        }
    }

    fn arith_mod(a: ZxValue, b: ZxValue) -> ZxValue {
        let bi = b.as_int();
        if bi == 0 {
            ZxValue::Int(0)
        } else {
            ZxValue::Int(a.as_int() % bi)
        }
    }

    fn arith_pow(a: ZxValue, b: ZxValue) -> ZxValue {
        if ZxValue::needs_float_promotion(&a, &b) {
            ZxValue::Float(a.as_float().powf(b.as_float()))
        } else {
            let base = a.as_int();
            let exp = b.as_int();
            if exp < 0 {
                ZxValue::Float((base as f64).powf(exp as f64))
            } else {
                ZxValue::Int(base.wrapping_pow(exp as u32))
            }
        }
    }

    // ── Main execution loop ──────────────────────────────────────────

    /// Execute the loaded bytecode program.
    /// Returns the final value (from RETURN or top of stack).
    pub fn execute(&mut self) -> Result<ZxValue, VmError> {
        let n_instrs = self.instructions.len();

        while self.ip < n_instrs {
            let (op, operand) = self.instructions[self.ip].clone();
            self.ip += 1;
            self.instructions_executed += 1;

            // Gas check
            self.consume_gas(op)?;

            match op {
                // ── Stack operations ─────────────────────────────
                Op::LOAD_CONST => {
                    if let Operand::U32(idx) = &operand {
                        self.push(self.const_val(*idx));
                    }
                }

                Op::LOAD_NAME => {
                    let name = match &operand {
                        Operand::U32(idx) => self.const_str(*idx),
                        _ => String::new(),
                    };
                    let val = self
                        .env
                        .get(&name)
                        .cloned()
                        .unwrap_or(ZxValue::Null);
                    self.push(val);
                }

                Op::STORE_NAME => {
                    let name = match &operand {
                        Operand::U32(idx) => self.const_str(*idx),
                        _ => String::new(),
                    };
                    let val = self.pop();
                    self.env.insert(name, val);
                }

                Op::STORE_FUNC => {
                    let name = match &operand {
                        Operand::U32(idx) => self.const_str(*idx),
                        _ => String::new(),
                    };
                    let val = self.pop();
                    self.env.insert(name, val);
                }

                Op::POP => {
                    self.pop();
                }

                Op::DUP => {
                    let v = self.peek().clone();
                    self.push(v);
                }

                // ── Arithmetic ───────────────────────────────────
                Op::ADD => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_add(a, b));
                }

                Op::SUB => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_sub(a, b));
                }

                Op::MUL => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_mul(a, b));
                }

                Op::DIV => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_div(a, b));
                }

                Op::MOD => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_mod(a, b));
                }

                Op::POW => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Self::arith_pow(a, b));
                }

                Op::NEG => {
                    let a = self.pop();
                    match a {
                        ZxValue::Int(i) => self.push(ZxValue::Int(-i)),
                        ZxValue::Float(f) => self.push(ZxValue::Float(-f)),
                        _ => self.push(ZxValue::Int(0)),
                    }
                }

                // ── Comparison ───────────────────────────────────
                Op::EQ => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a == b));
                }

                Op::NEQ => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a != b));
                }

                Op::LT => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a.partial_cmp(&b) == Some(std::cmp::Ordering::Less)));
                }

                Op::GT => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater)));
                }

                Op::LTE => {
                    let b = self.pop();
                    let a = self.pop();
                    let cmp = a.partial_cmp(&b);
                    self.push(ZxValue::Bool(
                        cmp == Some(std::cmp::Ordering::Less) || cmp == Some(std::cmp::Ordering::Equal),
                    ));
                }

                Op::GTE => {
                    let b = self.pop();
                    let a = self.pop();
                    let cmp = a.partial_cmp(&b);
                    self.push(ZxValue::Bool(
                        cmp == Some(std::cmp::Ordering::Greater) || cmp == Some(std::cmp::Ordering::Equal),
                    ));
                }

                // ── Logical ──────────────────────────────────────
                Op::AND => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a.is_truthy() && b.is_truthy()));
                }

                Op::OR => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(ZxValue::Bool(a.is_truthy() || b.is_truthy()));
                }

                Op::NOT => {
                    let a = self.pop();
                    self.push(ZxValue::Bool(!a.is_truthy()));
                }

                // ── Control flow ─────────────────────────────────
                Op::JUMP => {
                    if let Operand::U32(target) = &operand {
                        self.ip = *target as usize;
                    }
                }

                Op::JUMP_IF_FALSE => {
                    let cond = self.pop();
                    if !cond.is_truthy() {
                        if let Operand::U32(target) = &operand {
                            self.ip = *target as usize;
                        }
                    }
                }

                Op::JUMP_IF_TRUE => {
                    let cond = self.pop();
                    if cond.is_truthy() {
                        if let Operand::U32(target) = &operand {
                            self.ip = *target as usize;
                        }
                    }
                }

                Op::RETURN => {
                    return Ok(self.pop());
                }

                // ── Collections ──────────────────────────────────
                Op::BUILD_LIST => {
                    let count = match &operand {
                        Operand::U32(n) => *n as usize,
                        _ => 0,
                    };
                    let mut items = Vec::with_capacity(count);
                    for _ in 0..count {
                        items.push(self.pop());
                    }
                    items.reverse();
                    self.push(ZxValue::List(items));
                }

                Op::BUILD_MAP => {
                    let count = match &operand {
                        Operand::U32(n) => *n as usize,
                        _ => 0,
                    };
                    let mut pairs = Vec::with_capacity(count);
                    for _ in 0..count {
                        let val = self.pop();
                        let key = self.pop();
                        let key_str = match &key {
                            ZxValue::Str(s) => s.clone(),
                            other => other.display_str(),
                        };
                        pairs.push((key_str, val));
                    }
                    // Reverse to maintain insertion order
                    pairs.reverse();
                    self.push(ZxValue::Map(pairs));
                }

                Op::BUILD_SET => {
                    // Treat sets as lists (Zexus doesn't have a native set type in VM)
                    let count = match &operand {
                        Operand::U32(n) => *n as usize,
                        _ => 0,
                    };
                    let mut items = Vec::with_capacity(count);
                    for _ in 0..count {
                        items.push(self.pop());
                    }
                    items.reverse();
                    self.push(ZxValue::List(items));
                }

                Op::INDEX => {
                    let idx = self.pop();
                    let obj = self.pop();
                    match (&obj, &idx) {
                        (ZxValue::List(items), ZxValue::Int(i)) => {
                            let i = *i as usize;
                            self.push(items.get(i).cloned().unwrap_or(ZxValue::Null));
                        }
                        (ZxValue::Map(pairs), ZxValue::Str(key)) => {
                            let found = pairs.iter().find(|(k, _)| k == key);
                            self.push(found.map(|(_, v)| v.clone()).unwrap_or(ZxValue::Null));
                        }
                        (ZxValue::Str(s), ZxValue::Int(i)) => {
                            let i = *i as usize;
                            self.push(
                                s.chars()
                                    .nth(i)
                                    .map(|c| ZxValue::Str(c.to_string()))
                                    .unwrap_or(ZxValue::Null),
                            );
                        }
                        _ => self.push(ZxValue::Null),
                    }
                }

                Op::SLICE => {
                    let end = self.pop();
                    let start = self.pop();
                    let obj = self.pop();
                    match &obj {
                        ZxValue::List(items) => {
                            let s = start.as_int().max(0) as usize;
                            let e = end.as_int().max(0) as usize;
                            let e = e.min(items.len());
                            let s = s.min(e);
                            self.push(ZxValue::List(items[s..e].to_vec()));
                        }
                        ZxValue::Str(string) => {
                            let s = start.as_int().max(0) as usize;
                            let e = end.as_int().max(0) as usize;
                            let chars: Vec<char> = string.chars().collect();
                            let e = e.min(chars.len());
                            let s = s.min(e);
                            self.push(ZxValue::Str(chars[s..e].iter().collect()));
                        }
                        _ => self.push(ZxValue::Null),
                    }
                }

                Op::GET_ATTR => {
                    let attr = self.pop();
                    let obj = self.pop();
                    let attr_name = match &attr {
                        ZxValue::Str(s) => s.clone(),
                        other => other.display_str(),
                    };
                    match &obj {
                        ZxValue::Map(pairs) => {
                            let found = pairs.iter().find(|(k, _)| k == &attr_name);
                            self.push(found.map(|(_, v)| v.clone()).unwrap_or(ZxValue::Null));
                        }
                        _ => self.push(ZxValue::Null),
                    }
                }

                // ── Function calls ───────────────────────────────
                // These require Python interop for general callables.
                // For pure-Rust execution, we handle simple builtins
                // and signal NeedsPythonFallback for complex cases.
                Op::CALL_NAME | Op::CALL_TOP | Op::CALL_METHOD
                | Op::CALL_BUILTIN | Op::CALL_FUNC_CONST => {
                    // For Phase 2 MVP: signal fallback to Python for calls
                    // Phase 3+ will inline common builtins
                    return Err(VmError::NeedsPythonFallback);
                }

                // ── I/O ──────────────────────────────────────────
                Op::PRINT => {
                    let val = self.pop();
                    self.output.push(val.display_str());
                }

                // ── Blockchain opcodes ───────────────────────────
                Op::STATE_READ => {
                    let key = match &operand {
                        Operand::U32(idx) => self.const_str(*idx),
                        _ => {
                            let k = self.pop();
                            match k {
                                ZxValue::Str(s) => s,
                                other => other.display_str(),
                            }
                        }
                    };
                    let val = self
                        .blockchain_state
                        .get(&key)
                        .cloned()
                        .unwrap_or(ZxValue::Null);
                    self.push(val);
                }

                Op::STATE_WRITE => {
                    let val = self.pop();
                    let key = match &operand {
                        Operand::U32(idx) => self.const_str(*idx),
                        _ => {
                            let k = self.pop();
                            match k {
                                ZxValue::Str(s) => s,
                                other => other.display_str(),
                            }
                        }
                    };
                    if self.in_transaction {
                        self.tx_pending.insert(key, val);
                    } else {
                        self.blockchain_state.insert(key, val);
                    }
                }

                Op::TX_BEGIN => {
                    self.tx_stack.push(TxSnapshot {
                        state: self.blockchain_state.clone(),
                        pending: self.tx_pending.clone(),
                    });
                    self.in_transaction = true;
                    self.tx_snapshot = Some(self.blockchain_state.clone());
                    self.tx_pending.clear();
                }

                Op::TX_COMMIT => {
                    if self.in_transaction {
                        // Merge pending writes into blockchain state
                        for (k, v) in self.tx_pending.drain() {
                            self.blockchain_state.insert(k, v);
                        }
                        self.tx_snapshot = None;
                        // Restore outer transaction's pending writes
                        if let Some(outer) = self.tx_stack.pop() {
                            self.tx_pending = outer.pending;
                        }
                        self.in_transaction = !self.tx_stack.is_empty();
                    }
                }

                Op::TX_REVERT => {
                    if self.in_transaction {
                        if let Some(snapshot) = self.tx_snapshot.take() {
                            self.blockchain_state = snapshot;
                        }
                        self.tx_pending.clear();
                        if let Some(outer) = self.tx_stack.pop() {
                            if !self.tx_stack.is_empty() {
                                self.tx_snapshot = Some(outer.state);
                                self.tx_pending = outer.pending;
                            }
                        }
                        self.in_transaction = !self.tx_stack.is_empty();
                    }
                }

                Op::GAS_CHARGE => {
                    let amount = match &operand {
                        Operand::U32(n) => *n as u64,
                        _ => 0,
                    };
                    if self.gas_enabled {
                        self.gas_used += amount;
                        if self.gas_used > self.gas_limit {
                            self.revert_on_failure();
                            return Err(VmError::OutOfGas {
                                used: self.gas_used,
                                limit: self.gas_limit,
                                opcode: "GAS_CHARGE".to_string(),
                            });
                        }
                    }
                }

                Op::REQUIRE => {
                    let message = self.pop();
                    let condition = self.pop();
                    if !condition.is_truthy() {
                        self.revert_on_failure();
                        return Err(VmError::RequireFailed(message.display_str()));
                    }
                }

                Op::LEDGER_APPEND => {
                    let entry = self.pop();
                    if self.ledger.len() < 10000 {
                        self.ledger.push(entry);
                    }
                }

                Op::HASH_BLOCK => {
                    let data = self.pop();
                    let data_str = data.display_str();
                    let data_bytes = data_str.as_bytes();
                    use sha2::{Digest, Sha256};
                    let hash = Sha256::digest(data_bytes);
                    self.push(ZxValue::Str(hex::encode(hash)));
                }

                Op::VERIFY_SIGNATURE => {
                    // Simplified: pop pubkey, message, signature
                    // Full ECDSA verification would go through RustSignature
                    if self.stack.len() >= 3 {
                        let _pk = self.pop();
                        let _msg = self.pop();
                        let _sig = self.pop();
                        // TODO: wire through k256 verification
                        self.push(ZxValue::Bool(false));
                    } else {
                        self.push(ZxValue::Bool(false));
                    }
                }

                Op::MERKLE_ROOT => {
                    let leaf_count = match &operand {
                        Operand::U32(n) => *n as usize,
                        _ => 0,
                    };
                    if leaf_count == 0 || self.stack.len() < leaf_count {
                        self.push(ZxValue::Str(String::new()));
                    } else {
                        use sha2::{Digest, Sha256};
                        let mut leaves: Vec<ZxValue> = Vec::with_capacity(leaf_count);
                        for _ in 0..leaf_count {
                            leaves.push(self.pop());
                        }
                        leaves.reverse();
                        let mut hashes: Vec<String> = leaves
                            .iter()
                            .map(|leaf| {
                                let s = leaf.display_str();
                                hex::encode(Sha256::digest(s.as_bytes()))
                            })
                            .collect();
                        while hashes.len() > 1 {
                            if hashes.len() % 2 != 0 {
                                let last = hashes.last().unwrap().clone();
                                hashes.push(last);
                            }
                            let mut new_hashes = Vec::new();
                            for i in (0..hashes.len()).step_by(2) {
                                let combined = format!("{}{}", hashes[i], hashes[i + 1]);
                                new_hashes
                                    .push(hex::encode(Sha256::digest(combined.as_bytes())));
                            }
                            hashes = new_hashes;
                        }
                        self.push(ZxValue::Str(
                            hashes.into_iter().next().unwrap_or_default(),
                        ));
                    }
                }

                Op::EMIT_EVENT | Op::REGISTER_EVENT => {
                    // Events: pop data, store for Python to collect
                    let _data = self.pop();
                    // No-op in pure Rust — Python collects events post-execution
                }

                Op::AUDIT_LOG => {
                    let data = self.pop();
                    let action = self.pop();
                    self.audit_log.push((action.display_str(), data));
                }

                Op::RESTRICT_ACCESS => {
                    let _restriction = self.pop();
                    let _prop = self.pop();
                    let _obj = self.pop();
                    // Access control enforcement happens at contract level
                }

                // ── Exception handling ───────────────────────────
                Op::SETUP_TRY => {
                    let handler_ip = match &operand {
                        Operand::U32(target) => *target as usize,
                        _ => self.ip,
                    };
                    self.try_stack.push(handler_ip);
                }

                Op::POP_TRY => {
                    self.try_stack.pop();
                }

                Op::THROW => {
                    let exc = self.pop();
                    if let Some(handler_ip) = self.try_stack.pop() {
                        self.push(exc);
                        self.ip = handler_ip;
                    } else {
                        return Err(VmError::RuntimeError(exc.display_str()));
                    }
                }

                Op::ENABLE_ERROR_MODE => {
                    // No-op — error mode is a Python evaluator concept
                }

                // ── Marker ops ───────────────────────────────────
                Op::PARALLEL_START | Op::PARALLEL_END | Op::NOP => {}

                // ── Module / Contract ops (fallback to Python) ───
                Op::IMPORT | Op::EXPORT | Op::DEFINE_CONTRACT | Op::DEFINE_ENTITY
                | Op::SPAWN | Op::AWAIT | Op::SPAWN_CALL | Op::FOR_ITER => {
                    return Err(VmError::NeedsPythonFallback);
                }

                Op::UNKNOWN => {
                    return Err(VmError::UnsupportedOpcode(
                        self.instructions[self.ip - 1].0 as u16,
                    ));
                }
            }
        }

        // End of instructions — return top of stack or Null
        Ok(if self.stack.is_empty() {
            ZxValue::Null
        } else {
            self.pop()
        })
    }

    // ── Accessors for Python bridge ──────────────────────────────────

    /// Get the blockchain state after execution.
    pub fn get_blockchain_state(&self) -> &HashMap<String, ZxValue> {
        &self.blockchain_state
    }

    /// Get the execution stats.
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (self.instructions_executed, self.gas_used, self.gas_limit)
    }

    /// Get captured output (from PRINT opcodes).
    pub fn get_output(&self) -> &[String] {
        &self.output
    }

    /// Get the environment after execution.
    pub fn get_env(&self) -> &HashMap<String, ZxValue> {
        &self.env
    }

    /// Get the ledger entries.
    pub fn get_ledger(&self) -> &[ZxValue] {
        &self.ledger
    }
}

// ── PyO3 Bridge — RustVMExecutor ─────────────────────────────────────

/// Python-callable executor wrapping the Rust bytecode interpreter.
///
/// Usage from Python:
/// ```python
/// from zexus_core import RustVMExecutor
/// executor = RustVMExecutor()
/// result = executor.execute(zxc_bytes, env={}, state={}, gas_limit=1000000)
/// ```
#[pyclass]
pub struct RustVMExecutor {
    /// Stats from last execution
    last_instructions: u64,
    last_gas_used: u64,
}

#[pymethods]
impl RustVMExecutor {
    #[new]
    fn new() -> Self {
        RustVMExecutor {
            last_instructions: 0,
            last_gas_used: 0,
        }
    }

    /// Execute .zxc binary bytecode and return the result.
    ///
    /// Args:
    ///     data: bytes — serialized .zxc bytecode
    ///     env: dict — initial environment variables (optional)
    ///     state: dict — initial blockchain state (optional)
    ///     gas_limit: int — gas limit (0 = unlimited)
    ///
    /// Returns:
    ///     dict with keys: result, env, state, output, gas_used,
    ///                     instructions_executed, needs_fallback
    #[pyo3(signature = (data, env=None, state=None, gas_limit=0))]
    fn execute(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        env: Option<&Bound<'_, PyDict>>,
        state: Option<&Bound<'_, PyDict>>,
        gas_limit: u64,
    ) -> PyResult<PyObject> {
        let raw = data.as_bytes();

        // Deserialize .zxc → ZxcModule (GIL-free)
        let module = binary_bytecode::deserialize_zxc(raw, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        // Create VM
        let mut vm = RustVM::from_module(&module);

        // Set gas limit
        if gas_limit > 0 {
            vm.set_gas_limit(gas_limit);
        }

        // Convert Python env → Rust env
        if let Some(py_env) = env {
            let mut rust_env = HashMap::new();
            for (k, v) in py_env.iter() {
                let key = k.extract::<String>().unwrap_or_default();
                let val = py_to_zx(py, &v.to_object(py));
                rust_env.insert(key, val);
            }
            vm.set_env(rust_env);
        }

        // Convert Python state → Rust state
        if let Some(py_state) = state {
            let mut rust_state = HashMap::new();
            for (k, v) in py_state.iter() {
                let key = k.extract::<String>().unwrap_or_default();
                let val = py_to_zx(py, &v.to_object(py));
                rust_state.insert(key, val);
            }
            vm.set_blockchain_state(rust_state);
        }

        // Execute (GIL is held but computation is pure Rust)
        let exec_result = vm.execute();

        // Build result dict
        let result_dict = PyDict::new_bound(py);

        match exec_result {
            Ok(val) => {
                result_dict.set_item("result", zx_to_py(py, &val))?;
                result_dict.set_item("needs_fallback", false)?;
                result_dict.set_item("error", py.None())?;
            }
            Err(VmError::NeedsPythonFallback) => {
                result_dict.set_item("result", py.None())?;
                result_dict.set_item("needs_fallback", true)?;
                result_dict.set_item("error", py.None())?;
            }
            Err(VmError::OutOfGas { used, limit, opcode }) => {
                result_dict.set_item("result", py.None())?;
                result_dict.set_item("needs_fallback", false)?;
                result_dict
                    .set_item("error", format!("OutOfGas: used={}, limit={}, op={}", used, limit, opcode))?;
            }
            Err(VmError::RequireFailed(msg)) => {
                result_dict.set_item("result", py.None())?;
                result_dict.set_item("needs_fallback", false)?;
                result_dict.set_item("error", format!("RequireFailed: {}", msg))?;
            }
            Err(e) => {
                result_dict.set_item("result", py.None())?;
                result_dict.set_item("needs_fallback", false)?;
                result_dict.set_item("error", format!("{}", e))?;
            }
        }

        // Return environment
        let py_env_out = PyDict::new_bound(py);
        for (k, v) in vm.get_env() {
            py_env_out.set_item(k, zx_to_py(py, v))?;
        }
        result_dict.set_item("env", py_env_out)?;

        // Return blockchain state
        let py_state_out = PyDict::new_bound(py);
        for (k, v) in vm.get_blockchain_state() {
            py_state_out.set_item(k, zx_to_py(py, v))?;
        }
        result_dict.set_item("state", py_state_out)?;

        // Output
        let py_output = PyList::empty_bound(py);
        for line in vm.get_output() {
            py_output.append(line)?;
        }
        result_dict.set_item("output", py_output)?;

        // Stats
        let (instr_count, gas_used, _gas_limit) = vm.get_stats();
        result_dict.set_item("gas_used", gas_used)?;
        result_dict.set_item("instructions_executed", instr_count)?;

        self.last_instructions = instr_count;
        self.last_gas_used = gas_used;

        Ok(result_dict.to_object(py))
    }

    /// Get stats from the last execution.
    fn last_stats(&self) -> (u64, u64) {
        (self.last_instructions, self.last_gas_used)
    }

    /// Execute .zxc binary bytecode without any Python interop.
    /// Returns only numeric stats (for benchmarking).
    #[pyo3(signature = (data, iterations=1, gas_limit=0))]
    fn benchmark(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        iterations: u32,
        gas_limit: u64,
    ) -> PyResult<PyObject> {
        let raw = data.as_bytes();

        let module = binary_bytecode::deserialize_zxc(raw, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let start = std::time::Instant::now();
        let mut total_instrs: u64 = 0;
        let mut last_result = ZxValue::Null;

        for _ in 0..iterations {
            let mut vm = RustVM::from_module(&module);
            if gas_limit > 0 {
                vm.set_gas_limit(gas_limit);
            }
            match vm.execute() {
                Ok(val) => {
                    let (instrs, _, _) = vm.get_stats();
                    total_instrs += instrs;
                    last_result = val;
                }
                Err(VmError::NeedsPythonFallback) => {
                    let (instrs, _, _) = vm.get_stats();
                    total_instrs += instrs;
                }
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)));
                }
            }
        }

        let elapsed = start.elapsed();
        let result = PyDict::new_bound(py);
        result.set_item("iterations", iterations)?;
        result.set_item("total_instructions", total_instrs)?;
        result.set_item("elapsed_ms", elapsed.as_secs_f64() * 1000.0)?;
        result.set_item(
            "instructions_per_sec",
            if elapsed.as_secs_f64() > 0.0 {
                total_instrs as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
        )?;
        result.set_item("result", zx_to_py(py, &last_result))?;

        Ok(result.to_object(py))
    }
}
