// ─────────────────────────────────────────────────────────────────────
// Zexus Blockchain — Rust Execution Core
// ─────────────────────────────────────────────────────────────────────
//
// High-performance native execution engine exposed to Python via PyO3.
//
// Hot paths moved to Rust:
//   • Batch transaction execution (parallel via Rayon)
//   • SHA-256 / Keccak-256 hashing
//   • ECDSA-secp256k1 signature verification
//   • Merkle root computation
//   • Block header validation
//
// The Python `ExecutionAccelerator` detects this extension at import
// time and delegates to it automatically.  When the extension is not
// compiled the system falls back to the pure-Python implementation
// with zero breakage.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

mod executor;
mod hasher;
mod merkle;
mod signature;
mod validator;

use executor::{RustBatchExecutor, TxBatchResult as RustTxBatchResult};
use hasher::RustHasher;
use merkle::RustMerkle;
use signature::RustSignature;
use validator::RustBlockValidator;

// ── Python module definition ──────────────────────────────────────────

/// The native Zexus execution core.
#[pymodule]
fn zexus_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBatchExecutor>()?;
    m.add_class::<RustTxBatchResult>()?;
    m.add_class::<RustHasher>()?;
    m.add_class::<RustMerkle>()?;
    m.add_class::<RustSignature>()?;
    m.add_class::<RustBlockValidator>()?;

    // Convenience — quick check from Python:  `zexus_core.is_available()`
    #[pyfn(m)]
    fn is_available() -> bool {
        true
    }

    #[pyfn(m)]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    Ok(())
}
