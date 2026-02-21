// ─────────────────────────────────────────────────────────────────────
// Cryptographic Hashing — SHA-256 & Keccak-256
// ─────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use tiny_keccak::{Hasher as TinyHasher, Keccak};

#[pyclass]
pub struct RustHasher;

#[pymethods]
impl RustHasher {
    #[new]
    fn new() -> Self {
        RustHasher
    }

    /// SHA-256 of raw bytes → hex string.
    #[staticmethod]
    fn sha256(data: Vec<u8>) -> String {
        let hash = Sha256::digest(&data);
        hex::encode(hash)
    }

    /// SHA-256 of a UTF-8 string → hex string.
    #[staticmethod]
    fn sha256_str(text: &str) -> String {
        let hash = Sha256::digest(text.as_bytes());
        hex::encode(hash)
    }

    /// Double SHA-256 (Bitcoin-style) → hex string.
    #[staticmethod]
    fn sha256d(data: Vec<u8>) -> String {
        let first = Sha256::digest(&data);
        let second = Sha256::digest(&first);
        hex::encode(second)
    }

    /// Keccak-256 (Ethereum-style) → hex string.
    #[staticmethod]
    fn keccak256(data: Vec<u8>) -> String {
        let mut hasher = Keccak::v256();
        let mut output = [0u8; 32];
        hasher.update(&data);
        hasher.finalize(&mut output);
        hex::encode(output)
    }

    /// Keccak-256 of a UTF-8 string → hex string.
    #[staticmethod]
    fn keccak256_str(text: &str) -> String {
        let mut hasher = Keccak::v256();
        let mut output = [0u8; 32];
        hasher.update(text.as_bytes());
        hasher.finalize(&mut output);
        hex::encode(output)
    }

    /// Batch SHA-256: hash many byte-arrays in parallel → list of hex strings.
    #[staticmethod]
    fn sha256_batch(py: Python<'_>, items: Vec<Vec<u8>>) -> Vec<String> {
        use rayon::prelude::*;
        py.allow_threads(|| {
            items
                .par_iter()
                .map(|d| hex::encode(Sha256::digest(d)))
                .collect()
        })
    }

    /// Batch Keccak-256 in parallel.
    #[staticmethod]
    fn keccak256_batch(py: Python<'_>, items: Vec<Vec<u8>>) -> Vec<String> {
        use rayon::prelude::*;
        py.allow_threads(|| {
            items
                .par_iter()
                .map(|d| {
                    let mut h = Keccak::v256();
                    let mut out = [0u8; 32];
                    h.update(d);
                    h.finalize(&mut out);
                    hex::encode(out)
                })
                .collect()
        })
    }
}
