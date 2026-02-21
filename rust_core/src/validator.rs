// ─────────────────────────────────────────────────────────────────────
// Block Validation — fast parallel checks
// ─────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Validates block header integrity at native speed.
#[pyclass]
pub struct RustBlockValidator;

#[pymethods]
impl RustBlockValidator {
    #[new]
    fn new() -> Self {
        RustBlockValidator
    }

    /// Verify a block hash matches its header contents.
    ///
    /// `header_json` — JSON-encoded block header
    /// `expected_hash` — the hash claimed by the block
    ///
    /// Returns `true` if SHA-256(header_json) == expected_hash.
    #[staticmethod]
    fn verify_block_hash(header_json: &str, expected_hash: &str) -> bool {
        let hash = Sha256::digest(header_json.as_bytes());
        hex::encode(hash) == expected_hash
    }

    /// Validate a chain of block hashes in parallel.
    ///
    /// `blocks` — list of (header_json, claimed_hash, prev_hash) tuples.
    /// Returns a list of booleans — one per block.
    ///
    /// Checks:
    ///   1. SHA-256(header) == claimed_hash
    ///   2. prev_hash links correctly (sequential check)
    #[staticmethod]
    fn validate_chain(
        py: Python<'_>,
        blocks: Vec<(String, String, String)>,
    ) -> Vec<bool> {
        if blocks.is_empty() {
            return vec![];
        }

        // Step 1: verify hashes in parallel
        let hash_checks: Vec<bool> = py.allow_threads(|| {
            blocks
                .par_iter()
                .map(|(header, claimed, _)| {
                    let hash = Sha256::digest(header.as_bytes());
                    hex::encode(hash) == *claimed
                })
                .collect()
        });

        // Step 2: verify chain linkage (sequential — inherently ordered)
        let mut results = hash_checks;
        for i in 1..blocks.len() {
            let expected_prev = &blocks[i - 1].1; // previous block's hash
            let actual_prev = &blocks[i].2; // this block's prev_hash
            if expected_prev != actual_prev {
                results[i] = false;
            }
        }

        results
    }

    /// Check Proof-of-Work difficulty: does the block hash have the
    /// required number of leading zero bits?
    #[staticmethod]
    fn check_pow_difficulty(block_hash: &str, difficulty: u32) -> bool {
        let bytes = match hex::decode(block_hash) {
            Ok(b) => b,
            Err(_) => return false,
        };

        let mut leading_zeros = 0u32;
        for byte in &bytes {
            if *byte == 0 {
                leading_zeros += 8;
            } else {
                leading_zeros += byte.leading_zeros();
                break;
            }
        }

        leading_zeros >= difficulty
    }

    /// Batch-validate transaction signatures within a block.
    ///
    /// `tx_data` — list of (message_bytes, signature_bytes, pubkey_bytes).
    /// Returns the number of valid signatures.
    #[staticmethod]
    fn validate_tx_signatures(
        py: Python<'_>,
        tx_data: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    ) -> usize {
        let results: Vec<bool> = py.allow_threads(|| {
            use rayon::prelude::*;
            tx_data
                .par_iter()
                .map(|(msg, sig, pk)| {
                    crate::signature::verify_single(msg, sig, pk)
                })
                .collect()
        });
        results.iter().filter(|&&v| v).count()
    }
}
