// ─────────────────────────────────────────────────────────────────────
// Merkle Root Computation — parallelised
// ─────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

#[pyclass]
pub struct RustMerkle;

#[pymethods]
impl RustMerkle {
    #[new]
    fn new() -> Self {
        RustMerkle
    }

    /// Compute the Merkle root of a list of hex-encoded leaf hashes.
    ///
    /// Uses a standard binary Merkle tree (duplicate last leaf if odd).
    /// The hashing combines pairs with SHA-256(left || right).
    ///
    /// Returns the root as a hex string.
    #[staticmethod]
    fn compute_root(py: Python<'_>, leaves: Vec<String>) -> String {
        if leaves.is_empty() {
            return "0".repeat(64);
        }
        if leaves.len() == 1 {
            return leaves[0].clone();
        }

        // Decode hex leaves to raw bytes
        let mut current: Vec<[u8; 32]> = leaves
            .iter()
            .map(|h| {
                let bytes = hex::decode(h).unwrap_or_else(|_| {
                    // If not valid hex, hash the raw string
                    Sha256::digest(h.as_bytes()).to_vec()
                });
                let mut arr = [0u8; 32];
                let len = bytes.len().min(32);
                arr[..len].copy_from_slice(&bytes[..len]);
                arr
            })
            .collect();

        py.allow_threads(|| {
            while current.len() > 1 {
                // Duplicate last if odd
                if current.len() % 2 != 0 {
                    current.push(*current.last().unwrap());
                }

                // Pair and hash in parallel
                let pairs: Vec<([u8; 32], [u8; 32])> = current
                    .chunks(2)
                    .map(|c| (c[0], c[1]))
                    .collect();

                current = pairs
                    .par_iter()
                    .map(|(left, right)| {
                        let mut combined = Vec::with_capacity(64);
                        combined.extend_from_slice(left);
                        combined.extend_from_slice(right);
                        let hash = Sha256::digest(&combined);
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&hash);
                        arr
                    })
                    .collect();
            }

            hex::encode(current[0])
        })
    }

    /// Compute Merkle root from raw transaction data (list of byte arrays).
    /// Each item is SHA-256 hashed first to produce the leaf, then the
    /// tree is built.
    #[staticmethod]
    fn compute_root_from_data(py: Python<'_>, data: Vec<Vec<u8>>) -> String {
        let leaves: Vec<String> = py.allow_threads(|| {
            data.par_iter()
                .map(|d| hex::encode(Sha256::digest(d)))
                .collect()
        });
        // Re-acquire GIL for the tree computation call
        RustMerkle::compute_root(py, leaves)
    }

    /// Verify a Merkle proof.
    ///
    /// `leaf_hash` — the hex hash of the item to verify  
    /// `proof`     — list of (hex_hash, "left"|"right") pairs  
    /// `root`      — expected root hex  
    #[staticmethod]
    fn verify_proof(
        leaf_hash: &str,
        proof: Vec<(String, String)>,
        root: &str,
    ) -> bool {
        let mut current = hex::decode(leaf_hash).unwrap_or_default();
        if current.len() != 32 {
            return false;
        }

        for (sibling_hex, direction) in &proof {
            let sibling = hex::decode(sibling_hex).unwrap_or_default();
            if sibling.len() != 32 {
                return false;
            }
            let mut combined = Vec::with_capacity(64);
            if direction == "left" {
                combined.extend_from_slice(&sibling);
                combined.extend_from_slice(&current);
            } else {
                combined.extend_from_slice(&current);
                combined.extend_from_slice(&sibling);
            }
            current = Sha256::digest(&combined).to_vec();
        }

        hex::encode(&current) == root
    }
}
