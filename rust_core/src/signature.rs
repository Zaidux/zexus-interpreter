// ─────────────────────────────────────────────────────────────────────
// ECDSA-secp256k1 Signature Verification  (batch-parallel)
// ─────────────────────────────────────────────────────────────────────

use k256::ecdsa::{
    signature::Verifier, Signature, VerifyingKey,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

#[pyclass]
pub struct RustSignature;

#[pymethods]
impl RustSignature {
    #[new]
    fn new() -> Self {
        RustSignature
    }

    /// Verify a single ECDSA-secp256k1 signature.
    ///
    /// * `message`   — the raw message bytes that were signed
    /// * `signature` — 64-byte DER or compact signature
    /// * `public_key`— 33-byte (compressed) or 65-byte (uncompressed) public key
    ///
    /// Returns `true` if the signature is valid.
    #[staticmethod]
    fn verify(message: Vec<u8>, signature: Vec<u8>, public_key: Vec<u8>) -> bool {
        Self::_verify_inner(&message, &signature, &public_key)
    }

    /// Batch-verify multiple signatures in parallel.
    ///
    /// Each item is a tuple `(message_bytes, signature_bytes, pubkey_bytes)`.
    /// Returns a list of booleans (one per signature).
    #[staticmethod]
    fn verify_batch(
        py: Python<'_>,
        items: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    ) -> Vec<bool> {
        py.allow_threads(|| {
            items
                .par_iter()
                .map(|(msg, sig, pk)| Self::_verify_inner(msg, sig, pk))
                .collect()
        })
    }

    /// Hash a message with SHA-256 then verify the signature against the
    /// hash.  This matches the common blockchain pattern of sign(sha256(msg)).
    #[staticmethod]
    fn verify_hashed(message: Vec<u8>, signature: Vec<u8>, public_key: Vec<u8>) -> bool {
        let hash = Sha256::digest(&message);
        Self::_verify_inner(&hash, &signature, &public_key)
    }
}

impl RustSignature {
    fn _verify_inner(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
        verify_single(message, signature, public_key)
    }
}

/// Public helper so other modules (validator) can call it directly.
pub fn verify_single(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
    let vk = match VerifyingKey::from_sec1_bytes(public_key) {
        Ok(vk) => vk,
        Err(_) => return false,
    };

    let sig = match Signature::from_slice(signature) {
        Ok(s) => s,
        Err(_) => return false,
    };

    vk.verify(message, &sig).is_ok()
}
