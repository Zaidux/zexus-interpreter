"""
Zexus Blockchain — Rust Execution Core Bridge
===============================================

Provides a seamless Python interface to the native Rust execution core
(``zexus_core``).  When the Rust extension is compiled and installed the
bridge delegates hot-path operations to native code for maximum
throughput.  When the extension is **not** available the bridge falls
back to the pure-Python implementations with **zero breakage** — every
function has a Python fallback.

The bridge is transparent: callers use the same API regardless of
whether Rust is present.

Exposed APIs
------------
*   ``RustCoreBridge`` — unified façade
*   ``rust_core_available()`` — quick check
*   ``sha256()`` / ``keccak256()`` — hashing (native or fallback)
*   ``compute_merkle_root()`` — parallel Merkle root
*   ``verify_signature()`` / ``verify_signatures_batch()``
*   ``execute_batch()`` — parallel tx execution via Rayon
*   ``validate_chain()`` — parallel block-header validation

Build the Rust core
-------------------
::

    cd rust_core/
    pip install maturin
    maturin develop --release
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("zexus.blockchain.rust_bridge")

# ── Try to import the Rust extension ───────────────────────────────────

_RUST_AVAILABLE = False
_zexus_core = None

try:
    import zexus_core as _zexus_core  # type: ignore[import-untyped]
    _RUST_AVAILABLE = _zexus_core.is_available()
    if _RUST_AVAILABLE:
        logger.info(
            "Rust execution core loaded (v%s) — native acceleration enabled",
            _zexus_core.version(),
        )
except ImportError:
    logger.info(
        "Rust execution core not found — using pure-Python fallback.  "
        "For maximum throughput (~1800+ TPS) build the Rust core:  "
        "cd rust_core/ && maturin develop --release"
    )


def rust_core_available() -> bool:
    """Return ``True`` if the native Rust execution core is loaded."""
    return _RUST_AVAILABLE


# ── Hashing ────────────────────────────────────────────────────────────

def sha256(data: bytes) -> str:
    """SHA-256 hash → hex string.  Uses Rust if available."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.sha256(data)
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    """SHA-256 of UTF-8 string → hex."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.sha256_str(text)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256d(data: bytes) -> str:
    """Double SHA-256 (Bitcoin-style) → hex."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.sha256d(data)
    h1 = hashlib.sha256(data).digest()
    return hashlib.sha256(h1).hexdigest()


def keccak256(data: bytes) -> str:
    """Keccak-256 (Ethereum-style) → hex.  Uses Rust if available."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.keccak256(data)
    # Python fallback using hashlib if available, else pycryptodome
    try:
        import sha3  # type: ignore
        return sha3.keccak_256(data).hexdigest()
    except ImportError:
        # Minimal fallback — use SHA-256 as stand-in (not true keccak)
        logger.warning("No keccak library — falling back to SHA-256")
        return hashlib.sha256(data).hexdigest()


def sha256_batch(items: List[bytes]) -> List[str]:
    """Batch SHA-256 — parallel in Rust, sequential in Python."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.sha256_batch(items)
    return [hashlib.sha256(d).hexdigest() for d in items]


def keccak256_batch(items: List[bytes]) -> List[str]:
    """Batch Keccak-256 — parallel in Rust."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustHasher.keccak256_batch(items)
    return [keccak256(d) for d in items]


# ── Merkle ─────────────────────────────────────────────────────────────

def compute_merkle_root(leaves: List[str]) -> str:
    """Compute Merkle root from hex leaf hashes — parallel in Rust."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustMerkle.compute_root(leaves)
    return _merkle_root_py(leaves)


def verify_merkle_proof(
    leaf_hash: str,
    proof: List[Tuple[str, str]],
    root: str,
) -> bool:
    """Verify a Merkle inclusion proof."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustMerkle.verify_proof(leaf_hash, proof, root)
    return _merkle_verify_py(leaf_hash, proof, root)


def _merkle_root_py(leaves: List[str]) -> str:
    """Pure-Python Merkle root fallback."""
    if not leaves:
        return "0" * 64
    current = list(leaves)
    while len(current) > 1:
        if len(current) % 2 != 0:
            current.append(current[-1])
        next_level = []
        for i in range(0, len(current), 2):
            combined = bytes.fromhex(current[i]) + bytes.fromhex(current[i + 1])
            next_level.append(hashlib.sha256(combined).hexdigest())
        current = next_level
    return current[0]


def _merkle_verify_py(
    leaf_hash: str,
    proof: List[Tuple[str, str]],
    root: str,
) -> bool:
    """Pure-Python Merkle proof verification fallback."""
    current = bytes.fromhex(leaf_hash)
    for sibling_hex, direction in proof:
        sibling = bytes.fromhex(sibling_hex)
        if direction == "left":
            combined = sibling + current
        else:
            combined = current + sibling
        current = hashlib.sha256(combined).digest()
    return current.hex() == root


# ── Signature verification ─────────────────────────────────────────────

def verify_signature(
    message: bytes, signature: bytes, public_key: bytes
) -> bool:
    """Verify an ECDSA-secp256k1 signature.  Uses Rust if available."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustSignature.verify(message, signature, public_key)
    return _verify_sig_py(message, signature, public_key)


def verify_signatures_batch(
    items: List[Tuple[bytes, bytes, bytes]],
) -> List[bool]:
    """Batch-verify signatures — parallel in Rust."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustSignature.verify_batch(items)
    return [_verify_sig_py(m, s, p) for m, s, p in items]


def _verify_sig_py(
    message: bytes, signature: bytes, public_key: bytes
) -> bool:
    """Pure-Python ECDSA verification fallback."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ec, utils
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.exceptions import InvalidSignature

        vk = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key
        )
        vk.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


# ── Batch Execution ────────────────────────────────────────────────────

class BatchResult:
    """Wraps Rust's TxBatchResult or pure-Python equivalent."""

    def __init__(
        self,
        total: int = 0,
        succeeded: int = 0,
        failed: int = 0,
        gas_used: int = 0,
        elapsed: float = 0.0,
        receipts: Optional[List[Dict[str, Any]]] = None,
    ):
        self.total = total
        self.succeeded = succeeded
        self.failed = failed
        self.gas_used = gas_used
        self.elapsed = elapsed
        self.receipts = receipts or []

    @property
    def throughput(self) -> float:
        return self.total / self.elapsed if self.elapsed > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"BatchResult(total={self.total}, ok={self.succeeded}, "
            f"fail={self.failed}, gas={self.gas_used}, "
            f"{self.throughput:.1f} tx/s)"
        )


def execute_batch(
    transactions: List[Dict[str, Any]],
    vm_callback: Callable,
    max_workers: int = 0,
) -> BatchResult:
    """Execute a transaction batch using Rust parallelism or Python fallback.

    Parameters
    ----------
    transactions : list of dict
        Each dict: ``{contract, action, args, caller, gas_limit}``
    vm_callback : callable
        ``fn(contract, action, args_json, caller, gas_limit) -> dict``
    max_workers : int
        Rayon thread count (0 = auto-detect CPU cores).
    """
    if _RUST_AVAILABLE:
        # Rust executor expects args as JSON strings
        serialized = []
        for tx in transactions:
            serialized.append({
                "contract": str(tx.get("contract", "")),
                "action": str(tx.get("action", "")),
                "args": json.dumps(tx.get("args", {})) if not isinstance(tx.get("args"), str) else tx["args"],
                "caller": str(tx.get("caller", "")),
                "gas_limit": str(tx.get("gas_limit", "0")),
            })

        executor = _zexus_core.RustBatchExecutor(max_workers=max_workers)
        result = executor.execute_batch(serialized, vm_callback)

        receipts = []
        for r_json in result.receipts:
            try:
                receipts.append(json.loads(r_json))
            except json.JSONDecodeError:
                receipts.append({"success": False, "error": r_json})

        return BatchResult(
            total=result.total,
            succeeded=result.succeeded,
            failed=result.failed,
            gas_used=result.gas_used,
            elapsed=result.elapsed_secs,
            receipts=receipts,
        )

    # Pure-Python fallback
    return _execute_batch_py(transactions, vm_callback)


def _execute_batch_py(
    transactions: List[Dict[str, Any]],
    vm_callback: Callable,
) -> BatchResult:
    """Pure-Python sequential batch execution fallback."""
    start = time.time()
    result = BatchResult(total=len(transactions))

    for tx in transactions:
        args_json = json.dumps(tx.get("args", {}))
        try:
            receipt = vm_callback(
                tx.get("contract", ""),
                tx.get("action", ""),
                args_json,
                tx.get("caller", ""),
                tx.get("gas_limit", 0),
            )
            if isinstance(receipt, dict) and receipt.get("success"):
                result.succeeded += 1
                result.gas_used += receipt.get("gas_used", 0)
            else:
                result.failed += 1
            result.receipts.append(receipt if isinstance(receipt, dict) else {})
        except Exception as e:
            result.failed += 1
            result.receipts.append({"success": False, "error": str(e)})

    result.elapsed = time.time() - start
    return result


# ── Block Validation ───────────────────────────────────────────────────

def validate_chain_headers(
    blocks: List[Tuple[str, str, str]],
) -> List[bool]:
    """Validate chain of (header_json, hash, prev_hash) — parallel in Rust."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustBlockValidator.validate_chain(blocks)
    # Python fallback
    results = []
    for i, (header_json, claimed_hash, prev_hash) in enumerate(blocks):
        hash_ok = hashlib.sha256(header_json.encode()).hexdigest() == claimed_hash
        link_ok = True
        if i > 0:
            link_ok = blocks[i - 1][1] == prev_hash
        results.append(hash_ok and link_ok)
    return results


def check_pow_difficulty(block_hash: str, difficulty: int) -> bool:
    """Check PoW leading-zero-bits requirement — fast in Rust."""
    if _RUST_AVAILABLE:
        return _zexus_core.RustBlockValidator.check_pow_difficulty(
            block_hash, difficulty
        )
    # Python fallback
    hash_bytes = bytes.fromhex(block_hash)
    leading = 0
    for b in hash_bytes:
        if b == 0:
            leading += 8
        else:
            leading += (b ^ (b - 1)).bit_length() - 1 if b else 8
            break
    return leading >= difficulty


# ── Unified Bridge Class ───────────────────────────────────────────────

class RustCoreBridge:
    """Unified entry point providing all Rust-accelerated operations.

    Usage::

        bridge = RustCoreBridge()
        if bridge.is_native:
            print("Rust acceleration active")

        root = bridge.merkle_root(leaf_hashes)
        result = bridge.execute_batch(txs, vm_callback)
    """

    def __init__(self, max_workers: int = 0):
        self._max_workers = max_workers

    @property
    def is_native(self) -> bool:
        return _RUST_AVAILABLE

    @property
    def engine(self) -> str:
        return "rust/rayon" if _RUST_AVAILABLE else "python/threads"

    # Hashing
    sha256 = staticmethod(sha256)
    sha256_str = staticmethod(sha256_str)
    sha256d = staticmethod(sha256d)
    keccak256 = staticmethod(keccak256)
    sha256_batch = staticmethod(sha256_batch)
    keccak256_batch = staticmethod(keccak256_batch)

    # Merkle
    merkle_root = staticmethod(compute_merkle_root)
    merkle_verify = staticmethod(verify_merkle_proof)

    # Signatures
    verify_sig = staticmethod(verify_signature)
    verify_sigs_batch = staticmethod(verify_signatures_batch)

    # Execution
    def execute_batch(
        self,
        transactions: List[Dict[str, Any]],
        vm_callback: Callable,
    ) -> BatchResult:
        return execute_batch(
            transactions, vm_callback, max_workers=self._max_workers
        )

    # Validation
    validate_chain = staticmethod(validate_chain_headers)
    check_pow = staticmethod(check_pow_difficulty)
