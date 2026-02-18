"""
Zexus Post-Quantum Cryptography Bridge

Provides post-quantum digital signature support.  When the ``pqcrypto``
or ``oqs`` library is available, real SPHINCS+-SHA2-128f is used.
Otherwise, a pure-Python **WOTS+ (Winternitz One-Time Signature Plus)**
hash-based scheme is provided as a cryptographically real fallback.

WOTS+ is provably secure under the assumption that the underlying hash
function (SHA-256) is a pseudorandom function.  It is one of the building
blocks of XMSS (RFC 8391) and SPHINCS+.

Limitations of the WOTS+ fallback:
  - Each private key should only sign ONE message safely.  Re-using a
    WOTS+ key for a second message leaks partial private key information.
  - Signatures are ~2 KB (67 × 32 bytes).

For production deployments involving many signatures per key, install a
real post-quantum library:
    pip install pqcrypto   # or:  pip install oqs
"""

import hashlib
import secrets
from typing import Dict, Tuple

# ── Try real post-quantum libraries first ────────────────────────────

_PQ_BACKEND = "wots"  # default fallback

try:
    import oqs  # liboqs-python
    _PQ_BACKEND = "oqs"
except ImportError:
    pass

if _PQ_BACKEND == "wots":
    try:
        from pqcrypto.sign.sphincs_sha2_128f import (
            generate_keypair as _sphincs_keygen,
            sign as _sphincs_sign_raw,
            verify as _sphincs_verify_raw,
        )
        _PQ_BACKEND = "pqcrypto"
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  WOTS+ (Winternitz One-Time Signature Plus) — Pure Python Fallback
# ═══════════════════════════════════════════════════════════════════════

_WOTS_N = 32          # security parameter (hash output bytes = 256 bits)
_WOTS_W = 16          # Winternitz parameter
_WOTS_LEN1 = 64       # ceil(8*N / log2(W))  = ceil(256/4) = 64
_WOTS_LEN2 = 3        # floor(log2(LEN1 * (W-1)) / log2(W)) + 1 = 3
_WOTS_LEN = _WOTS_LEN1 + _WOTS_LEN2  # 67 chains total

# Fixed public seed for domain separation (safe — only provides randomization)
_WOTS_PUB_SEED = hashlib.sha256(b"zexus-wots-public-seed-v1").digest()


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _wots_chain(value: bytes, start: int, steps: int, addr: int) -> bytes:
    """Iterate the hash chain ``steps`` times starting at index ``start``.

    Uses bitmask indices start, start+1, ..., start+steps-1 so that
    chain(x, 0, a, addr) followed by chain(result, a, b, addr) equals
    chain(x, 0, a+b, addr).  This composability is essential for WOTS+.
    """
    result = value
    for i in range(start, start + steps):
        bitmask = _hash(_WOTS_PUB_SEED + addr.to_bytes(4, "big") + i.to_bytes(4, "big"))
        result = _hash(bytes(a ^ b for a, b in zip(result, bitmask)))
    return result


def _base_w(msg_hash: bytes, w: int = _WOTS_W) -> list:
    """Convert hash to base-w representation (4-bit nibbles for w=16)."""
    out = []
    for byte in msg_hash:
        out.append(byte >> 4)
        out.append(byte & 0x0F)
    return out


def _checksum(msg_base_w: list) -> list:
    """Compute WOTS+ checksum digits."""
    total = sum(_WOTS_W - 1 - v for v in msg_base_w)
    cs = []
    for _ in range(_WOTS_LEN2):
        cs.append(total % _WOTS_W)
        total //= _WOTS_W
    cs.reverse()
    return cs


def wots_keygen(seed: bytes = b"") -> Tuple[bytes, bytes]:
    """Generate a WOTS+ keypair.

    Returns:
        (private_key, public_key)
        private_key = 32-byte seed
        public_key  = 67 × 32 = 2144 bytes
    """
    if not seed:
        seed = secrets.token_bytes(_WOTS_N)

    pk_parts = []
    for i in range(_WOTS_LEN):
        sk_i = _hash(seed + i.to_bytes(4, "big"))
        pk_i = _wots_chain(sk_i, 0, _WOTS_W - 1, i)
        pk_parts.append(pk_i)

    return seed, b"".join(pk_parts)


def wots_sign(message: bytes, private_key: bytes) -> bytes:
    """Sign a message using WOTS+.

    Returns signature bytes (67 × 32 = 2144 bytes).
    """
    seed = private_key[:_WOTS_N]
    msg_hash = _hash(message)
    msg_bw = _base_w(msg_hash)
    msg_bw += _checksum(msg_bw)

    sig_parts = []
    for i in range(_WOTS_LEN):
        sk_i = _hash(seed + i.to_bytes(4, "big"))
        sig_i = _wots_chain(sk_i, 0, msg_bw[i], i)
        sig_parts.append(sig_i)

    return b"".join(sig_parts)


def wots_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify a WOTS+ signature.

    Returns True only if the signature is cryptographically valid.
    """
    if len(signature) != _WOTS_LEN * _WOTS_N:
        return False
    if len(public_key) != _WOTS_LEN * _WOTS_N:
        return False

    msg_hash = _hash(message)
    msg_bw = _base_w(msg_hash)
    msg_bw += _checksum(msg_bw)

    for i in range(_WOTS_LEN):
        sig_i = signature[i * _WOTS_N: (i + 1) * _WOTS_N]
        pk_i = public_key[i * _WOTS_N: (i + 1) * _WOTS_N]
        remaining = _WOTS_W - 1 - msg_bw[i]
        computed = _wots_chain(sig_i, msg_bw[i], remaining, i)
        if computed != pk_i:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════
#  Public API — auto-selects the best available backend
# ═══════════════════════════════════════════════════════════════════════

def sha256_hash(data):
    """SHA-256 hash (hex)."""
    return hashlib.sha256(data.encode()).hexdigest()


def generate_sphincs_keypair() -> Dict[str, str]:
    """Generate a post-quantum keypair.

    Returns dict with ``public_key`` and ``private_key`` (hex-encoded),
    plus ``algorithm`` indicating which backend was used.
    """
    if _PQ_BACKEND == "oqs":
        signer = oqs.Signature("SPHINCS+-SHA2-128f-simple")
        pk = signer.generate_keypair()
        sk = signer.export_secret_key()
        return {
            "public_key": pk.hex(),
            "private_key": sk.hex(),
            "algorithm": "SPHINCS+-SHA2-128f (liboqs)",
        }

    if _PQ_BACKEND == "pqcrypto":
        pk, sk = _sphincs_keygen()
        return {
            "public_key": pk.hex(),
            "private_key": sk.hex(),
            "algorithm": "SPHINCS+-SHA2-128f (pqcrypto)",
        }

    # WOTS+ fallback
    sk, pk = wots_keygen()
    return {
        "public_key": pk.hex(),
        "private_key": sk.hex(),
        "algorithm": "WOTS+ (hash-based OTS, pure Python)",
    }


def sphincs_sign(message, private_key):
    """Sign a message with the post-quantum private key.

    Returns hex-encoded signature.
    """
    msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
    sk_bytes = bytes.fromhex(private_key) if isinstance(private_key, str) else private_key

    if _PQ_BACKEND == "oqs":
        signer = oqs.Signature("SPHINCS+-SHA2-128f-simple", sk_bytes)
        sig = signer.sign(msg_bytes)
        return sig.hex()

    if _PQ_BACKEND == "pqcrypto":
        sig = _sphincs_sign_raw(sk_bytes, msg_bytes)
        return sig.hex()

    # WOTS+ fallback
    sig = wots_sign(msg_bytes, sk_bytes)
    return sig.hex()


def sphincs_verify(message, signature, public_key):
    """Verify a post-quantum signature.

    Returns True ONLY if the signature is cryptographically valid.
    No longer returns True unconditionally.
    """
    msg_bytes = message.encode("utf-8") if isinstance(message, str) else message

    try:
        sig_bytes = bytes.fromhex(signature) if isinstance(signature, str) else signature
        pk_bytes = bytes.fromhex(public_key) if isinstance(public_key, str) else public_key
    except (ValueError, AttributeError):
        return False

    if _PQ_BACKEND == "oqs":
        verifier = oqs.Signature("SPHINCS+-SHA2-128f-simple")
        return verifier.verify(msg_bytes, sig_bytes, pk_bytes)

    if _PQ_BACKEND == "pqcrypto":
        try:
            _sphincs_verify_raw(pk_bytes, msg_bytes, sig_bytes)
            return True
        except Exception:
            return False

    # WOTS+ fallback
    return wots_verify(msg_bytes, sig_bytes, pk_bytes)
