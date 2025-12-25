"""Crypto module for Zexus standard library."""

import hashlib
import secrets
import hmac


class CryptoModule:
    """Provides cryptographic operations."""

    @staticmethod
    def hash_sha256(data: str) -> str:
        """Calculate SHA-256 hash."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def hash_sha512(data: str) -> str:
        """Calculate SHA-512 hash."""
        return hashlib.sha512(data.encode()).hexdigest()

    @staticmethod
    def hash_md5(data: str) -> str:
        """Calculate MD5 hash (not recommended for security)."""
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def hash_blake2b(data: str, digest_size: int = 32) -> str:
        """Calculate BLAKE2b hash."""
        return hashlib.blake2b(data.encode(), digest_size=digest_size).hexdigest()

    @staticmethod
    def hash_blake2s(data: str, digest_size: int = 32) -> str:
        """Calculate BLAKE2s hash."""
        return hashlib.blake2s(data.encode(), digest_size=digest_size).hexdigest()

    @staticmethod
    def hmac_sha256(data: str, key: str) -> str:
        """Calculate HMAC-SHA256."""
        return hmac.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def hmac_sha512(data: str, key: str) -> str:
        """Calculate HMAC-SHA512."""
        return hmac.new(key.encode(), data.encode(), hashlib.sha512).hexdigest()

    @staticmethod
    def random_bytes(size: int = 32) -> str:
        """Generate random bytes (hex encoded)."""
        return secrets.token_hex(size)

    @staticmethod
    def random_int(min_val: int = 0, max_val: int = 2**32) -> int:
        """Generate random integer."""
        return secrets.randbelow(max_val - min_val) + min_val

    @staticmethod
    def compare_digest(a: str, b: str) -> bool:
        """Constant-time string comparison."""
        return hmac.compare_digest(a, b)

    @staticmethod
    def keccak256(data: str) -> str:
        """Calculate Keccak-256 hash (Ethereum-style)."""
        try:
            from Crypto.Hash import keccak
            k = keccak.new(digest_bits=256)
            k.update(data.encode())
            return k.hexdigest()
        except ImportError:
            # Fallback to SHA3-256 if pycryptodome not available
            return hashlib.sha3_256(data.encode()).hexdigest()

    @staticmethod
    def sha3_256(data: str) -> str:
        """Calculate SHA3-256 hash."""
        return hashlib.sha3_256(data.encode()).hexdigest()

    @staticmethod
    def sha3_512(data: str) -> str:
        """Calculate SHA3-512 hash."""
        return hashlib.sha3_512(data.encode()).hexdigest()

    @staticmethod
    def pbkdf2(password: str, salt: str, iterations: int = 100000, 
               key_length: int = 32, algorithm: str = 'sha256') -> str:
        """Derive key from password using PBKDF2."""
        hash_name = algorithm.lower().replace('-', '')
        derived_key = hashlib.pbkdf2_hmac(
            hash_name,
            password.encode(),
            salt.encode(),
            iterations,
            dklen=key_length
        )
        return derived_key.hex()

    @staticmethod
    def generate_salt(size: int = 16) -> str:
        """Generate random salt."""
        return secrets.token_hex(size)

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Constant-time comparison (alias for compare_digest)."""
        return hmac.compare_digest(a, b)


# Export functions for easy access
hash_sha256 = CryptoModule.hash_sha256
hash_sha512 = CryptoModule.hash_sha512
hash_md5 = CryptoModule.hash_md5
hash_blake2b = CryptoModule.hash_blake2b
hash_blake2s = CryptoModule.hash_blake2s
hmac_sha256 = CryptoModule.hmac_sha256
hmac_sha512 = CryptoModule.hmac_sha512
random_bytes = CryptoModule.random_bytes
random_int = CryptoModule.random_int
compare_digest = CryptoModule.compare_digest
keccak256 = CryptoModule.keccak256
sha3_256 = CryptoModule.sha3_256
sha3_512 = CryptoModule.sha3_512
pbkdf2 = CryptoModule.pbkdf2
generate_salt = CryptoModule.generate_salt
constant_time_compare = CryptoModule.constant_time_compare
