"""Tests for stdlib CryptoModule."""

import hashlib
import hmac as pyhmac
import pytest
from src.zexus.stdlib.crypto import CryptoModule


@pytest.fixture
def c():
    return CryptoModule()


# ── Hash functions ───────────────────────────────────────────────────────
class TestHashFunctions:
    def test_sha256(self, c):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert c.hash_sha256("hello") == expected

    def test_sha512(self, c):
        expected = hashlib.sha512(b"hello").hexdigest()
        assert c.hash_sha512("hello") == expected

    def test_md5(self, c):
        expected = hashlib.md5(b"hello").hexdigest()
        assert c.hash_md5("hello") == expected

    def test_blake2b(self, c):
        result = c.hash_blake2b("hello")
        assert isinstance(result, str) and len(result) > 0

    def test_blake2s(self, c):
        result = c.hash_blake2s("hello")
        assert isinstance(result, str) and len(result) > 0

    def test_sha3_256(self, c):
        expected = hashlib.sha3_256(b"hello").hexdigest()
        assert c.sha3_256("hello") == expected

    def test_sha3_512(self, c):
        expected = hashlib.sha3_512(b"hello").hexdigest()
        assert c.sha3_512("hello") == expected

    def test_keccak256(self, c):
        result = c.keccak256("hello")
        assert isinstance(result, str) and len(result) == 64


# ── HMAC ─────────────────────────────────────────────────────────────────
class TestHMAC:
    def test_hmac_sha256(self, c):
        expected = pyhmac.new(b"key", b"msg", hashlib.sha256).hexdigest()
        assert c.hmac_sha256("msg", "key") == expected

    def test_hmac_sha512(self, c):
        expected = pyhmac.new(b"key", b"msg", hashlib.sha512).hexdigest()
        assert c.hmac_sha512("msg", "key") == expected


# ── Random bytes / int ───────────────────────────────────────────────────
class TestRandomGeneration:
    def test_random_bytes_length(self, c):
        result = c.random_bytes(16)
        assert isinstance(result, str) and len(result) == 32

    def test_random_int_range(self, c):
        for _ in range(20):
            val = c.random_int(1, 100)
            assert 1 <= val <= 100


# ── Comparison ───────────────────────────────────────────────────────────
class TestComparison:
    def test_compare_digest_same(self, c):
        assert c.compare_digest("abc", "abc") is True

    def test_compare_digest_different(self, c):
        assert c.compare_digest("abc", "xyz") is False

    def test_constant_time_compare(self, c):
        assert c.constant_time_compare("a", "a") is True
        assert c.constant_time_compare("a", "b") is False


# ── PBKDF2 ───────────────────────────────────────────────────────────────
class TestPBKDF2:
    def test_pbkdf2_deterministic(self, c):
        r1 = c.pbkdf2("password", "salt", 1000, 32)
        r2 = c.pbkdf2("password", "salt", 1000, 32)
        assert r1 == r2
        assert isinstance(r1, str)

    def test_pbkdf2_different_passwords(self, c):
        r1 = c.pbkdf2("pass1", "salt", 1000, 32)
        r2 = c.pbkdf2("pass2", "salt", 1000, 32)
        assert r1 != r2


# ── Salt ─────────────────────────────────────────────────────────────────
class TestSalt:
    def test_generate_salt_length(self, c):
        result = c.generate_salt(16)
        assert isinstance(result, str) and len(result) == 32

    def test_generate_salt_unique(self, c):
        s1 = c.generate_salt()
        s2 = c.generate_salt()
        assert s1 != s2
