"""Tests for stdlib EncodingModule."""

import zlib
import json
import pytest
from src.zexus.stdlib.encoding import EncodingModule


@pytest.fixture
def enc():
    return EncodingModule()


# ── Base64 ───────────────────────────────────────────────────────────────
class TestBase64:
    def test_encode_decode(self, enc):
        encoded = enc.base64_encode("hello world")
        decoded = enc.base64_decode(encoded)
        assert decoded == "hello world"

    def test_urlsafe_encode_decode(self, enc):
        data = "data with +/= chars"
        encoded = enc.base64_urlsafe_encode(data)
        decoded = enc.base64_urlsafe_decode(encoded)
        assert decoded == data

    def test_empty_string(self, enc):
        assert enc.base64_decode(enc.base64_encode("")) == ""


# ── Base32 ───────────────────────────────────────────────────────────────
class TestBase32:
    def test_encode_decode(self, enc):
        encoded = enc.base32_encode("test data")
        decoded = enc.base32_decode(encoded)
        assert decoded == "test data"


# ── Hex ──────────────────────────────────────────────────────────────────
class TestHex:
    def test_encode_decode(self, enc):
        encoded = enc.hex_encode("ABC")
        decoded = enc.hex_decode(encoded)
        assert decoded == "ABC"

    def test_base16_encode_decode(self, enc):
        encoded = enc.base16_encode("hello")
        decoded = enc.base16_decode(encoded)
        assert decoded == "hello"


# ── Base85 / ASCII85 ────────────────────────────────────────────────────
class TestBase85:
    def test_base85_roundtrip(self, enc):
        encoded = enc.base85_encode("test85")
        decoded = enc.base85_decode(encoded)
        assert decoded == "test85"

    def test_ascii85_roundtrip(self, enc):
        encoded = enc.ascii85_encode("test85")
        decoded = enc.ascii85_decode(encoded)
        assert decoded == "test85"


# ── URL encoding ─────────────────────────────────────────────────────────
class TestURLEncoding:
    def test_url_encode_decode(self, enc):
        encoded = enc.url_encode("hello world&foo=bar")
        decoded = enc.url_decode(encoded)
        assert decoded == "hello world&foo=bar"

    def test_url_encode_plus(self, enc):
        encoded = enc.url_encode_plus("hello world")
        assert "+" in encoded
        decoded = enc.url_decode_plus(encoded)
        assert decoded == "hello world"


# ── HTML encoding ────────────────────────────────────────────────────────
class TestHTMLEncoding:
    def test_html_encode_decode(self, enc):
        encoded = enc.html_encode("<script>alert('xss')</script>")
        assert "<" not in encoded
        decoded = enc.html_decode(encoded)
        assert decoded == "<script>alert('xss')</script>"


# ── Unicode ──────────────────────────────────────────────────────────────
class TestUnicode:
    def test_unicode_encode_decode(self, enc):
        encoded = enc.unicode_encode("café")
        decoded = enc.unicode_decode(encoded)
        assert decoded == "café"

    def test_unicode_normalize(self, enc):
        result = enc.unicode_normalize("café", "NFC")
        assert isinstance(result, str)


# ── Binary ───────────────────────────────────────────────────────────────
class TestBinary:
    def test_to_from_binary(self, enc):
        binary = enc.to_binary("A")
        result = enc.from_binary(binary)
        assert result == "A"


# ── ASCII ────────────────────────────────────────────────────────────────
class TestASCII:
    def test_to_from_ascii_codes(self, enc):
        codes = enc.to_ascii_codes("Hi")
        assert codes == [72, 105]
        result = enc.from_ascii_codes(codes)
        assert result == "Hi"


# ── ROT13 ────────────────────────────────────────────────────────────────
class TestROT13:
    def test_rot13_roundtrip(self, enc):
        encoded = enc.rot13("Hello World")
        assert encoded != "Hello World"
        decoded = enc.rot13(encoded)
        assert decoded == "Hello World"


# ── Checksums ────────────────────────────────────────────────────────────
class TestChecksums:
    def test_crc32(self, enc):
        result = enc.crc32("hello")
        expected = zlib.crc32(b"hello") & 0xFFFFFFFF
        assert result == expected

    def test_adler32(self, enc):
        result = enc.adler32("hello")
        expected = zlib.adler32(b"hello") & 0xFFFFFFFF
        assert result == expected


# ── JSON wrappers ────────────────────────────────────────────────────────
class TestJSONWrappers:
    def test_json_encode_decode(self, enc):
        obj = {"a": 1, "b": [2, 3]}
        encoded = enc.json_encode(obj)
        decoded = enc.json_decode(encoded)
        assert decoded == obj

    def test_json_encode_pretty(self, enc):
        result = enc.json_encode({"x": 1}, pretty=True)
        assert "\n" in result
