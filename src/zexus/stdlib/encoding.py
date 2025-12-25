"""Encoding module for Zexus standard library."""

import base64
import binascii
import json
from typing import Dict, Any


class EncodingModule:
    """Provides encoding/decoding operations."""

    # Base64 encoding/decoding
    @staticmethod
    def base64_encode(data: str) -> str:
        """Encode string to base64."""
        return base64.b64encode(data.encode()).decode('ascii')

    @staticmethod
    def base64_decode(data: str) -> str:
        """Decode base64 to string."""
        return base64.b64decode(data).decode('utf-8')

    @staticmethod
    def base64_urlsafe_encode(data: str) -> str:
        """Encode string to URL-safe base64."""
        return base64.urlsafe_b64encode(data.encode()).decode('ascii')

    @staticmethod
    def base64_urlsafe_decode(data: str) -> str:
        """Decode URL-safe base64 to string."""
        return base64.urlsafe_b64decode(data).decode('utf-8')

    # Base32 encoding/decoding
    @staticmethod
    def base32_encode(data: str) -> str:
        """Encode string to base32."""
        return base64.b32encode(data.encode()).decode('ascii')

    @staticmethod
    def base32_decode(data: str) -> str:
        """Decode base32 to string."""
        return base64.b32decode(data).decode('utf-8')

    # Base16/Hex encoding/decoding
    @staticmethod
    def hex_encode(data: str) -> str:
        """Encode string to hexadecimal."""
        return data.encode().hex()

    @staticmethod
    def hex_decode(data: str) -> str:
        """Decode hexadecimal to string."""
        return bytes.fromhex(data).decode('utf-8')

    @staticmethod
    def base16_encode(data: str) -> str:
        """Encode string to base16."""
        return base64.b16encode(data.encode()).decode('ascii')

    @staticmethod
    def base16_decode(data: str) -> str:
        """Decode base16 to string."""
        return base64.b16decode(data).decode('utf-8')

    # Base85 encoding/decoding
    @staticmethod
    def base85_encode(data: str) -> str:
        """Encode string to base85."""
        return base64.b85encode(data.encode()).decode('ascii')

    @staticmethod
    def base85_decode(data: str) -> str:
        """Decode base85 to string."""
        return base64.b85decode(data).decode('utf-8')

    # ASCII85 encoding/decoding
    @staticmethod
    def ascii85_encode(data: str) -> str:
        """Encode string to ASCII85."""
        return base64.a85encode(data.encode()).decode('ascii')

    @staticmethod
    def ascii85_decode(data: str) -> str:
        """Decode ASCII85 to string."""
        return base64.a85decode(data).decode('utf-8')

    # URL encoding/decoding
    @staticmethod
    def url_encode(data: str) -> str:
        """URL encode string."""
        import urllib.parse
        return urllib.parse.quote(data)

    @staticmethod
    def url_decode(data: str) -> str:
        """URL decode string."""
        import urllib.parse
        return urllib.parse.unquote(data)

    @staticmethod
    def url_encode_plus(data: str) -> str:
        """URL encode with + for spaces."""
        import urllib.parse
        return urllib.parse.quote_plus(data)

    @staticmethod
    def url_decode_plus(data: str) -> str:
        """URL decode with + for spaces."""
        import urllib.parse
        return urllib.parse.unquote_plus(data)

    # HTML encoding/decoding
    @staticmethod
    def html_encode(data: str) -> str:
        """HTML encode string."""
        import html
        return html.escape(data)

    @staticmethod
    def html_decode(data: str) -> str:
        """HTML decode string."""
        import html
        return html.unescape(data)

    # Unicode operations
    @staticmethod
    def unicode_encode(data: str, encoding: str = 'utf-8') -> str:
        """Encode string to bytes (hex representation)."""
        return data.encode(encoding).hex()

    @staticmethod
    def unicode_decode(data: str, encoding: str = 'utf-8') -> str:
        """Decode bytes (hex representation) to string."""
        return bytes.fromhex(data).decode(encoding)

    @staticmethod
    def unicode_normalize(data: str, form: str = 'NFC') -> str:
        """Normalize Unicode string (NFC, NFD, NFKC, NFKD)."""
        import unicodedata
        return unicodedata.normalize(form, data)

    # Binary/ASCII operations
    @staticmethod
    def to_binary(data: str) -> str:
        """Convert string to binary representation."""
        return ' '.join(format(ord(c), '08b') for c in data)

    @staticmethod
    def from_binary(data: str) -> str:
        """Convert binary representation to string."""
        binary_values = data.split()
        return ''.join(chr(int(b, 2)) for b in binary_values)

    @staticmethod
    def to_ascii_codes(data: str) -> list:
        """Convert string to ASCII codes."""
        return [ord(c) for c in data]

    @staticmethod
    def from_ascii_codes(codes: list) -> str:
        """Convert ASCII codes to string."""
        return ''.join(chr(code) for code in codes)

    # ROT13
    @staticmethod
    def rot13(data: str) -> str:
        """ROT13 encoding/decoding."""
        import codecs
        return codecs.encode(data, 'rot_13')

    # CRC32 checksum
    @staticmethod
    def crc32(data: str) -> int:
        """Calculate CRC32 checksum."""
        return binascii.crc32(data.encode()) & 0xffffffff

    # Adler32 checksum
    @staticmethod
    def adler32(data: str) -> int:
        """Calculate Adler32 checksum."""
        import zlib
        return zlib.adler32(data.encode()) & 0xffffffff

    # JSON encoding/decoding (for convenience)
    @staticmethod
    def json_encode(obj: Any, pretty: bool = False) -> str:
        """Encode object to JSON."""
        if pretty:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def json_decode(data: str) -> Any:
        """Decode JSON to object."""
        return json.loads(data)


# Export functions for easy access
base64_encode = EncodingModule.base64_encode
base64_decode = EncodingModule.base64_decode
base64_urlsafe_encode = EncodingModule.base64_urlsafe_encode
base64_urlsafe_decode = EncodingModule.base64_urlsafe_decode
base32_encode = EncodingModule.base32_encode
base32_decode = EncodingModule.base32_decode
hex_encode = EncodingModule.hex_encode
hex_decode = EncodingModule.hex_decode
base16_encode = EncodingModule.base16_encode
base16_decode = EncodingModule.base16_decode
base85_encode = EncodingModule.base85_encode
base85_decode = EncodingModule.base85_decode
ascii85_encode = EncodingModule.ascii85_encode
ascii85_decode = EncodingModule.ascii85_decode
url_encode = EncodingModule.url_encode
url_decode = EncodingModule.url_decode
url_encode_plus = EncodingModule.url_encode_plus
url_decode_plus = EncodingModule.url_decode_plus
html_encode = EncodingModule.html_encode
html_decode = EncodingModule.html_decode
unicode_encode = EncodingModule.unicode_encode
unicode_decode = EncodingModule.unicode_decode
unicode_normalize = EncodingModule.unicode_normalize
to_binary = EncodingModule.to_binary
from_binary = EncodingModule.from_binary
to_ascii_codes = EncodingModule.to_ascii_codes
from_ascii_codes = EncodingModule.from_ascii_codes
rot13 = EncodingModule.rot13
crc32 = EncodingModule.crc32
adler32 = EncodingModule.adler32
json_encode = EncodingModule.json_encode
json_decode = EncodingModule.json_decode
