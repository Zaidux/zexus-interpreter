"""Compression module for Zexus standard library."""

import zlib
import gzip
import bz2
import lzma
import base64
from typing import Dict, Any


class CompressionModule:
    """Provides compression/decompression operations."""

    # Zlib compression
    @staticmethod
    def zlib_compress(data: str, level: int = 6) -> str:
        """Compress string using zlib (returns base64)."""
        compressed = zlib.compress(data.encode(), level)
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def zlib_decompress(data: str) -> str:
        """Decompress zlib data (from base64)."""
        compressed = base64.b64decode(data)
        return zlib.decompress(compressed).decode('utf-8')

    # Gzip compression
    @staticmethod
    def gzip_compress(data: str, level: int = 6) -> str:
        """Compress string using gzip (returns base64)."""
        compressed = gzip.compress(data.encode(), compresslevel=level)
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def gzip_decompress(data: str) -> str:
        """Decompress gzip data (from base64)."""
        compressed = base64.b64decode(data)
        return gzip.decompress(compressed).decode('utf-8')

    # Bzip2 compression
    @staticmethod
    def bzip2_compress(data: str, level: int = 9) -> str:
        """Compress string using bzip2 (returns base64)."""
        compressed = bz2.compress(data.encode(), compresslevel=level)
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def bzip2_decompress(data: str) -> str:
        """Decompress bzip2 data (from base64)."""
        compressed = base64.b64decode(data)
        return bz2.decompress(compressed).decode('utf-8')

    # LZMA compression
    @staticmethod
    def lzma_compress(data: str, preset: int = 6) -> str:
        """Compress string using LZMA (returns base64)."""
        compressed = lzma.compress(data.encode(), preset=preset)
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def lzma_decompress(data: str) -> str:
        """Decompress LZMA data (from base64)."""
        compressed = base64.b64decode(data)
        return lzma.decompress(compressed).decode('utf-8')

    # File compression helpers
    @staticmethod
    def compress_file(input_path: str, output_path: str, method: str = 'gzip', level: int = 6) -> Dict[str, Any]:
        """Compress file using specified method."""
        try:
            with open(input_path, 'rb') as f_in:
                data = f_in.read()
            
            if method == 'gzip':
                with gzip.open(output_path, 'wb', compresslevel=level) as f_out:
                    f_out.write(data)
            elif method == 'bzip2':
                with bz2.open(output_path, 'wb', compresslevel=level) as f_out:
                    f_out.write(data)
            elif method == 'lzma':
                with lzma.open(output_path, 'wb', preset=level) as f_out:
                    f_out.write(data)
            elif method == 'zlib':
                compressed = zlib.compress(data, level)
                with open(output_path, 'wb') as f_out:
                    f_out.write(compressed)
            else:
                return {'success': False, 'error': f'Unknown compression method: {method}'}
            
            import os
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            return {
                'success': True,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': round(ratio, 2)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def decompress_file(input_path: str, output_path: str, method: str = 'gzip') -> Dict[str, Any]:
        """Decompress file using specified method."""
        try:
            if method == 'gzip':
                with gzip.open(input_path, 'rb') as f_in:
                    data = f_in.read()
            elif method == 'bzip2':
                with bz2.open(input_path, 'rb') as f_in:
                    data = f_in.read()
            elif method == 'lzma':
                with lzma.open(input_path, 'rb') as f_in:
                    data = f_in.read()
            elif method == 'zlib':
                with open(input_path, 'rb') as f_in:
                    data = zlib.decompress(f_in.read())
            else:
                return {'success': False, 'error': f'Unknown decompression method: {method}'}
            
            with open(output_path, 'wb') as f_out:
                f_out.write(data)
            
            import os
            return {
                'success': True,
                'decompressed_size': os.path.getsize(output_path)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # Compression ratio calculation
    @staticmethod
    def calculate_ratio(original: str, compressed: str) -> float:
        """Calculate compression ratio percentage."""
        if len(original) == 0:
            return 0.0
        return (1 - len(compressed) / len(original)) * 100

    # CRC32 checksum for integrity
    @staticmethod
    def crc32(data: str) -> int:
        """Calculate CRC32 checksum."""
        return zlib.crc32(data.encode()) & 0xffffffff

    @staticmethod
    def adler32(data: str) -> int:
        """Calculate Adler32 checksum."""
        return zlib.adler32(data.encode()) & 0xffffffff


# Export functions for easy access
zlib_compress = CompressionModule.zlib_compress
zlib_decompress = CompressionModule.zlib_decompress
gzip_compress = CompressionModule.gzip_compress
gzip_decompress = CompressionModule.gzip_decompress
bzip2_compress = CompressionModule.bzip2_compress
bzip2_decompress = CompressionModule.bzip2_decompress
lzma_compress = CompressionModule.lzma_compress
lzma_decompress = CompressionModule.lzma_decompress
compress_file = CompressionModule.compress_file
decompress_file = CompressionModule.decompress_file
calculate_ratio = CompressionModule.calculate_ratio
crc32 = CompressionModule.crc32
adler32 = CompressionModule.adler32
