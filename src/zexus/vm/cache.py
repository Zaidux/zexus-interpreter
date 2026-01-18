"""
Bytecode Caching System for Zexus VM

This module provides a comprehensive bytecode caching system to avoid recompiling
the same code multiple times. Features include:
- LRU (Least Recently Used) eviction policy
- Cache statistics tracking
- AST-based cache keys
- File-based cache keys (path + mtime for cross-run persistence)
- Pattern recognition cache (reuse bytecode for similar AST shapes)
- Optional persistent disk cache
- Memory-efficient storage

Part of Phase 4: Bytecode Caching Enhancement
Enhanced: File-based persistent caching for faster repeat runs
"""

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bytecode import Bytecode


@dataclass
class FileMetadata:
    """Metadata for file-based cache entries"""
    file_path: str
    mtime: float
    size: int
    content_hash: str  # Hash of file content for extra validation


@dataclass
class CacheStats:
    """Statistics for bytecode cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    total_entries: int = 0
    hit_rate: float = 0.0
    file_hits: int = 0      # File-based cache hits
    file_misses: int = 0    # File-based cache misses
    pattern_hits: int = 0   # Pattern cache hits
    
    def update_hit_rate(self):
        """Update hit rate percentage"""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'memory_bytes': self.memory_bytes,
            'total_entries': self.total_entries,
            'hit_rate': round(self.hit_rate, 2),
            'file_hits': self.file_hits,
            'file_misses': self.file_misses,
            'pattern_hits': self.pattern_hits
        }


@dataclass
class CacheEntry:
    """Entry in the bytecode cache"""
    bytecode: Bytecode
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self):
        """Update access timestamp and count"""
        self.timestamp = time.time()
        self.access_count += 1


class BytecodeCache:
    """
    LRU cache for compiled bytecode
    
    Features:
    - AST-based cache keys (hash of AST structure)
    - LRU eviction when cache size limit is reached
    - Statistics tracking (hits, misses, evictions)
    - Optional persistent disk cache
    - Memory-efficient storage
    
    Usage:
        cache = BytecodeCache(max_size=1000, persistent=False)
        
        # Check cache
        bytecode = cache.get(ast_node)
        if bytecode is None:
            # Compile and store
            bytecode = compiler.compile(ast_node)
            cache.put(ast_node, bytecode)
        
        # Get statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats.hit_rate}%")
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        persistent: bool = False,
        cache_dir: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize bytecode cache
        
        Args:
            max_size: Maximum number of entries (default 1000)
            max_memory_mb: Maximum memory usage in MB (default 100)
            persistent: Enable disk-based persistent cache
            cache_dir: Directory for persistent cache
            debug: Enable debug output
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.persistent = persistent
        self.debug = debug
        
        # LRU cache using OrderedDict (insertion order preserved)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # File-based cache: maps (file_path, mtime) -> list of bytecode entries
        # This persists across interpreter runs when persistent=True
        self._file_cache: Dict[str, Dict] = {}  # file_path -> {mtime, content_hash, bytecodes: []}
        
        # Pattern cache: maps AST structure hash -> bytecode
        # Allows reusing bytecode for similar code patterns across files
        self._pattern_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_patterns = 500  # Max pattern cache entries
        self._pattern_memory_bytes = 0
        self._max_pattern_memory_bytes = max(1, self.max_memory_bytes // 4)
        
        # Statistics
        self.stats = CacheStats()
        
        # Persistent cache
        self.cache_dir = None
        if persistent:
            self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.zexus' / 'cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Load file cache index from disk
            self._load_file_cache_index()
            if self.debug:
                print(f"📦 Cache: Persistent cache enabled at {self.cache_dir}")
    
    def _hash_ast(self, ast_node: Any) -> str:
        """
        Generate unique hash for AST node
        
        Uses JSON serialization of AST structure to create deterministic hash.
        Handles circular references and complex nested structures.
        
        Args:
            ast_node: AST node to hash
            
        Returns:
            MD5 hash string (32 characters)
        """
        try:
            # Convert AST to hashable representation
            ast_repr = self._ast_to_dict(ast_node)
            ast_json = json.dumps(ast_repr, sort_keys=True)
            return hashlib.md5(ast_json.encode()).hexdigest()
        except Exception as e:
            # Fallback to string representation
            if self.debug:
                print(f"⚠️ Cache: AST hashing fallback ({e})")
            return hashlib.md5(str(ast_node).encode()).hexdigest()
    
    def _ast_to_dict(self, node: Any, depth: int = 0, max_depth: int = 50) -> Any:
        """
        Convert AST node to dictionary for hashing
        
        Recursively converts AST nodes to dictionaries, handling:
        - Node types and attributes
        - Lists and tuples
        - Nested nodes
        - Circular references (via depth limit)
        
        Args:
            node: AST node or value
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Hashable representation (dict, list, or primitive)
        """
        if depth > max_depth:
            return f"<max_depth_{type(node).__name__}>"
        
        # Handle None
        if node is None:
            return None
        
        # Handle primitives
        if isinstance(node, (int, float, str, bool)):
            return node
        
        # Handle lists/tuples
        if isinstance(node, (list, tuple)):
            return [self._ast_to_dict(item, depth + 1, max_depth) for item in node]
        
        # Handle dictionaries
        if isinstance(node, dict):
            return {k: self._ast_to_dict(v, depth + 1, max_depth) for k, v in node.items()}
        
        # Handle AST nodes (objects with __dict__)
        if hasattr(node, '__dict__'):
            result = {'__type__': type(node).__name__}
            for key, value in node.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._ast_to_dict(value, depth + 1, max_depth)
            return result
        
        # Fallback to string representation
        return f"<{type(node).__name__}>"
    
    def _estimate_size(self, bytecode: Bytecode) -> int:
        """
        Estimate bytecode size in bytes
        
        Approximates memory usage by counting:
        - Instructions (each ~100 bytes)
        - Constants (pickle size)
        - Metadata (small overhead)
        
        Args:
            bytecode: Bytecode object
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Count instructions
            instruction_size = len(bytecode.instructions) * 100  # ~100 bytes per instruction
            
            # Estimate constants size
            constants_size = 0
            for const in bytecode.constants:
                try:
                    constants_size += len(pickle.dumps(const))
                except (TypeError, pickle.PicklingError):
                    constants_size += 100  # Fallback estimate
            
            # Add metadata overhead
            metadata_size = 200  # Small overhead for name, line_map, etc.
            
            return instruction_size + constants_size + metadata_size
        except (AttributeError, TypeError):
            # Fallback to conservative estimate
            return len(bytecode.instructions) * 150
    
    def _evict_lru(self):
        """
        Evict least recently used entry
        
        Removes the oldest entry (first in OrderedDict) and updates statistics.
        """
        if not self._cache:
            return
        
        # Remove oldest entry (LRU)
        key, entry = self._cache.popitem(last=False)
        
        # Update statistics
        self.stats.evictions += 1
        self.stats.memory_bytes -= entry.size_bytes
        self.stats.total_entries -= 1
        
        if self.debug:
            print(f"🗑️ Cache: Evicted LRU entry {key[:8]}... (freed {entry.size_bytes} bytes)")
    
    def _evict_to_fit(self, new_size: int):
        """
        Evict entries until new entry fits
        
        Keeps evicting LRU entries until:
        1. Cache size < max_size
        2. Memory usage + new_size < max_memory_bytes
        
        Args:
            new_size: Size of new entry in bytes
        """
        # Evict by count
        while len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # Evict by memory
        while self._cache and (self.stats.memory_bytes + new_size) > self.max_memory_bytes:
            self._evict_lru()
    
    def get(self, ast_node: Any) -> Optional[Bytecode]:
        """
        Get bytecode from cache
        
        If found:
        - Returns cached bytecode
        - Updates access timestamp and count
        - Moves entry to end (most recent in LRU)
        - Increments hit counter
        
        If not found:
        - Increments miss counter
        - Returns None
        
        Args:
            ast_node: AST node to look up
            
        Returns:
            Cached bytecode or None
        """
        key = self._hash_ast(ast_node)
        
        if key in self._cache:
            # Cache hit
            entry = self._cache[key]
            entry.update_access()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Update statistics
            self.stats.hits += 1
            self.stats.update_hit_rate()
            
            if self.debug:
                print(f"✅ Cache: HIT {key[:8]}... (access #{entry.access_count})")
            
            return entry.bytecode
        else:
            # Cache miss
            self.stats.misses += 1
            self.stats.update_hit_rate()
            
            if self.debug:
                print(f"❌ Cache: MISS {key[:8]}...")
            
            # Try persistent cache if enabled
            if self.persistent:
                bytecode = self._load_from_disk(key)
                if bytecode:
                    # Found in disk cache, add to memory cache
                    self.put(ast_node, bytecode, skip_disk=True)
                    return bytecode
            
            return None
    
    def put(self, ast_node: Any, bytecode: Bytecode, skip_disk: bool = False):
        """
        Store bytecode in cache
        
        Process:
        1. Hash AST node to create cache key
        2. Estimate bytecode size
        3. Evict entries if needed to fit new entry
        4. Store in memory cache
        5. Optionally save to disk cache
        
        Args:
            ast_node: AST node (cache key)
            bytecode: Compiled bytecode
            skip_disk: Skip disk cache (used when loading from disk)
        """
        key = self._hash_ast(ast_node)
        size = self._estimate_size(bytecode)
        
        # Evict if needed
        self._evict_to_fit(size)
        
        # Create entry
        entry = CacheEntry(
            bytecode=bytecode,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size
        )
        
        # Store in cache
        self._cache[key] = entry
        
        # Update statistics
        self.stats.memory_bytes += size
        self.stats.total_entries += 1
        
        if self.debug:
            print(f"💾 Cache: PUT {key[:8]}... ({size} bytes, {len(self._cache)} entries)")
        
        # Save to disk if persistent
        if self.persistent and not skip_disk:
            self._save_to_disk(key, bytecode)
    
    def invalidate(self, ast_node: Any):
        """
        Remove entry from cache
        
        Args:
            ast_node: AST node to invalidate
        """
        key = self._hash_ast(ast_node)
        
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats.memory_bytes -= entry.size_bytes
            self.stats.total_entries -= 1
            
            if self.debug:
                print(f"🗑️ Cache: Invalidated {key[:8]}...")
            
            # Remove from disk cache
            if self.persistent:
                self._delete_from_disk(key)
    
    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        self._file_cache.clear()
        self._pattern_cache.clear()
        self._pattern_memory_bytes = 0
        self.stats = CacheStats()
        
        if self.debug:
            print("🗑️ Cache: Cleared all entries")
        
        # Clear disk cache
        if self.persistent and self.cache_dir:
            for cache_file in self.cache_dir.glob('*.cache'):
                cache_file.unlink()
            # Clear file cache index
            index_file = self.cache_dir / 'file_index.cache'
            if index_file.exists():
                index_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        self.stats.update_hit_rate()
        base = self.stats.to_dict()
        base['pattern_memory_bytes'] = self._pattern_memory_bytes
        return base

    def _evict_pattern_lru(self) -> None:
        if not self._pattern_cache:
            return
        _, entry = self._pattern_cache.popitem(last=False)
        self._pattern_memory_bytes = max(0, self._pattern_memory_bytes - entry.size_bytes)
    
    def reset_stats(self):
        """Reset statistics (keeps cache entries)"""
        self.stats = CacheStats(
            total_entries=len(self._cache),
            memory_bytes=self.stats.memory_bytes
        )
    
    # ==================== Persistent Cache Methods ====================
    
    def _save_to_disk(self, key: str, bytecode: Bytecode):
        """Save bytecode to disk cache"""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(bytecode, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.debug:
                print(f"💾 Cache: Saved to disk {key[:8]}...")
        except Exception as e:
            if self.debug:
                print(f"⚠️ Cache: Failed to save to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Bytecode]:
        """Load bytecode from disk cache"""
        if not self.cache_dir:
            return None
        
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    bytecode = pickle.load(f)
                
                if self.debug:
                    print(f"💾 Cache: Loaded from disk {key[:8]}...")
                
                return bytecode
        except Exception as e:
            if self.debug:
                print(f"⚠️ Cache: Failed to load from disk: {e}")
        
        return None
    
    def _delete_from_disk(self, key: str):
        """Delete cache entry from disk"""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            if self.debug:
                print(f"⚠️ Cache: Failed to delete from disk: {e}")
    
    # ==================== File-Based Cache Methods ====================
    # These methods enable faster repeat runs by caching based on file path + mtime
    
    def _get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for a source file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            # Read file content for hash
            content = path.read_text(encoding='utf-8')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            return FileMetadata(
                file_path=str(path.resolve()),
                mtime=stat.st_mtime,
                size=stat.st_size,
                content_hash=content_hash
            )
        except Exception as e:
            if self.debug:
                print(f"⚠️ Cache: Failed to get file metadata: {e}")
            return None
    
    def _file_cache_key(self, file_path: str) -> str:
        """Generate cache key from file path"""
        # Normalize path and create stable key
        normalized = str(Path(file_path).resolve())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_by_file(self, file_path: str) -> Optional[List[Bytecode]]:
        """
        Get all cached bytecode for a source file
        
        This enables faster repeat runs - if the file hasn't changed,
        we can reuse all previously compiled bytecode.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            List of cached bytecode objects, or None if cache invalid/missing
        """
        metadata = self._get_file_metadata(file_path)
        if not metadata:
            self.stats.file_misses += 1
            return None
        
        file_key = self._file_cache_key(file_path)
        
        # Check memory cache first
        if file_key in self._file_cache:
            cached = self._file_cache[file_key]
            # Validate: file hasn't been modified
            if (cached.get('mtime') == metadata.mtime and 
                cached.get('content_hash') == metadata.content_hash):
                self.stats.file_hits += 1
                if self.debug:
                    print(f"✅ FileCache: HIT {file_path} ({len(cached.get('bytecodes', []))} entries)")
                return cached.get('bytecodes', [])
            else:
                # File changed, invalidate
                if self.debug:
                    print(f"🔄 FileCache: STALE {file_path} (file modified)")
                del self._file_cache[file_key]
        
        # Check disk cache if persistent
        if self.persistent:
            loaded = self._load_file_bytecode(file_key, metadata)
            if loaded:
                self.stats.file_hits += 1
                return loaded
        
        self.stats.file_misses += 1
        if self.debug:
            print(f"❌ FileCache: MISS {file_path}")
        return None
    
    def put_by_file(self, file_path: str, bytecodes: List[Bytecode]):
        """
        Store all bytecode for a source file
        
        Args:
            file_path: Path to the source file
            bytecodes: List of compiled bytecode objects
        """
        metadata = self._get_file_metadata(file_path)
        if not metadata:
            return
        
        file_key = self._file_cache_key(file_path)
        
        # Store in memory cache
        self._file_cache[file_key] = {
            'file_path': metadata.file_path,
            'mtime': metadata.mtime,
            'size': metadata.size,
            'content_hash': metadata.content_hash,
            'bytecodes': bytecodes,
            'cached_at': time.time()
        }
        
        if self.debug:
            print(f"💾 FileCache: PUT {file_path} ({len(bytecodes)} entries)")
        
        # Save to disk if persistent
        if self.persistent:
            self._save_file_bytecode(file_key, metadata, bytecodes)
            self._save_file_cache_index()
    
    def invalidate_file(self, file_path: str):
        """Invalidate cache for a specific file"""
        file_key = self._file_cache_key(file_path)
        
        if file_key in self._file_cache:
            del self._file_cache[file_key]
            if self.debug:
                print(f"🗑️ FileCache: Invalidated {file_path}")
        
        # Remove from disk
        if self.persistent:
            self._delete_file_bytecode(file_key)
            self._save_file_cache_index()
    
    def _save_file_bytecode(self, file_key: str, metadata: FileMetadata, bytecodes: List[Bytecode]):
        """Save file bytecode to disk"""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"file_{file_key}.cache"
            data = {
                'metadata': {
                    'file_path': metadata.file_path,
                    'mtime': metadata.mtime,
                    'size': metadata.size,
                    'content_hash': metadata.content_hash
                },
                'bytecodes': bytecodes,
                'cached_at': time.time()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.debug:
                print(f"💾 FileCache: Saved to disk {file_key[:8]}...")
        except Exception as e:
            if self.debug:
                print(f"⚠️ FileCache: Failed to save: {e}")
    
    def _load_file_bytecode(self, file_key: str, current_metadata: FileMetadata) -> Optional[List[Bytecode]]:
        """Load file bytecode from disk and validate"""
        if not self.cache_dir:
            return None
        
        try:
            cache_file = self.cache_dir / f"file_{file_key}.cache"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            cached_meta = data.get('metadata', {})
            # Validate metadata matches
            if (cached_meta.get('mtime') == current_metadata.mtime and
                cached_meta.get('content_hash') == current_metadata.content_hash):
                
                bytecodes = data.get('bytecodes', [])
                # Store in memory cache too
                self._file_cache[file_key] = {
                    'file_path': current_metadata.file_path,
                    'mtime': current_metadata.mtime,
                    'size': current_metadata.size,
                    'content_hash': current_metadata.content_hash,
                    'bytecodes': bytecodes,
                    'cached_at': data.get('cached_at', time.time())
                }
                
                if self.debug:
                    print(f"💾 FileCache: Loaded from disk {file_key[:8]}... ({len(bytecodes)} entries)")
                return bytecodes
            else:
                # Stale cache, remove it
                cache_file.unlink()
                if self.debug:
                    print(f"🔄 FileCache: Removed stale disk cache {file_key[:8]}...")
        except Exception as e:
            if self.debug:
                print(f"⚠️ FileCache: Failed to load: {e}")
        
        return None
    
    def _delete_file_bytecode(self, file_key: str):
        """Delete file bytecode from disk"""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"file_{file_key}.cache"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            if self.debug:
                print(f"⚠️ FileCache: Failed to delete: {e}")
    
    def _load_file_cache_index(self):
        """Load file cache index from disk on startup"""
        if not self.cache_dir:
            return
        
        try:
            index_file = self.cache_dir / 'file_index.cache'
            if index_file.exists():
                with open(index_file, 'rb') as f:
                    index = pickle.load(f)
                if self.debug:
                    print(f"📂 FileCache: Loaded index with {len(index)} files")
        except Exception as e:
            if self.debug:
                print(f"⚠️ FileCache: Failed to load index: {e}")
    
    def _save_file_cache_index(self):
        """Save file cache index to disk"""
        if not self.cache_dir:
            return
        
        try:
            index_file = self.cache_dir / 'file_index.cache'
            # Just save file keys and metadata (not bytecode)
            index = {
                key: {
                    'file_path': data.get('file_path'),
                    'mtime': data.get('mtime'),
                    'content_hash': data.get('content_hash'),
                    'cached_at': data.get('cached_at')
                }
                for key, data in self._file_cache.items()
            }
            with open(index_file, 'wb') as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            if self.debug:
                print(f"⚠️ FileCache: Failed to save index: {e}")
    
    # ==================== Pattern Cache Methods ====================
    # These methods enable reusing bytecode for similar code patterns
    
    def _get_pattern_hash(self, ast_node: Any) -> str:
        """
        Generate a structural hash for AST pattern matching
        
        Unlike _hash_ast which includes literal values, this creates
        a hash of just the AST structure (node types, not values).
        This allows matching similar patterns like:
            for i in range(100)  <->  for j in range(500)
        """
        try:
            pattern = self._ast_to_pattern(ast_node)
            return hashlib.md5(json.dumps(pattern, sort_keys=True).encode()).hexdigest()
        except Exception:
            return ""
    
    def _ast_to_pattern(self, node: Any, depth: int = 0, max_depth: int = 30) -> Any:
        """
        Convert AST to structural pattern (ignoring literal values)
        
        This extracts just the shape of the code, not specific values.
        """
        if depth > max_depth:
            return {'_type': 'MAX_DEPTH'}
        
        if node is None:
            return None
        
        # For primitives, just record the type not the value
        if isinstance(node, (int, float)):
            return {'_type': 'number'}
        if isinstance(node, str):
            return {'_type': 'string'}
        if isinstance(node, bool):
            return {'_type': 'bool'}
        
        if isinstance(node, (list, tuple)):
            return [self._ast_to_pattern(item, depth + 1, max_depth) for item in node]
        
        if isinstance(node, dict):
            return {k: self._ast_to_pattern(v, depth + 1, max_depth) for k, v in node.items()}
        
        if hasattr(node, '__dict__'):
            result = {'_type': type(node).__name__}
            for key, value in node.__dict__.items():
                if not key.startswith('_'):
                    # Skip 'name' and 'value' fields as they vary
                    if key in ('name', 'value', 'literal', 'identifier'):
                        result[key] = {'_type': type(value).__name__ if value else 'none'}
                    else:
                        result[key] = self._ast_to_pattern(value, depth + 1, max_depth)
            return result
        
        return {'_type': type(node).__name__}
    
    def get_by_pattern(self, ast_node: Any) -> Optional[Bytecode]:
        """
        Get cached bytecode matching AST pattern
        
        This enables reusing compiled bytecode for similar code shapes.
        E.g., two for-loops with different bounds can share bytecode.
        
        Args:
            ast_node: AST node to match
            
        Returns:
            Matching bytecode or None
        """
        pattern_hash = self._get_pattern_hash(ast_node)
        if not pattern_hash:
            return None
        
        if pattern_hash in self._pattern_cache:
            entry = self._pattern_cache[pattern_hash]
            entry.update_access()
            self._pattern_cache.move_to_end(pattern_hash)
            self.stats.pattern_hits += 1
            
            if self.debug:
                print(f"✅ PatternCache: HIT {pattern_hash[:8]}...")
            
            return entry.bytecode
        
        return None
    
    def put_by_pattern(self, ast_node: Any, bytecode: Bytecode):
        """
        Store bytecode by pattern for future reuse
        
        Args:
            ast_node: AST node (pattern source)
            bytecode: Compiled bytecode
        """
        pattern_hash = self._get_pattern_hash(ast_node)
        if not pattern_hash:
            return
        
        # Evict if at capacity or over memory budget
        while len(self._pattern_cache) >= self._max_patterns:
            self._evict_pattern_lru()
        
        size = self._estimate_size(bytecode)
        while self._pattern_cache and (self._pattern_memory_bytes + size) > self._max_pattern_memory_bytes:
            self._evict_pattern_lru()
        entry = CacheEntry(
            bytecode=bytecode,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size
        )
        
        self._pattern_cache[pattern_hash] = entry
        self._pattern_memory_bytes += size
        
        if self.debug:
            print(f"💾 PatternCache: PUT {pattern_hash[:8]}...")
    
    def is_file_cached(self, file_path: str) -> bool:
        """
        Check if a file has valid cached bytecode
        
        Args:
            file_path: Path to source file
            
        Returns:
            True if valid cache exists
        """
        metadata = self._get_file_metadata(file_path)
        if not metadata:
            return False
        
        file_key = self._file_cache_key(file_path)
        
        # Check memory cache
        if file_key in self._file_cache:
            cached = self._file_cache[file_key]
            if (cached.get('mtime') == metadata.mtime and
                cached.get('content_hash') == metadata.content_hash):
                return True
        
        # Check disk cache
        if self.persistent and self.cache_dir:
            cache_file = self.cache_dir / f"file_{file_key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    cached_meta = data.get('metadata', {})
                    return (cached_meta.get('mtime') == metadata.mtime and
                            cached_meta.get('content_hash') == metadata.content_hash)
                except Exception:
                    pass
        
        return False
    
    def get_file_cache_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about cached file"""
        file_key = self._file_cache_key(file_path)
        
        if file_key in self._file_cache:
            cached = self._file_cache[file_key]
            return {
                'file_path': cached.get('file_path'),
                'mtime': cached.get('mtime'),
                'content_hash': cached.get('content_hash'),
                'bytecode_count': len(cached.get('bytecodes', [])),
                'cached_at': cached.get('cached_at')
            }
        
        return None
    
    # ==================== Utility Methods ====================
    
    def size(self) -> int:
        """Get current cache size (number of entries)"""
        return len(self._cache)
    
    def memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.stats.memory_bytes
    
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.stats.memory_bytes / (1024 * 1024)
    
    def contains(self, ast_node: Any) -> bool:
        """Check if AST node is in cache"""
        key = self._hash_ast(ast_node)
        return key in self._cache
    
    def get_entry_info(self, ast_node: Any) -> Optional[Dict[str, Any]]:
        """Get information about cached entry"""
        key = self._hash_ast(ast_node)
        
        if key in self._cache:
            entry = self._cache[key]
            return {
                'key': key,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'instruction_count': len(entry.bytecode.instructions),
                'constant_count': len(entry.bytecode.constants)
            }
        
        return None
    
    def get_all_keys(self) -> list:
        """Get all cache keys"""
        return list(self._cache.keys())
    
    def __len__(self) -> int:
        """Get cache size"""
        return len(self._cache)
    
    def __contains__(self, ast_node: Any) -> bool:
        """Check if AST node is cached"""
        return self.contains(ast_node)
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"BytecodeCache(size={len(self._cache)}/{self.max_size}, "
                f"files={len(self._file_cache)}, patterns={len(self._pattern_cache)}, "
                f"memory={self.memory_usage_mb():.2f}MB, "
                f"hit_rate={self.stats.hit_rate:.1f}%)")

    def __bool__(self) -> bool:
        """Ensure the cache instance is truthy even when empty."""
        return True
