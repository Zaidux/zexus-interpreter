"""
Bytecode Caching System for Zexus VM

This module provides a comprehensive bytecode caching system to avoid recompiling
the same code multiple times. Features include:
- LRU (Least Recently Used) eviction policy
- Cache statistics tracking
- AST-based cache keys
- Optional persistent disk cache
- Memory-efficient storage

Part of Phase 4: Bytecode Caching Enhancement
"""

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .bytecode import Bytecode


@dataclass
class CacheStats:
    """Statistics for bytecode cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    total_entries: int = 0
    hit_rate: float = 0.0
    
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
            'hit_rate': round(self.hit_rate, 2)
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
        
        # Statistics
        self.stats = CacheStats()
        
        # Persistent cache
        self.cache_dir = None
        if persistent:
            self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.zexus' / 'cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        self.stats = CacheStats()
        
        if self.debug:
            print("🗑️ Cache: Cleared all entries")
        
        # Clear disk cache
        if self.persistent and self.cache_dir:
            for cache_file in self.cache_dir.glob('*.cache'):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        self.stats.update_hit_rate()
        return self.stats.to_dict()
    
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
                f"memory={self.memory_usage_mb():.2f}MB, "
                f"hit_rate={self.stats.hit_rate:.1f}%)")
