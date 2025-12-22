"""
Memory Pool Optimization for Zexus VM (Phase 8.2)

Provides object pooling to reduce allocation overhead and GC pressure:
- Integer pooling for common values (-128 to 256)
- String pooling for small strings (<64 chars)
- List pooling for small lists (<16 elements)
- Automatic pool statistics and monitoring
"""

import time
from typing import Any, Optional, List, Dict, Set
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PoolStats:
    """Statistics for an object pool"""
    allocations: int = 0      # Total allocation requests
    hits: int = 0             # Requests served from pool
    misses: int = 0           # Requests that needed new objects
    returns: int = 0          # Objects returned to pool
    reuses: int = 0           # Objects reused from pool
    current_size: int = 0     # Current pool size
    max_size: int = 0         # Maximum pool size
    total_created: int = 0    # Total objects created
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def reuse_rate(self) -> float:
        """Calculate object reuse rate"""
        return (self.reuses / self.total_created * 100) if self.total_created > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'allocations': self.allocations,
            'hits': self.hits,
            'misses': self.misses,
            'returns': self.returns,
            'reuses': self.reuses,
            'current_size': self.current_size,
            'max_size': self.max_size,
            'total_created': self.total_created,
            'hit_rate': self.hit_rate,
            'reuse_rate': self.reuse_rate
        }


class ObjectPool:
    """
    Generic object pool with configurable size and factory
    """
    
    def __init__(self, max_size: int = 1000, factory=None):
        """
        Initialize object pool
        
        Args:
            max_size: Maximum number of objects to pool
            factory: Optional factory function to create new objects
        """
        self.max_size = max_size
        self.factory = factory
        self.pool: deque = deque(maxlen=max_size)
        self.stats = PoolStats(max_size=max_size)
        self.in_use: Set[int] = set()  # Track objects in use
    
    def acquire(self, default_value: Any = None) -> Any:
        """
        Acquire an object from pool or create new one
        
        Args:
            default_value: Default value if pool is empty and no factory
            
        Returns:
            Object from pool or newly created
        """
        self.stats.allocations += 1
        
        if self.pool:
            obj = self.pool.popleft()
            self.stats.hits += 1
            self.stats.reuses += 1
            self.stats.current_size = len(self.pool)
            obj_id = id(obj)
            self.in_use.add(obj_id)
            return obj
        else:
            self.stats.misses += 1
            if self.factory:
                obj = self.factory()
            else:
                obj = default_value
            self.stats.total_created += 1
            obj_id = id(obj)
            self.in_use.add(obj_id)
            return obj
    
    def release(self, obj: Any) -> bool:
        """
        Return an object to the pool
        
        Args:
            obj: Object to return
            
        Returns:
            True if object was returned to pool, False if pool is full
        """
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
        
        if len(self.pool) < self.max_size:
            self.pool.append(obj)
            self.stats.returns += 1
            self.stats.current_size = len(self.pool)
            return True
        return False
    
    def clear(self):
        """Clear the pool"""
        self.pool.clear()
        self.in_use.clear()
        self.stats.current_size = 0
    
    def size(self) -> int:
        """Get current pool size"""
        return len(self.pool)


class IntegerPool:
    """
    Specialized pool for integer values
    
    Uses a fixed pool for common integers (-128 to 256) and a dynamic
    pool for other values.
    """
    
    # Pre-allocated integers for common range
    SMALL_INT_MIN = -128
    SMALL_INT_MAX = 256
    
    def __init__(self, max_size: int = 1000):
        """Initialize integer pool"""
        self.max_size = max_size
        
        # Pre-allocated small integers (like Python's integer cache)
        self.small_ints: Dict[int, int] = {
            i: i for i in range(self.SMALL_INT_MIN, self.SMALL_INT_MAX + 1)
        }
        
        # Dynamic pool for other integers
        self.pool: Dict[int, int] = {}
        self.stats = PoolStats(max_size=max_size)
        self.access_count: Dict[int, int] = {}  # LRU tracking
    
    def get(self, value: int) -> int:
        """
        Get integer from pool
        
        Args:
            value: Integer value
            
        Returns:
            Pooled integer instance
        """
        self.stats.allocations += 1
        
        # Check small int cache first
        if self.SMALL_INT_MIN <= value <= self.SMALL_INT_MAX:
            self.stats.hits += 1
            return self.small_ints[value]
        
        # Check dynamic pool
        if value in self.pool:
            self.stats.hits += 1
            self.stats.reuses += 1
            self.access_count[value] = self.access_count.get(value, 0) + 1
            return self.pool[value]
        
        # Create new integer
        self.stats.misses += 1
        self.stats.total_created += 1
        
        # Add to pool if not full
        if len(self.pool) < self.max_size:
            self.pool[value] = value
            self.access_count[value] = 1
            self.stats.current_size = len(self.pool)
        elif self.access_count:
            # Evict least recently used
            lru_value = min(self.access_count, key=self.access_count.get)
            del self.pool[lru_value]
            del self.access_count[lru_value]
            self.pool[value] = value
            self.access_count[value] = 1
        
        return value
    
    def clear(self):
        """Clear dynamic pool (keep small ints)"""
        self.pool.clear()
        self.access_count.clear()
        self.stats.current_size = 0


class StringPool:
    """
    String interning pool for small strings
    
    Pools strings up to 64 characters to reduce memory overhead
    and improve comparison performance.
    """
    
    MAX_STRING_LENGTH = 64
    
    def __init__(self, max_size: int = 10000):
        """Initialize string pool"""
        self.max_size = max_size
        self.pool: Dict[str, str] = {}
        self.stats = PoolStats(max_size=max_size)
        self.access_count: Dict[str, int] = {}
    
    def get(self, value: str) -> str:
        """
        Get string from pool (intern)
        
        Args:
            value: String value
            
        Returns:
            Pooled string instance
        """
        self.stats.allocations += 1
        
        # Don't pool long strings
        if len(value) > self.MAX_STRING_LENGTH:
            self.stats.misses += 1
            return value
        
        # Check if already pooled
        if value in self.pool:
            self.stats.hits += 1
            self.stats.reuses += 1
            self.access_count[value] = self.access_count.get(value, 0) + 1
            return self.pool[value]
        
        # Add to pool
        self.stats.misses += 1
        self.stats.total_created += 1
        
        if len(self.pool) < self.max_size:
            self.pool[value] = value
            self.access_count[value] = 1
            self.stats.current_size = len(self.pool)
        elif self.access_count:
            # Evict LRU
            lru_str = min(self.access_count, key=self.access_count.get)
            del self.pool[lru_str]
            del self.access_count[lru_str]
            self.pool[value] = value
            self.access_count[value] = 1
        
        return value
    
    def clear(self):
        """Clear the pool"""
        self.pool.clear()
        self.access_count.clear()
        self.stats.current_size = 0


class ListPool:
    """
    Pool for small list objects
    
    Pools empty lists and small lists (<16 elements) for reuse.
    """
    
    MAX_LIST_SIZE = 16
    
    def __init__(self, max_pool_size: int = 500):
        """Initialize list pool"""
        self.max_pool_size = max_pool_size
        self.MAX_LIST_SIZE = 16
        # Separate pools by size
        self.pools: Dict[int, ObjectPool] = {
            i: ObjectPool(max_size=max_pool_size // 16, factory=list)
            for i in range(self.MAX_LIST_SIZE + 1)
        }
        self.stats = PoolStats(max_size=max_pool_size)
    
    def acquire(self, size: int = 0) -> List:
        """
        Acquire a list from pool
        
        Args:
            size: Desired list size
            
        Returns:
            List from pool or newly created
        """
        self.stats.allocations += 1
        
        if size > self.MAX_LIST_SIZE:
            self.stats.misses += 1
            self.stats.total_created += 1
            return [None] * size
        
        pool = self.pools[size]
        lst = pool.acquire([None] * size)
        
        # Update aggregate stats
        if pool.stats.hits > 0:
            self.stats.hits += 1
            self.stats.reuses += 1
        else:
            self.stats.misses += 1
            self.stats.total_created += 1
        
        # Ensure list is the right size and cleared
        if len(lst) != size:
            lst.clear()
            lst.extend([None] * size)
        
        return lst
    
    def release(self, lst: List) -> bool:
        """
        Return a list to the pool
        
        Args:
            lst: List to return
            
        Returns:
            True if returned to pool, False otherwise
        """
        size = len(lst)
        if size > self.MAX_LIST_SIZE:
            return False
        
        # Clear list before returning to pool
        lst.clear()
        
        pool = self.pools[size]
        success = pool.release(lst)
        
        if success:
            self.stats.returns += 1
        
        return success
    
    def clear(self):
        """Clear all pools"""
        for pool in self.pools.values():
            pool.clear()
        self.stats.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            'aggregate': self.stats.to_dict(),
            'by_size': {
                size: pool.stats.to_dict()
                for size, pool in self.pools.items()
                if pool.stats.allocations > 0
            }
        }


class MemoryPoolManager:
    """
    Central manager for all memory pools
    
    Coordinates integer, string, and list pools and provides
    unified statistics and control.
    """
    
    def __init__(
        self,
        enable_int_pool: bool = True,
        enable_str_pool: bool = True,
        enable_list_pool: bool = True,
        int_pool_size: int = 1000,
        str_pool_size: int = 10000,
        list_pool_size: int = 500
    ):
        """
        Initialize memory pool manager
        
        Args:
            enable_int_pool: Enable integer pooling
            enable_str_pool: Enable string pooling
            enable_list_pool: Enable list pooling
            int_pool_size: Max integer pool size
            str_pool_size: Max string pool size
            list_pool_size: Max list pool size
        """
        self.enable_int_pool = enable_int_pool
        self.enable_str_pool = enable_str_pool
        self.enable_list_pool = enable_list_pool
        
        self.int_pool = IntegerPool(max_size=int_pool_size) if enable_int_pool else None
        self.str_pool = StringPool(max_size=str_pool_size) if enable_str_pool else None
        self.list_pool = ListPool(max_pool_size=list_pool_size) if enable_list_pool else None
        
        self.start_time = time.time()
    
    def get_int(self, value: int) -> int:
        """Get pooled integer"""
        if self.int_pool:
            return self.int_pool.get(value)
        return value
    
    def get_str(self, value: str) -> str:
        """Get pooled string"""
        if self.str_pool:
            return self.str_pool.get(value)
        return value
    
    def acquire_list(self, size: int = 0) -> List:
        """Acquire pooled list"""
        if self.list_pool:
            return self.list_pool.acquire(size)
        return [None] * size
    
    def release_list(self, lst: List) -> bool:
        """Release list back to pool"""
        if self.list_pool:
            return self.list_pool.release(lst)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'enabled': {
                'int_pool': self.enable_int_pool,
                'str_pool': self.enable_str_pool,
                'list_pool': self.enable_list_pool
            },
            'uptime_seconds': time.time() - self.start_time
        }
        
        if self.int_pool:
            stats['int_pool'] = self.int_pool.stats.to_dict()
        
        if self.str_pool:
            stats['str_pool'] = self.str_pool.stats.to_dict()
        
        if self.list_pool:
            stats['list_pool'] = self.list_pool.get_stats()
        
        # Calculate aggregate savings
        total_allocations = 0
        total_hits = 0
        
        if self.int_pool:
            total_allocations += self.int_pool.stats.allocations
            total_hits += self.int_pool.stats.hits
        
        if self.str_pool:
            total_allocations += self.str_pool.stats.allocations
            total_hits += self.str_pool.stats.hits
        
        if self.list_pool:
            total_allocations += self.list_pool.stats.allocations
            total_hits += self.list_pool.stats.hits
        
        stats['aggregate'] = {
            'total_allocations': total_allocations,
            'total_hits': total_hits,
            'overall_hit_rate': (total_hits / total_allocations * 100) if total_allocations > 0 else 0.0
        }
        
        return stats
    
    def clear_all(self):
        """Clear all pools"""
        if self.int_pool:
            self.int_pool.clear()
        if self.str_pool:
            self.str_pool.clear()
        if self.list_pool:
            self.list_pool.clear()
    
    def memory_usage_bytes(self) -> int:
        """Estimate memory usage of pools"""
        import sys
        total = 0
        
        if self.int_pool:
            total += sys.getsizeof(self.int_pool.pool)
            total += sys.getsizeof(self.int_pool.small_ints)
        
        if self.str_pool:
            total += sys.getsizeof(self.str_pool.pool)
            for s in self.str_pool.pool.values():
                total += sys.getsizeof(s)
        
        if self.list_pool:
            for pool in self.list_pool.pools.values():
                total += sys.getsizeof(pool.pool)
        
        return total
