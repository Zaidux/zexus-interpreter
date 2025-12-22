# Memory Pool Usage Guide

## Overview

The Zexus VM Memory Pool system provides object pooling and caching for common types to reduce memory allocations and improve performance. It implements intelligent caching strategies for integers, strings, and lists.

**Created**: Phase 8.2 VM Optimization  
**Version**: 1.0.0  
**Status**: Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Pool Types](#pool-types)
4. [Statistics & Monitoring](#statistics--monitoring)
5. [Performance Tuning](#performance-tuning)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

---

## Quick Start

### Basic Usage

```python
from zexus.vm.memory_pool import MemoryPoolManager

# Create pool manager
manager = MemoryPoolManager()

# Get pooled integers
x = manager.get_int(42)
y = manager.get_int(100)

# Get pooled strings
name = manager.get_str("hello")

# Acquire and release lists
lst = manager.acquire_list(10)  # List with 10 elements
# ... use list ...
manager.release_list(lst)  # Return to pool

# Get statistics
stats = manager.get_stats()
print(f"Overall hit rate: {stats['aggregate']['overall_hit_rate']:.1f}%")
```

### Integration with VM

```python
from zexus.vm.vm import VM
from zexus.vm.memory_pool import MemoryPoolManager

class OptimizedVM(VM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_manager = MemoryPoolManager()
    
    def _execute_LOAD_CONST(self, instruction):
        """Load constant with pooling"""
        value = instruction.arg
        
        # Use pooled values
        if isinstance(value, int):
            value = self.pool_manager.get_int(value)
        elif isinstance(value, str):
            value = self.pool_manager.get_str(value)
        
        self.push(value)
    
    def _execute_BUILD_LIST(self, instruction):
        """Build list using pool"""
        count = instruction.arg
        lst = self.pool_manager.acquire_list(count)
        
        # Populate from stack
        for i in range(count - 1, -1, -1):
            lst[i] = self.pop()
        
        self.push(lst)
```

---

## Architecture

### Component Overview

```
MemoryPoolManager
├── IntegerPool
│   ├── Small Int Cache (-128 to 256)
│   └── LRU Pool (larger integers)
├── StringPool
│   ├── Intern Map (strings ≤ 64 chars)
│   └── LRU Eviction
└── ListPool
    └── Size-based Pools (0 to 16)
```

### Design Principles

1. **Type-Specific Optimization**: Different pooling strategies for different types
2. **LRU Eviction**: Automatic cleanup when pools grow too large
3. **Statistics Tracking**: Comprehensive metrics for performance monitoring
4. **Configurable Limits**: Tunable pool sizes and thresholds

---

## Pool Types

### 1. Integer Pool

Optimized for small integers and frequently used values.

**Features:**
- Static cache for small integers (-128 to 256)
- LRU-based dynamic pool for larger integers
- Zero allocation for cached values

**Configuration:**
```python
pool = IntegerPool(max_pool_size=1000)

# Small integers always cached
x = pool.get(42)  # Cache hit
y = pool.get(100)  # Cache hit

# Large integers pooled with LRU
a = pool.get(1000000)  # First access: miss
b = pool.get(1000000)  # Second access: hit
```

**When to Use:**
- Loop counters and indices
- Configuration values
- Mathematical constants
- Enumeration values

### 2. String Pool

String interning for short, frequently used strings.

**Features:**
- Automatic interning for strings ≤ 64 characters
- LRU eviction when pool exceeds size limit
- Identity comparison for interned strings

**Configuration:**
```python
pool = StringPool(max_interned=500)

# Short strings interned
s1 = pool.get("hello")
s2 = pool.get("hello")
assert s1 is s2  # Same object

# Long strings not pooled
long_str = pool.get("x" * 100)  # Not interned
```

**When to Use:**
- Variable names
- Function names
- Common literals ("", "error", "true", etc.)
- Dictionary keys

### 3. List Pool

Size-based pools for list reuse.

**Features:**
- Separate pools for each size (0 to 16)
- Automatic clearing on release
- Large lists (>16) not pooled

**Configuration:**
```python
pool = ListPool(max_pool_size=100)

# Acquire list
lst = pool.acquire(5)  # List of size 5
assert len(lst) == 5

# Use list
lst[0] = 10
lst[1] = 20

# Release back to pool
pool.release(lst)  # Cleared and pooled

# Large lists not pooled
large = pool.acquire(100)  # New allocation
pool.release(large)  # Not pooled
```

**When to Use:**
- Temporary work arrays
- Stack frames
- Loop buffers
- Argument lists

---

## Statistics & Monitoring

### Collecting Statistics

```python
manager = MemoryPoolManager()

# ... perform operations ...

stats = manager.get_stats()
```

### Statistics Structure

```python
{
    'int_pool': {
        'hits': 1500,
        'misses': 200,
        'allocations': 200,
        'releases': 0,  # Integers not released
        'pool_size': 150,
        'hit_rate': 88.2,
        'reuse_rate': 0.0
    },
    'str_pool': {
        'hits': 800,
        'misses': 100,
        'allocations': 100,
        'releases': 0,  # Strings not released
        'pool_size': 75,
        'hit_rate': 88.9,
        'reuse_rate': 0.0
    },
    'list_pool': {
        'hits': 0,
        'misses': 300,
        'allocations': 300,
        'releases': 280,
        'pool_size': 280,
        'hit_rate': 0.0,
        'reuse_rate': 93.3
    },
    'aggregate': {
        'total_hits': 2300,
        'total_misses': 600,
        'total_operations': 2900,
        'overall_hit_rate': 79.3
    }
}
```

### Key Metrics

1. **Hit Rate**: Percentage of requests served from cache
   - `hit_rate = (hits / (hits + misses)) * 100`
   - Higher is better (less allocation)

2. **Reuse Rate**: Percentage of allocated objects returned to pool
   - `reuse_rate = (releases / allocations) * 100`
   - Applies mainly to lists

3. **Pool Size**: Current number of pooled objects
   - Monitor to prevent unbounded growth

### Performance Monitoring

```python
def monitor_pool_performance(manager: MemoryPoolManager):
    """Monitor and report pool performance"""
    stats = manager.get_stats()
    
    # Check overall efficiency
    overall_hit_rate = stats['aggregate']['overall_hit_rate']
    if overall_hit_rate < 50:
        print(f"⚠️  Low hit rate: {overall_hit_rate:.1f}%")
    else:
        print(f"✅ Good hit rate: {overall_hit_rate:.1f}%")
    
    # Check integer pool
    int_stats = stats['int_pool']
    print(f"Integer pool: {int_stats['hit_rate']:.1f}% hits, "
          f"{int_stats['pool_size']} cached")
    
    # Check string pool
    str_stats = stats['str_pool']
    print(f"String pool: {str_stats['hit_rate']:.1f}% hits, "
          f"{str_stats['pool_size']} interned")
    
    # Check list pool
    list_stats = stats['list_pool']
    print(f"List pool: {list_stats['reuse_rate']:.1f}% reuse, "
          f"{list_stats['pool_size']} pooled")
```

---

## Performance Tuning

### Pool Size Configuration

```python
# Low-memory environment
manager = MemoryPoolManager(
    int_pool_size=500,      # Smaller int cache
    str_pool_size=200,      # Fewer interned strings
    list_pool_size=50       # Smaller list pools
)

# High-performance environment
manager = MemoryPoolManager(
    int_pool_size=5000,     # Larger int cache
    str_pool_size=2000,     # More interned strings
    list_pool_size=500      # Larger list pools
)

# Balanced (default)
manager = MemoryPoolManager()  # Uses reasonable defaults
```

### Selective Pool Enabling

```python
manager = MemoryPoolManager()

# Disable string pooling if not beneficial
manager.enable_pool('str', False)

# Check which pools are enabled
stats = manager.get_stats()
if 'str_pool' not in stats:
    print("String pooling disabled")
```

### Memory Pressure Management

```python
def manage_memory_pressure(manager: MemoryPoolManager):
    """Clear pools under memory pressure"""
    stats = manager.get_stats()
    
    # Get total pooled objects
    total_pooled = (
        stats['int_pool']['pool_size'] +
        stats['str_pool']['pool_size'] +
        stats['list_pool']['pool_size']
    )
    
    # Clear if too many pooled objects
    if total_pooled > 10000:
        manager.clear_all_pools()
        print("Cleared pools due to memory pressure")
```

---

## Integration Examples

### Example 1: VM Integration

```python
class ZexusVM:
    def __init__(self):
        self.pool_manager = MemoryPoolManager(
            int_pool_size=2000,
            str_pool_size=1000,
            list_pool_size=200
        )
        self.stack = []
        self.locals = {}
    
    def load_constant(self, value):
        """Load constant with pooling"""
        if isinstance(value, int):
            value = self.pool_manager.get_int(value)
        elif isinstance(value, str):
            value = self.pool_manager.get_str(value)
        self.stack.append(value)
    
    def build_list(self, count):
        """Build list from stack using pool"""
        lst = self.pool_manager.acquire_list(count)
        for i in range(count - 1, -1, -1):
            lst[i] = self.stack.pop()
        self.stack.append(lst)
    
    def cleanup_list(self, lst):
        """Return list to pool"""
        if isinstance(lst, list):
            self.pool_manager.release_list(lst)
```

### Example 2: Compiler Integration

```python
class Compiler:
    def __init__(self):
        self.pool_manager = MemoryPoolManager()
        self.string_table = {}
    
    def intern_string(self, s: str) -> str:
        """Intern string constant"""
        return self.pool_manager.get_str(s)
    
    def get_constant_int(self, n: int) -> int:
        """Get constant integer"""
        return self.pool_manager.get_int(n)
    
    def compile_list_literal(self, elements):
        """Compile list literal with pooling metadata"""
        size = len(elements)
        
        # Add hint for runtime pooling
        return {
            'type': 'list_literal',
            'size': size,
            'poolable': size <= 16,
            'elements': elements
        }
```

### Example 3: Test Framework

```python
import unittest
from zexus.vm.memory_pool import MemoryPoolManager

class PooledTestCase(unittest.TestCase):
    def setUp(self):
        """Create fresh pool manager for each test"""
        self.pool_manager = MemoryPoolManager()
    
    def tearDown(self):
        """Report pool statistics after test"""
        stats = self.pool_manager.get_stats()
        hit_rate = stats['aggregate']['overall_hit_rate']
        
        # Log performance
        print(f"\nPool hit rate: {hit_rate:.1f}%")
        
        # Clear pools
        self.pool_manager.clear_all_pools()
    
    def test_with_pooling(self):
        """Test that uses pooled objects"""
        # Use pooled integers
        for i in range(100):
            x = self.pool_manager.get_int(i % 10)
            self.assertIsInstance(x, int)
        
        # Check pooling is effective
        stats = self.pool_manager.get_stats()
        self.assertGreater(stats['int_pool']['hit_rate'], 80.0)
```

---

## Best Practices

### 1. Pool Small, Frequently Used Objects

✅ **Good:**
```python
# Small integers
for i in range(1000):
    x = manager.get_int(i % 10)  # Only 10 unique values

# Short strings
name = manager.get_str("temp")  # Reused often

# Small lists
buffer = manager.acquire_list(5)  # Temporary buffer
```

❌ **Avoid:**
```python
# Large, unique integers
x = manager.get_int(random.randint(0, 10**9))  # Unlikely to reuse

# Long strings
text = manager.get_str("x" * 1000)  # Won't be pooled anyway

# Large lists
big_list = manager.acquire_list(10000)  # Won't be pooled
```

### 2. Always Release Lists

✅ **Good:**
```python
lst = manager.acquire_list(10)
try:
    # Use list
    process_data(lst)
finally:
    manager.release_list(lst)  # Always release
```

❌ **Avoid:**
```python
lst = manager.acquire_list(10)
process_data(lst)
# Forgot to release - memory leak!
```

### 3. Monitor Pool Performance

✅ **Good:**
```python
# Regular monitoring
if iterations % 1000 == 0:
    stats = manager.get_stats()
    if stats['aggregate']['overall_hit_rate'] < 30:
        # Adjust pooling strategy
        manager.enable_pool('int', False)
```

### 4. Clear Pools Periodically

✅ **Good:**
```python
# Clear pools after major operations
def process_batch(items):
    for item in items:
        process_item(item)
    
    # Clear pools after batch
    manager.clear_all_pools()
```

### 5. Use Context Managers

✅ **Good:**
```python
from contextlib import contextmanager

@contextmanager
def pooled_list(manager, size):
    """Context manager for pooled lists"""
    lst = manager.acquire_list(size)
    try:
        yield lst
    finally:
        manager.release_list(lst)

# Usage
with pooled_list(manager, 10) as lst:
    lst[0] = 42
    process(lst)
# Automatically released
```

---

## API Reference

### MemoryPoolManager

Main interface for memory pooling.

#### Constructor

```python
MemoryPoolManager(
    int_pool_size: int = 1000,
    str_pool_size: int = 500,
    list_pool_size: int = 100
)
```

**Parameters:**
- `int_pool_size`: Maximum size of integer pool
- `str_pool_size`: Maximum number of interned strings
- `list_pool_size`: Maximum size of each list pool

#### Methods

##### `get_int(value: int) -> int`

Get pooled integer.

**Returns:** Pooled integer (may be same object for small integers)

```python
x = manager.get_int(42)
```

##### `get_str(value: str) -> str`

Get pooled string.

**Returns:** Interned string if ≤64 chars, otherwise original

```python
s = manager.get_str("hello")
```

##### `acquire_list(size: int) -> list`

Acquire list from pool.

**Parameters:**
- `size`: Desired list size (0-16 for pooling)

**Returns:** List of specified size

```python
lst = manager.acquire_list(10)
```

##### `release_list(lst: list) -> None`

Return list to pool.

**Parameters:**
- `lst`: List to return

```python
manager.release_list(lst)
```

##### `get_stats() -> dict`

Get pool statistics.

**Returns:** Dictionary with detailed statistics

```python
stats = manager.get_stats()
print(stats['aggregate']['overall_hit_rate'])
```

##### `clear_all_pools() -> None`

Clear all pools.

```python
manager.clear_all_pools()
```

##### `enable_pool(pool_name: str, enabled: bool) -> None`

Enable or disable specific pool.

**Parameters:**
- `pool_name`: Pool to configure ('int', 'str', or 'list')
- `enabled`: Whether to enable the pool

```python
manager.enable_pool('str', False)
```

### IntegerPool

Low-level integer pooling.

#### Constructor

```python
IntegerPool(max_pool_size: int = 1000)
```

#### Methods

##### `get(value: int) -> int`

Get pooled integer.

```python
pool = IntegerPool()
x = pool.get(42)
```

##### `clear() -> None`

Clear the pool.

```python
pool.clear()
```

### StringPool

Low-level string pooling.

#### Constructor

```python
StringPool(max_interned: int = 500)
```

#### Methods

##### `get(value: str) -> str`

Get interned string.

```python
pool = StringPool()
s = pool.get("hello")
```

##### `clear() -> None`

Clear the pool.

```python
pool.clear()
```

### ListPool

Low-level list pooling.

#### Constructor

```python
ListPool(max_pool_size: int = 100)
```

#### Methods

##### `acquire(size: int) -> list`

Acquire list from pool.

```python
pool = ListPool()
lst = pool.acquire(10)
```

##### `release(lst: list) -> None`

Return list to pool.

```python
pool.release(lst)
```

##### `clear() -> None`

Clear all pools.

```python
pool.clear()
```

---

## Performance Characteristics

### Time Complexity

| Operation | Best Case | Worst Case | Average |
|-----------|-----------|------------|---------|
| get_int (small) | O(1) | O(1) | O(1) |
| get_int (large) | O(1) | O(n) | O(1) |
| get_str | O(1) | O(n) | O(1) |
| acquire_list | O(1) | O(1) | O(1) |
| release_list | O(n) | O(n) | O(n) |

*n = pool size*

### Space Complexity

| Pool Type | Space Usage |
|-----------|-------------|
| IntegerPool | O(cached_ints + pooled_ints) |
| StringPool | O(interned_strings) |
| ListPool | O(pooled_lists × avg_size) |

### Performance Tips

1. **Small Integers**: Almost free (static cache)
2. **Short Strings**: Very fast (hash lookup)
3. **Lists**: Fast acquire, moderate release cost
4. **LRU Eviction**: Occasional O(n) cleanup

---

## Troubleshooting

### Low Hit Rates

**Problem:** Overall hit rate < 30%

**Solutions:**
1. Check if values are being reused
2. Increase pool sizes
3. Profile to find pooling opportunities
4. Consider disabling ineffective pools

```python
stats = manager.get_stats()
for pool_name in ['int_pool', 'str_pool', 'list_pool']:
    pool_stats = stats.get(pool_name, {})
    hit_rate = pool_stats.get('hit_rate', 0)
    if hit_rate < 20:
        print(f"⚠️  {pool_name} ineffective: {hit_rate:.1f}% hit rate")
```

### Memory Leaks

**Problem:** Pool sizes growing unbounded

**Solutions:**
1. Ensure lists are released
2. Add periodic pool clearing
3. Reduce max pool sizes
4. Monitor pool statistics

```python
def check_pool_sizes(manager):
    stats = manager.get_stats()
    
    total_size = (
        stats['int_pool']['pool_size'] +
        stats['str_pool']['pool_size'] +
        stats['list_pool']['pool_size']
    )
    
    if total_size > 5000:
        print(f"⚠️  Large pool size: {total_size}")
        manager.clear_all_pools()
```

### Poor Performance

**Problem:** Pooling slower than direct allocation

**Causes:**
1. Pool sizes too small (excessive LRU)
2. Values not reused enough
3. Overhead of pool management

**Solutions:**
```python
# Disable pooling if not beneficial
stats = manager.get_stats()
if stats['aggregate']['overall_hit_rate'] < 10:
    manager.enable_pool('int', False)
    manager.enable_pool('str', False)
    manager.enable_pool('list', False)
```

---

## Advanced Topics

### Custom Pool Implementation

```python
from zexus.vm.memory_pool import ObjectPool, PoolStats

class TuplePool:
    """Pool for small tuples"""
    
    def __init__(self, max_pool_size=100):
        # Separate pool for each size
        self.pools = {
            i: ObjectPool(lambda: tuple([None] * i), max_pool_size)
            for i in range(17)
        }
    
    def acquire(self, size: int) -> tuple:
        """Acquire tuple from pool"""
        if size > 16:
            return tuple([None] * size)
        return self.pools[size].acquire()
    
    def release(self, tpl: tuple) -> None:
        """Return tuple to pool"""
        size = len(tpl)
        if size <= 16:
            self.pools[size].release(tpl)
```

### Integration with GC

```python
import gc
from zexus.vm.memory_pool import MemoryPoolManager

class GCIntegratedPoolManager(MemoryPoolManager):
    """Pool manager that clears on GC"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Register GC callback
        self.gc_count = gc.get_count()[0]
    
    def maybe_gc_clear(self):
        """Clear pools on GC collection"""
        current_count = gc.get_count()[0]
        
        if current_count < self.gc_count:
            # GC occurred
            self.clear_all_pools()
            print("Cleared pools after GC")
        
        self.gc_count = current_count
```

---

## Changelog

### Version 1.0.0 (Phase 8.2)
- Initial implementation
- IntegerPool with small int cache
- StringPool with interning
- ListPool with size-based pools
- Comprehensive statistics
- Full test coverage

---

## See Also

- [VM Optimization Phase 8 Master List](VM_OPTIMIZATION_PHASE_8_MASTER_LIST.md)
- [Profiler Usage Guide](PROFILER_USAGE_GUIDE.md)
- VM Architecture Documentation
- Performance Tuning Guide
