"""
Tests for Memory Pool Optimization (Phase 8.2)

Tests cover:
- Object pool functionality
- Integer pooling
- String pooling
- List pooling
- Pool statistics
- Memory savings
"""

import sys
import unittest
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.zexus.vm.memory_pool import (
    ObjectPool, IntegerPool, StringPool, ListPool,
    MemoryPoolManager, PoolStats
)


class TestPoolStats(unittest.TestCase):
    """Test pool statistics"""
    
    def test_pool_stats_creation(self):
        """Test creating pool stats"""
        stats = PoolStats()
        self.assertEqual(stats.allocations, 0)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 0)
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = PoolStats()
        stats.hits = 80
        stats.misses = 20
        
        self.assertAlmostEqual(stats.hit_rate, 80.0, places=1)
    
    def test_reuse_rate_calculation(self):
        """Test reuse rate calculation"""
        stats = PoolStats()
        stats.total_created = 100
        stats.reuses = 400
        
        self.assertAlmostEqual(stats.reuse_rate, 400.0, places=1)
    
    def test_stats_to_dict(self):
        """Test stats serialization"""
        stats = PoolStats()
        stats.hits = 10
        stats.misses = 5
        
        data = stats.to_dict()
        self.assertEqual(data['hits'], 10)
        self.assertEqual(data['misses'], 5)
        self.assertIn('hit_rate', data)


class TestObjectPool(unittest.TestCase):
    """Test generic object pool"""
    
    def test_pool_creation(self):
        """Test creating object pool"""
        pool = ObjectPool(max_size=10)
        self.assertEqual(pool.max_size, 10)
        self.assertEqual(pool.size(), 0)
    
    def test_acquire_miss(self):
        """Test acquiring when pool is empty"""
        pool = ObjectPool(max_size=10)
        obj = pool.acquire(default_value=42)
        
        self.assertEqual(obj, 42)
        self.assertEqual(pool.stats.misses, 1)
        self.assertEqual(pool.stats.hits, 0)
    
    def test_acquire_and_release(self):
        """Test acquire and release cycle"""
        pool = ObjectPool(max_size=10, factory=list)
        
        # Acquire object
        obj1 = pool.acquire()
        self.assertIsInstance(obj1, list)
        self.assertEqual(pool.stats.misses, 1)
        
        # Release object
        success = pool.release(obj1)
        self.assertTrue(success)
        self.assertEqual(pool.stats.returns, 1)
        self.assertEqual(pool.size(), 1)
        
        # Acquire again - should hit pool
        obj2 = pool.acquire()
        self.assertIs(obj1, obj2)  # Same object
        self.assertEqual(pool.stats.hits, 1)
        self.assertEqual(pool.stats.reuses, 1)
    
    def test_pool_max_size(self):
        """Test pool respects max size"""
        pool = ObjectPool(max_size=2)
        
        objs = [pool.acquire(i) for i in range(5)]
        
        # Release all
        for obj in objs:
            pool.release(obj)
        
        # Pool should only hold 2
        self.assertEqual(pool.size(), 2)
    
    def test_pool_clear(self):
        """Test clearing pool"""
        pool = ObjectPool(max_size=10)
        
        for i in range(5):
            pool.release(i)
        
        self.assertEqual(pool.size(), 5)
        
        pool.clear()
        self.assertEqual(pool.size(), 0)


class TestIntegerPool(unittest.TestCase):
    """Test integer pooling"""
    
    def test_small_int_caching(self):
        """Test small integers are cached"""
        pool = IntegerPool()
        
        # Small integers should always hit cache
        count = 0
        for i in range(-128, 257):
            val = pool.get(i)
            self.assertEqual(val, i)
            count += 1
        
        # All should be hits
        self.assertEqual(pool.stats.hits, count)
    
    def test_large_int_pooling(self):
        """Test large integers are pooled"""
        pool = IntegerPool(max_size=100)
        
        # First access - miss
        val1 = pool.get(1000)
        self.assertEqual(val1, 1000)
        self.assertEqual(pool.stats.misses, 1)
        
        # Second access - hit
        val2 = pool.get(1000)
        self.assertEqual(val2, 1000)
        self.assertEqual(pool.stats.hits, 1)
    
    def test_int_pool_lru_eviction(self):
        """Test LRU eviction for integers"""
        pool = IntegerPool(max_size=3)
        
        # Add 3 large integers
        pool.get(1000)
        pool.get(2000)
        pool.get(3000)
        
        # Access 1000 and 2000 multiple times
        for _ in range(5):
            pool.get(1000)
            pool.get(2000)
        
        # Add new integer - should evict 3000 (least accessed)
        pool.get(4000)
        
        # 1000 and 2000 should still be in pool
        hits_before = pool.stats.hits
        pool.get(1000)
        pool.get(2000)
        self.assertEqual(pool.stats.hits, hits_before + 2)
    
    def test_int_pool_clear(self):
        """Test clearing integer pool"""
        pool = IntegerPool()
        
        # Add some large integers
        for i in range(1000, 1010):
            pool.get(i)
        
        pool.clear()
        
        # Small ints should still work
        self.assertEqual(pool.get(0), 0)
        self.assertEqual(pool.get(100), 100)


class TestStringPool(unittest.TestCase):
    """Test string pooling"""
    
    def test_string_interning(self):
        """Test basic string interning"""
        pool = StringPool()
        
        str1 = pool.get("hello")
        str2 = pool.get("hello")
        
        # Should be same object (interned)
        self.assertIs(str1, str2)
        self.assertEqual(pool.stats.hits, 1)
    
    def test_long_strings_not_pooled(self):
        """Test long strings are not pooled"""
        pool = StringPool()
        
        long_str = "x" * 100  # > MAX_STRING_LENGTH (64)
        
        result = pool.get(long_str)
        self.assertEqual(result, long_str)
        self.assertEqual(pool.stats.hits, 0)
        self.assertEqual(pool.stats.misses, 1)
    
    def test_string_pool_lru(self):
        """Test LRU eviction for strings"""
        pool = StringPool(max_size=3)
        
        # Add 3 strings
        pool.get("a")
        pool.get("b")
        pool.get("c")
        
        # Access "a" and "b" multiple times
        for _ in range(5):
            pool.get("a")
            pool.get("b")
        
        # Add new string - should evict "c"
        pool.get("d")
        
        # "a" and "b" should still hit
        hits_before = pool.stats.hits
        pool.get("a")
        pool.get("b")
        self.assertEqual(pool.stats.hits, hits_before + 2)
    
    def test_string_pool_stats(self):
        """Test string pool statistics"""
        pool = StringPool(max_size=100)
        
        # Add unique strings
        for i in range(50):
            pool.get(f"string{i}")
        
        self.assertEqual(pool.stats.misses, 50)
        self.assertEqual(pool.stats.total_created, 50)
        
        # Access again - all hits
        for i in range(50):
            pool.get(f"string{i}")
        
        self.assertEqual(pool.stats.hits, 50)


class TestListPool(unittest.TestCase):
    """Test list pooling"""
    
    def test_acquire_empty_list(self):
        """Test acquiring empty list"""
        pool = ListPool()
        
        lst = pool.acquire(0)
        self.assertIsInstance(lst, list)
        self.assertEqual(len(lst), 0)
    
    def test_acquire_sized_list(self):
        """Test acquiring sized list"""
        pool = ListPool()
        
        lst = pool.acquire(5)
        self.assertEqual(len(lst), 5)
        self.assertEqual(lst, [None, None, None, None, None])
    
    def test_list_reuse(self):
        """Test list reuse"""
        pool = ListPool()
        
        # Acquire list
        lst1 = pool.acquire(3)
        lst1[0] = "test"
        
        # Release list
        pool.release(lst1)
        
        # Acquire again - should get same list (cleared)
        lst2 = pool.acquire(3)
        self.assertIs(lst1, lst2)
        self.assertEqual(lst2, [None, None, None])  # Should be cleared
    
    def test_large_list_not_pooled(self):
        """Test large lists are not pooled"""
        pool = ListPool()
        
        large_list = pool.acquire(20)  # > MAX_LIST_SIZE (16)
        self.assertEqual(len(large_list), 20)
        
        # Should not be able to release
        success = pool.release(large_list)
        self.assertFalse(success)
    
    def test_list_pool_by_size(self):
        """Test separate pools by size"""
        pool = ListPool()
        
        # Acquire different sizes
        lst0 = pool.acquire(0)
        lst5 = pool.acquire(5)
        lst10 = pool.acquire(10)
        
        # Release all
        pool.release(lst0)
        pool.release(lst5)
        pool.release(lst10)
        
        # Acquire same sizes - should get same lists
        new_lst5 = pool.acquire(5)
        self.assertIs(lst5, new_lst5)
    
    def test_list_pool_stats(self):
        """Test list pool statistics"""
        pool = ListPool()
        
        # Acquire lists
        lists = [pool.acquire(i % 5) for i in range(20)]
        
        self.assertEqual(pool.stats.allocations, 20)
        
        # Release lists
        for lst in lists:
            pool.release(lst)
        
        # Reacquire - should have high hit rate
        for i in range(20):
            pool.acquire(i % 5)
        
        self.assertGreater(pool.stats.hits, 0)


class TestMemoryPoolManager(unittest.TestCase):
    """Test memory pool manager"""
    
    def test_manager_creation(self):
        """Test creating pool manager"""
        manager = MemoryPoolManager()
        
        self.assertTrue(manager.enable_int_pool)
        self.assertTrue(manager.enable_str_pool)
        self.assertTrue(manager.enable_list_pool)
        self.assertIsNotNone(manager.int_pool)
        self.assertIsNotNone(manager.str_pool)
        self.assertIsNotNone(manager.list_pool)
    
    def test_get_int(self):
        """Test getting pooled integer"""
        manager = MemoryPoolManager()
        
        val1 = manager.get_int(42)
        val2 = manager.get_int(42)
        
        self.assertEqual(val1, 42)
        self.assertEqual(val2, 42)
    
    def test_get_str(self):
        """Test getting pooled string"""
        manager = MemoryPoolManager()
        
        str1 = manager.get_str("hello")
        str2 = manager.get_str("hello")
        
        self.assertIs(str1, str2)
    
    def test_list_acquire_release(self):
        """Test list acquire/release"""
        manager = MemoryPoolManager()
        
        lst = manager.acquire_list(5)
        self.assertEqual(len(lst), 5)
        
        success = manager.release_list(lst)
        self.assertTrue(success)
    
    def test_get_stats(self):
        """Test getting comprehensive stats"""
        manager = MemoryPoolManager()
        
        # Use all pools
        manager.get_int(100)
        manager.get_str("test")
        manager.acquire_list(3)
        
        stats = manager.get_stats()
        
        self.assertIn('enabled', stats)
        self.assertIn('int_pool', stats)
        self.assertIn('str_pool', stats)
        self.assertIn('list_pool', stats)
        self.assertIn('aggregate', stats)
    
    def test_clear_all_pools(self):
        """Test clearing all pools"""
        manager = MemoryPoolManager()
        
        # Populate pools
        for i in range(10):
            manager.get_int(1000 + i)
            manager.get_str(f"str{i}")
        
        manager.clear_all()
        
        stats = manager.get_stats()
        self.assertEqual(stats['int_pool']['current_size'], 0)
        self.assertEqual(stats['str_pool']['current_size'], 0)
    
    def test_selective_pool_enabling(self):
        """Test enabling/disabling specific pools"""
        manager = MemoryPoolManager(
            enable_int_pool=True,
            enable_str_pool=False,
            enable_list_pool=True
        )
        
        self.assertIsNotNone(manager.int_pool)
        self.assertIsNone(manager.str_pool)
        self.assertIsNotNone(manager.list_pool)


class TestPoolPerformance(unittest.TestCase):
    """Test pool performance characteristics"""
    
    def test_int_pool_performance(self):
        """Test integer pool performance"""
        pool = IntegerPool(max_size=1000)
        
        # Warm up pool
        for i in range(100):
            pool.get(i)
        
        # Measure hit rate
        start = time.perf_counter()
        for _ in range(10000):
            for i in range(100):
                pool.get(i)
        elapsed = time.perf_counter() - start
        
        self.assertGreater(pool.stats.hit_rate, 50.0)
        self.assertLess(elapsed, 0.5)  # Should be fast
    
    def test_string_pool_performance(self):
        """Test string pool performance"""
        pool = StringPool(max_size=1000)
        
        strings = [f"string{i}" for i in range(100)]
        
        # Warm up
        for s in strings:
            pool.get(s)
        
        # Measure performance
        start = time.perf_counter()
        for _ in range(1000):
            for s in strings:
                pool.get(s)
        elapsed = time.perf_counter() - start
        
        self.assertGreater(pool.stats.hit_rate, 90.0)
        self.assertLess(elapsed, 0.5)
    
    def test_list_pool_overhead(self):
        """Test list pool overhead is acceptable"""
        pool = ListPool(max_pool_size=500)
        
        # With pool
        start = time.perf_counter()
        for _ in range(1000):
            lst = pool.acquire(5)
            pool.release(lst)
        pooled_time = time.perf_counter() - start
        
        # Without pool
        start = time.perf_counter()
        for _ in range(1000):
            lst = [None] * 5
        direct_time = time.perf_counter() - start
        
        # Pool should not be more than 30x slower (interpreted Python overhead)
        overhead = pooled_time / direct_time if direct_time > 0 else 1
        self.assertLess(overhead, 30.0, f"Overhead: {overhead:.1f}x")
    
    def test_memory_savings(self):
        """Test that pooling saves memory allocations"""
        manager = MemoryPoolManager()
        
        # Create many objects with high reuse
        for i in range(1000):
            # Reuse same values frequently
            manager.get_int(i % 10)  # Only 10 unique values
            manager.get_str(f"str{i % 5}")  # Only 5 unique strings
            lst = manager.acquire_list(i % 3)  # Only 3 sizes
            manager.release_list(lst)
        
        stats = manager.get_stats()
        
        # Should have high overall hit rate due to reuse
        overall_hit_rate = stats['aggregate']['overall_hit_rate']
        self.assertGreater(overall_hit_rate, 70.0, f"Hit rate: {overall_hit_rate:.1f}%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
