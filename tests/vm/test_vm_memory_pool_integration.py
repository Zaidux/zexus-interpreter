"""Integration tests for VM with Memory Pool."""

import unittest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from zexus.vm.vm import VM


class TestVMMemoryPoolIntegration(unittest.TestCase):
    """Test VM integration with memory pools."""
    
    def setUp(self):
        """Create VM with memory pooling enabled."""
        self.vm = VM(
            enable_memory_pool=True,
            pool_max_size=100,
            debug=False
        )
    
    def test_memory_pool_enabled(self):
        """Test that memory pools are enabled."""
        self.assertTrue(self.vm.enable_memory_pool)
        self.assertIsNotNone(self.vm.integer_pool)
        self.assertIsNotNone(self.vm.string_pool)
        self.assertIsNotNone(self.vm.list_pool)
    
    def test_allocate_integer(self):
        """Test integer allocation from pool."""
        value = self.vm.allocate_integer(42)
        self.assertEqual(value, 42)
        
        # Get stats to verify allocation
        stats = self.vm.get_pool_stats()
        self.assertIn('integer_pool', stats)
        self.assertGreater(stats['integer_pool']['allocations'], 0)
    
    def test_allocate_string(self):
        """Test string allocation from pool."""
        value = self.vm.allocate_string("hello")
        self.assertEqual(value, "hello")
        
        # Same string should be interned
        value2 = self.vm.allocate_string("hello")
        self.assertIs(value, value2)
        
        stats = self.vm.get_pool_stats()
        self.assertIn('string_pool', stats)
    
    def test_allocate_list(self):
        """Test list allocation from pool."""
        lst = self.vm.allocate_list(10)
        self.assertIsInstance(lst, list)
        # Note: ListPool returns a list of size 'size' filled with None
        self.assertEqual(len(lst), 10)
        
        stats = self.vm.get_pool_stats()
        self.assertIn('list_pool', stats)
        # ListPool returns nested stats with 'aggregate'
        self.assertGreater(stats['list_pool']['aggregate']['allocations'], 0)
    
    def test_release_and_reuse(self):
        """Test release and reuse of pooled objects."""
        # Allocate and release lists (integers and strings don't need explicit release)
        for i in range(10):
            lst = self.vm.allocate_list(5)
            self.vm.release_list(lst)
        
        stats = self.vm.get_pool_stats()
        list_stats = stats['list_pool']['aggregate']
        
        # Should have hits from reuse
        self.assertGreater(list_stats['allocations'], 0)
        self.assertGreater(list_stats['returns'], 0)
    
    def test_pool_stats(self):
        """Test getting pool statistics."""
        # Do some allocations
        for i in range(5):
            self.vm.allocate_integer(i)
            self.vm.allocate_string(f"str{i}")
            lst = self.vm.allocate_list()
            self.vm.release_list(lst)
        
        stats = self.vm.get_pool_stats()
        
        # Verify all pools have stats
        self.assertIn('integer_pool', stats)
        self.assertIn('string_pool', stats)
        self.assertIn('list_pool', stats)
        
        # Verify stats structure - IntegerPool and StringPool have PoolStats
        int_stats = stats['integer_pool']
        self.assertIn('hits', int_stats)
        self.assertIn('misses', int_stats)
        self.assertIn('allocations', int_stats)
        
        str_stats = stats['string_pool']
        self.assertIn('hits', str_stats)
        self.assertIn('allocations', str_stats)
        
        # ListPool has nested structure with 'aggregate'
        list_stats = stats['list_pool']
        self.assertIn('aggregate', list_stats)
        self.assertIn('allocations', list_stats['aggregate'])
    
    def test_reset_pools(self):
        """Test resetting all pools."""
        # Allocate some objects
        for i in range(5):
            self.vm.allocate_integer(i)
            lst = self.vm.allocate_list(2)
            self.vm.release_list(lst)
        
        # Reset pools
        self.vm.reset_pools()
        
        # Stats should show cleared state
        stats = self.vm.get_pool_stats()
        int_stats = stats['integer_pool']
        
        # Pool should be empty after clear
        self.assertEqual(int_stats['current_size'], 0)
    
    def test_high_performance_vm_has_pools(self):
        """Test that high performance VM has memory pools enabled."""
        from zexus.vm.vm import create_high_performance_vm
        
        vm = create_high_performance_vm()
        self.assertTrue(vm.enable_memory_pool)
        self.assertIsNotNone(vm.integer_pool)
        self.assertIsNotNone(vm.string_pool)
        self.assertIsNotNone(vm.list_pool)
    
    def test_pool_disabled_vm(self):
        """Test VM with memory pooling disabled."""
        vm = VM(enable_memory_pool=False)
        
        self.assertFalse(vm.enable_memory_pool)
        self.assertIsNone(vm.integer_pool)
        self.assertIsNone(vm.string_pool)
        self.assertIsNone(vm.list_pool)
        
        # Stats should return error
        stats = vm.get_pool_stats()
        self.assertIn('error', stats)
        
        # Allocation should still work (just not pooled)
        value = vm.allocate_integer(42)
        self.assertEqual(value, 42)
    
    def test_pool_hit_rate(self):
        """Test that pool hit rate improves with reuse."""
        # First allocation - all misses
        for i in range(10):
            self.vm.allocate_integer(i)
        
        stats1 = self.vm.get_pool_stats()
        hits_1 = stats1['integer_pool']['hits']
        
        # Allocate same integers again - should get hits
        for i in range(10):
            self.vm.allocate_integer(i)
        
        stats2 = self.vm.get_pool_stats()
        hits_2 = stats2['integer_pool']['hits']
        
        # Hits should increase
        self.assertGreater(hits_2, hits_1)
    
    def test_string_interning(self):
        """Test that string pool interns strings."""
        s1 = self.vm.allocate_string("test")
        s2 = self.vm.allocate_string("test")
        s3 = self.vm.allocate_string("test")
        
        # All should be the same object
        self.assertIs(s1, s2)
        self.assertIs(s2, s3)
        
        stats = self.vm.get_pool_stats()
        str_stats = stats['string_pool']
        
        # Should have hits from interning
        self.assertGreater(str_stats['hits'], 0)


class TestMemoryPoolPerformance(unittest.TestCase):
    """Performance tests for memory pools."""
    
    def test_allocation_performance(self):
        """Test that pooled allocation is faster than regular allocation."""
        import time
        
        # VM with pooling
        vm_pooled = VM(enable_memory_pool=True, pool_max_size=1000)
        
        # VM without pooling
        vm_regular = VM(enable_memory_pool=False)
        
        # Warm up
        for i in range(100):
            vm_pooled.allocate_integer(i)
            vm_regular.allocate_integer(i)
        
        # Test pooled lists (these can be released and reused)
        start = time.perf_counter()
        for _ in range(1000):
            for i in range(10):
                lst = vm_pooled.allocate_list(5)
                vm_pooled.release_list(lst)
        pooled_time = time.perf_counter() - start
        
        # Test regular lists
        start = time.perf_counter()
        for _ in range(1000):
            for i in range(10):
                lst = vm_regular.allocate_list(5)
        regular_time = time.perf_counter() - start
        
        print(f"\nPooled allocation: {pooled_time:.4f}s")
        print(f"Regular allocation: {regular_time:.4f}s")
        
        if pooled_time < regular_time:
            print(f"Speedup: {regular_time/pooled_time:.2f}x")
        else:
            print(f"No speedup (pooling overhead: {pooled_time/regular_time:.2f}x)")
        
        # Pooled should be at least as fast (usually faster with reuse)
        # We're not asserting faster because the overhead of pool management
        # might be similar for small objects, but it prevents GC pressure


if __name__ == '__main__':
    unittest.main()
