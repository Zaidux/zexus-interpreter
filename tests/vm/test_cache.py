"""
Test Suite for Bytecode Caching System

Tests all aspects of the bytecode cache including:
- Cache hits and misses
- LRU eviction
- Memory limits
- Cache statistics
- Persistent cache
- AST hashing
- Cache invalidation

Part of Phase 4: Bytecode Caching Enhancement
"""

import sys
import unittest
import time
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.zexus.vm.cache import BytecodeCache, CacheStats, CacheEntry
from src.zexus.vm.bytecode import Bytecode, BytecodeBuilder
from src.zexus import zexus_ast


class TestCacheBasics(unittest.TestCase):
    """Test basic cache operations"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=100, debug=False)
    
    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        self.assertEqual(self.cache.size(), 0)
        self.assertEqual(self.cache.memory_usage(), 0)
        self.assertEqual(self.cache.stats.hits, 0)
        self.assertEqual(self.cache.stats.misses, 0)
    
    def test_cache_miss(self):
        """Test cache miss increments miss counter"""
        node = zexus_ast.IntegerLiteral(42)
        result = self.cache.get(node)
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.stats.misses, 1)
        self.assertEqual(self.cache.stats.hits, 0)
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache"""
        node = zexus_ast.IntegerLiteral(42)
        
        # Create bytecode
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        # Store in cache
        self.cache.put(node, bytecode)
        
        # Retrieve from cache
        cached = self.cache.get(node)
        
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached.instructions), 1)
        self.assertEqual(cached.constants[0], 42)
        self.assertEqual(self.cache.stats.hits, 1)
        self.assertEqual(self.cache.size(), 1)
    
    def test_cache_hit(self):
        """Test cache hit increments hit counter"""
        node = zexus_ast.IntegerLiteral(42)
        
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.cache.put(node, bytecode)
        
        # First get - hit
        cached1 = self.cache.get(node)
        self.assertIsNotNone(cached1)
        self.assertEqual(self.cache.stats.hits, 1)
        
        # Second get - another hit
        cached2 = self.cache.get(node)
        self.assertIsNotNone(cached2)
        self.assertEqual(self.cache.stats.hits, 2)
    
    def test_different_nodes_different_cache(self):
        """Test different AST nodes have different cache entries"""
        node1 = zexus_ast.IntegerLiteral(42)
        node2 = zexus_ast.IntegerLiteral(100)
        
        builder1 = BytecodeBuilder()
        builder1.emit_constant(42)
        bytecode1 = builder1.build()
        
        builder2 = BytecodeBuilder()
        builder2.emit_constant(100)
        bytecode2 = builder2.build()
        
        self.cache.put(node1, bytecode1)
        self.cache.put(node2, bytecode2)
        
        # Both should be cached separately
        cached1 = self.cache.get(node1)
        cached2 = self.cache.get(node2)
        
        self.assertEqual(cached1.constants[0], 42)
        self.assertEqual(cached2.constants[0], 100)
        self.assertEqual(self.cache.size(), 2)


class TestCacheEviction(unittest.TestCase):
    """Test LRU eviction policy"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=3, debug=False)
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        # Create 4 nodes
        nodes = [
            zexus_ast.IntegerLiteral(i)
            for i in range(4)
        ]
        
        # Create bytecode for each
        bytecodes = []
        for i in range(4):
            builder = BytecodeBuilder()
            builder.emit_constant(i)
            bytecodes.append(builder.build())
        
        # Add first 3 (cache full)
        for i in range(3):
            self.cache.put(nodes[i], bytecodes[i])
        
        self.assertEqual(self.cache.size(), 3)
        self.assertEqual(self.cache.stats.evictions, 0)
        
        # Add 4th - should evict first (LRU)
        self.cache.put(nodes[3], bytecodes[3])
        
        self.assertEqual(self.cache.size(), 3)
        self.assertEqual(self.cache.stats.evictions, 1)
        
        # First should be evicted
        self.assertIsNone(self.cache.get(nodes[0]))
        
        # Others should still be there
        self.assertIsNotNone(self.cache.get(nodes[1]))
        self.assertIsNotNone(self.cache.get(nodes[2]))
        self.assertIsNotNone(self.cache.get(nodes[3]))
    
    def test_lru_updates_on_access(self):
        """Test that accessing an entry updates its LRU position"""
        nodes = [zexus_ast.IntegerLiteral(i) for i in range(4)]
        bytecodes = []
        for i in range(4):
            builder = BytecodeBuilder()
            builder.emit_constant(i)
            bytecodes.append(builder.build())
        
        # Fill cache
        for i in range(3):
            self.cache.put(nodes[i], bytecodes[i])
        
        # Access first node (makes it most recent)
        self.cache.get(nodes[0])
        
        # Add 4th node - should evict 2nd (oldest now)
        self.cache.put(nodes[3], bytecodes[3])
        
        # First should still be there (was accessed)
        self.assertIsNotNone(self.cache.get(nodes[0]))
        
        # Second should be evicted
        self.assertIsNone(self.cache.get(nodes[1]))


class TestCacheMemoryLimits(unittest.TestCase):
    """Test memory-based eviction"""
    
    def setUp(self):
        # Very small memory limit for testing
        self.cache = BytecodeCache(
            max_size=1000,
            max_memory_mb=0.001,  # 1KB
            debug=False
        )
    
    def test_memory_limit_triggers_eviction(self):
        """Test eviction when memory limit is reached"""
        # Create many nodes to exceed memory limit
        for i in range(20):
            node = zexus_ast.IntegerLiteral(i)
            builder = BytecodeBuilder()
            for j in range(10):  # Create larger bytecode
                builder.emit_constant(i * 100 + j)
            bytecode = builder.build()
            self.cache.put(node, bytecode)
        
        # Cache should have evicted some entries
        self.assertLess(self.cache.size(), 20)
        self.assertGreater(self.cache.stats.evictions, 0)
        
        # Memory usage should be close to limit (within 50%)
        # Note: Eviction happens before adding, so we might exceed slightly
        self.assertLess(self.cache.memory_usage(), self.cache.max_memory_bytes * 1.5)


class TestCacheStatistics(unittest.TestCase):
    """Test cache statistics tracking"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=10, debug=False)
    
    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        # 1 miss
        self.cache.get(node)
        
        # Store
        self.cache.put(node, bytecode)
        
        # 3 hits
        self.cache.get(node)
        self.cache.get(node)
        self.cache.get(node)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['hits'], 3)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 75.0)  # 3 hits / 4 total = 75%
    
    def test_memory_usage_tracking(self):
        """Test memory usage is tracked"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        initial_memory = self.cache.memory_usage()
        self.cache.put(node, bytecode)
        after_memory = self.cache.memory_usage()
        
        self.assertGreater(after_memory, initial_memory)
    
    def test_stats_reset(self):
        """Test statistics can be reset"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.cache.put(node, bytecode)
        self.cache.get(node)
        self.cache.get(node)
        
        # Reset stats
        self.cache.reset_stats()
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['total_entries'], 1)  # Entry still there


class TestCacheInvalidation(unittest.TestCase):
    """Test cache invalidation"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=10, debug=False)
    
    def test_invalidate_entry(self):
        """Test invalidating a cache entry"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.cache.put(node, bytecode)
        self.assertEqual(self.cache.size(), 1)
        
        # Invalidate
        self.cache.invalidate(node)
        
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.get(node))
    
    def test_clear_cache(self):
        """Test clearing entire cache"""
        # Add multiple entries
        for i in range(5):
            node = zexus_ast.IntegerLiteral(i)
            builder = BytecodeBuilder()
            builder.emit_constant(i)
            bytecode = builder.build()
            self.cache.put(node, bytecode)
        
        self.assertEqual(self.cache.size(), 5)
        
        # Clear
        self.cache.clear()
        
        self.assertEqual(self.cache.size(), 0)
        self.assertEqual(self.cache.memory_usage(), 0)


class TestASTHashing(unittest.TestCase):
    """Test AST hashing functionality"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=10, debug=False)
    
    def test_same_ast_same_hash(self):
        """Test same AST structure produces same hash"""
        node1 = zexus_ast.IntegerLiteral(42)
        node2 = zexus_ast.IntegerLiteral(42)
        
        hash1 = self.cache._hash_ast(node1)
        hash2 = self.cache._hash_ast(node2)
        
        self.assertEqual(hash1, hash2)
    
    def test_different_ast_different_hash(self):
        """Test different AST structures produce different hashes"""
        node1 = zexus_ast.IntegerLiteral(42)
        node2 = zexus_ast.IntegerLiteral(100)
        
        hash1 = self.cache._hash_ast(node1)
        hash2 = self.cache._hash_ast(node2)
        
        self.assertNotEqual(hash1, hash2)
    
    def test_complex_ast_hashing(self):
        """Test hashing of complex nested AST"""
        # Create complex expression: (10 + 20) * 30
        inner = zexus_ast.InfixExpression(
            left=zexus_ast.IntegerLiteral(10),
            operator='+',
            right=zexus_ast.IntegerLiteral(20)
        )
        outer = zexus_ast.InfixExpression(
            left=inner,
            operator='*',
            right=zexus_ast.IntegerLiteral(30)
        )
        
        hash1 = self.cache._hash_ast(outer)
        
        # Create same structure again
        inner2 = zexus_ast.InfixExpression(
            left=zexus_ast.IntegerLiteral(10),
            operator='+',
            right=zexus_ast.IntegerLiteral(20)
        )
        outer2 = zexus_ast.InfixExpression(
            left=inner2,
            operator='*',
            right=zexus_ast.IntegerLiteral(30)
        )
        
        hash2 = self.cache._hash_ast(outer2)
        
        self.assertEqual(hash1, hash2)


class TestPersistentCache(unittest.TestCase):
    """Test disk-based persistent cache"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = BytecodeCache(
            max_size=10,
            persistent=True,
            cache_dir=self.temp_dir,
            debug=False
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_to_disk(self):
        """Test bytecode is saved to disk"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.cache.put(node, bytecode)
        
        # Check file was created
        cache_files = list(Path(self.temp_dir).glob('*.cache'))
        self.assertEqual(len(cache_files), 1)
    
    def test_load_from_disk(self):
        """Test bytecode can be loaded from disk"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        # Save to cache
        self.cache.put(node, bytecode)
        
        # Create new cache instance (clears memory)
        new_cache = BytecodeCache(
            max_size=10,
            persistent=True,
            cache_dir=self.temp_dir,
            debug=False
        )
        
        # Should load from disk
        cached = new_cache.get(node)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.constants[0], 42)
    
    def test_clear_removes_disk_files(self):
        """Test clearing cache removes disk files"""
        # Add entries
        for i in range(3):
            node = zexus_ast.IntegerLiteral(i)
            builder = BytecodeBuilder()
            builder.emit_constant(i)
            bytecode = builder.build()
            self.cache.put(node, bytecode)
        
        # Check files exist
        cache_files = list(Path(self.temp_dir).glob('*.cache'))
        self.assertEqual(len(cache_files), 3)
        
        # Clear cache
        self.cache.clear()
        
        # Files should be gone
        cache_files = list(Path(self.temp_dir).glob('*.cache'))
        self.assertEqual(len(cache_files), 0)


class TestCacheUtilities(unittest.TestCase):
    """Test cache utility methods"""
    
    def setUp(self):
        self.cache = BytecodeCache(max_size=10, debug=False)
    
    def test_contains(self):
        """Test cache contains check"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.assertFalse(self.cache.contains(node))
        
        self.cache.put(node, bytecode)
        
        self.assertTrue(self.cache.contains(node))
    
    def test_get_entry_info(self):
        """Test getting entry information"""
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        
        self.cache.put(node, bytecode)
        
        info = self.cache.get_entry_info(node)
        
        self.assertIsNotNone(info)
        self.assertIn('key', info)
        self.assertIn('timestamp', info)
        self.assertIn('access_count', info)
        self.assertIn('size_bytes', info)
        self.assertIn('instruction_count', info)
        self.assertEqual(info['access_count'], 1)
    
    def test_len(self):
        """Test __len__ method"""
        self.assertEqual(len(self.cache), 0)
        
        node = zexus_ast.IntegerLiteral(42)
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        bytecode = builder.build()
        self.cache.put(node, bytecode)
        
        self.assertEqual(len(self.cache), 1)
    
    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.cache)
        self.assertIn('BytecodeCache', repr_str)
        self.assertIn('size=', repr_str)
        self.assertIn('memory=', repr_str)


class TestCacheWithCompiler(unittest.TestCase):
    """Test cache integration with bytecode compiler"""
    
    def setUp(self):
        try:
            from src.zexus.evaluator.bytecode_compiler import EvaluatorBytecodeCompiler
            self.compiler = EvaluatorBytecodeCompiler(use_cache=True, cache_size=100)
            self.compiler_available = True
        except (ImportError, ModuleNotFoundError, AttributeError):
            self.compiler_available = False
    
    def test_compiler_uses_cache(self):
        """Test compiler uses cache for repeated compilations"""
        if not self.compiler_available:
            self.skipTest("Compiler not available")
        
        # Check if cache is available
        if not self.compiler.cache:
            self.skipTest("Cache not available in compiler")
        
        node = zexus_ast.IntegerLiteral(42)
        
        # First compile - cache miss
        bytecode1 = self.compiler.compile(node)
        stats1 = self.compiler.get_cache_stats()
        
        self.assertIsNotNone(bytecode1)
        self.assertIsNotNone(stats1)
        self.assertEqual(stats1['misses'], 1)
        self.assertEqual(stats1['hits'], 0)
        
        # Second compile - cache hit
        bytecode2 = self.compiler.compile(node)
        stats2 = self.compiler.get_cache_stats()
        
        self.assertIsNotNone(bytecode2)
        self.assertIsNotNone(stats2)
        self.assertEqual(stats2['hits'], 1)
    
    def test_compiler_cache_stats(self):
        """Test compiler cache statistics"""
        if not self.compiler_available:
            self.skipTest("Compiler not available")
        
        # Check if cache is available
        if not self.compiler.cache:
            self.skipTest("Cache not available in compiler")
        
        # Compile different nodes
        for i in range(5):
            node = zexus_ast.IntegerLiteral(i)
            self.compiler.compile(node)
        
        # Compile same nodes again (cache hits)
        for i in range(5):
            node = zexus_ast.IntegerLiteral(i)
            self.compiler.compile(node)
        
        stats = self.compiler.get_cache_stats()
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['misses'], 5)
        self.assertEqual(stats['hits'], 5)
        self.assertEqual(stats['hit_rate'], 50.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
    
    # Print summary
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("BYTECODE CACHE TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
