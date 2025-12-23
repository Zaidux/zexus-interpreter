"""
Tests for Async/Await Optimizer

Tests optimization strategies:
- Coroutine pooling
- Fast path for resolved futures
- Batch operation detection
- Statistics tracking
"""

import unittest
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from zexus.vm.async_optimizer import (
    AsyncOptimizer,
    AsyncOptimizationLevel,
    AsyncStats,
    FastFuture,
    CoroutinePool,
    BatchAwaitDetector,
    is_coroutine_like,
    is_immediately_available,
    fast_await,
)


class TestAsyncStats(unittest.TestCase):
    """Test AsyncStats class"""
    
    def test_stats_creation(self):
        """Test creating stats"""
        stats = AsyncStats()
        self.assertEqual(stats.total_spawns, 0)
        self.assertEqual(stats.total_awaits, 0)
        self.assertEqual(stats.fast_path_hits, 0)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = AsyncStats(
            total_spawns=10,
            total_awaits=20,
            fast_path_hits=15,
            pooled_coroutines=10,
            coroutine_reuses=7
        )
        
        d = stats.to_dict()
        self.assertEqual(d['total_spawns'], 10)
        self.assertEqual(d['total_awaits'], 20)
        self.assertEqual(d['fast_path_hits'], 15)
        self.assertEqual(d['pool_hit_rate'], 70.0)
        self.assertEqual(d['fast_path_rate'], 75.0)


class TestFastFuture(unittest.TestCase):
    """Test FastFuture class"""
    
    def test_resolved_future(self):
        """Test creating resolved future"""
        future = FastFuture(value=42)
        self.assertTrue(future.done())
        self.assertEqual(future.result(), 42)
        self.assertIsNone(future.exception())
    
    def test_rejected_future(self):
        """Test creating rejected future"""
        exc = ValueError("test error")
        future = FastFuture(exception=exc)
        self.assertTrue(future.done())
        self.assertEqual(future.exception(), exc)
        
        with self.assertRaises(ValueError):
            future.result()
    
    def test_await_resolved(self):
        """Test awaiting resolved future"""
        async def test():
            future = FastFuture(value=123)
            result = await future
            return result
        
        result = asyncio.run(test())
        self.assertEqual(result, 123)
    
    def test_await_rejected(self):
        """Test awaiting rejected future"""
        async def test():
            future = FastFuture(exception=RuntimeError("fail"))
            return await future
        
        with self.assertRaises(RuntimeError):
            asyncio.run(test())


class TestCoroutinePool(unittest.TestCase):
    """Test CoroutinePool class"""
    
    def test_pool_creation(self):
        """Test creating pool"""
        pool = CoroutinePool(max_size=10)
        self.assertEqual(pool.max_size, 10)
        self.assertEqual(len(pool.pool), 0)
    
    def test_pool_reuse(self):
        """Test coroutine reuse"""
        pool = CoroutinePool(max_size=10)
        
        async def dummy():
            return 42
        
        # Get wrapper
        wrapper1 = pool.get_wrapper(dummy())
        self.assertEqual(pool.stats.pooled_coroutines, 1)
        self.assertEqual(pool.stats.coroutine_reuses, 0)
        
        # Release it
        pool.release_wrapper(wrapper1)
        self.assertEqual(len(pool.pool), 1)
        
        # Get another - should reuse
        wrapper2 = pool.get_wrapper(dummy())
        self.assertEqual(pool.stats.pooled_coroutines, 2)
        self.assertEqual(pool.stats.coroutine_reuses, 1)
    
    def test_pool_max_size(self):
        """Test pool respects max size"""
        pool = CoroutinePool(max_size=2)
        
        async def dummy():
            return 42
        
        # Release 3 wrappers
        wrappers = []
        for i in range(3):
            coro = dummy()
            wrapper = pool.get_wrapper(coro)
            wrappers.append(wrapper)
            # Close coroutine before releasing
            coro.close()
        
        for wrapper in wrappers:
            pool.release_wrapper(wrapper)
        
        # Pool should only have 2 (deque maxlen limits it)
        self.assertLessEqual(len(pool.pool), 2)
    
    def test_pool_clear(self):
        """Test clearing pool"""
        pool = CoroutinePool(max_size=10)
        
        async def dummy():
            return 42
        
        wrappers = []
        for i in range(5):
            coro = dummy()
            wrapper = pool.get_wrapper(coro)
            wrappers.append(wrapper)
            # Close coroutine before releasing
            coro.close()
        
        for wrapper in wrappers:
            pool.release_wrapper(wrapper)
        
        self.assertGreater(len(pool.pool), 0)
        
        pool.clear()
        self.assertEqual(len(pool.pool), 0)


class TestBatchAwaitDetector(unittest.TestCase):
    """Test BatchAwaitDetector class"""
    
    def test_batch_detection(self):
        """Test detecting batch of awaits"""
        detector = BatchAwaitDetector()
        
        async def dummy(x):
            return x
        
        # Add first await - not enough for batch
        batch = detector.add_await(dummy(1))
        self.assertIsNone(batch)
        
        # Add second await - triggers batch
        batch = detector.add_await(dummy(2))
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 2)
    
    def test_flush(self):
        """Test flushing pending awaits"""
        detector = BatchAwaitDetector()
        
        async def dummy(x):
            return x
        
        detector.add_await(dummy(1))
        
        batch = detector.flush()
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(detector.pending_awaits), 0)


class TestAsyncOptimizer(unittest.TestCase):
    """Test AsyncOptimizer class"""
    
    def test_optimizer_creation(self):
        """Test creating optimizer"""
        optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
        self.assertEqual(optimizer.level, AsyncOptimizationLevel.MODERATE)
        self.assertIsNotNone(optimizer.coroutine_pool)
    
    def test_spawn_with_pooling(self):
        """Test spawning with coroutine pooling"""
        optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.BASIC)
        
        async def dummy():
            return 42
        
        coro = dummy()
        wrapped = optimizer.spawn(coro)
        
        self.assertEqual(optimizer.stats.total_spawns, 1)
        self.assertIsNotNone(wrapped)
    
    def test_await_optimized_fast_path(self):
        """Test optimized await with fast path"""
        async def test():
            optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
            
            # Create fast future
            future = FastFuture(value=123)
            result = await optimizer.await_optimized(future)
            
            self.assertEqual(result, 123)
            self.assertEqual(optimizer.stats.total_awaits, 1)
            self.assertEqual(optimizer.stats.fast_path_hits, 1)
            self.assertEqual(optimizer.stats.event_loop_skips, 1)
        
        asyncio.run(test())
    
    def test_await_optimized_done_future(self):
        """Test optimized await with done asyncio Future"""
        async def test():
            optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
            
            # Create done future
            future = asyncio.Future()
            future.set_result(456)
            
            result = await optimizer.await_optimized(future)
            
            self.assertEqual(result, 456)
            self.assertEqual(optimizer.stats.fast_path_hits, 1)
        
        asyncio.run(test())
    
    def test_create_resolved_future(self):
        """Test creating resolved future"""
        optimizer = AsyncOptimizer()
        future = optimizer.create_resolved_future(789)
        
        self.assertIsInstance(future, FastFuture)
        self.assertEqual(future.result(), 789)
    
    def test_create_rejected_future(self):
        """Test creating rejected future"""
        optimizer = AsyncOptimizer()
        exc = ValueError("test")
        future = optimizer.create_rejected_future(exc)
        
        self.assertIsInstance(future, FastFuture)
        self.assertEqual(future.exception(), exc)
    
    def test_gather_batch_single(self):
        """Test gathering single coroutine"""
        async def test():
            optimizer = AsyncOptimizer()
            
            async def dummy():
                return 42
            
            results = await optimizer.gather_batch([dummy()])
            self.assertEqual(results, [42])
        
        asyncio.run(test())
    
    def test_gather_batch_multiple(self):
        """Test gathering multiple coroutines"""
        async def test():
            optimizer = AsyncOptimizer()
            
            async def dummy(x):
                await asyncio.sleep(0)  # Yield control
                return x * 2
            
            results = await optimizer.gather_batch([
                dummy(1),
                dummy(2),
                dummy(3)
            ])
            
            self.assertEqual(results, [2, 4, 6])
            self.assertEqual(optimizer.stats.batched_operations, 3)
        
        asyncio.run(test())
    
    def test_gather_batch_empty(self):
        """Test gathering empty batch"""
        async def test():
            optimizer = AsyncOptimizer()
            results = await optimizer.gather_batch([])
            self.assertEqual(results, [])
        
        asyncio.run(test())
    
    def test_get_stats(self):
        """Test getting statistics"""
        optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
        
        stats = optimizer.get_stats()
        self.assertIn('total_spawns', stats)
        self.assertIn('total_awaits', stats)
        self.assertIn('pool_stats', stats)
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        async def test():
            optimizer = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
            
            # Do some operations
            future = FastFuture(value=123)
            await optimizer.await_optimized(future)
            
            self.assertEqual(optimizer.stats.total_awaits, 1)
            
            # Reset
            optimizer.reset_stats()
            self.assertEqual(optimizer.stats.total_awaits, 0)
        
        asyncio.run(test())
    
    def test_optimization_levels(self):
        """Test different optimization levels"""
        # NONE - no pooling
        opt_none = AsyncOptimizer(level=AsyncOptimizationLevel.NONE)
        self.assertIsNone(opt_none.coroutine_pool)
        self.assertIsNone(opt_none.batch_detector)
        
        # BASIC - pooling only
        opt_basic = AsyncOptimizer(level=AsyncOptimizationLevel.BASIC)
        self.assertIsNotNone(opt_basic.coroutine_pool)
        self.assertIsNone(opt_basic.batch_detector)
        
        # MODERATE - pooling + fast paths
        opt_moderate = AsyncOptimizer(level=AsyncOptimizationLevel.MODERATE)
        self.assertIsNotNone(opt_moderate.coroutine_pool)
        self.assertIsNone(opt_moderate.batch_detector)
        
        # AGGRESSIVE - all optimizations
        opt_aggressive = AsyncOptimizer(level=AsyncOptimizationLevel.AGGRESSIVE)
        self.assertIsNotNone(opt_aggressive.coroutine_pool)
        self.assertIsNotNone(opt_aggressive.batch_detector)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_is_coroutine_like(self):
        """Test coroutine detection"""
        async def coro():
            return 42
        
        # Test coroutine
        c = coro()
        self.assertTrue(is_coroutine_like(c))
        c.close()  # Clean up
        
        # Test FastFuture
        self.assertTrue(is_coroutine_like(FastFuture(42)))
        
        # Test non-coroutines
        self.assertFalse(is_coroutine_like(42))
        self.assertFalse(is_coroutine_like("string"))
    
    def test_is_immediately_available(self):
        """Test checking if awaitable is ready"""
        # FastFuture is always available
        self.assertTrue(is_immediately_available(FastFuture(42)))
        
        # Non-futures are not available
        self.assertFalse(is_immediately_available(42))
    
    def test_fast_await(self):
        """Test fast await utility"""
        async def test():
            # Fast path for FastFuture
            result1 = await fast_await(FastFuture(123))
            self.assertEqual(result1, 123)
            
            # Fast path for done Future
            future = asyncio.Future()
            future.set_result(456)
            result2 = await fast_await(future)
            self.assertEqual(result2, 456)
            
            # Regular await for coroutine
            async def coro():
                return 789
            result3 = await fast_await(coro())
            self.assertEqual(result3, 789)
        
        asyncio.run(test())


class TestPerformance(unittest.TestCase):
    """Performance tests (informational only)"""
    
    def test_fast_future_vs_regular(self):
        """Compare FastFuture vs asyncio.Future performance"""
        import time
        
        async def test_fast_future():
            for _ in range(1000):
                future = FastFuture(value=42)
                result = await future
        
        async def test_regular_future():
            for _ in range(1000):
                future = asyncio.Future()
                future.set_result(42)
                result = await future
        
        # Time FastFuture
        start = time.perf_counter()
        asyncio.run(test_fast_future())
        fast_time = time.perf_counter() - start
        
        # Time regular Future
        start = time.perf_counter()
        asyncio.run(test_regular_future())
        regular_time = time.perf_counter() - start
        
        print(f"\nFastFuture: {fast_time:.4f}s")
        print(f"Regular Future: {regular_time:.4f}s")
        
        if fast_time < regular_time:
            speedup = regular_time / fast_time
            print(f"FastFuture is {speedup:.2f}x faster")
        
        # FastFuture should be competitive at minimum
        self.assertLess(fast_time, regular_time * 2)


if __name__ == '__main__':
    unittest.main()
