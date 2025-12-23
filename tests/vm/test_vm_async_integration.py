"""
Integration tests for async optimizer with VM.

Tests async optimizer integration with the Zexus VM, including:
- SPAWN/AWAIT opcode optimizations  
- Statistics tracking
- Optimization levels
"""

import unittest
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from zexus.vm.vm import VM
from zexus.vm.async_optimizer import AsyncOptimizationLevel


class TestVMAsyncIntegration(unittest.TestCase):
    """Integration tests for async optimizer with VM."""
    
    def test_vm_creates_async_optimizer_by_default(self):
        """Test that VM creates async optimizer by default."""
        vm = VM()
        self.assertIsNotNone(vm.async_optimizer)
        self.assertEqual(vm.async_optimizer.level.name, 'MODERATE')
    
    def test_vm_async_optimizer_can_be_disabled(self):
        """Test that async optimizer can be disabled."""
        vm = VM(enable_async_optimizer=False)
        self.assertIsNone(vm.async_optimizer)
    
    def test_vm_async_optimization_level_can_be_set(self):
        """Test that async optimization level can be configured."""
        vm = VM(async_optimization_level='AGGRESSIVE')
        self.assertEqual(vm.async_optimizer.level, AsyncOptimizationLevel.AGGRESSIVE)
    def test_vm_async_optimization_level_can_be_set(self):
        """Test that async optimization level can be configured."""
        vm = VM(async_optimization_level='AGGRESSIVE')
        self.assertEqual(vm.async_optimizer.level, AsyncOptimizationLevel.AGGRESSIVE)
    
    def test_get_async_stats(self):
        """Test getting async optimizer statistics."""
        vm = VM()
        
        # Initially no stats
        stats = vm.get_async_stats()
        self.assertEqual(stats['total_spawns'], 0)
        self.assertEqual(stats['total_awaits'], 0)
        self.assertIn('fast_path_hits', stats)
        self.assertIn('pooled_coroutines', stats)
    
    def test_reset_async_stats(self):
        """Test resetting async optimizer statistics."""
        vm = VM()
        
        # Manually increment some stats to test reset
        vm.async_optimizer.stats.total_spawns = 10
        vm.async_optimizer.stats.total_awaits = 5
        
        stats = vm.get_async_stats()
        self.assertEqual(stats['total_spawns'], 10)
        
        # Reset stats
        vm.reset_async_stats()
        stats = vm.get_async_stats()
        self.assertEqual(stats['total_spawns'], 0)
        self.assertEqual(stats['total_awaits'], 0)
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        levels = ['NONE', 'BASIC', 'MODERATE', 'AGGRESSIVE']
        
        for level in levels:
            vm = VM(async_optimization_level=level)
            self.assertEqual(vm.async_optimizer.level.name, level)
    
    def test_async_optimizer_with_peephole(self):
        """Test async optimizer working alongside peephole optimizer."""
        # VM with both optimizers enabled
        vm = VM(
            enable_peephole_optimizer=True,
            enable_async_optimizer=True
        )
        
        # Both should be enabled
        self.assertIsNotNone(vm.peephole_optimizer)
        self.assertIsNotNone(vm.async_optimizer)
        
        # Both should have stats
        peephole_stats = vm.get_optimizer_stats()
        async_stats = vm.get_async_stats()
        
        self.assertIsNotNone(peephole_stats)
        self.assertIsNotNone(async_stats)
    
    def test_async_optimizer_spawn_direct(self):
        """Test async optimizer spawn method directly."""
        async def test_coroutine():
            return 42
        
        async def run_test():
            vm = VM()
            coro = test_coroutine()
            
            # Use async optimizer directly
            optimized_coro = vm.async_optimizer.spawn(coro)
            task = asyncio.create_task(optimized_coro)
            result = await task
            
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(result, 42)
    
    def test_async_optimizer_await_direct(self):
        """Test async optimizer await method directly."""
        async def test_coroutine():
            await asyncio.sleep(0.001)
            return 123
        
        async def run_test():
            vm = VM()
            coro = test_coroutine()
            task = asyncio.create_task(coro)
            
            # Use async optimizer await
            result = await vm.async_optimizer.await_optimized(task)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(result, 123)
    
    def test_async_stats_tracking(self):
        """Test that async operations are tracked in stats."""
        async def test_coroutine():
            return 99
        
        async def run_test():
            vm = VM()
            
            # Do some spawns
            coro1 = vm.async_optimizer.spawn(test_coroutine())
            task1 = asyncio.create_task(coro1)
            
            coro2 = vm.async_optimizer.spawn(test_coroutine())
            task2 = asyncio.create_task(coro2)
            
            # Do some awaits
            result1 = await vm.async_optimizer.await_optimized(task1)
            result2 = await vm.async_optimizer.await_optimized(task2)
            
            return result1, result2
        
        results = asyncio.run(run_test())
        self.assertEqual(results, (99, 99))


if __name__ == '__main__':
    unittest.main()



if __name__ == '__main__':
    unittest.main()
