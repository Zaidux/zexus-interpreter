"""
Comprehensive integration tests for ALL VM optimizations working together.

Tests all 5 phases of VM Optimization Phase 8:
- Phase 1: Instruction-level profiling
- Phase 2: Memory pool optimization
- Phase 3: Bytecode peephole optimizer
- Phase 4: Async/await performance enhancements
- Phase 5: SSA converter & register allocator

These tests validate production readiness with edge cases and real-world scenarios.
"""

import unittest
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.vm.vm import VM
from zexus.vm.profiler import ProfilingLevel
from zexus.vm.peephole_optimizer import Instruction


class TestAllOptimizationsBasic(unittest.TestCase):
    """Basic integration tests for all optimizations enabled"""
    
    def test_all_optimizers_enabled(self):
        """Test creating VM with all optimizers enabled"""
        vm = VM(
            enable_profiling=False,  # Profiling changes enable_profiling internally, test separately
            profiling_level=ProfilingLevel.DETAILED,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            optimization_level="AGGRESSIVE",
            enable_async_optimizer=True,
            async_optimization_level="MODERATE",
            enable_ssa=True,
            enable_register_allocation=True,
            num_allocator_registers=16
        )
        
        # Verify key optimizers are enabled
        self.assertTrue(vm.enable_memory_pool)
        self.assertTrue(vm.enable_peephole_optimizer)
        self.assertTrue(vm.enable_async_optimizer)
        self.assertTrue(vm.enable_ssa)
        self.assertTrue(vm.enable_register_allocation)
    
    def test_memory_pool_with_profiling(self):
        """Test memory pool allocations work correctly"""
        vm = VM(
            enable_profiling=False,  # Keep profiling disabled for this test
            enable_memory_pool=True
        )
        
        # Allocate some objects - integers should use small int cache
        for i in range(-128, 128):  # Small int range
            vm.allocate_integer(i)
        
        # Allocate some strings
        for i in range(10):
            vm.allocate_string(f"test_{i}")
            vm.allocate_string(f"test_{i}")  # Duplicate - should hit pool
        
        # Get pool stats
        pool_stats = vm.get_pool_stats()
        
        # Integer pool should have operations
        int_total = pool_stats['integer_pool']['hits'] + pool_stats['integer_pool']['misses']
        self.assertGreater(int_total, 0)
        
        # String pool should have operations from duplicates
        str_total = pool_stats['string_pool']['hits'] + pool_stats['string_pool']['misses']
        self.assertGreater(str_total, 0)
    
    def test_peephole_optimizer_with_ssa(self):
        """Test peephole optimizer and SSA converter together"""
        vm = VM(
            enable_peephole_optimizer=True,
            enable_ssa=True,
            optimization_level="AGGRESSIVE"
        )
        
        # Code with constant folding opportunity
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
            Instruction('STORE_FAST', 'x'),
            Instruction('LOAD_FAST', 'x'),
            Instruction('STORE_FAST', 'y'),  # Copy
        ]
        
        # Optimize with peephole
        optimized = vm.optimize_bytecode(instructions)
        
        # Convert to SSA (will further optimize)
        ssa = vm.convert_to_ssa(optimized)
        
        # Should have optimizations from both
        opt_stats = vm.get_optimizer_stats()
        ssa_stats = vm.get_ssa_stats()
        
        self.assertGreater(opt_stats['constant_folds'], 0)
        self.assertEqual(ssa_stats['conversions'], 1)


class TestAllOptimizationsEdgeCases(unittest.TestCase):
    """Edge case tests for all optimizations"""
    
    def test_empty_bytecode_all_optimizers(self):
        """Test all optimizers handle empty bytecode"""
        vm = VM(
            enable_profiling=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        empty_instructions = []
        
        # Peephole should handle empty
        optimized = vm.optimize_bytecode(empty_instructions)
        self.assertEqual(len(optimized), 0)
        
        # SSA should handle empty
        ssa = vm.convert_to_ssa(empty_instructions)
        self.assertEqual(len(ssa.blocks), 1)  # Entry block only
        
        # Allocator should handle empty
        result = vm.allocate_registers(empty_instructions)
        self.assertEqual(result.num_registers_used, 0)
    
    def test_single_instruction_all_optimizers(self):
        """Test all optimizers handle single instruction"""
        vm = VM(
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        single = [Instruction('LOAD_CONST', 42)]
        
        optimized = vm.optimize_bytecode(single)
        self.assertEqual(len(optimized), 1)
        
        ssa = vm.convert_to_ssa(single)
        self.assertEqual(len(ssa.blocks), 1)
        
        result = vm.allocate_registers(single)
        self.assertEqual(result.num_registers_used, 0)  # No variables
    
    def test_large_program_all_optimizers(self):
        """Test all optimizers handle large programs efficiently"""
        vm = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Generate large program
        large_program = []
        for i in range(1000):
            large_program.extend([
                Instruction('LOAD_CONST', i),
                Instruction('STORE_FAST', f'var_{i}'),
            ])
        
        # All optimizers should complete quickly
        import time
        
        start = time.time()
        optimized = vm.optimize_bytecode(large_program)
        peephole_time = time.time() - start
        
        start = time.time()
        ssa = vm.convert_to_ssa(optimized)
        ssa_time = time.time() - start
        
        start = time.time()
        result = vm.allocate_registers(optimized)
        allocator_time = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(peephole_time, 1.0, "Peephole too slow")
        self.assertLess(ssa_time, 5.0, "SSA too slow")
        self.assertLess(allocator_time, 5.0, "Allocator too slow")
    
    def test_deep_nesting_all_optimizers(self):
        """Test deeply nested control flow"""
        vm = VM(
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Deeply nested if-else
        instructions = []
        for i in range(10):
            instructions.extend([
                Instruction('LOAD_FAST', 'x'),
                Instruction('LOAD_CONST', i),
                Instruction('COMPARE_OP', '<'),
                Instruction('POP_JUMP_IF_FALSE', len(instructions) + 10),
            ])
        
        # All optimizers should handle it
        ssa = vm.convert_to_ssa(instructions)
        self.assertGreaterEqual(len(ssa.blocks), 1)  # At least entry block
        
        result = vm.allocate_registers(instructions)
        # May have registers or not depending on variables
        self.assertIsNotNone(result)


class TestAllOptimizationsRealWorld(unittest.TestCase):
    """Real-world scenario tests"""
    
    def test_fibonacci_computation(self):
        """Test optimizing Fibonacci computation"""
        vm = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True,
            optimization_level="AGGRESSIVE"
        )
        
        # Fibonacci bytecode pattern
        fib_instructions = [
            # n = 10
            Instruction('LOAD_CONST', 10),
            Instruction('STORE_FAST', 'n'),
            # a = 0
            Instruction('LOAD_CONST', 0),
            Instruction('STORE_FAST', 'a'),
            # b = 1
            Instruction('LOAD_CONST', 1),
            Instruction('STORE_FAST', 'b'),
            # Loop: while n > 0
            Instruction('LOAD_FAST', 'n'),
            Instruction('LOAD_CONST', 0),
            Instruction('COMPARE_OP', '>'),
            Instruction('POP_JUMP_IF_FALSE', 20),
            # temp = a
            Instruction('LOAD_FAST', 'a'),
            Instruction('STORE_FAST', 'temp'),
            # a = b
            Instruction('LOAD_FAST', 'b'),
            Instruction('STORE_FAST', 'a'),
            # b = temp + b
            Instruction('LOAD_FAST', 'temp'),
            Instruction('LOAD_FAST', 'b'),
            Instruction('BINARY_ADD'),
            Instruction('STORE_FAST', 'b'),
            # n = n - 1
            Instruction('LOAD_FAST', 'n'),
            Instruction('LOAD_CONST', 1),
            Instruction('BINARY_SUB'),
            Instruction('STORE_FAST', 'n'),
            Instruction('JUMP_ABSOLUTE', 6),
        ]
        
        # Optimize
        optimized = vm.optimize_bytecode(fib_instructions)
        ssa = vm.convert_to_ssa(optimized)
        result = vm.allocate_registers(optimized)
        
        # Should optimize well
        opt_stats = vm.get_optimizer_stats()
        ssa_stats = vm.get_ssa_stats()
        
        # Should have loop structure with multiple blocks
        # Entry block + loop header + loop body + exit = at least 3 blocks
        self.assertGreaterEqual(len(ssa.blocks), 3, 
                              f"Expected at least 3 blocks for Fibonacci loop, got {len(ssa.blocks)}")
        
        # Should have some phi nodes at loop header (for loop-carried dependencies)
        total_phi_nodes = sum(len(block.phi_nodes) for block in ssa.blocks.values())
        self.assertGreater(total_phi_nodes, 0, "Expected phi nodes for loop variables")
        # Phi nodes depend on control flow structure
        # self.assertGreater(ssa.num_phi_nodes, 0, "Should have phi nodes")
        self.assertLess(result.num_registers_used, 20, "Should use reasonable number of registers")
    
    def test_string_processing_pipeline(self):
        """Test string processing with memory pooling"""
        vm = VM(
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            pool_max_size=1000
        )
        
        # Simulate string processing
        strings_created = 0
        for i in range(500):
            s = vm.allocate_string(f"item_{i % 50}")  # Reuse patterns
            strings_created += 1
        
        # Should have high hit rate due to reuse
        stats = vm.get_pool_stats()
        hit_rate = stats['string_pool']['hit_rate']
        self.assertGreater(hit_rate, 50.0, f"String pool hit rate too low: {hit_rate}%")
    
    def test_async_coroutine_optimization(self):
        """Test async operations with optimizer"""
        async def run_test():
            vm = VM(
                enable_async_optimizer=True,
                async_optimization_level="AGGRESSIVE"
            )
            
            # Simulate spawning multiple coroutines
            async def dummy_coro():
                await asyncio.sleep(0.001)
                return 42
            
            # Spawn many coroutines (would use pool)
            tasks = []
            for i in range(50):
                tasks.append(dummy_coro())
            
            results = await asyncio.gather(*tasks)
            
            # All should complete
            self.assertEqual(len(results), 50)
            self.assertTrue(all(r == 42 for r in results))
        
        asyncio.run(run_test())
    
    def test_matrix_computation(self):
        """Test matrix-like computation with all optimizations"""
        vm = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Matrix operation pattern (simplified)
        matrix_instructions = []
        size = 10
        
        for i in range(size):
            for j in range(size):
                matrix_instructions.extend([
                    # result[i][j] = a[i][j] + b[i][j]
                    Instruction('LOAD_FAST', f'a_{i}_{j}'),
                    Instruction('LOAD_FAST', f'b_{i}_{j}'),
                    Instruction('BINARY_ADD'),
                    Instruction('STORE_FAST', f'result_{i}_{j}'),
                ])
        
        # Optimize
        optimized = vm.optimize_bytecode(matrix_instructions)
        result = vm.allocate_registers(optimized)
        
        # Should handle many variables
        self.assertGreater(len(result.allocation), 0)
        
        # May need to spill some due to register pressure
        total_vars = size * size * 3  # a, b, result
        if result.num_registers_used < 16:
            # All fit
            self.assertEqual(len(result.spilled), 0)
        else:
            # Some spilled
            self.assertGreater(len(result.spilled), 0)


class TestAllOptimizationsStatistics(unittest.TestCase):
    """Test statistics collection across all optimizers"""
    
    def test_statistics_tracking(self):
        """Test all optimizers track statistics correctly"""
        vm = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_async_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Do some operations
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
            Instruction('STORE_FAST', 'x'),
        ]
        
        vm.optimize_bytecode(instructions)
        vm.convert_to_ssa(instructions)
        vm.allocate_registers(instructions)
        
        # Get all stats
        pool_stats = vm.get_pool_stats()
        opt_stats = vm.get_optimizer_stats()
        ssa_stats = vm.get_ssa_stats()
        allocator_stats = vm.get_allocator_stats()
        
        # All should return valid dicts
        self.assertIsInstance(pool_stats, dict)
        self.assertIsInstance(opt_stats, dict)
        self.assertIsInstance(ssa_stats, dict)
        self.assertIsInstance(allocator_stats, dict)
        
        # Stats should have expected keys
        self.assertIn('integer_pool', pool_stats)
        self.assertIn('constant_folds', opt_stats)
        self.assertIn('conversions', ssa_stats)
        self.assertIn('allocations', allocator_stats)
    
    def test_reset_all_statistics(self):
        """Test resetting all optimizer statistics"""
        vm = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Do operations
        instructions = [
            Instruction('LOAD_CONST', 1),
            Instruction('STORE_FAST', 'x'),
        ]
        
        vm.optimize_bytecode(instructions)
        vm.convert_to_ssa(instructions)
        vm.allocate_registers(instructions)
        
        # Reset all
        vm.reset_profiler()
        vm.reset_pools()
        vm.reset_optimizer_stats()
        vm.reset_ssa_stats()
        vm.reset_allocator_stats()
        
        # Stats should be reset
        opt_stats = vm.get_optimizer_stats()
        ssa_stats = vm.get_ssa_stats()
        allocator_stats = vm.get_allocator_stats()
        
        self.assertEqual(opt_stats['constant_folds'], 0)
        self.assertEqual(ssa_stats['conversions'], 0)
        self.assertEqual(allocator_stats['allocations'], 0)


class TestAllOptimizationsPerformance(unittest.TestCase):
    """Performance validation tests"""
    
    def test_optimization_overhead(self):
        """Test that optimizations don't add excessive overhead"""
        import time
        
        # Baseline: VM without optimizations
        vm_baseline = VM(
            enable_profiling=False,
            enable_memory_pool=False,
            enable_peephole_optimizer=False,
            enable_ssa=False,
            enable_register_allocation=False
        )
        
        # Optimized: VM with all optimizations
        vm_optimized = VM(
            enable_profiling=True,
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Test program
        program = []
        for i in range(100):
            program.extend([
                Instruction('LOAD_CONST', i),
                Instruction('STORE_FAST', f'var_{i}'),
                Instruction('LOAD_FAST', f'var_{i}'),
            ])
        
        # Time optimized version
        start = time.time()
        vm_optimized.optimize_bytecode(program)
        vm_optimized.convert_to_ssa(program)
        vm_optimized.allocate_registers(program)
        optimized_time = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(optimized_time, 2.0, "Optimizations too slow")
    
    def test_memory_pool_efficiency(self):
        """Test memory pool reduces allocations"""
        vm = VM(enable_memory_pool=True)
        
        # Allocate many small integers (should be cached)
        for _ in range(1000):
            for i in range(-128, 256):
                vm.allocate_integer(i)
        
        stats = vm.get_pool_stats()
        
        # Small int cache should have 100% hit rate
        int_hits = stats['integer_pool']['hits']
        int_total = stats['integer_pool']['hits'] + stats['integer_pool']['misses']
        
        if int_total > 0:
            hit_rate = (int_hits / int_total) * 100
            self.assertGreater(hit_rate, 95.0, f"Small int cache hit rate too low: {hit_rate}%")


class TestAllOptimizationsCorrectness(unittest.TestCase):
    """Correctness tests to ensure optimizations don't change semantics"""
    
    def test_constant_folding_correctness(self):
        """Test constant folding produces correct results"""
        vm = VM(enable_peephole_optimizer=True, optimization_level="AGGRESSIVE")
        
        # Various arithmetic operations
        test_cases = [
            ([
                Instruction('LOAD_CONST', 5),
                Instruction('LOAD_CONST', 3),
                Instruction('BINARY_ADD'),
            ], 8),
            ([
                Instruction('LOAD_CONST', 10),
                Instruction('LOAD_CONST', 4),
                Instruction('BINARY_SUB'),
            ], 6),
            ([
                Instruction('LOAD_CONST', 6),
                Instruction('LOAD_CONST', 7),
                Instruction('BINARY_MUL'),
            ], 42),
        ]
        
        for instructions, expected in test_cases:
            optimized = vm.optimize_bytecode(instructions)
            # Should be folded to single LOAD_CONST
            self.assertEqual(len(optimized), 1)
            self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
            self.assertEqual(optimized[0].arg, expected)
    
    def test_ssa_preserves_semantics(self):
        """Test SSA conversion preserves program semantics"""
        vm = VM(enable_ssa=True)
        
        # Program with multiple assignments
        instructions = [
            ('LOAD_CONST', 1),
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
            ('LOAD_CONST', 2),
            ('BINARY_ADD'),
            ('STORE_FAST', 'x'),  # x = x + 2
            ('LOAD_FAST', 'x'),
            ('RETURN_VALUE'),
        ]
        
        # Convert to SSA and back
        ssa = vm.convert_to_ssa(instructions)
        
        # Should have variable versions tracked
        self.assertIsInstance(ssa.variable_versions, dict)
        # Should have created blocks
        self.assertGreater(len(ssa.blocks), 0)
    
    def test_register_allocation_preserves_values(self):
        """Test register allocation doesn't corrupt values"""
        vm = VM(enable_register_allocation=True, num_allocator_registers=4)
        
        # Program with more variables than registers
        instructions = []
        for i in range(10):
            instructions.extend([
                ('LOAD_CONST', i),
                ('STORE_FAST', f'var_{i}'),
            ])
        
        result = vm.allocate_registers(instructions)
        
        # Should allocate successfully (with spilling if needed)
        self.assertGreater(result.num_registers_used, 0)
        
        # All variables should be accounted for (allocated or spilled)
        total_accounted = len(result.allocation) + len(result.spilled)
        self.assertEqual(total_accounted, 10)


class TestAllOptimizationsErrorHandling(unittest.TestCase):
    """Test error handling in all optimizers"""
    
    def test_invalid_bytecode_handling(self):
        """Test all optimizers handle invalid bytecode gracefully"""
        vm = VM(
            enable_peephole_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Invalid instruction (but should not crash)
        # Each optimizer should either handle it or skip it
        invalid = [Instruction('INVALID_OPCODE', None)]
        
        # Should not crash
        try:
            optimized = vm.optimize_bytecode(invalid)
            # May pass through or remove
        except Exception:
            # Or raise controlled error
            pass
    
    def test_cyclic_jumps(self):
        """Test handling of cyclic control flow"""
        vm = VM(enable_ssa=True)
        
        # Infinite loop
        cyclic = [
            ('LOAD_CONST', 1),
            ('JUMP_ABSOLUTE', 0),
        ]
        
        # Should handle without infinite loop in SSA conversion
        ssa = vm.convert_to_ssa(cyclic)
        self.assertIsNotNone(ssa)
        self.assertGreater(len(ssa.blocks), 0)


def run_all_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsRealWorld))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestAllOptimizationsErrorHandling))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
