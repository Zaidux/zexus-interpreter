"""
Integration tests for SSA and Register Allocation with VM

Tests complete integration of Phase 8.5 optimizations:
- SSA conversion through VM interface
- Register allocation through VM interface
- Combined SSA + register allocation workflow
- Statistics tracking
- Optimization levels
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from zexus.vm.vm import VM


class TestVMSSAIntegration(unittest.TestCase):
    """Test VM integration with SSA converter"""
    
    def test_vm_ssa_disabled_by_default(self):
        """Test that SSA is disabled by default"""
        vm = VM()
        self.assertIsNone(vm.ssa_converter)
    
    def test_vm_ssa_can_be_enabled(self):
        """Test that SSA can be enabled"""
        vm = VM(enable_ssa=True)
        self.assertIsNotNone(vm.ssa_converter)
    
    def test_convert_to_ssa(self):
        """Test converting instructions to SSA"""
        vm = VM(enable_ssa=True)
        
        instructions = [
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
            ('STORE_FAST', 'y'),
        ]
        
        ssa_program = vm.convert_to_ssa(instructions)
        
        self.assertIsNotNone(ssa_program)
        self.assertGreater(len(ssa_program.blocks), 0)
    
    def test_ssa_stats(self):
        """Test getting SSA statistics"""
        vm = VM(enable_ssa=True)
        
        instructions = [
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
        ]
        
        vm.convert_to_ssa(instructions)
        
        stats = vm.get_ssa_stats()
        self.assertIn('conversions', stats)
        self.assertEqual(stats['conversions'], 1)
    
    def test_reset_ssa_stats(self):
        """Test resetting SSA statistics"""
        vm = VM(enable_ssa=True)
        
        instructions = [('STORE_FAST', 'x')]
        vm.convert_to_ssa(instructions)
        
        vm.reset_ssa_stats()
        stats = vm.get_ssa_stats()
        self.assertEqual(stats['conversions'], 0)


class TestVMRegisterAllocationIntegration(unittest.TestCase):
    """Test VM integration with register allocator"""
    
    def test_vm_allocator_disabled_by_default(self):
        """Test that register allocator is disabled by default"""
        vm = VM()
        self.assertIsNone(vm.register_allocator)
    
    def test_vm_allocator_can_be_enabled(self):
        """Test that register allocator can be enabled"""
        vm = VM(enable_register_allocation=True)
        self.assertIsNotNone(vm.register_allocator)
    
    def test_allocate_registers(self):
        """Test allocating registers"""
        vm = VM(enable_register_allocation=True)
        
        instructions = [
            ('LOAD_FAST', 'x'),
            ('LOAD_FAST', 'y'),
            ('STORE_FAST', 'z'),
        ]
        
        result = vm.allocate_registers(instructions)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.allocation)
    
    def test_allocator_stats(self):
        """Test getting register allocator statistics"""
        vm = VM(enable_register_allocation=True)
        
        instructions = [('LOAD_FAST', 'x')]
        vm.allocate_registers(instructions)
        
        stats = vm.get_allocator_stats()
        self.assertIn('allocations', stats)
        self.assertEqual(stats['allocations'], 1)
    
    def test_reset_allocator_stats(self):
        """Test resetting register allocator statistics"""
        vm = VM(enable_register_allocation=True)
        
        instructions = [('LOAD_FAST', 'x')]
        vm.allocate_registers(instructions)
        
        vm.reset_allocator_stats()
        stats = vm.get_allocator_stats()
        self.assertEqual(stats['allocations'], 0)
    
    def test_num_registers_configuration(self):
        """Test configuring number of registers"""
        vm = VM(enable_register_allocation=True, num_allocator_registers=32)
        self.assertEqual(vm.register_allocator.num_registers, 32)


class TestCombinedSSAAndAllocation(unittest.TestCase):
    """Test combined SSA and register allocation workflow"""
    
    def test_ssa_then_allocation(self):
        """Test SSA conversion followed by register allocation"""
        vm = VM(enable_ssa=True, enable_register_allocation=True)
        
        instructions = [
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
            ('STORE_FAST', 'y'),
            ('LOAD_FAST', 'y'),
        ]
        
        # Convert to SSA
        ssa_program = vm.convert_to_ssa(instructions)
        self.assertIsNotNone(ssa_program)
        
        # Allocate registers (on original instructions for simplicity)
        result = vm.allocate_registers(instructions)
        self.assertIsNotNone(result)
    
    def test_all_optimizers_together(self):
        """Test all optimizers working together"""
        vm = VM(
            enable_memory_pool=True,
            enable_peephole_optimizer=True,
            enable_async_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True,
            debug=False
        )
        
        # All should be enabled (check the right attributes)
        self.assertTrue(vm.enable_memory_pool)
        self.assertIsNotNone(vm.peephole_optimizer)
        self.assertIsNotNone(vm.async_optimizer)
        self.assertIsNotNone(vm.ssa_converter)
        self.assertIsNotNone(vm.register_allocator)
    
    def test_statistics_from_all_optimizers(self):
        """Test getting statistics from all optimizers"""
        vm = VM(
            enable_peephole_optimizer=True,
            enable_async_optimizer=True,
            enable_ssa=True,
            enable_register_allocation=True
        )
        
        # Get all stats
        peephole_stats = vm.get_optimizer_stats()
        async_stats = vm.get_async_stats()
        ssa_stats = vm.get_ssa_stats()
        allocator_stats = vm.get_allocator_stats()
        
        self.assertIsNotNone(peephole_stats)
        self.assertIsNotNone(async_stats)
        self.assertIsNotNone(ssa_stats)
        self.assertIsNotNone(allocator_stats)


if __name__ == '__main__':
    unittest.main()
