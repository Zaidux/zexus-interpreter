"""Integration tests for VM with Peephole Optimizer."""

import unittest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from zexus.vm.vm import VM
from zexus.vm.peephole_optimizer import Instruction, OptimizationLevel


class TestVMPeepholeIntegration(unittest.TestCase):
    """Test VM integration with peephole optimizer."""
    
    def setUp(self):
        """Create VM with peephole optimizer enabled."""
        self.vm = VM(
            enable_peephole_optimizer=True,
            optimization_level="MODERATE",
            debug=False
        )
    
    def test_optimizer_enabled(self):
        """Test that peephole optimizer is enabled."""
        self.assertTrue(self.vm.enable_peephole_optimizer)
        self.assertIsNotNone(self.vm.peephole_optimizer)
        self.assertEqual(
            self.vm.peephole_optimizer.level,
            OptimizationLevel.MODERATE
        )
    
    def test_optimize_bytecode_constant_folding(self):
        """Test optimizing bytecode with constant folding."""
        # Create instructions: 5 + 3
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
            Instruction('RETURN'),
        ]
        
        optimized = self.vm.optimize_bytecode(instructions)
        
        # Should be folded to LOAD_CONST 8
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 8)
        self.assertEqual(optimized[1].opcode, 'RETURN')
    
    def test_optimize_bytecode_dead_code(self):
        """Test optimizing bytecode with dead code elimination."""
        instructions = [
            Instruction('LOAD_CONST', 42),
            Instruction('NOP'),
            Instruction('NOP'),
            Instruction('RETURN'),
        ]
        
        optimized = self.vm.optimize_bytecode(instructions)
        
        # NOPs should be removed
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[1].opcode, 'RETURN')
    
    def test_optimize_bytecode_strength_reduction(self):
        """Test optimizing bytecode with strength reduction."""
        # Strength reduction requires AGGRESSIVE level
        vm = VM(enable_peephole_optimizer=True, optimization_level="AGGRESSIVE")
        
        # x * 0 should be replaced with 0
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 0),
            Instruction('MUL'),
            Instruction('RETURN'),
        ]
        
        optimized = vm.optimize_bytecode(instructions)
        
        # Should be replaced with LOAD_CONST 0, RETURN (strength reduction removes LOAD_FAST and MUL)
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 0)
        self.assertEqual(optimized[1].opcode, 'RETURN')
    
    def test_get_optimizer_stats(self):
        """Test getting optimizer statistics."""
        # Perform some optimizations
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
        ]
        
        self.vm.optimize_bytecode(instructions)
        
        stats = self.vm.get_optimizer_stats()
        
        # Check stats structure
        self.assertIn('constant_folds', stats)
        self.assertIn('dead_code_eliminated', stats)
        self.assertIn('original_size', stats)
        self.assertIn('optimized_size', stats)
        self.assertIn('reduction_percent', stats)
        
        # Should have at least one constant fold
        self.assertGreater(stats['constant_folds'], 0)
    
    def test_reset_optimizer_stats(self):
        """Test resetting optimizer statistics."""
        # Perform optimization
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
        ]
        
        self.vm.optimize_bytecode(instructions)
        stats1 = self.vm.get_optimizer_stats()
        self.assertGreater(stats1['constant_folds'], 0)
        
        # Reset
        self.vm.reset_optimizer_stats()
        
        # Stats should be reset
        stats2 = self.vm.get_optimizer_stats()
        self.assertEqual(stats2['constant_folds'], 0)
    
    def test_optimizer_disabled_vm(self):
        """Test VM with optimizer disabled."""
        vm = VM(enable_peephole_optimizer=False)
        
        self.assertFalse(vm.enable_peephole_optimizer)
        self.assertIsNone(vm.peephole_optimizer)
        
        # Stats should return error
        stats = vm.get_optimizer_stats()
        self.assertIn('error', stats)
        
        # Optimization should return unchanged bytecode
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
        ]
        
        optimized = vm.optimize_bytecode(instructions)
        self.assertEqual(len(optimized), len(instructions))
    
    def test_multiple_optimization_passes(self):
        """Test that optimizer runs multiple passes."""
        # This pattern requires 2 passes to fully optimize
        instructions = [
            Instruction('LOAD_CONST', 1),
            Instruction('LOAD_CONST', 2),
            Instruction('BINARY_ADD'),  # -> LOAD_CONST 3
            Instruction('LOAD_CONST', 4),
            Instruction('BINARY_ADD'),  # -> LOAD_CONST 7
            Instruction('RETURN'),
        ]
        
        optimized = self.vm.optimize_bytecode(instructions)
        
        # Should be fully optimized to LOAD_CONST 7
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 7)
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        # Basic level
        vm_basic = VM(
            enable_peephole_optimizer=True,
            optimization_level="BASIC"
        )
        self.assertEqual(
            vm_basic.peephole_optimizer.level,
            OptimizationLevel.BASIC
        )
        
        # Moderate level
        vm_moderate = VM(
            enable_peephole_optimizer=True,
            optimization_level="MODERATE"
        )
        self.assertEqual(
            vm_moderate.peephole_optimizer.level,
            OptimizationLevel.MODERATE
        )
        
        # Aggressive level
        vm_aggressive = VM(
            enable_peephole_optimizer=True,
            optimization_level="AGGRESSIVE"
        )
        self.assertEqual(
            vm_aggressive.peephole_optimizer.level,
            OptimizationLevel.AGGRESSIVE
        )
    
    def test_high_performance_vm_has_optimizer(self):
        """Test that high performance VM has optimizer enabled."""
        from zexus.vm.vm import create_high_performance_vm
        
        vm = create_high_performance_vm()
        self.assertTrue(vm.enable_peephole_optimizer)
        self.assertIsNotNone(vm.peephole_optimizer)
        # Should use aggressive optimization
        self.assertEqual(
            vm.peephole_optimizer.level,
            OptimizationLevel.AGGRESSIVE
        )


class TestOptimizationBenefits(unittest.TestCase):
    """Test real-world optimization benefits."""
    
    def test_arithmetic_optimization(self):
        """Test arithmetic expression optimization."""
        vm = VM(enable_peephole_optimizer=True, optimization_level="AGGRESSIVE")
        
        # Complex arithmetic: (5 + 3) * 2 - 1
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 3),
            Instruction('BINARY_ADD'),
            Instruction('LOAD_CONST', 2),
            Instruction('MUL'),
            Instruction('LOAD_CONST', 1),
            Instruction('BINARY_SUB'),
            Instruction('RETURN'),
        ]
        
        optimized = vm.optimize_bytecode(instructions)
        
        # Should be fully folded to LOAD_CONST 15
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 15)
        
        stats = vm.get_optimizer_stats()
        # Should have performed optimizations
        self.assertGreater(stats['constant_folds'], 0)
        self.assertGreater(stats['reduction_percent'], 50.0)
    
    def test_code_size_reduction(self):
        """Test that optimization reduces code size."""
        vm = VM(enable_peephole_optimizer=True, optimization_level="MODERATE")
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('NOP'),
            Instruction('LOAD_CONST', 5),
            Instruction('NOP'),
            Instruction('BINARY_ADD'),
            Instruction('NOP'),
            Instruction('RETURN'),
        ]
        
        optimized = vm.optimize_bytecode(instructions)
        
        # Should remove NOPs and fold constants
        self.assertLess(len(optimized), len(instructions))
        
        stats = vm.get_optimizer_stats()
        reduction = stats['reduction_percent']
        # Should have significant reduction
        self.assertGreater(reduction, 40.0)


if __name__ == '__main__':
    unittest.main()
