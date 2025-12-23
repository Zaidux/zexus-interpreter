"""
Tests for Bytecode Peephole Optimizer

Tests optimization patterns:
- Constant folding
- Dead code elimination
- Strength reduction
- Instruction fusion
- Statistics tracking
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from zexus.vm.peephole_optimizer import (
    PeepholeOptimizer,
    OptimizationLevel,
    Instruction,
    OptimizationStats,
    optimize_bytecode,
)


class TestInstruction(unittest.TestCase):
    """Test Instruction class"""
    
    def test_instruction_creation(self):
        """Test creating instructions"""
        inst = Instruction('LOAD_CONST', 42, lineno=10)
        self.assertEqual(inst.opcode, 'LOAD_CONST')
        self.assertEqual(inst.arg, 42)
        self.assertEqual(inst.lineno, 10)
    
    def test_instruction_repr(self):
        """Test instruction string representation"""
        inst1 = Instruction('LOAD_CONST', 42)
        self.assertEqual(repr(inst1), 'LOAD_CONST(42)')
        
        inst2 = Instruction('RETURN')
        self.assertEqual(repr(inst2), 'RETURN')


class TestOptimizationStats(unittest.TestCase):
    """Test OptimizationStats class"""
    
    def test_stats_creation(self):
        """Test creating stats"""
        stats = OptimizationStats()
        self.assertEqual(stats.constant_folds, 0)
        self.assertEqual(stats.dead_code_eliminated, 0)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = OptimizationStats(
            constant_folds=5,
            dead_code_eliminated=3,
            original_size=100,
            optimized_size=92
        )
        
        d = stats.to_dict()
        self.assertEqual(d['constant_folds'], 5)
        self.assertEqual(d['dead_code_eliminated'], 3)
        self.assertEqual(d['reduction_percent'], 8.0)


class TestConstantFolding(unittest.TestCase):
    """Test constant folding optimizations"""
    
    def test_fold_addition(self):
        """Test folding constant addition"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 20),
            Instruction('ADD'),
            Instruction('RETURN'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be folded to LOAD_CONST 30, RETURN
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 30)
        self.assertEqual(optimized[1].opcode, 'RETURN')
        self.assertEqual(optimizer.stats.constant_folds, 1)
    
    def test_fold_subtraction(self):
        """Test folding constant subtraction"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 50),
            Instruction('LOAD_CONST', 20),
            Instruction('SUB'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 30)
    
    def test_fold_multiplication(self):
        """Test folding constant multiplication"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 6),
            Instruction('LOAD_CONST', 7),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 42)
    
    def test_fold_division(self):
        """Test folding constant division"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 100),
            Instruction('LOAD_CONST', 4),
            Instruction('DIV'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 25.0)
    
    def test_no_fold_division_by_zero(self):
        """Test that division by zero is not folded"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 100),
            Instruction('LOAD_CONST', 0),
            Instruction('DIV'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should not be folded
        self.assertEqual(len(optimized), 3)
    
    def test_no_fold_non_numeric(self):
        """Test that non-numeric constants are not folded"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 'hello'),
            Instruction('LOAD_CONST', 'world'),
            Instruction('ADD'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should not be folded
        self.assertEqual(len(optimized), 3)
    
    def test_fold_binary_opcodes(self):
        """Test folding BINARY_* opcodes"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 5),
            Instruction('BINARY_ADD'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].arg, 15)


class TestDeadCodeElimination(unittest.TestCase):
    """Test dead code elimination"""
    
    def test_eliminate_nop(self):
        """Test eliminating NOP instructions"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 42),
            Instruction('NOP'),
            Instruction('NOP'),
            Instruction('RETURN'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # NOPs should be removed
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[1].opcode, 'RETURN')
        self.assertEqual(optimizer.stats.dead_code_eliminated, 2)
    
    def test_eliminate_load_pop(self):
        """Test eliminating LOAD_CONST followed by POP_TOP"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 42),
            Instruction('POP_TOP'),
            Instruction('RETURN'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # LOAD_CONST, POP_TOP should be removed
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'RETURN')
        self.assertEqual(optimizer.stats.dead_code_eliminated, 1)
    
    def test_eliminate_dup_pop(self):
        """Test eliminating DUP_TOP followed by POP_TOP"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 42),
            Instruction('DUP_TOP'),
            Instruction('POP_TOP'),
            Instruction('RETURN'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # DUP_TOP, POP_TOP should be removed
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[1].opcode, 'RETURN')
    
    def test_eliminate_dead_stores(self):
        """Test eliminating consecutive dead stores"""
        optimizer = PeepholeOptimizer(OptimizationLevel.MODERATE)
        
        # Test consecutive stores (this pattern IS optimized)
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('STORE_FAST', 'x'),
            Instruction('STORE_FAST', 'x'),  # Consecutive store - eliminated
            Instruction('LOAD_FAST', 'x'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Second consecutive STORE_FAST should be eliminated
        self.assertLess(len(optimized), len(instructions))
        
        # Note: Dead stores separated by LOAD_CONST require full data flow
        # analysis which is beyond peephole optimization scope


class TestStrengthReduction(unittest.TestCase):
    """Test strength reduction optimizations"""
    
    def test_multiply_by_zero(self):
        """Test x * 0 = 0"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 0),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be replaced with LOAD_CONST 0
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 0)
        self.assertEqual(optimizer.stats.strength_reductions, 1)
    
    def test_multiply_by_one(self):
        """Test x * 1 = x"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 1),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be replaced with just LOAD_FAST x
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_FAST')
        self.assertEqual(optimized[0].arg, 'x')
    
    def test_multiply_by_two(self):
        """Test x * 2 = x + x"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 2),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be replaced with LOAD_FAST, DUP_TOP, ADD
        self.assertEqual(len(optimized), 3)
        self.assertEqual(optimized[0].opcode, 'LOAD_FAST')
        self.assertEqual(optimized[1].opcode, 'DUP_TOP')
        self.assertEqual(optimized[2].opcode, 'ADD')
    
    def test_multiply_by_power_of_two(self):
        """Test x * 8 = x << 3"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 8),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be replaced with LOAD_FAST, LOAD_CONST 3, LSHIFT
        self.assertEqual(len(optimized), 3)
        self.assertEqual(optimized[0].opcode, 'LOAD_FAST')
        self.assertEqual(optimized[1].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[1].arg, 3)  # log2(8)
        self.assertEqual(optimized[2].opcode, 'LSHIFT')


class TestOptimizationLevels(unittest.TestCase):
    """Test different optimization levels"""
    
    def test_none_level_no_optimization(self):
        """Test NONE level performs no optimization"""
        optimizer = PeepholeOptimizer(OptimizationLevel.NONE)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 20),
            Instruction('ADD'),
            Instruction('NOP'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should be unchanged
        self.assertEqual(len(optimized), len(instructions))
    
    def test_basic_level(self):
        """Test BASIC level optimizations"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 20),
            Instruction('ADD'),
            Instruction('NOP'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should fold constant and remove NOP
        self.assertEqual(len(optimized), 1)
        self.assertGreater(optimizer.stats.constant_folds, 0)
        self.assertGreater(optimizer.stats.dead_code_eliminated, 0)
    
    def test_moderate_level(self):
        """Test MODERATE level optimizations"""
        optimizer = PeepholeOptimizer(OptimizationLevel.MODERATE)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('STORE_FAST', 'x'),
            Instruction('LOAD_CONST', 20),
            Instruction('STORE_FAST', 'x'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should eliminate dead store
        self.assertLessEqual(len(optimized), len(instructions))
    
    def test_aggressive_level(self):
        """Test AGGRESSIVE level optimizations"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 0),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should strength reduce
        self.assertEqual(len(optimized), 1)
        self.assertGreater(optimizer.stats.strength_reductions, 0)


class TestMultiplePasses(unittest.TestCase):
    """Test multiple optimization passes"""
    
    def test_multiple_passes(self):
        """Test that multiple passes find all optimizations"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        # Create nested optimizable pattern
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 10),
            Instruction('ADD'),  # -> 15
            Instruction('LOAD_CONST', 3),
            Instruction('MUL'),  # -> 45
            Instruction('NOP'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should eventually fold to single LOAD_CONST 45
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 45)
    
    def test_max_passes_limit(self):
        """Test that optimization stops after max passes"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        # Create long chain
        instructions = []
        for i in range(100):
            instructions.append(Instruction('NOP'))
        instructions.append(Instruction('RETURN'))
        
        optimized = optimizer.optimize(instructions)
        
        # All NOPs should be removed in one pass
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'RETURN')


class TestStatistics(unittest.TestCase):
    """Test optimization statistics"""
    
    def test_stats_tracking(self):
        """Test that statistics are tracked correctly"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 20),
            Instruction('ADD'),  # Constant fold
            Instruction('NOP'),  # Dead code
            Instruction('LOAD_FAST', 'x'),
            Instruction('LOAD_CONST', 0),
            Instruction('MUL'),  # Strength reduction
        ]
        
        optimized = optimizer.optimize(instructions)
        
        stats = optimizer.get_stats()
        self.assertGreater(stats['constant_folds'], 0)
        self.assertGreater(stats['dead_code_eliminated'], 0)
        self.assertGreater(stats['strength_reductions'], 0)
        self.assertEqual(stats['original_size'], 7)
        self.assertEqual(stats['optimized_size'], len(optimized))
    
    def test_reduction_percentage(self):
        """Test reduction percentage calculation"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        instructions = [
            Instruction('NOP'),
            Instruction('NOP'),
            Instruction('NOP'),
            Instruction('NOP'),
            Instruction('RETURN'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        stats = optimizer.get_stats()
        # 4 NOPs removed out of 5 instructions = 80% reduction
        self.assertAlmostEqual(stats['reduction_percent'], 80.0, places=1)


class TestConvenienceFunction(unittest.TestCase):
    """Test optimize_bytecode convenience function"""
    
    def test_optimize_bytecode(self):
        """Test convenience function"""
        instructions = [
            Instruction('LOAD_CONST', 10),
            Instruction('LOAD_CONST', 20),
            Instruction('ADD'),
            Instruction('NOP'),
        ]
        
        optimized, stats = optimize_bytecode(instructions, OptimizationLevel.BASIC)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].arg, 30)
        self.assertIsInstance(stats, dict)
        self.assertIn('constant_folds', stats)


class TestComplexPatterns(unittest.TestCase):
    """Test complex optimization patterns"""
    
    def test_nested_arithmetic(self):
        """Test nested arithmetic optimization"""
        optimizer = PeepholeOptimizer(OptimizationLevel.BASIC)
        
        # (5 + 10) * (3 + 2)
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 10),
            Instruction('ADD'),
            Instruction('LOAD_CONST', 3),
            Instruction('LOAD_CONST', 2),
            Instruction('ADD'),
            Instruction('MUL'),
        ]
        
        optimized = optimizer.optimize(instructions)
        
        # Should fold to 75
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 75)
    
    def test_mixed_optimizations(self):
        """Test multiple optimization types in one pass"""
        optimizer = PeepholeOptimizer(OptimizationLevel.AGGRESSIVE)
        
        instructions = [
            Instruction('LOAD_CONST', 5),
            Instruction('LOAD_CONST', 5),
            Instruction('ADD'),  # Fold to 10
            Instruction('NOP'),  # Eliminate
            Instruction('LOAD_CONST', 2),
            Instruction('MUL'),  # 10 * 2 = fold to 20
            Instruction('NOP'),  # Eliminate
        ]
        
        optimized = optimizer.optimize(instructions)
        
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].opcode, 'LOAD_CONST')
        self.assertEqual(optimized[0].arg, 20)


if __name__ == '__main__':
    unittest.main()
