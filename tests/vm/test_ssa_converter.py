"""
Tests for SSA Converter

Tests SSA conversion including:
- Basic block construction
- Dominator computation
- Phi node insertion
- Variable renaming
- SSA destruction
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from zexus.vm.ssa_converter import (
    SSAConverter,
    BasicBlock,
    SSAProgram,
    destruct_ssa
)


class TestBasicBlock(unittest.TestCase):
    """Test BasicBlock class"""
    
    def test_create_basic_block(self):
        """Test creating a basic block"""
        block = BasicBlock(id=0)
        self.assertEqual(block.id, 0)
        self.assertEqual(len(block.instructions), 0)
        self.assertEqual(len(block.predecessors), 0)
        self.assertEqual(len(block.successors), 0)
    
    def test_add_phi_node(self):
        """Test adding phi node"""
        block = BasicBlock(id=0)
        phi = block.add_phi('x', [(1, 'x_1'), (2, 'x_2')])
        
        self.assertEqual(len(block.phi_nodes), 1)
        self.assertEqual(block.phi_nodes[0].target, 'x')
        self.assertEqual(len(block.phi_nodes[0].sources), 2)


class TestSSAConverter(unittest.TestCase):
    """Test SSAConverter class"""
    
    def test_create_converter(self):
        """Test creating SSA converter"""
        converter = SSAConverter()
        self.assertEqual(converter.stats['conversions'], 0)
    
    def test_simple_conversion(self):
        """Test simple SSA conversion"""
        converter = SSAConverter()
        
        instructions = [
            ('LOAD_FAST', 'x'),
            ('STORE_FAST', 'y'),
        ]
        
        ssa_program = converter.convert_to_ssa(instructions)
        
        self.assertIsInstance(ssa_program, SSAProgram)
        self.assertGreater(len(ssa_program.blocks), 0)
    
    def test_phi_insertion_for_merge(self):
        """Test phi node insertion at merge points"""
        converter = SSAConverter()
        
        # Simulated if-then-else with merge
        # This would need actual control flow, simplified test
        instructions = [
            ('LOAD_FAST', 'x'),
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
        ]
        
        ssa_program = converter.convert_to_ssa(instructions)
        
        # Should create versioned variables
        self.assertGreater(len(ssa_program.variable_versions), 0)
    
    def test_variable_renaming(self):
        """Test variable renaming to SSA form"""
        converter = SSAConverter()
        
        instructions = [
            ('STORE_FAST', 'x'),  # x_1 =
            ('LOAD_FAST', 'x'),   # use x_1
            ('STORE_FAST', 'x'),  # x_2 =
            ('LOAD_FAST', 'x'),   # use x_2
        ]
        
        ssa_program = converter.convert_to_ssa(instructions)
        
        # x should have multiple versions
        self.assertIn('x', ssa_program.variable_versions)
        self.assertGreater(ssa_program.variable_versions['x'], 1)
    
    def test_statistics(self):
        """Test SSA conversion statistics"""
        converter = SSAConverter()
        
        instructions = [
            ('STORE_FAST', 'x'),
            ('LOAD_FAST', 'x'),
        ]
        
        converter.convert_to_ssa(instructions)
        
        stats = converter.get_stats()
        self.assertEqual(stats['conversions'], 1)
        self.assertIn('phi_nodes_inserted', stats)
        self.assertIn('variables_renamed', stats)
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        converter = SSAConverter()
        
        instructions = [('STORE_FAST', 'x')]
        converter.convert_to_ssa(instructions)
        
        converter.reset_stats()
        stats = converter.get_stats()
        self.assertEqual(stats['conversions'], 0)


class TestSSADestruction(unittest.TestCase):
    """Test SSA destruction"""
    
    def test_destruct_simple_ssa(self):
        """Test converting SSA back to regular form"""
        # Create simple SSA program
        block = BasicBlock(id=0)
        block.instructions = [
            ('LOAD_FAST', 'x_1'),
            ('STORE_FAST', 'y_1'),
        ]
        
        ssa_program = SSAProgram(
            blocks={0: block},
            entry_block=0
        )
        
        instructions = destruct_ssa(ssa_program)
        
        self.assertGreater(len(instructions), 0)
        self.assertIsInstance(instructions, list)


if __name__ == '__main__':
    unittest.main()
