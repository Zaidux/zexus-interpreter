"""
Tests for Register Allocator

Tests graph coloring register allocation including:
- Live range computation
- Interference graph construction
- Graph coloring
- Register coalescing
- Spilling
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from zexus.vm.register_allocator import (
    RegisterAllocator,
    LiveRange,
    InterferenceGraph,
    AllocationResult,
    compute_live_ranges,
    _extract_vars
)


class TestLiveRange(unittest.TestCase):
    """Test LiveRange class"""
    
    def test_create_live_range(self):
        """Test creating a live range"""
        lr = LiveRange(variable='x', start=0, end=10)
        self.assertEqual(lr.variable, 'x')
        self.assertEqual(lr.start, 0)
        self.assertEqual(lr.end, 10)
    
    def test_overlaps_true(self):
        """Test overlapping live ranges"""
        lr1 = LiveRange(variable='x', start=0, end=10)
        lr2 = LiveRange(variable='y', start=5, end=15)
        self.assertTrue(lr1.overlaps(lr2))
        self.assertTrue(lr2.overlaps(lr1))
    
    def test_overlaps_false(self):
        """Test non-overlapping live ranges"""
        lr1 = LiveRange(variable='x', start=0, end=5)
        lr2 = LiveRange(variable='y', start=6, end=10)
        self.assertFalse(lr1.overlaps(lr2))
        self.assertFalse(lr2.overlaps(lr1))
    
    def test_overlaps_edge_case(self):
        """Test edge case where ranges touch but don't overlap"""
        lr1 = LiveRange(variable='x', start=0, end=5)
        lr2 = LiveRange(variable='y', start=5, end=10)
        # Ranges that touch at a point do overlap
        self.assertTrue(lr1.overlaps(lr2))


class TestInterferenceGraph(unittest.TestCase):
    """Test InterferenceGraph class"""
    
    def test_create_graph(self):
        """Test creating an interference graph"""
        graph = InterferenceGraph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
    
    def test_add_node(self):
        """Test adding nodes to graph"""
        graph = InterferenceGraph()
        graph.add_node('x')
        graph.add_node('y')
        self.assertEqual(len(graph.nodes), 2)
        self.assertIn('x', graph.nodes)
        self.assertIn('y', graph.nodes)
    
    def test_add_edge(self):
        """Test adding edges to graph"""
        graph = InterferenceGraph()
        graph.add_edge('x', 'y')
        self.assertIn('y', graph.neighbors('x'))
        self.assertIn('x', graph.neighbors('y'))
    
    def test_degree(self):
        """Test getting node degree"""
        graph = InterferenceGraph()
        graph.add_edge('x', 'y')
        graph.add_edge('x', 'z')
        self.assertEqual(graph.degree('x'), 2)
        self.assertEqual(graph.degree('y'), 1)
        self.assertEqual(graph.degree('z'), 1)
    
    def test_remove_node(self):
        """Test removing node from graph"""
        graph = InterferenceGraph()
        graph.add_edge('x', 'y')
        graph.add_edge('x', 'z')
        graph.remove_node('x')
        
        self.assertNotIn('x', graph.nodes)
        self.assertEqual(graph.degree('y'), 0)
        self.assertEqual(graph.degree('z'), 0)


class TestRegisterAllocator(unittest.TestCase):
    """Test RegisterAllocator class"""
    
    def test_create_allocator(self):
        """Test creating register allocator"""
        allocator = RegisterAllocator(num_registers=16)
        self.assertEqual(allocator.num_registers, 16)
        self.assertEqual(len(allocator.available_registers), 14)  # 16 - 2 reserved
    
    def test_simple_allocation(self):
        """Test simple register allocation with no conflicts"""
        allocator = RegisterAllocator(num_registers=8)
        
        # Non-overlapping live ranges
        live_ranges = {
            'x': LiveRange(variable='x', start=0, end=2),
            'y': LiveRange(variable='y', start=3, end=5),
            'z': LiveRange(variable='z', start=6, end=8),
        }
        
        instructions = []
        result = allocator.allocate(instructions, live_ranges)
        
        # All should be allocated, none spilled
        self.assertEqual(len(result.spilled), 0)
        self.assertEqual(len(result.allocation), 3)
    
    def test_interference_allocation(self):
        """Test allocation with interfering variables"""
        allocator = RegisterAllocator(num_registers=8)
        
        # Overlapping live ranges
        live_ranges = {
            'x': LiveRange(variable='x', start=0, end=10),
            'y': LiveRange(variable='y', start=5, end=15),
            'z': LiveRange(variable='z', start=8, end=12),
        }
        
        instructions = []
        result = allocator.allocate(instructions, live_ranges)
        
        # All should be allocated to different registers
        self.assertEqual(len(result.spilled), 0)
        self.assertEqual(len(result.allocation), 3)
        # x, y, z should have different registers
        self.assertNotEqual(result.allocation['x'], result.allocation['y'])
        self.assertNotEqual(result.allocation['y'], result.allocation['z'])
    
    def test_spilling(self):
        """Test spilling when not enough registers"""
        # Only 2 available registers (after reserved)
        allocator = RegisterAllocator(num_registers=4)
        
        # 3 overlapping variables - must spill one
        live_ranges = {
            'x': LiveRange(variable='x', start=0, end=10),
            'y': LiveRange(variable='y', start=0, end=10),
            'z': LiveRange(variable='z', start=0, end=10),
        }
        
        instructions = []
        result = allocator.allocate(instructions, live_ranges)
        
        # Should spill at least one variable
        self.assertGreater(len(result.spilled), 0)
    
    def test_coalescing(self):
        """Test register coalescing for moves"""
        allocator = RegisterAllocator(num_registers=8)
        
        # Two variables with move between them
        live_ranges = {
            'x': LiveRange(variable='x', start=0, end=5),
            'y': LiveRange(variable='y', start=6, end=10),
        }
        
        # Move instruction
        instructions = [
            ('MOVE', 'x', 'y'),
        ]
        
        result = allocator.allocate(instructions, live_ranges)
        
        # Non-interfering variables can be coalesced
        # They might get the same register
        self.assertEqual(len(result.spilled), 0)
    
    def test_statistics(self):
        """Test allocation statistics"""
        allocator = RegisterAllocator(num_registers=8)
        
        live_ranges = {
            'x': LiveRange(variable='x', start=0, end=10),
        }
        
        instructions = []
        allocator.allocate(instructions, live_ranges)
        
        stats = allocator.get_stats()
        self.assertEqual(stats['allocations'], 1)
        self.assertIn('spills', stats)
        self.assertIn('coalesced_moves', stats)
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        allocator = RegisterAllocator(num_registers=8)
        
        live_ranges = {'x': LiveRange(variable='x', start=0, end=10)}
        instructions = []
        allocator.allocate(instructions, live_ranges)
        
        allocator.reset_stats()
        stats = allocator.get_stats()
        self.assertEqual(stats['allocations'], 0)


class TestLiveRangeComputation(unittest.TestCase):
    """Test live range computation"""
    
    def test_compute_simple_live_ranges(self):
        """Test computing live ranges for simple code"""
        instructions = [
            ('LOAD_FAST', 'x'),
            ('LOAD_FAST', 'y'),
            ('STORE_FAST', 'z'),
        ]
        
        live_ranges = compute_live_ranges(instructions)
        
        self.assertIn('x', live_ranges)
        self.assertIn('y', live_ranges)
        self.assertIn('z', live_ranges)
    
    def test_live_range_extends(self):
        """Test that live ranges extend properly"""
        instructions = [
            ('LOAD_FAST', 'x'),      # 0
            ('NOOP',),               # 1
            ('LOAD_FAST', 'x'),      # 2
        ]
        
        live_ranges = compute_live_ranges(instructions)
        
        # x used at 0 and 2, so range should cover both
        self.assertEqual(live_ranges['x'].start, 0)
        self.assertEqual(live_ranges['x'].end, 2)


class TestExtractVars(unittest.TestCase):
    """Test variable extraction from instructions"""
    
    def test_extract_load_fast(self):
        """Test extracting vars from LOAD_FAST"""
        instr = ('LOAD_FAST', 'x')
        defs, uses = _extract_vars(instr)
        
        self.assertEqual(len(defs), 0)
        self.assertEqual(uses, ['x'])
    
    def test_extract_store_fast(self):
        """Test extracting vars from STORE_FAST"""
        instr = ('STORE_FAST', 'x')
        defs, uses = _extract_vars(instr)
        
        self.assertEqual(defs, ['x'])
        self.assertEqual(len(uses), 0)
    
    def test_extract_binary_op(self):
        """Test extracting vars from binary operation"""
        instr = ('BINARY_ADD', 'z', 'x', 'y')
        defs, uses = _extract_vars(instr)
        
        self.assertEqual(defs, ['z'])
        self.assertEqual(set(uses), {'x', 'y'})


if __name__ == '__main__':
    unittest.main()
