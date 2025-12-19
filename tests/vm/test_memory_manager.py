"""
Comprehensive test suite for Zexus Memory Manager (Phase 7)

Tests heap allocation, garbage collection, memory profiling,
leak detection, and VM integration.

Author: Zexus Team
Date: December 19, 2025
"""

import unittest
import time
import sys

from src.zexus.vm.memory_manager import (
    MemoryManager, Heap, GarbageCollector,
    MemoryObject, MemoryStats, ObjectState,
    create_memory_manager
)
from src.zexus.vm.vm import VM
from src.zexus.vm.bytecode import Bytecode, Opcode


class TestHeap(unittest.TestCase):
    """Test heap allocator"""
    
    def setUp(self):
        self.heap = Heap(max_size=1024 * 1024)  # 1MB for testing
    
    def test_heap_creation(self):
        """Test creating heap"""
        self.assertIsNotNone(self.heap)
        self.assertEqual(self.heap.max_size, 1024 * 1024)
        self.assertEqual(len(self.heap.objects), 0)
    
    def test_allocate_object(self):
        """Test allocating an object"""
        obj = {"key": "value"}
        obj_id = self.heap.allocate(obj)
        
        self.assertGreaterEqual(obj_id, 0)
        self.assertEqual(len(self.heap.objects), 1)
        self.assertEqual(self.heap.stats.allocation_count, 1)
        self.assertGreater(self.heap.stats.current_usage, 0)
    
    def test_allocate_multiple(self):
        """Test allocating multiple objects"""
        obj_ids = []
        for i in range(10):
            obj_id = self.heap.allocate(f"object_{i}")
            obj_ids.append(obj_id)
        
        self.assertEqual(len(self.heap.objects), 10)
        self.assertEqual(len(set(obj_ids)), 10)  # All unique
        self.assertEqual(self.heap.stats.allocation_count, 10)
    
    def test_deallocate_object(self):
        """Test deallocating an object"""
        obj_id = self.heap.allocate("test")
        initial_usage = self.heap.stats.current_usage
        
        success = self.heap.deallocate(obj_id)
        
        self.assertTrue(success)
        self.assertEqual(len(self.heap.objects), 0)
        self.assertEqual(self.heap.stats.current_usage, 0)
        self.assertEqual(self.heap.stats.deallocation_count, 1)
    
    def test_deallocate_nonexistent(self):
        """Test deallocating nonexistent object"""
        success = self.heap.deallocate(9999)
        self.assertFalse(success)
    
    def test_get_object(self):
        """Test retrieving object by ID"""
        obj = [1, 2, 3, 4, 5]
        obj_id = self.heap.allocate(obj)
        
        retrieved = self.heap.get_object(obj_id)
        
        self.assertEqual(retrieved, obj)
    
    def test_get_nonexistent_object(self):
        """Test getting nonexistent object"""
        obj = self.heap.get_object(9999)
        self.assertIsNone(obj)
    
    def test_heap_limit(self):
        """Test heap size limit enforcement"""
        small_heap = Heap(max_size=100)
        
        # Try to allocate object larger than heap
        large_obj = "x" * 1000
        
        with self.assertRaises(MemoryError):
            small_heap.allocate(large_obj)
    
    def test_free_list_reuse(self):
        """Test that deallocated IDs are reused"""
        obj_id1 = self.heap.allocate("first")
        self.heap.deallocate(obj_id1)
        
        obj_id2 = self.heap.allocate("second")
        
        # Should reuse the freed ID
        self.assertEqual(obj_id1, obj_id2)
    
    def test_peak_usage_tracking(self):
        """Test peak memory usage tracking"""
        # Allocate some objects
        ids = [self.heap.allocate(f"obj_{i}") for i in range(5)]
        peak_after_alloc = self.heap.stats.peak_usage
        
        # Deallocate all
        for obj_id in ids:
            self.heap.deallocate(obj_id)
        
        # Peak should remain
        self.assertEqual(self.heap.stats.peak_usage, peak_after_alloc)
        self.assertGreater(self.heap.stats.peak_usage, self.heap.stats.current_usage)


class TestGarbageCollector(unittest.TestCase):
    """Test garbage collector"""
    
    def setUp(self):
        self.heap = Heap(max_size=1024 * 1024)
        self.gc = GarbageCollector(heap=self.heap, threshold=10)
    
    def test_gc_creation(self):
        """Test creating garbage collector"""
        self.assertIsNotNone(self.gc)
        self.assertEqual(self.gc.threshold, 10)
        self.assertEqual(len(self.gc.root_set), 0)
    
    def test_add_remove_root(self):
        """Test adding and removing roots"""
        obj_id = self.heap.allocate("root_object")
        
        self.gc.add_root(obj_id)
        self.assertIn(obj_id, self.gc.root_set)
        
        self.gc.remove_root(obj_id)
        self.assertNotIn(obj_id, self.gc.root_set)
    
    def test_mark_phase(self):
        """Test mark phase finds reachable objects"""
        # Allocate objects
        root_id = self.heap.allocate("root")
        other_id = self.heap.allocate("other")
        
        # Mark root as root
        self.gc.add_root(root_id)
        
        # Run mark phase
        marked = self.gc.mark_phase()
        
        self.assertIn(root_id, marked)
    
    def test_sweep_phase(self):
        """Test sweep phase collects unmarked objects"""
        # Allocate objects
        root_id = self.heap.allocate("root")
        garbage_id = self.heap.allocate("garbage")
        
        # Mark only root
        self.gc.add_root(root_id)
        marked = self.gc.mark_phase()
        
        # Sweep should collect garbage
        collected = self.gc.sweep_phase(marked)
        
        self.assertEqual(collected, 1)
        self.assertIsNone(self.heap.get_object(garbage_id))
        self.assertIsNotNone(self.heap.get_object(root_id))
    
    def test_full_gc_cycle(self):
        """Test complete garbage collection cycle"""
        # Allocate objects
        root_id = self.heap.allocate("root")
        garbage_ids = [self.heap.allocate(f"garbage_{i}") for i in range(5)]
        
        self.gc.add_root(root_id)
        
        # Force collection
        collected, gc_time = self.gc.collect(force=True)
        
        self.assertEqual(collected, 5)
        self.assertGreater(gc_time, 0)
        self.assertEqual(self.heap.stats.gc_runs, 1)
        self.assertEqual(len(self.heap.objects), 1)
    
    def test_threshold_triggering(self):
        """Test GC triggers at threshold"""
        self.gc.threshold = 5
        
        # Allocate below threshold (manually track)
        for i in range(4):
            self.gc.allocations_since_gc += 1
        
        self.assertFalse(self.gc.should_collect())
        
        # One more should trigger
        self.gc.allocations_since_gc += 1
        self.assertTrue(self.gc.should_collect())
    
    def test_auto_collection(self):
        """Test automatic collection on threshold"""
        self.gc.threshold = 3
        initial_runs = self.heap.stats.gc_runs
        
        # Allocate garbage
        for i in range(5):
            self.heap.allocate(f"garbage_{i}")
            self.gc.record_allocation()
        
        # Should have triggered GC
        self.assertGreater(self.heap.stats.gc_runs, initial_runs)
    
    def test_protected_roots(self):
        """Test that root objects are never collected"""
        root_id = self.heap.allocate("protected")
        self.gc.add_root(root_id)
        
        # Run GC multiple times
        for _ in range(3):
            self.gc.collect(force=True)
        
        # Root should still exist
        self.assertIsNotNone(self.heap.get_object(root_id))


class TestMemoryManager(unittest.TestCase):
    """Test memory manager integration"""
    
    def setUp(self):
        self.mm = MemoryManager(
            max_heap_size=1024 * 1024,
            gc_threshold=10,
            enable_profiling=True
        )
    
    def test_mm_creation(self):
        """Test creating memory manager"""
        self.assertIsNotNone(self.mm)
        self.assertIsNotNone(self.mm.heap)
        self.assertIsNotNone(self.mm.gc)
        self.assertTrue(self.mm.enable_profiling)
    
    def test_allocate_deallocate(self):
        """Test allocation and deallocation"""
        obj_id = self.mm.allocate({"data": "test"})
        self.assertGreaterEqual(obj_id, 0)
        
        obj = self.mm.get(obj_id)
        self.assertEqual(obj["data"], "test")
        
        success = self.mm.deallocate(obj_id)
        self.assertTrue(success)
        
        self.assertIsNone(self.mm.get(obj_id))
    
    def test_allocate_as_root(self):
        """Test allocating root objects"""
        obj_id = self.mm.allocate("root", root=True)
        
        # Run GC
        collected, _ = self.mm.collect_garbage(force=True)
        
        # Root should survive
        self.assertIsNotNone(self.mm.get(obj_id))
    
    def test_get_stats(self):
        """Test getting memory statistics"""
        # Allocate some objects
        for i in range(5):
            self.mm.allocate(f"object_{i}")
        
        stats = self.mm.get_stats()
        
        self.assertIn('total_allocated', stats)
        self.assertIn('current_usage', stats)
        self.assertIn('allocation_count', stats)
    
    def test_memory_report(self):
        """Test generating memory report"""
        self.mm.allocate("test")
        
        report = self.mm.get_memory_report()
        
        self.assertIn("Memory Manager Report", report)
        self.assertIn("Memory Usage", report)
        self.assertIn("Garbage Collection", report)
    
    def test_leak_detection(self):
        """Test detecting potential memory leaks"""
        # Allocate object and wait
        obj_id = self.mm.allocate("old_object")
        
        # Manually set allocation time to simulate age
        mem_obj = self.mm.heap.get_memory_object(obj_id)
        if mem_obj:
            mem_obj.allocated_at = time.time() - 70  # 70 seconds ago
        
        leaks = self.mm.detect_leaks()
        
        # Should detect as potential leak
        self.assertGreater(len(leaks), 0)
    
    def test_profiling_history(self):
        """Test allocation history tracking"""
        self.mm.allocate("obj1")
        self.mm.allocate("obj2")
        
        self.assertGreater(len(self.mm.allocation_history), 0)
    
    def test_collect_garbage_api(self):
        """Test garbage collection API"""
        # Allocate some garbage
        for i in range(5):
            self.mm.allocate(f"garbage_{i}")
        
        collected, gc_time = self.mm.collect_garbage(force=True)
        
        self.assertGreaterEqual(collected, 0)
        self.assertGreaterEqual(gc_time, 0)


class TestMemoryStats(unittest.TestCase):
    """Test memory statistics"""
    
    def test_stats_creation(self):
        """Test creating stats object"""
        stats = MemoryStats()
        self.assertEqual(stats.total_allocated, 0)
        self.assertEqual(stats.gc_runs, 0)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = MemoryStats()
        stats.total_allocated = 1000
        stats.allocation_count = 10
        
        stats_dict = stats.to_dict()
        
        self.assertIn('total_allocated', stats_dict)
        self.assertIn('allocation_count', stats_dict)


class TestVMIntegration(unittest.TestCase):
    """Test VM integration with memory manager"""
    
    def test_vm_with_memory_manager(self):
        """Test VM with memory manager enabled"""
        vm = VM(use_memory_manager=True, max_heap_mb=10)
        
        self.assertTrue(vm.use_memory_manager)
        self.assertIsNotNone(vm.memory_manager)
    
    def test_vm_without_memory_manager(self):
        """Test VM without memory manager"""
        vm = VM(use_memory_manager=False)
        
        self.assertFalse(vm.use_memory_manager)
        self.assertIsNone(vm.memory_manager)
    
    def test_vm_memory_stats(self):
        """Test getting memory stats from VM"""
        vm = VM(use_memory_manager=True)
        
        stats = vm.get_memory_stats()
        
        self.assertIn('total_allocated', stats)
    
    def test_vm_collect_garbage(self):
        """Test triggering GC from VM"""
        vm = VM(use_memory_manager=True)
        
        result = vm.collect_garbage(force=True)
        
        self.assertIn('collected', result)
        self.assertIn('gc_time', result)
    
    def test_vm_memory_report(self):
        """Test getting memory report from VM"""
        vm = VM(use_memory_manager=True)
        
        report = vm.get_memory_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("Memory Manager", report)
    
    def test_vm_bytecode_execution_with_mm(self):
        """Test bytecode execution with memory management"""
        vm = VM(use_memory_manager=True, max_heap_mb=10)
        
        # Create simple bytecode
        bytecode = Bytecode()
        bytecode.constants = [10, 20, "x"]
        bytecode.instructions = [
            ("LOAD_CONST", 0),   # Load 10
            ("LOAD_CONST", 1),   # Load 20
            ("ADD", None),        # Add
            ("STORE_NAME", 2),   # Store to x
        ]
        
        result = vm.execute(bytecode)
        
        # Check memory stats
        stats = vm.get_memory_stats()
        self.assertGreater(stats.get('allocation_count', 0), 0)
    
    def test_vm_managed_variables(self):
        """Test VM tracks managed variables"""
        vm = VM(use_memory_manager=True)
        
        bytecode = Bytecode()
        bytecode.constants = [42, "value"]
        bytecode.instructions = [
            ("LOAD_CONST", 0),
            ("STORE_NAME", 1),
        ]
        
        vm.execute(bytecode)
        
        # Should have tracked the variable
        self.assertIn("value", vm._managed_objects)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function"""
    
    def test_create_memory_manager(self):
        """Test creating memory manager with defaults"""
        mm = create_memory_manager()
        
        self.assertIsNotNone(mm)
        self.assertIsNotNone(mm.heap)
        self.assertIsNotNone(mm.gc)
    
    def test_create_with_custom_settings(self):
        """Test creating with custom settings"""
        mm = create_memory_manager(max_heap_mb=50, gc_threshold=500)
        
        self.assertEqual(mm.heap.max_size, 50 * 1024 * 1024)
        self.assertEqual(mm.gc.threshold, 500)


class TestMemoryObject(unittest.TestCase):
    """Test memory object dataclass"""
    
    def test_memory_object_creation(self):
        """Test creating memory object"""
        obj = MemoryObject(
            id=0,
            obj="test",
            size=100,
            allocated_at=time.time()
        )
        
        self.assertEqual(obj.id, 0)
        self.assertEqual(obj.state, ObjectState.ALLOCATED)
    
    def test_memory_object_hash(self):
        """Test memory object hashing"""
        obj1 = MemoryObject(0, "test", 100, time.time())
        obj2 = MemoryObject(0, "test2", 100, time.time())
        
        self.assertEqual(hash(obj1), hash(obj2))  # Same ID
    
    def test_memory_object_equality(self):
        """Test memory object equality"""
        obj1 = MemoryObject(0, "test", 100, time.time())
        obj2 = MemoryObject(0, "other", 100, time.time())
        obj3 = MemoryObject(1, "test", 100, time.time())
        
        self.assertEqual(obj1, obj2)  # Same ID
        self.assertNotEqual(obj1, obj3)  # Different ID


class TestStressScenarios(unittest.TestCase):
    """Stress testing and edge cases"""
    
    def test_many_allocations(self):
        """Test many allocations"""
        mm = MemoryManager(max_heap_size=10 * 1024 * 1024, gc_threshold=100)
        
        # Allocate 1000 objects
        obj_ids = []
        for i in range(1000):
            obj_id = mm.allocate(f"object_{i}")
            obj_ids.append(obj_id)
        
        self.assertEqual(len(obj_ids), 1000)
        stats = mm.heap.stats
        self.assertEqual(stats.allocation_count, 1000)
    
    def test_rapid_alloc_dealloc(self):
        """Test rapid allocation and deallocation"""
        mm = MemoryManager(gc_threshold=50)
        
        for _ in range(100):
            obj_id = mm.allocate("temp")
            mm.deallocate(obj_id)
        
        stats = mm.heap.stats
        self.assertEqual(stats.allocation_count, 100)
        self.assertEqual(stats.deallocation_count, 100)
    
    def test_gc_under_pressure(self):
        """Test GC under memory pressure"""
        mm = MemoryManager(max_heap_size=1024, gc_threshold=5)
        
        # Allocate until near limit
        try:
            for i in range(100):
                mm.allocate("x" * 10)
        except MemoryError:
            pass
        
        # GC should have run
        self.assertGreater(mm.heap.stats.gc_runs, 0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency calculation"""
        mm = MemoryManager()
        
        # Allocate and deallocate
        for i in range(10):
            obj_id = mm.allocate(f"object_{i}")
            mm.deallocate(obj_id)
        
        stats_dict = mm.heap.stats.to_dict()
        self.assertIn('memory_efficiency', stats_dict)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
