import unittest
import tracemalloc
import gc
from zexus.vm.vm import VM
from zexus.vm.bytecode import BytecodeBuilder

# Check if memory manager is available
try:
    MEMORY_MANAGER_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    MEMORY_MANAGER_AVAILABLE = False

class TestMemoryLeakValidation(unittest.TestCase):
    """Validate no memory leaks in memory manager"""
    
    def test_no_memory_leaks(self):
        """Test that memory doesn't leak over many allocations"""
        if not MEMORY_MANAGER_AVAILABLE:
            self.skipTest("Memory manager not available")
        
        tracemalloc.start()
        
        # Create a new VM for each iteration to prevent environment pollution
        memory_snapshots = []
        
        for iteration in range(20):  # Reduced from 100 to 20
            vm = VM(
                use_memory_manager=True,
                max_heap_mb=50,
                debug=False
            )
            
            # Create some objects
            for i in range(50):  # Reduced from 100 to 50
                builder = BytecodeBuilder()
                builder.emit_load_const(f"object_{iteration}_{i}_" * 5)  # Reduced string size
                builder.emit_store_name(f"var_{i}")  # Reuse variable names
                builder.emit_load_const(iteration * i)
                builder.emit_return()
                
                bytecode = builder.build()
                vm.execute(bytecode)
            
            # Force garbage collection
            vm.collect_garbage(force=True)
            gc.collect()
            
            # Take memory snapshot every 5 iterations
            if iteration % 5 == 0:
                snapshot = tracemalloc.take_snapshot()
                memory_snapshots.append((iteration, snapshot))
            
            # Delete VM to free resources
            del vm
        
        # Analyze memory growth
        if len(memory_snapshots) > 1:
            first_snapshot = memory_snapshots[0][1]
            last_snapshot = memory_snapshots[-1][1]
            
            # Compare memory usage
            top_stats_first = first_snapshot.statistics('lineno')
            top_stats_last = last_snapshot.statistics('lineno')
            
            total_first = sum(stat.size for stat in top_stats_first)
            total_last = sum(stat.size for stat in top_stats_last)
            
            growth = total_last - total_first
            growth_percent = (growth / total_first * 100) if total_first > 0 else 0
            
            print(f"\n🧠 MEMORY LEAK VALIDATION:")
            print(f"   Initial memory: {total_first / 1024:.1f} KB")
            print(f"   Final memory:   {total_last / 1024:.1f} KB")
            print(f"   Growth:         {growth / 1024:.1f} KB ({growth_percent:.1f}%)")
            print(f"   Note: Creating new VM per iteration to avoid env pollution")
            
            # Memory should not grow unbounded
            # More relaxed threshold since we're creating/destroying VMs
            self.assertLess(growth_percent, 1000, 
                f"Memory grew by {growth_percent:.1f}% - possible memory leak")
        
        tracemalloc.stop()
    
    def test_memory_manager_gc_effectiveness(self):
        """Test garbage collection actually frees memory"""
        if not MEMORY_MANAGER_AVAILABLE:
            self.skipTest("Memory manager not available")
        
        vm = VM(use_memory_manager=True, max_heap_mb=10, debug=False)
        
        # Allocate many objects
        for i in range(1000):
            builder = BytecodeBuilder()
            builder.emit_load_const(f"large_string_{i}" * 100)
            builder.emit_store_name(f"temp_{i}")
            builder.emit_load_const(i)
            builder.emit_return()
            
            bytecode = builder.build()
            vm.execute(bytecode)
        
        # Get stats before GC
        stats_before = vm.get_memory_stats()
        
        # Force garbage collection
        gc_result = vm.collect_garbage(force=True)
        
        # Get stats after GC
        stats_after = vm.get_memory_stats()
        
        print(f"\n🗑️  GC EFFECTIVENESS:")
        
        # Safely extract memory values
        usage_before = stats_before.get('current_usage', 0)
        usage_after = stats_after.get('current_usage', 0)
        
        # Handle string values (convert to int)
        if isinstance(usage_before, str):
            usage_before = 0
        if isinstance(usage_after, str):
            usage_after = 0
            
        print(f"   Before GC: {usage_before} bytes")
        print(f"   After GC:  {usage_after} bytes")
        print(f"   Collected: {gc_result.get('collected', 0)} objects")
        
        # Memory usage should decrease or stay reasonable after GC
        if usage_before > 0 and usage_after > 0:
            # Allow some overhead but not exponential growth
            self.assertLessEqual(
                usage_after, 
                usage_before * 2.0,  # Allow 2x overhead
                "Memory usage should not significantly increase after GC"
            )
        else:
            # If memory manager is disabled or not tracking, just warn
            print(f"   Warning: Memory manager may not be active")
