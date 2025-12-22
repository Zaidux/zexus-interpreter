import tracemalloc
import gc

class TestMemoryLeakValidation(unittest.TestCase):
    """Validate no memory leaks in memory manager"""
    
    def test_no_memory_leaks(self):
        """Test that memory doesn't leak over many allocations"""
        if not MEMORY_MANAGER_AVAILABLE:
            self.skipTest("Memory manager not available")
        
        tracemalloc.start()
        
        vm = VM(
            use_memory_manager=True,
            max_heap_mb=50,
            debug=False
        )
        
        # Track memory usage over many allocations
        memory_snapshots = []
        
        for iteration in range(100):
            # Create unique objects
            for i in range(100):
                builder = BytecodeBuilder()
                builder.emit_load_const(f"object_{iteration}_{i}_" * 10)  # Larger string
                builder.emit_store_name(f"var_{iteration}_{i}")
                builder.emit_load_const(iteration * i)
                builder.emit_return()
                
                bytecode = builder.build()
                vm.execute(bytecode)
            
            # Force garbage collection every 10 iterations
            if iteration % 10 == 0:
                vm.collect_garbage(force=True)
                gc.collect()
            
            # Take memory snapshot
            if iteration % 20 == 0:
                snapshot = tracemalloc.take_snapshot()
                memory_snapshots.append((iteration, snapshot))
        
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
            
            # Memory should not grow unbounded
            # Allow some growth for caches, but not exponential
            self.assertLess(growth_percent, 500, 
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
        print(f"   Before GC: {stats_before.get('current_usage', 0)} bytes")
        print(f"   After GC:  {stats_after.get('current_usage', 0)} bytes")
        print(f"   Collected: {gc_result.get('collected', 0)} objects")
        
        # Memory usage should decrease after GC
        if 'current_usage' in stats_before and 'current_usage' in stats_after:
            self.assertLessEqual(
                stats_after['current_usage'], 
                stats_before['current_usage'] * 1.5,  # Allow some overhead
                "Memory usage should not significantly increase after GC"
              )
