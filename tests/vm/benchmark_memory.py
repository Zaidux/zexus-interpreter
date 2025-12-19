"""
Memory Manager Benchmark Suite (Phase 7)

Benchmarks memory usage, garbage collection overhead, and performance
characteristics of the Zexus memory manager.

Author: Zexus Team
Date: December 19, 2025
"""

import time
import sys
import gc as python_gc
from typing import List, Dict, Any
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.zexus.vm.memory_manager import create_memory_manager, MemoryManager
from src.zexus.vm.vm import VM
from src.zexus.vm.bytecode import Bytecode


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def format_time(seconds: float) -> str:
    """Format time in appropriate unit"""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


class Benchmark:
    """Base benchmark class"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
    
    def run(self):
        """Run benchmark and collect results"""
        raise NotImplementedError
    
    def report(self):
        """Print benchmark results"""
        print(f"\n{'='*70}")
        print(f"  {self.name}")
        print(f"{'='*70}")
        for key, value in self.results.items():
            print(f"  {key:<40} {value}")


class AllocationBenchmark(Benchmark):
    """Benchmark allocation performance"""
    
    def __init__(self):
        super().__init__("Allocation Performance")
    
    def run(self):
        """Run allocation benchmark"""
        mm = create_memory_manager(max_heap_mb=100, gc_threshold=10000)
        
        # Allocate many small objects
        num_objects = 10000
        start = time.time()
        
        obj_ids = []
        for i in range(num_objects):
            obj_id = mm.allocate(f"object_{i}")
            obj_ids.append(obj_id)
        
        alloc_time = time.time() - start
        
        stats = mm.heap.stats  # Access stats directly, not through to_dict()
        
        self.results = {
            'Objects Allocated': f"{num_objects:,}",
            'Total Time': format_time(alloc_time),
            'Time per Allocation': format_time(alloc_time / num_objects),
            'Allocations per Second': f"{num_objects / alloc_time:,.0f}",
            'Memory Used': format_bytes(stats.current_usage),
            'Peak Memory': format_bytes(stats.peak_usage),
        }


class DeallocationBenchmark(Benchmark):
    """Benchmark deallocation performance"""
    
    def __init__(self):
        super().__init__("Deallocation Performance")
    
    def run(self):
        """Run deallocation benchmark"""
        mm = create_memory_manager(max_heap_mb=100, gc_threshold=10000)
        
        # Allocate objects first
        num_objects = 10000
        obj_ids = [mm.allocate(f"object_{i}") for i in range(num_objects)]
        
        # Deallocate them
        start = time.time()
        for obj_id in obj_ids:
            mm.deallocate(obj_id)
        dealloc_time = time.time() - start
        
        stats = mm.heap.stats
        memory_eff = (stats.total_freed / max(stats.total_allocated, 1))
        
        self.results = {
            'Objects Deallocated': f"{num_objects:,}",
            'Total Time': format_time(dealloc_time),
            'Time per Deallocation': format_time(dealloc_time / num_objects),
            'Deallocations per Second': f"{num_objects / dealloc_time:,.0f}",
            'Memory Efficiency': f"{memory_eff:.2%}",
        }


class GarbageCollectionBenchmark(Benchmark):
    """Benchmark garbage collection"""
    
    def __init__(self):
        super().__init__("Garbage Collection Performance")
    
    def run(self):
        """Run GC benchmark"""
        mm = create_memory_manager(max_heap_mb=100, gc_threshold=10000)
        
        # Create garbage
        num_garbage = 5000
        num_roots = 100
        
        # Allocate roots
        root_ids = [mm.allocate(f"root_{i}", root=True) for i in range(num_roots)]
        
        # Allocate garbage
        for i in range(num_garbage):
            mm.allocate(f"garbage_{i}")
        
        # Force GC
        start = time.time()
        collected, gc_time = mm.collect_garbage(force=True)
        total_time = time.time() - start
        
        self.results = {
            'Total Objects': f"{num_garbage + num_roots:,}",
            'Root Objects': f"{num_roots:,}",
            'Garbage Objects': f"{num_garbage:,}",
            'Objects Collected': f"{collected:,}",
            'GC Time': format_time(gc_time),
            'Total Time (with overhead)': format_time(total_time),
            'Collection Rate': f"{collected / gc_time:,.0f} objects/sec" if gc_time > 0 else "N/A",
            'Objects Remaining': f"{len(mm.heap.objects):,}",
        }


class MemoryUsageBenchmark(Benchmark):
    """Benchmark memory usage patterns"""
    
    def __init__(self):
        super().__init__("Memory Usage Patterns")
    
    def run(self):
        """Run memory usage benchmark"""
        mm = create_memory_manager(max_heap_mb=50, gc_threshold=1000)
        
        # Pattern: Allocate, use, deallocate in waves
        waves = 10
        objects_per_wave = 1000
        
        peak_usage = 0
        total_allocated = 0
        total_freed = 0
        
        for wave in range(waves):
            # Allocate wave
            obj_ids = []
            for i in range(objects_per_wave):
                obj_id = mm.allocate({"data": "x" * 100, "index": i})
                obj_ids.append(obj_id)
            
            stats = mm.heap.stats
            peak_usage = max(peak_usage, stats.current_usage)
            total_allocated += stats.total_allocated
            
            # Deallocate half
            for obj_id in obj_ids[:objects_per_wave // 2]:
                mm.deallocate(obj_id)
            
            total_freed += stats.total_freed
        
        final_stats = mm.heap.stats
        memory_eff = (final_stats.total_freed / max(final_stats.total_allocated, 1))
        
        self.results = {
            'Waves': f"{waves}",
            'Objects per Wave': f"{objects_per_wave:,}",
            'Peak Memory Usage': format_bytes(peak_usage),
            'Final Memory Usage': format_bytes(final_stats.current_usage),
            'Memory Efficiency': f"{memory_eff:.2%}",
            'GC Runs': f"{final_stats.gc_runs:,}",
            'Objects Collected by GC': f"{final_stats.objects_collected:,}",
        }


class VMIntegrationBenchmark(Benchmark):
    """Benchmark VM integration"""
    
    def __init__(self):
        super().__init__("VM Integration Performance")
    
    def run(self):
        """Run VM integration benchmark"""
        # Without memory manager
        vm_without = VM(use_memory_manager=False)
        
        bytecode = Bytecode()
        bytecode.constants = list(range(100)) + [f"var_{i}" for i in range(100)]
        bytecode.instructions = []
        
        # Create bytecode that allocates many variables
        for i in range(100):
            bytecode.instructions.append(("LOAD_CONST", i))
            bytecode.instructions.append(("STORE_NAME", 100 + i))
        
        # Benchmark without memory manager
        start = time.time()
        vm_without.execute(bytecode)
        time_without = time.time() - start
        
        # With memory manager
        vm_with = VM(use_memory_manager=True, max_heap_mb=10)
        
        start = time.time()
        vm_with.execute(bytecode)
        time_with = time.time() - start
        
        mm_stats = vm_with.memory_manager.heap.stats if vm_with.memory_manager else None
        
        overhead = ((time_with - time_without) / time_without * 100) if time_without > 0 else 0
        
        self.results = {
            'Variables Stored': "100",
            'Execution Time (without MM)': format_time(time_without),
            'Execution Time (with MM)': format_time(time_with),
            'Memory Manager Overhead': f"{overhead:.2f}%",
            'Objects Tracked': f"{mm_stats.allocation_count:,}" if mm_stats else "0",
            'Memory Used': format_bytes(mm_stats.current_usage) if mm_stats else "0 B",
        }


class ComparisonBenchmark(Benchmark):
    """Compare with Python's default GC"""
    
    def __init__(self):
        super().__init__("Comparison with Python GC")
    
    def run(self):
        """Compare with Python's garbage collector"""
        num_objects = 5000
        
        # Zexus Memory Manager
        mm = create_memory_manager(max_heap_mb=100, gc_threshold=10000)
        
        start = time.time()
        obj_ids = [mm.allocate({"data": f"object_{i}"}) for i in range(num_objects)]
        # Clear half (make garbage)
        for obj_id in obj_ids[:num_objects // 2]:
            mm.deallocate(obj_id)
        collected, gc_time = mm.collect_garbage(force=True)
        zexus_total = time.time() - start
        
        # Python GC
        python_gc.disable()
        
        start = time.time()
        py_objects = [{"data": f"object_{i}"} for i in range(num_objects)]
        # Clear half
        py_objects = py_objects[num_objects // 2:]
        python_gc.collect()
        python_total = time.time() - start
        
        python_gc.enable()
        
        zexus_stats = mm.heap.stats
        memory_eff = (zexus_stats.total_freed / max(zexus_stats.total_allocated, 1))
        
        speedup = (python_total / zexus_total) if zexus_total > 0 else 0
        
        self.results = {
            'Objects Created': f"{num_objects:,}",
            'Zexus Total Time': format_time(zexus_total),
            'Python Total Time': format_time(python_total),
            'Relative Performance': f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower",
            'Zexus GC Time': format_time(gc_time),
            'Zexus Objects Collected': f"{collected:,}",
            'Zexus Memory Efficiency': f"{memory_eff:.2%}",
        }


class StressBenchmark(Benchmark):
    """Stress test memory manager"""
    
    def __init__(self):
        super().__init__("Stress Test")
    
    def run(self):
        """Run stress test"""
        mm = create_memory_manager(max_heap_mb=10, gc_threshold=500)
        
        num_iterations = 1000
        objects_per_iter = 100
        
        start = time.time()
        errors = 0
        
        for i in range(num_iterations):
            try:
                # Rapid allocate/deallocate
                obj_ids = []
                for j in range(objects_per_iter):
                    obj_id = mm.allocate(f"temp_{j}")
                    obj_ids.append(obj_id)
                
                # Deallocate half randomly
                import random
                random.shuffle(obj_ids)
                for obj_id in obj_ids[:objects_per_iter // 2]:
                    mm.deallocate(obj_id)
            except Exception as e:
                errors += 1
        
        total_time = time.time() - start
        stats = mm.heap.stats
        
        self.results = {
            'Iterations': f"{num_iterations:,}",
            'Objects per Iteration': f"{objects_per_iter:,}",
            'Total Objects Created': f"{num_iterations * objects_per_iter:,}",
            'Total Time': format_time(total_time),
            'Errors': f"{errors:,}",
            'GC Runs': f"{stats.gc_runs:,}",
            'Objects Collected': f"{stats.objects_collected:,}",
            'Final Memory Usage': format_bytes(stats.current_usage),
            'Peak Memory Usage': format_bytes(stats.peak_usage),
        }


def run_all_benchmarks():
    """Run all benchmarks and display results"""
    print("\n" + "="*70)
    print("  ZEXUS MEMORY MANAGER BENCHMARK SUITE (Phase 7)")
    print("="*70)
    
    benchmarks = [
        AllocationBenchmark(),
        DeallocationBenchmark(),
        GarbageCollectionBenchmark(),
        MemoryUsageBenchmark(),
        VMIntegrationBenchmark(),
        ComparisonBenchmark(),
        StressBenchmark(),
    ]
    
    for benchmark in benchmarks:
        benchmark.run()
        benchmark.report()
    
    print("\n" + "="*70)
    print("  Benchmark Suite Complete")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_all_benchmarks()
