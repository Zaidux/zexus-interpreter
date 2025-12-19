"""
Benchmark for Bytecode Caching System

Measures performance improvements from caching:
- Compilation time savings
- Cache hit rates
- Memory overhead
- Throughput improvements
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.zexus.vm.cache import BytecodeCache
from src.zexus.vm.bytecode import BytecodeBuilder
from src.zexus import zexus_ast


def benchmark_cache_performance():
    """Benchmark cache hit/miss performance"""
    print("=" * 60)
    print("BYTECODE CACHE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    cache = BytecodeCache(max_size=1000, debug=False)
    
    # Create test nodes
    nodes = [zexus_ast.IntegerLiteral(i) for i in range(100)]
    bytecodes = []
    for i in range(100):
        builder = BytecodeBuilder()
        builder.emit_constant(i)
        for j in range(10):  # Make it more realistic
            builder.emit_constant(i * 10 + j)
            builder.emit("ADD")
        bytecode = builder.build()
        bytecodes.append(bytecode)
    
    # Benchmark: First access (all misses)
    print("\n1. First access (cache misses):")
    start = time.time()
    for i, node in enumerate(nodes):
        cache.put(node, bytecodes[i])
    miss_time = time.time() - start
    print(f"   Time: {miss_time*1000:.2f}ms")
    print(f"   Rate: {len(nodes)/miss_time:.0f} operations/sec")
    
    # Benchmark: Second access (all hits)
    print("\n2. Second access (cache hits):")
    start = time.time()
    for node in nodes:
        cached = cache.get(node)
    hit_time = time.time() - start
    print(f"   Time: {hit_time*1000:.2f}ms")
    print(f"   Rate: {len(nodes)/hit_time:.0f} operations/sec")
    
    # Calculate speedup
    speedup = miss_time / hit_time if hit_time > 0 else 0
    print(f"\n3. Cache speedup: {speedup:.1f}x faster")
    
    # Statistics
    stats = cache.get_stats()
    print(f"\n4. Cache statistics:")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")
    print(f"   Memory usage: {cache.memory_usage_mb():.2f} MB")
    print(f"   Entries: {stats['total_entries']}")


def benchmark_compilation_savings():
    """Benchmark compilation time savings"""
    print("\n" + "=" * 60)
    print("COMPILATION TIME SAVINGS BENCHMARK")
    print("=" * 60)
    
    # Simulate compilation (create realistic bytecode)
    def simulate_compile(value):
        """Simulate compilation time"""
        builder = BytecodeBuilder()
        for i in range(50):  # Simulate complex compilation
            builder.emit_constant(value + i)
            builder.emit("ADD")
        time.sleep(0.0001)  # Simulate compilation overhead
        return builder.build()
    
    cache = BytecodeCache(max_size=100, debug=False)
    test_node = zexus_ast.IntegerLiteral(42)
    
    # Without cache (compile every time)
    print("\n1. Without cache (100 compilations):")
    start = time.time()
    for _ in range(100):
        bytecode = simulate_compile(42)
    no_cache_time = time.time() - start
    print(f"   Time: {no_cache_time*1000:.2f}ms")
    
    # With cache (compile once, reuse 99 times)
    print("\n2. With cache (1 compile + 99 cache hits):")
    start = time.time()
    # First compilation
    bytecode = simulate_compile(42)
    cache.put(test_node, bytecode)
    # 99 cache hits
    for _ in range(99):
        cached = cache.get(test_node)
    with_cache_time = time.time() - start
    print(f"   Time: {with_cache_time*1000:.2f}ms")
    
    # Calculate savings
    speedup = no_cache_time / with_cache_time if with_cache_time > 0 else 0
    time_saved = no_cache_time - with_cache_time
    percent_saved = (time_saved / no_cache_time * 100) if no_cache_time > 0 else 0
    
    print(f"\n3. Performance improvement:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Time saved: {time_saved*1000:.2f}ms")
    print(f"   Savings: {percent_saved:.1f}%")


def benchmark_memory_overhead():
    """Benchmark memory overhead of caching"""
    print("\n" + "=" * 60)
    print("MEMORY OVERHEAD BENCHMARK")
    print("=" * 60)
    
    cache = BytecodeCache(max_size=1000, debug=False)
    
    # Add various sizes of bytecode
    sizes = [10, 50, 100, 500]
    for size in sizes:
        node = zexus_ast.IntegerLiteral(size)
        builder = BytecodeBuilder()
        for i in range(size):
            builder.emit_constant(i)
        bytecode = builder.build()
        cache.put(node, bytecode)
        
        print(f"\n{size} instructions:")
        print(f"   Cache entries: {cache.size()}")
        print(f"   Memory usage: {cache.memory_usage_mb():.4f} MB")
        print(f"   Avg per entry: {cache.memory_usage() / cache.size():.0f} bytes")


def benchmark_eviction_performance():
    """Benchmark LRU eviction performance"""
    print("\n" + "=" * 60)
    print("LRU EVICTION PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    cache = BytecodeCache(max_size=100, debug=False)
    
    # Fill cache
    nodes = [zexus_ast.IntegerLiteral(i) for i in range(100)]
    for i, node in enumerate(nodes):
        builder = BytecodeBuilder()
        builder.emit_constant(i)
        bytecode = builder.build()
        cache.put(node, bytecode)
    
    print(f"\n1. Cache filled: {cache.size()} entries")
    
    # Add more to trigger eviction
    print("\n2. Adding 50 more entries (triggers eviction):")
    start = time.time()
    for i in range(100, 150):
        node = zexus_ast.IntegerLiteral(i)
        builder = BytecodeBuilder()
        builder.emit_constant(i)
        bytecode = builder.build()
        cache.put(node, bytecode)
    eviction_time = time.time() - start
    
    stats = cache.get_stats()
    print(f"   Time: {eviction_time*1000:.2f}ms")
    print(f"   Evictions: {stats['evictions']}")
    print(f"   Final size: {cache.size()}")
    print(f"   Time per eviction: {eviction_time/stats['evictions']*1000:.2f}ms")


def benchmark_realistic_workload():
    """Benchmark realistic workload with mixed hits/misses"""
    print("\n" + "=" * 60)
    print("REALISTIC WORKLOAD BENCHMARK")
    print("=" * 60)
    
    cache = BytecodeCache(max_size=50, debug=False)
    
    # Create bytecodes for common patterns
    common_nodes = [zexus_ast.IntegerLiteral(i) for i in range(10)]
    for i, node in enumerate(common_nodes):
        builder = BytecodeBuilder()
        builder.emit_constant(i)
        bytecode = builder.build()
        cache.put(node, bytecode)
    
    print("\n1. Workload: 80% hits, 20% misses")
    print("   Simulating 1000 operations...")
    
    start = time.time()
    hits = 0
    misses = 0
    
    for i in range(1000):
        # 80% chance of accessing common node (cache hit)
        # 20% chance of new node (cache miss)
        if i % 5 == 0:  # 20% new
            node = zexus_ast.IntegerLiteral(1000 + i)
            builder = BytecodeBuilder()
            builder.emit_constant(1000 + i)
            bytecode = builder.build()
            cache.put(node, bytecode)
            misses += 1
        else:  # 80% common
            node = common_nodes[i % 10]
            cached = cache.get(node)
            hits += 1
    
    workload_time = time.time() - start
    
    stats = cache.get_stats()
    print(f"\n2. Results:")
    print(f"   Time: {workload_time*1000:.2f}ms")
    print(f"   Operations/sec: {1000/workload_time:.0f}")
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")
    print(f"   Cache size: {cache.size()}")
    print(f"   Evictions: {stats['evictions']}")


if __name__ == '__main__':
    print("\n🚀 Starting Bytecode Cache Benchmarks...\n")
    
    benchmark_cache_performance()
    benchmark_compilation_savings()
    benchmark_memory_overhead()
    benchmark_eviction_performance()
    benchmark_realistic_workload()
    
    print("\n" + "=" * 60)
    print("✅ All benchmarks complete!")
    print("=" * 60)
